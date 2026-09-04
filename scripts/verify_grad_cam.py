#!/usr/bin/env python3
"""Headless sanity check for src/grad_cam.py.

Builds fresh (random-init, offline-safe) copies of the two specialist
architectures used by the GUI, runs Grad-CAM against real sample images from
data/, and checks the mechanics that matter: output shape/range, hook
cleanup across repeated calls, and graceful failure on a bad input. This does
not (and cannot) validate that a heatmap is clinically meaningful -- the
checkpoints loaded here are randomly initialized, not the trained models in
models/*.pt (which are Git LFS pointers in a fresh checkout, not weights).

Usage:
    python3 scripts/verify_grad_cam.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.radiology_report_gui import (  # noqa: E402
    ALZ_CLASSES_4,
    TUMOR_CLASSES_4,
    build_alzheimers_model,
    build_tumor_model,
    get_alzheimers_transform,
    get_tumor_transform,
)
from src.grad_cam import GradCAM, get_target_layer, overlay_cam_on_image  # noqa: E402


def print_step(message: str) -> None:
    print(f"[verify-grad-cam] {message}", flush=True)


def load_pretrained_backbone(model, torchvision_ctor, weights_enum) -> bool:
    """Best-effort: copy ImageNet-pretrained `.features` weights onto `model`.

    `build_tumor_model`/`build_alzheimers_model` construct with `weights=None`
    (the GUI always loads a trained checkpoint on top, so a pretrained
    backbone would just be overwritten there). For this standalone check,
    though, a fully random-init backbone tends to produce a degenerate,
    all-zero Grad-CAM map after the ReLU in the CAM formula (there's no
    class-discriminative signal yet) -- correctly triggering grad_cam.py's
    flat-map safeguard, but useless for exercising the non-degenerate path.
    Loading real backbone weights (classifier head stays untrained/random,
    which is fine -- only classification accuracy would be affected, not
    Grad-CAM mechanics) gives a meaningful check. Falls back to random-init
    if offline.
    """

    try:
        pretrained = torchvision_ctor(weights=weights_enum)
        model.features.load_state_dict(pretrained.features.state_dict())
        return True
    except Exception as e:
        print_step(f"Could not fetch pretrained backbone weights ({e}); continuing with random-init backbone.")
        return False


def find_sample_images(directory: Path, limit: int = 2) -> list[Path]:
    images = sorted(directory.glob("*.jpg"))[:limit]
    if not images:
        raise SystemExit(f"No sample images found under {directory}")
    return images


def check_specialist(name: str, model, transform, class_names: list[str], image_dir: Path, output_dir: Path) -> None:
    import torch
    from PIL import Image

    print_step(f"Checking {name} against images in {image_dir}")
    target_layer = get_target_layer(model)
    images = find_sample_images(image_dir)

    for image_path in images:
        pil_image = Image.open(image_path).convert("RGB")
        tensor = transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            predicted_idx = int(torch.argmax(logits, dim=1).item())

        # Hooks must be empty before the call.
        assert not target_layer._forward_hooks, f"{name}: stale forward hook before generate()"
        assert not target_layer._backward_hooks, f"{name}: stale backward hook before generate()"

        cam = GradCAM(model, target_layer).generate(tensor, predicted_idx)

        # Hooks must be empty again immediately after, whether generate() succeeded or not.
        assert not target_layer._forward_hooks, f"{name}: forward hook leaked after generate()"
        assert not target_layer._backward_hooks, f"{name}: backward hook leaked after generate()"

        assert cam is not None, f"{name}: generate() unexpectedly returned None for {image_path.name}"
        assert cam.ndim == 2, f"{name}: expected a 2D CAM, got shape {cam.shape}"
        assert 2 <= cam.shape[0] <= 64 and 2 <= cam.shape[1] <= 64, (
            f"{name}: CAM spatial size {cam.shape} looks implausible for a 224x224 input"
        )
        assert cam.min() >= 0.0 - 1e-6 and cam.max() <= 1.0 + 1e-6, f"{name}: CAM values out of [0,1]: {cam.min()}..{cam.max()}"

        overlay = overlay_cam_on_image(pil_image, cam)
        assert overlay.size == pil_image.size, (
            f"{name}: overlay size {overlay.size} does not match source image size {pil_image.size}"
        )

        out_path = output_dir / f"{name}_{image_path.stem}_predicted_{class_names[predicted_idx]}.png"
        overlay.save(out_path)
        print_step(f"  {image_path.name} -> predicted={class_names[predicted_idx]}, saved {out_path.name}")

    # Repeated-call hook-leak check (simulates repeated GUI "Analyze" clicks).
    pil_image = Image.open(images[0]).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0)
    for _ in range(20):
        GradCAM(model, target_layer).generate(tensor, 0)
    assert not target_layer._forward_hooks, f"{name}: forward hooks leaked across repeated calls"
    assert not target_layer._backward_hooks, f"{name}: backward hooks leaked across repeated calls"
    print_step(f"  {name}: no hook leakage across 20 repeated generate() calls")

    # Graceful-failure check: an out-of-range class index must return None, not raise.
    out_of_range = GradCAM(model, target_layer).generate(tensor, len(class_names) + 5)
    assert out_of_range is None, f"{name}: expected None for an out-of-range class_idx, got {out_of_range!r}"
    print_step(f"  {name}: out-of-range class_idx returns None instead of raising")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to save overlay PNGs for inspection.")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="grad_cam_verify_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print_step(f"Writing overlay PNGs to {output_dir}")

    from torchvision.models import (
        EfficientNet_B3_Weights,
        MobileNet_V3_Large_Weights,
        efficientnet_b3,
        mobilenet_v3_large,
    )

    tumor_model = build_tumor_model("efficientnet_b3", len(TUMOR_CLASSES_4))
    load_pretrained_backbone(tumor_model, efficientnet_b3, EfficientNet_B3_Weights.DEFAULT)
    tumor_model.eval()
    check_specialist(
        "tumor",
        tumor_model,
        get_tumor_transform(),
        TUMOR_CLASSES_4,
        PROJECT_ROOT / "data" / "brain_tumor" / "Testing" / "glioma",
        output_dir,
    )

    alz_model = build_alzheimers_model("mobilenet_v3_large", len(ALZ_CLASSES_4))
    load_pretrained_backbone(alz_model, mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT)
    alz_model.eval()
    check_specialist(
        "dementia",
        alz_model,
        get_alzheimers_transform(),
        ALZ_CLASSES_4,
        PROJECT_ROOT / "data" / "alzheimers" / "MildDemented",
        output_dir,
    )

    print_step("All checks passed.")
    print_step(
        "Note: these models use an ImageNet-pretrained backbone with an untrained classifier head "
        "(models/*.pt are Git LFS pointers in this checkout, not the real trained weights), so the "
        "predicted labels and heatmap focus are not clinically meaningful -- this only validates "
        "Grad-CAM's mechanics (shapes, value ranges, hook cleanup, graceful failure)."
    )


if __name__ == "__main__":
    main()
