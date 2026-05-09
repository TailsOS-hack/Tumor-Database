"""Single-image hierarchical inference for the binary-router pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from src.experiment_pipeline import (
        DEMENTIA_CLASSES,
        DOMAIN_CLASSES,
        MODELS_DIR,
        TASK_DEFAULT_MODEL_PATH,
        TUMOR_CLASSES,
        build_model,
        build_transforms,
    )
except ModuleNotFoundError:
    from experiment_pipeline import (
        DEMENTIA_CLASSES,
        DOMAIN_CLASSES,
        MODELS_DIR,
        TASK_DEFAULT_MODEL_PATH,
        TUMOR_CLASSES,
        build_model,
        build_transforms,
    )


@dataclass
class HierarchicalPrediction:
    domain: str
    subtype: str
    label: str
    confidence: float
    router_confidence: float
    specialist_confidence: float
    router_checkpoint: str
    specialist_checkpoint: str


def _default_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_pipeline_checkpoint(path: Path, device):
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"{path} is not a pipeline checkpoint with model_state metadata.")

    class_names = checkpoint["class_names"]
    model = build_model(
        checkpoint.get("arch", "resnet50"),
        len(class_names),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    transform = build_transforms(train=False, image_size=int(checkpoint.get("image_size", 224)))
    return model, transform, checkpoint


def _softmax_prediction(logits, class_names: list[str]) -> tuple[str, float]:
    import torch

    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_prob, top_idx = torch.topk(probs, 1)
    return class_names[int(top_idx.item())], float(top_prob.item())


class HierarchicalInferencePipeline:
    def __init__(
        self,
        binary_checkpoint: Path = TASK_DEFAULT_MODEL_PATH["binary"],
        tumor_checkpoint: Path = TASK_DEFAULT_MODEL_PATH["tumor"],
        dementia_checkpoint: Path = TASK_DEFAULT_MODEL_PATH["dementia"],
        device: Optional[object] = None,
    ) -> None:
        self.device = device or _default_device()
        self.binary_checkpoint = Path(binary_checkpoint)
        self.tumor_checkpoint = Path(tumor_checkpoint)
        self.dementia_checkpoint = Path(dementia_checkpoint)

        self.binary_model, self.binary_transform, self.binary_meta = _load_pipeline_checkpoint(
            self.binary_checkpoint, self.device
        )
        self.tumor_model, self.tumor_transform, self.tumor_meta = _load_pipeline_checkpoint(
            self.tumor_checkpoint, self.device
        )
        self.dementia_model, self.dementia_transform, self.dementia_meta = _load_pipeline_checkpoint(
            self.dementia_checkpoint, self.device
        )

    def predict(self, image_path: Path) -> HierarchicalPrediction:
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            router_logits = self.binary_model(self.binary_transform(image).unsqueeze(0).to(self.device))
            router_classes = self.binary_meta.get("class_names", DOMAIN_CLASSES)
            domain, router_conf = _softmax_prediction(router_logits, router_classes)

            if domain == "tumor":
                logits = self.tumor_model(self.tumor_transform(image).unsqueeze(0).to(self.device))
                specialist_classes = self.tumor_meta.get("class_names", TUMOR_CLASSES)
                subtype, specialist_conf = _softmax_prediction(logits, specialist_classes)
                specialist_checkpoint = self.tumor_checkpoint
            else:
                logits = self.dementia_model(self.dementia_transform(image).unsqueeze(0).to(self.device))
                specialist_classes = self.dementia_meta.get("class_names", DEMENTIA_CLASSES)
                subtype, specialist_conf = _softmax_prediction(logits, specialist_classes)
                specialist_checkpoint = self.dementia_checkpoint

        return HierarchicalPrediction(
            domain=domain,
            subtype=subtype,
            label=f"{domain}_{subtype}",
            confidence=router_conf * specialist_conf,
            router_confidence=router_conf,
            specialist_confidence=specialist_conf,
            router_checkpoint=str(self.binary_checkpoint),
            specialist_checkpoint=str(specialist_checkpoint),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="MRI image path")
    parser.add_argument("--binary-checkpoint", default=str(MODELS_DIR / "binary_router.pt"))
    parser.add_argument("--tumor-checkpoint", default=str(MODELS_DIR / "brain_tumor_classifier.pt"))
    parser.add_argument("--dementia-checkpoint", default=str(MODELS_DIR / "alzheimers_classifier.pt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = HierarchicalInferencePipeline(
        binary_checkpoint=Path(args.binary_checkpoint),
        tumor_checkpoint=Path(args.tumor_checkpoint),
        dementia_checkpoint=Path(args.dementia_checkpoint),
    )
    prediction = pipeline.predict(Path(args.image))
    print(json.dumps(asdict(prediction), indent=2))


if __name__ == "__main__":
    main()
