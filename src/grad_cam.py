"""Hook-based Grad-CAM for the tumor and dementia specialist classifiers.

Grad-CAM needs a backward pass, but every inference path in
`radiology_report_gui.py` runs under `@torch.no_grad()` for speed. This
module is intentionally decoupled from that path: callers pass in an
already-loaded model and a single input tensor, and `GradCAM.generate`
runs its own short-lived forward+backward pass under `torch.enable_grad()`
regardless of the caller's grad context.

Failures (including an unsupported backward op on an unusual device backend
such as DirectML) are caught and reported as `None` rather than raised, so a
Grad-CAM failure never breaks the classification result it is explaining.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last conv block before pooling for the supported specialists.

    Both `efficientnet_b3` and `mobilenet_v3_large` (the only two
    architectures ever used for the tumor/dementia specialists) expose a
    `.features` Sequential whose last element is the final
    Conv2dNormActivation block before avgpool. No per-architecture branching
    is needed as long as that holds.
    """

    if hasattr(model, "features") and len(model.features) > 0:
        return model.features[-1]
    raise ValueError(
        f"Don't know how to find a Grad-CAM target layer for {type(model).__name__}; "
        "expected a `.features` Sequential (efficientnet_b3 / mobilenet_v3_large)."
    )


class GradCAM:
    """Hook-based Grad-CAM against a single target layer of `model`."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> Optional[np.ndarray]:
        """Return a (H, W) float array in [0, 1], or None on failure.

        `input_tensor` should be a single unbatched or batch-of-1 tensor
        already on the model's device. This never raises.
        """

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(_module, _inputs, output):
            activations["value"] = output

        def backward_hook(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0]

        was_training = self.model.training
        forward_handle = None
        backward_handle = None
        try:
            self.model.eval()
            forward_handle = self.target_layer.register_forward_hook(forward_hook)
            backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

            with torch.enable_grad():
                x = input_tensor.clone().detach()
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                x.requires_grad_(True)

                self.model.zero_grad(set_to_none=True)
                output = self.model(x)
                if class_idx < 0 or class_idx >= output.shape[1]:
                    logger.warning("Grad-CAM class_idx %s out of range for output shape %s", class_idx, output.shape)
                    return None

                score = output[0, class_idx]
                score.backward()

                if "value" not in activations or "value" not in gradients:
                    logger.warning("Grad-CAM hooks did not fire; target layer may be unreachable in the forward graph")
                    return None

                activation = activations["value"].detach()[0]  # (C, h, w)
                gradient = gradients["value"].detach()[0]  # (C, h, w)

                weights = gradient.mean(dim=(1, 2))  # (C,)
                cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))  # (h, w)

                cam_min = cam.min()
                cam_max = cam.max()
                if float(cam_max - cam_min) < 1e-12:
                    logger.warning("Grad-CAM produced a flat map (max - min ~= 0); skipping heatmap")
                    return None
                cam = (cam - cam_min) / (cam_max - cam_min)

                return cam.cpu().numpy().astype(np.float32)

        except Exception:
            logger.exception("Grad-CAM generation failed")
            return None
        finally:
            if forward_handle is not None:
                forward_handle.remove()
            if backward_handle is not None:
                backward_handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)


def overlay_cam_on_image(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Upsample `cam` to `pil_image`'s resolution and alpha-blend a jet colormap over it."""

    import matplotlib

    rgb_image = pil_image.convert("RGB")
    width, height = rgb_image.size

    cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
    resized = F.interpolate(cam_tensor, size=(height, width), mode="bilinear", align_corners=False)
    resized = resized.squeeze(0).squeeze(0).clamp(0, 1).numpy()  # (H, W) in [0, 1]

    colormap = matplotlib.colormaps["jet"]
    heatmap_rgba = colormap(resized)  # (H, W, 4) floats in [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.float32)

    base_rgb = np.asarray(rgb_image, dtype=np.float32)
    blended = base_rgb * (1.0 - alpha) + heatmap_rgb * alpha
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended, mode="RGB")
