"""Grad-CAM for TwoStageResNet.

Hooks into the last ResNet block (``backbone.layer4[-1]``) and back-propagates
through any output head to produce a spatial heatmap highlighting which image
regions drove that prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM wrapper for TwoStageResNet.

    Usage::

        gcam = GradCAM(model)
        heatmaps, classes = gcam.compute(image_tensor, target_key="AdoptionSpeed")
        gcam.remove_hooks()

    Or as a context manager::

        with GradCAM(model) as gcam:
            heatmaps, classes = gcam.compute(image_tensor)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        """
        Args:
            model: TwoStageResNet instance (extract_features=False).
            target_layer: layer to hook.  Defaults to model.backbone.layer4[-1].
        """
        self.model = model
        self.target_layer = target_layer or model.backbone.layer4[-1]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles: list = []

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        def _fwd(module, input, output):
            self._activations = output

        def _bwd(module, grad_in, grad_out):
            self._gradients = grad_out[0]

        self._handles = [
            self.target_layer.register_forward_hook(_fwd),
            self.target_layer.register_full_backward_hook(_bwd),
        ]

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute(
        self,
        image_tensor: torch.Tensor,
        target_key: str = "AdoptionSpeed",
        target_class: int | torch.Tensor | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """Compute Grad-CAM heatmaps.

        Args:
            image_tensor: float tensor of shape (B, C, H, W).
            target_key: which output head to explain (e.g. ``"AdoptionSpeed"``
                        or ``"Type"``).
            target_class: class index to explain per sample.  ``None`` uses the
                          predicted (argmax) class.  A single int is broadcast
                          to the whole batch.

        Returns:
            heatmaps: numpy array (B, H, W), values in [0, 1].
            predicted_classes: list of int, one per sample.
        """
        self._register_hooks()
        self.model.eval()

        h, w = image_tensor.shape[-2:]

        with torch.enable_grad():
            x = image_tensor.detach().requires_grad_(False)
            outputs = self.model(x)
            logits = outputs[target_key] if isinstance(outputs, dict) else outputs

            if target_class is None:
                cls_idx = logits.argmax(dim=1)
            elif isinstance(target_class, int):
                cls_idx = torch.full(
                    (image_tensor.size(0),), target_class,
                    dtype=torch.long, device=logits.device,
                )
            else:
                cls_idx = target_class.to(logits.device)

            scores = logits[torch.arange(image_tensor.size(0), device=logits.device), cls_idx]
            self.model.zero_grad()
            scores.sum().backward()

        self.remove_hooks()

        # weights: global-average-pooled gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1))      # (B, H', W')

        # Upsample to input resolution
        cam = F.interpolate(
            cam.unsqueeze(1).float(),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)                                                 # (B, H, W)

        # Per-sample min-max normalisation
        flat = cam.flatten(1)
        cam_min = flat.min(dim=1).values.view(-1, 1, 1)
        cam_max = flat.max(dim=1).values.view(-1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu().numpy(), cls_idx.cpu().tolist()


def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_key: str = "AdoptionSpeed",
    target_class: int | None = None,
    target_layer: nn.Module | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Functional wrapper around :class:`GradCAM`.

    Registers hooks, computes heatmaps, and removes hooks in one call.
    Prefer this for one-shot usage; use :class:`GradCAM` directly when
    explaining multiple batches to avoid repeated hook registration.
    """
    with GradCAM(model, target_layer=target_layer) as gcam:
        return gcam.compute(image_tensor, target_key=target_key, target_class=target_class)
