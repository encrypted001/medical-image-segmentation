"""Shared checkpoint loading and single-image preprocessing/inference helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.dataset import SUPPORTED_EXTENSIONS
from src.model import UNet


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[UNet, dict]:
    """Load a compatible binary U-Net checkpoint and return its config."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a supported checkpoint dictionary")
    required = ("model_state_dict", "image_mode", "image_size", "in_channels", "out_channels", "base_channels")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(missing)}")
    if checkpoint["image_mode"] not in ("grayscale", "rgb"):
        raise ValueError(f"Unsupported checkpoint image_mode: {checkpoint['image_mode']!r}")
    if int(checkpoint["out_channels"]) != 1:
        raise ValueError("This inference workflow supports binary checkpoints with one output channel")
    try:
        height, width = (int(value) for value in checkpoint["image_size"])
        in_channels = int(checkpoint["in_channels"])
        base_channels = int(checkpoint["base_channels"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint contains invalid model dimensions") from exc
    if height <= 0 or width <= 0 or in_channels not in (1, 3) or base_channels <= 0:
        raise ValueError("Checkpoint contains unsupported image or model dimensions")
    expected_channels = 1 if checkpoint["image_mode"] == "grayscale" else 3
    if in_channels != expected_channels:
        raise ValueError(
            f"Checkpoint image mode {checkpoint['image_mode']!r} implies {expected_channels} channels, "
            f"but it declares {in_channels}"
        )

    model = UNet(in_channels=in_channels, out_channels=1, base_channels=base_channels).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Checkpoint weights do not match this U-Net: {exc}") from exc
    model.eval()
    return model, checkpoint


def predict_image(
    image: Image.Image,
    model: UNet,
    checkpoint: dict,
) -> tuple[Image.Image, Image.Image]:
    """Predict a binary mask and red overlay while retaining input dimensions.

    Preprocessing matches SegmentationDataset: trained grayscale/RGB conversion,
    bilinear resize to the checkpoint size, uint8 scaling by 255, then CHW.
    The binary output threshold is sigmoid(logit) >= 0.5 (logit >= 0).
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL image")
    original_rgb = image.convert("RGB")
    original_size = image.size
    mode = checkpoint["image_mode"]
    height, width = (int(value) for value in checkpoint["image_size"])
    prepared = image.convert("L" if mode == "grayscale" else "RGB")
    prepared = prepared.resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    if array.ndim == 2:
        tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
    if tensor.shape[1] != int(checkpoint["in_channels"]):
        raise ValueError(
            f"Preprocessing produced {tensor.shape[1]} channel(s), checkpoint expects "
            f"{checkpoint['in_channels']}"
        )

    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(tensor.to(device))[0, 0]
        mask_array = (logits >= 0).to(torch.uint8).cpu().numpy() * 255
    mask = Image.fromarray(mask_array)
    if mask.size != original_size:
        mask = mask.resize(original_size, Image.Resampling.NEAREST)

    red = Image.new("RGB", original_size, (255, 0, 0))
    tinted = Image.blend(original_rgb, red, alpha=0.40)
    overlay = Image.composite(tinted, original_rgb, mask)
    return mask, overlay


def validate_image_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension {suffix!r}; supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
