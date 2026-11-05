"""Predict a binary mask for one MRI image using the saved U-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from src.inference import load_model, predict_image, validate_image_extension


def predict(
    image_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, Path]:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    validate_image_extension(image_path.name)
    model, checkpoint = load_model(checkpoint_path, device)
    try:
        with Image.open(image_path) as source:
            original = source.copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Input is not a readable image ({image_path}): {exc}") from exc

    mask, overlay = predict_image(original, model, checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{image_path.stem}_pred_mask.png"
    overlay_path = output_dir / f"{image_path.stem}_overlay.png"
    mask.save(mask_path)
    overlay.save(overlay_path)
    print(f"Input image: {image_path} ({original.width}x{original.height})")
    print(f"Predicted foreground pixels: {mask.histogram()[255]}")
    print(f"Saved predicted mask: {mask_path}")
    print(f"Saved visualization overlay: {overlay_path}")
    print("This model output is an image segmentation prediction, not a medical diagnosis.")
    return mask_path, overlay_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="MRI image path")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    try:
        predict(args.image, args.checkpoint, args.output_dir, device)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(f"Prediction could not be completed: {exc}")


if __name__ == "__main__":
    main()
