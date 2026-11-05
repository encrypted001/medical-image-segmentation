"""Create a labeled side-by-side PNG of an MRI and its segmentation prediction."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from src.inference import load_model, predict_image, validate_image_extension


def make_visualization(
    image_path: Path,
    checkpoint_path: Path,
    output_path: Path,
    device: torch.device,
    ground_truth_path: Path | None = None,
    show: bool = False,
) -> Path:
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    validate_image_extension(image_path.name)
    model, checkpoint = load_model(checkpoint_path, device)
    try:
        with Image.open(image_path) as opened:
            source = opened.copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read image {image_path}: {exc}") from exc

    predicted_mask, overlay = predict_image(source, model, checkpoint)
    ground_truth = None
    if ground_truth_path is not None:
        if not ground_truth_path.is_file():
            raise FileNotFoundError(f"Ground-truth mask not found: {ground_truth_path}")
        try:
            with Image.open(ground_truth_path) as opened_mask:
                ground_truth = opened_mask.copy()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read ground-truth mask {ground_truth_path}: {exc}") from exc
        if ground_truth.mode not in ("1", "L", "I", "I;16", "P"):
            raise ValueError("Ground-truth mask must be a single-channel label image")
        ground_truth = ground_truth.convert("L")
        if ground_truth.size != source.size:
            ground_truth = ground_truth.resize(source.size, Image.Resampling.NEAREST)
        ground_truth = ground_truth.point(lambda value: 255 if value >= 128 else 0)

    canvas = compose_visualization(source, predicted_mask, overlay, ground_truth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    print(f"Saved visualization: {output_path}")
    if show:
        _open_in_viewer(output_path)
    return output_path


def compose_visualization(
    source: Image.Image,
    predicted_mask: Image.Image,
    overlay: Image.Image,
    ground_truth: Image.Image | None = None,
) -> Image.Image:
    """Compose the same labeled panel used by the CLI and Streamlit interface."""
    panels: list[tuple[str, Image.Image]] = [
        ("MRI image", source.convert("RGB")),
        ("Predicted mask", predicted_mask.convert("RGB")),
        ("Prediction overlay", overlay),
    ]
    if ground_truth is not None:
        panels.insert(1, ("Ground-truth mask", ground_truth.convert("RGB")))

    panel_width = max(panel.width for _, panel in panels)
    panel_height = max(panel.height for _, panel in panels)
    label_height = 36
    gap = 12
    canvas = Image.new(
        "RGB",
        (len(panels) * panel_width + (len(panels) - 1) * gap, panel_height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * (panel_width + gap)
        draw.text((x + 4, 10), label, fill="black")
        fitted = ImageOps.contain(panel.convert("RGB"), (panel_width, panel_height))
        canvas.paste(fitted, (x, label_height))

    return canvas


def _open_in_viewer(path: Path) -> None:
    """Ask the OS to open a saved image, without making viewer failure fatal."""
    absolute_path = str(path.resolve())
    try:
        if os.name == "nt":
            os.startfile(absolute_path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", absolute_path])
        else:
            subprocess.Popen(["xdg-open", absolute_path])
        print(f"Opening visualization: {absolute_path}")
    except (OSError, AttributeError) as exc:
        print(f"Could not launch an image viewer: {exc}")
        print(f"The visualization is saved; open it manually: {absolute_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="MRI image path")
    parser.add_argument("--mask", type=Path, help="Optional ground-truth mask path")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("outputs/segmentation_visualization.png"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--show", action="store_true", help="Open the saved visualization in the default image viewer")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    try:
        make_visualization(args.image, args.checkpoint, args.output, device, args.mask, args.show)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(f"Visualization could not be created: {exc}")


if __name__ == "__main__":
    main()
