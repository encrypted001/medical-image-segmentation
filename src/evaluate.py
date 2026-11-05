"""Evaluate a saved binary U-Net on the held-out patient-group test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from src.dataset import SegmentationDataset, split_by_group
from src.model import UNet


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {path} does not contain a model_state_dict")
    required = ("image_mode", "image_size", "in_channels", "out_channels", "base_channels")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing model configuration fields: {', '.join(missing)}")
    if checkpoint["out_channels"] != 1:
        raise ValueError("This evaluator supports binary checkpoints with exactly one output channel")
    return checkpoint


def _metric_counts(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int, int, int]:
    """Return Dice, IoU, intersection, predicted-positive, and target-positive counts.

    A pixel is foreground when sigmoid(logit) >= 0.5 (equivalently logit >= 0).
    If both prediction and ground truth are empty, Dice and IoU are defined as
    1.0. If only one is empty, the corresponding score is 0.0.
    """
    prediction = logits >= 0
    truth = target >= 0.5
    intersection = int((prediction & truth).sum().item())
    predicted_positive = int(prediction.sum().item())
    target_positive = int(truth.sum().item())
    dice_denominator = predicted_positive + target_positive
    union = predicted_positive + target_positive - intersection
    dice = 1.0 if dice_denominator == 0 else 2.0 * intersection / dice_denominator
    iou = 1.0 if union == 0 else intersection / union
    return dice, iou, intersection, predicted_positive, target_positive


def evaluate(
    image_dir: Path,
    mask_dir: Path,
    checkpoint_path: Path,
    output_csv: Path,
    device: torch.device,
) -> tuple[float, float]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    image_size = tuple(int(value) for value in checkpoint["image_size"])
    dataset = SegmentationDataset(
        image_dir,
        mask_dir,
        image_size=image_size,
        image_mode=checkpoint["image_mode"],
        mask_mode="binary",
    )
    _, _, test_subset = split_by_group(
        dataset,
        train_fraction=float(checkpoint.get("train_fraction", 0.70)),
        validation_fraction=float(checkpoint.get("validation_fraction", 0.15)),
        test_fraction=float(checkpoint.get("test_fraction", 0.15)),
        seed=int(checkpoint.get("seed", 42)),
    )
    if len(test_subset) == 0:
        raise ValueError("The held-out test split is empty")

    model = UNet(
        in_channels=int(checkpoint["in_channels"]),
        out_channels=1,
        base_channels=int(checkpoint["base_channels"]),
    ).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match the configured U-Net: {exc}") from exc
    model.eval()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total_intersection = 0
    total_predicted = 0
    total_truth = 0
    total_dice = 0.0
    total_iou = 0.0
    empty_truth_count = 0
    nonempty_dice = 0.0
    nonempty_iou = 0.0
    nonempty_count = 0
    with output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["image", "mask", "dice", "iou", "ground_truth_empty", "predicted_pixels", "ground_truth_pixels"],
        )
        writer.writeheader()
        with torch.inference_mode():
            for dataset_index in test_subset.indices:
                image, mask = dataset[dataset_index]
                logits = model(image.unsqueeze(0).to(device))[0]
                dice, iou, intersection, predicted_positive, target_positive = _metric_counts(logits, mask.to(device))
                image_path, mask_path = dataset.pairs[dataset_index]
                row = {
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "dice": f"{dice:.6f}",
                    "iou": f"{iou:.6f}",
                    "ground_truth_empty": target_positive == 0,
                    "predicted_pixels": predicted_positive,
                    "ground_truth_pixels": target_positive,
                }
                writer.writerow(row)
                rows.append(row)
                total_dice += dice
                total_iou += iou
                total_intersection += intersection
                total_predicted += predicted_positive
                total_truth += target_positive
                empty_truth_count += target_positive == 0
                if target_positive > 0:
                    nonempty_dice += dice
                    nonempty_iou += iou
                    nonempty_count += 1

    macro_dice = total_dice / len(rows)
    macro_iou = total_iou / len(rows)
    micro_dice_denominator = total_predicted + total_truth
    micro_union = total_predicted + total_truth - total_intersection
    micro_dice = 1.0 if micro_dice_denominator == 0 else 2 * total_intersection / micro_dice_denominator
    micro_iou = 1.0 if micro_union == 0 else total_intersection / micro_union
    test_groups = sorted({dataset.groups[index] for index in test_subset.indices})
    print(f"Held-out test images: {len(rows)}")
    print(f"Held-out patient group(s): {', '.join(test_groups)}")
    print(f"Macro Dice: {macro_dice:.4f} | Macro IoU: {macro_iou:.4f}")
    print(f"Micro Dice: {micro_dice:.4f} | Micro IoU: {micro_iou:.4f}")
    print(f"Empty ground-truth masks: {empty_truth_count}/{len(rows)}")
    if nonempty_count:
        print(
            f"Non-empty ground-truth images ({nonempty_count}): "
            f"mean Dice={nonempty_dice / nonempty_count:.4f}, "
            f"mean IoU={nonempty_iou / nonempty_count:.4f}"
        )
    print(f"Predicted foreground pixels: {total_predicted}; ground-truth foreground pixels: {total_truth}")
    print("Empty-mask convention: both empty => Dice/IoU 1; only one empty => 0.")
    print(f"Per-image results: {output_csv}")
    return macro_dice, macro_iou


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/masks"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation_results.csv"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    try:
        evaluate(args.images, args.masks, args.checkpoint, args.output, device)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(f"Evaluation could not be completed: {exc}")


if __name__ == "__main__":
    main()
