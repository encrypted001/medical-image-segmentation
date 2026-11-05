"""Train U-Net with patient-grouped splits and validation-only model selection."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
import random
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.dataset import SegmentationDataset, split_by_group
from src.model import UNet


def combined_bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Stable binary objective: mean BCE-with-logits plus soft Dice loss."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    soft_dice = (2 * intersection + 1.0) / (denominator + 1.0)
    dice_loss = 1.0 - soft_dice.mean()
    return 0.5 * bce + 0.5 * dice_loss


@torch.no_grad()
def batch_dice(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Threshold logits at zero and compute per-sample binary Dice."""
    predicted = (logits >= 0).to(targets.dtype)
    intersection = (predicted * targets).sum(dim=(1, 2, 3))
    denominator = predicted.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    # A pair of empty masks receives Dice 1, a useful convention for empty slices.
    return (2 * intersection + 1.0) / (denominator + 1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    source_dataset = loader.dataset
    while isinstance(source_dataset, Subset):
        source_dataset = source_dataset.dataset
    if isinstance(source_dataset, SegmentationDataset):
        source_dataset.augment = training
    model.train(training)
    loss_total = 0.0
    dice_total = 0.0
    sample_total = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = combined_bce_dice_loss(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

        count = images.shape[0]
        loss_total += loss.item() * count
        dice_total += batch_dice(logits.detach(), masks).sum().item()
        sample_total += count

    if sample_total == 0:
        raise RuntimeError("A data split is empty; check the patient-group split settings.")
    return loss_total / sample_total, dice_total / sample_total


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _load_checkpoint(path: Path, model: UNet, optimizer: torch.optim.Optimizer, device: torch.device,
                     image_mode: str, image_size: tuple[int, int], base_channels: int) -> tuple[int, float]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load resume checkpoint {path}: {exc}") from exc
    expected = {
        "image_mode": image_mode,
        "image_size": list(image_size),
        "base_channels": base_channels,
        "in_channels": model.in_channels,
    }
    differences = [
        f"{key}: checkpoint={checkpoint.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if checkpoint.get(key) != value
    ]
    if differences:
        raise ValueError("Resume checkpoint configuration mismatch: " + "; ".join(differences))
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Resume checkpoint is missing compatible model/optimizer state: {exc}") from exc
    return int(checkpoint["epoch"]) + 1, float(checkpoint["best_validation_dice"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/masks"))
    parser.add_argument("--epochs", type=int, default=30, help="Total target epochs, including resumed epochs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=(256, 256))
    parser.add_argument("--image-mode", choices=("grayscale", "rgb"), default="grayscale")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--history", type=Path, default=Path("outputs/training_history.csv"))
    parser.add_argument("--resume", type=Path, help="Resume model and optimizer state from a checkpoint")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.base_channels < 1:
        parser.error("--epochs, --batch-size, and --base-channels must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if any(dimension <= 0 for dimension in args.image_size):
        parser.error("Both --image-size dimensions must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch environment")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    loader_generator = _seed_everything(args.seed)

    try:
        dataset = SegmentationDataset(
            args.images,
            args.masks,
            image_size=tuple(args.image_size),
            image_mode=args.image_mode,
            mask_mode="binary",
        )
        # Read one item lazily to confirm file decoding, configured channels,
        # output mask shape, and binary labels before constructing the optimizer.
        sample_image, sample_mask = dataset[0]
        expected_channels = 1 if args.image_mode == "grayscale" else 3
        expected_hw = tuple(args.image_size)
        if tuple(sample_image.shape) != (expected_channels, *expected_hw):
            raise ValueError(
                f"Dataset produced image shape {tuple(sample_image.shape)}, expected "
                f"({expected_channels}, {expected_hw[0]}, {expected_hw[1]})"
            )
        if tuple(sample_mask.shape) != (1, *expected_hw):
            raise ValueError(
                f"Binary training needs mask shape (1, {expected_hw[0]}, {expected_hw[1]}), "
                f"got {tuple(sample_mask.shape)}"
            )
        if not torch.all((sample_mask == 0) | (sample_mask == 1)):
            raise ValueError("Binary masks must contain only 0 and 1 after dataset conversion")

        # Check the real dataset tensor against the selected model before
        # splitting or constructing an optimizer.
        input_channels = 1 if args.image_mode == "grayscale" else 3
        model = UNet(
            in_channels=input_channels,
            out_channels=1,
            base_channels=args.base_channels,
        ).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(sample_image.unsqueeze(0).to(device))
        if tuple(logits.shape) != (1, 1, *expected_hw):
            raise ValueError(
                f"Model returned {tuple(logits.shape)} for configured image size {expected_hw}"
            )

        train_set, validation_set, test_set = split_by_group(
            dataset,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(f"Dataset/model preflight failed; no training was started: {exc}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch, best_validation_dice = 1, -1.0
    if args.resume:
        try:
            start_epoch, best_validation_dice = _load_checkpoint(
                args.resume, model, optimizer, device, args.image_mode,
                expected_hw, args.base_channels,
            )
        except ValueError as exc:
            parser.error(f"Resume failed; no training was started: {exc}")
        if start_epoch > args.epochs:
            parser.error(
                f"Checkpoint is already at epoch {start_epoch - 1}; set --epochs above that value to continue"
            )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(
        f"Verified dataset/model compatibility. Device={device}; "
        f"groups: train={len(train_set)} slices, validation={len(validation_set)} slices, "
        f"test reserved={len(test_set)} slices.",
        flush=True,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.resume or not args.history.exists()
    history_mode = "w" if not args.resume else "a"
    fields = ["epoch", "train_loss", "train_dice", "validation_loss", "validation_dice"]
    training_started_at = datetime.now(timezone.utc)
    training_timer = monotonic()
    with args.history.open(history_mode, newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            train_loss, train_dice = run_epoch(model, train_loader, device, optimizer)
            validation_loss, validation_dice = run_epoch(model, validation_loader, device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_dice": f"{train_dice:.6f}",
                    "validation_loss": f"{validation_loss:.6f}",
                    "validation_dice": f"{validation_dice:.6f}",
                }
            )
            history_file.flush()
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss={train_loss:.4f}, Dice={train_dice:.4f} | "
                f"validation loss={validation_loss:.4f}, Dice={validation_dice:.4f}",
                flush=True,
            )

            if validation_dice > best_validation_dice:
                best_validation_dice = validation_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_validation_dice": best_validation_dice,
                        "image_mode": args.image_mode,
                        "image_size": list(expected_hw),
                        "in_channels": input_channels,
                        "out_channels": 1,
                        "base_channels": args.base_channels,
                        "seed": args.seed,
                        "train_fraction": args.train_fraction,
                        "validation_fraction": args.validation_fraction,
                        "test_fraction": args.test_fraction,
                    },
                    args.checkpoint,
                )

    training_duration_seconds = monotonic() - training_timer
    training_finished_at = datetime.now(timezone.utc)
    run_metadata_path = args.history.with_name("training_run.json")
    run_metadata_path.write_text(
        json.dumps(
            {
                "started_at_utc": training_started_at.isoformat(),
                "finished_at_utc": training_finished_at.isoformat(),
                "training_duration_seconds": round(training_duration_seconds, 3),
                "training_duration_hms": str(timedelta(seconds=round(training_duration_seconds))),
                "epochs_completed_this_run": args.epochs - start_epoch + 1,
                "resumed_from_checkpoint": str(args.resume) if args.resume else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Training finished. Duration: {training_duration_seconds:.1f} seconds")
    print(f"Best validation Dice: {best_validation_dice:.4f}")
    print(f"Best checkpoint: {args.checkpoint}")
    print(f"Epoch history: {args.history}")
    print(f"Run metadata: {run_metadata_path}")
    print("The test subset was reserved and was not used for model selection.")


if __name__ == "__main__":
    main()
