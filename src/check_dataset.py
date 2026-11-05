"""Load a few dataset samples and print their tensor shapes and value ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset import SegmentationDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/masks"))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--image-mode", choices=("grayscale", "rgb"), default="grayscale")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    dataset = SegmentationDataset(
        args.images, args.masks, image_size=(args.height, args.width),
        image_mode=args.image_mode, mask_mode="binary"
    )
    for index in range(min(args.samples, len(dataset))):
        image, mask = dataset[index]
        image_path, mask_path = dataset.pairs[index]
        print(f"image: {image_path}")
        print(f"mask:  {mask_path}")
        print(
            f"  image tensor: shape={tuple(image.shape)}, dtype={image.dtype}, "
            f"range=[{image.min().item():.3f}, {image.max().item():.3f}]"
        )
        print(
            f"  mask tensor:  shape={tuple(mask.shape)}, dtype={mask.dtype}, "
            f"values={mask.unique().tolist()}"
        )


if __name__ == "__main__":
    main()
