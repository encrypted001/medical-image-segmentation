"""Inspect recursively stored image/mask pairs without changing the dataset.

Pairing follows the project convention: images and masks share a relative
path and filename stem; their supported file extensions may differ.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def _files_by_key(root: Path) -> dict[Path, list[Path]]:
    files: dict[Path, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files[path.relative_to(root).with_suffix("")].append(path)
    return dict(files)


def _mask_key(key: Path) -> Path:
    """Recognize the source dataset's documented-in-files `_mask` suffix."""
    if key.name.lower().endswith("_mask"):
        return key.with_name(key.name[:-5])
    return key


def inspect(image_root: Path, mask_root: Path, examples: int = 5) -> int:
    """Print dataset summary; return 1 if the dataset is empty or has issues."""
    if not image_root.is_dir() or not mask_root.is_dir():
        missing = [str(p) for p in (image_root, mask_root) if not p.is_dir()]
        print("Missing dataset directory: " + ", ".join(missing))
        return 1

    images = _files_by_key(image_root)
    raw_masks = _files_by_key(mask_root)
    masks: dict[Path, list[Path]] = defaultdict(list)
    for key, paths in raw_masks.items():
        masks[_mask_key(key)].extend(paths)
    masks = dict(masks)
    image_keys, mask_keys = set(images), set(masks)
    matched_keys = sorted(image_keys & mask_keys)
    unmatched_images = sorted(image_keys - mask_keys)
    unmatched_masks = sorted(mask_keys - image_keys)
    duplicate_keys = {key for key, paths in images.items() if len(paths) > 1}
    duplicate_keys |= {key for key, paths in masks.items() if len(paths) > 1}

    print(f"Supported image files: {sum(map(len, images.values()))}")
    print(f"Supported mask files: {sum(map(len, masks.values()))}")
    print(f"Matched image-mask pairs: {len(matched_keys)}")
    print(f"Unmatched images: {sum(len(images[k]) for k in unmatched_images)}")
    print(f"Unmatched masks: {sum(len(masks[k]) for k in unmatched_masks)}")

    mismatches = []
    unreadable = []
    for key in matched_keys:
        if len(images[key]) != 1 or len(masks[key]) != 1:
            print(f"Ambiguous pair for {key}: {len(images[key])} image(s), {len(masks[key])} mask(s)")
            continue
        image_path, mask_path = images[key][0], masks[key][0]
        try:
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                image_size, mask_size = image.size, mask.size
        except (OSError, UnidentifiedImageError) as exc:
            unreadable.append((image_path, mask_path, str(exc)))
            continue
        if image_size != mask_size:
            mismatches.append((image_path, image_size, mask_path, mask_size))

    print(f"Dimension-compatible pairs: {len(matched_keys) - len(mismatches) - len(unreadable) - len(duplicate_keys)}")
    print(f"Dimension mismatches: {len(mismatches)}")
    for image_path, image_size, mask_path, mask_size in mismatches:
        print(f"  {image_path} {image_size} != {mask_path} {mask_size}")
    print(f"Unreadable pairs: {len(unreadable)}")
    for image_path, mask_path, error in unreadable:
        print(f"  {image_path} / {mask_path}: {error}")

    print(f"\nFirst {min(examples, len(matched_keys))} matched path(s):")
    for key in matched_keys[:examples]:
        for image_path in images[key]:
            for mask_path in masks[key]:
                print(f"  image: {image_path}\n  mask:  {mask_path}")
    if unmatched_images:
        print("\nUnmatched image paths:")
        for key in unmatched_images:
            for path in images[key]:
                print(f"  {path}")
    if unmatched_masks:
        print("\nUnmatched mask paths:")
        for key in unmatched_masks:
            for path in masks[key]:
                print(f"  {path}")

    if not images and not masks:
        print("\nDataset is empty. Add supported images and corresponding masks.")
        return 1
    if duplicate_keys or unmatched_images or unmatched_masks or mismatches or unreadable:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/masks"))
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(inspect(args.images, args.masks, max(0, args.examples)))


if __name__ == "__main__":
    main()
