"""Image and segmentation-mask loading plus patient-grouped data splitting.

For the bundled TIFF data, images are RGB and masks are grayscale with values
0 (background) and 255 (foreground). Binary mode explicitly maps those values
to 0/1; it does not treat every arbitrary nonzero value as foreground.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, Subset


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MaskMode = Literal["binary", "multiclass"]
ImageMode = Literal["grayscale", "rgb"]


class SegmentationDataset(Dataset):
    """Load paired images and categorical masks from recursive directories.

    Images are float tensors in [0, 1], scaled from their 8-bit intensity
    values by 255. By default, images are converted to grayscale and returned
    as [1,H,W]; set ``image_mode="rgb"`` to preserve all three input channels.
    In ``binary`` mode masks are float tensors shaped [1,H,W]
    and use explicitly configured background/foreground pixel values. In
    ``multiclass`` mode masks are integer tensors shaped [H,W], preserving
    their class IDs for a multiclass loss such as CrossEntropyLoss.

    Pair keys must have the same relative path and image stem. A trailing
    ``_mask`` is stripped from mask stems (the convention in this dataset).
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: tuple[int, int] | None = (256, 256),
        *,
        image_mode: ImageMode = "grayscale",
        mask_mode: MaskMode = "binary",
        background_value: int = 0,
        foreground_values: tuple[int, ...] = (255,),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.image_mode = image_mode
        self.mask_mode = mask_mode
        self.background_value = background_value
        self.foreground_values = tuple(foreground_values)

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory does not exist: {self.mask_dir}")
        if mask_mode not in ("binary", "multiclass"):
            raise ValueError("mask_mode must be 'binary' or 'multiclass'")
        if image_mode not in ("grayscale", "rgb"):
            raise ValueError("image_mode must be 'grayscale' or 'rgb'")
        if image_size is not None and (
            len(image_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in image_size)
        ):
            raise ValueError("image_size must be a pair of positive integers (height, width)")
        if not self.foreground_values or self.background_value in self.foreground_values:
            raise ValueError("foreground_values must be non-empty and exclude background_value")

        image_files = self._collect(self.image_dir, is_mask=False)
        mask_files = self._collect(self.mask_dir, is_mask=True)
        image_keys, mask_keys = set(image_files), set(mask_files)
        unmatched_images = sorted(image_keys - mask_keys)
        unmatched_masks = sorted(mask_keys - image_keys)
        if unmatched_images or unmatched_masks:
            image_examples = [str(image_files[k]) for k in unmatched_images[:3]]
            mask_examples = [str(mask_files[k]) for k in unmatched_masks[:3]]
            raise ValueError(
                f"Mismatched image/mask pairs: {len(unmatched_images)} unmatched image(s) "
                f"and {len(unmatched_masks)} unmatched mask(s). "
                f"Image examples: {image_examples}; mask examples: {mask_examples}"
            )
        if not image_files:
            raise FileNotFoundError(
                f"No supported image-mask pairs found in {self.image_dir} and {self.mask_dir}. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        keys = sorted(image_files)
        self.pairs = [(image_files[key], mask_files[key]) for key in keys]
        self.groups = tuple(self._infer_group(image_path) for image_path, _ in self.pairs)

    @staticmethod
    def _collect(root: Path, *, is_mask: bool) -> dict[Path, Path]:
        result: dict[Path, Path] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            key = path.relative_to(root).with_suffix("")
            if is_mask and key.name.lower().endswith("_mask"):
                key = key.with_name(key.name[:-5])
            if key in result:
                raise ValueError(f"Multiple {'mask' if is_mask else 'image'} files share pair key {key}")
            result[key] = path
        return result

    def _infer_group(self, image_path: Path) -> str:
        """Infer a conservative group ID from a patient folder or numbered slices.

        Nested datasets use their first relative directory as the group. For
        flat slice datasets, a trailing _<slice number> is removed. If neither
        exists, the entire flat directory is treated as one group to avoid
        accidentally putting potentially related samples in different splits.
        """
        relative = image_path.relative_to(self.image_dir)
        if len(relative.parts) > 1:
            return relative.parts[0]
        prefix = re.sub(r"_\d+$", "", image_path.stem)
        return prefix or "."

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[index]
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("L" if self.image_mode == "grayscale" else "RGB")
            with Image.open(mask_path) as source_mask:
                raw_mask = np.asarray(source_mask)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read image/mask pair {image_path} / {mask_path}: {exc}") from exc

        if raw_mask.ndim != 2 or not np.issubdtype(raw_mask.dtype, np.integer):
            raise ValueError(
                f"Mask must be a single-channel integer label image; got shape {raw_mask.shape} "
                f"and dtype {raw_mask.dtype}: {mask_path}"
            )
        values = np.unique(raw_mask)
        if self.mask_mode == "binary":
            allowed = {self.background_value, *self.foreground_values}
            invalid = [int(value) for value in values if int(value) not in allowed]
            if invalid:
                raise ValueError(
                    f"Invalid binary mask values {invalid[:20]} in {mask_path}. "
                    f"Allowed values are {sorted(allowed)}; configure foreground_values "
                    "only after confirming the dataset encoding."
                )
        if image.size != (raw_mask.shape[1], raw_mask.shape[0]):
            raise ValueError(
                f"Image/mask dimensions differ: {image_path} is {image.size}, "
                f"but {mask_path} is {(raw_mask.shape[1], raw_mask.shape[0])}"
            )

        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            # Nearest-neighbor preserves the original categorical label values.
            mask_image = Image.fromarray(raw_mask)
            raw_mask = np.asarray(mask_image.resize((width, height), Image.Resampling.NEAREST))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if image_array.ndim == 2:
            image_tensor = torch.from_numpy(image_array.copy()).unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image_array.copy()).permute(2, 0, 1).contiguous()

        if self.mask_mode == "binary":
            foreground = np.isin(raw_mask, self.foreground_values)
            mask_tensor = torch.from_numpy(foreground.astype(np.float32, copy=False)).unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(raw_mask.astype(np.int64, copy=True))
        return image_tensor, mask_tensor


def split_by_group(
    dataset: SegmentationDataset,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split into train/validation/test subsets without separating inferred groups.

    At least three distinct subject groups are required. The bundled data has
    only one group, so a valid three-way patient-independent split requires
    additional patients and this function deliberately raises an actionable
    error instead of leaking slices across splits.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Train, validation, and test fractions must be positive and sum to 1")

    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(dataset.groups):
        grouped.setdefault(group, []).append(index)
    groups = list(grouped)
    if len(groups) < 3:
        raise ValueError(
            f"Cannot create leakage-free train/validation/test splits: found {len(groups)} "
            f"distinct subject group(s), need at least 3. Current dataset groups: {groups}. "
            "Add data from at least two more independent patients (with patient identifiers "
            "or separate patient folders); do not split slices from one patient across sets."
        )

    random.Random(seed).shuffle(groups)
    train_count = max(1, round(len(groups) * train_fraction))
    validation_count = max(1, round(len(groups) * validation_fraction))
    if train_count + validation_count >= len(groups):
        train_count = len(groups) - 2
        validation_count = 1
    train_groups = set(groups[:train_count])
    validation_groups = set(groups[train_count : train_count + validation_count])
    test_groups = set(groups[train_count + validation_count :])

    indices = [[], [], []]
    for index, group in enumerate(dataset.groups):
        target = 0 if group in train_groups else 1 if group in validation_groups else 2
        indices[target].append(index)
    return tuple(Subset(dataset, selected) for selected in indices)  # type: ignore[return-value]
