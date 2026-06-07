from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def _find_array(root: Path, names: Iterable[str], required: bool) -> np.ndarray | None:
    for name in names:
        path = root / name
        if path.exists():
            return np.load(path, mmap_mode="r")
    if required:
        raise FileNotFoundError(f"None of {tuple(names)} exists in {root}")
    return None


def _image_tensor(array: np.ndarray) -> torch.Tensor:
    sample = np.asarray(array)
    if np.issubdtype(sample.dtype, np.integer):
        sample = sample.astype(np.float32) / np.iinfo(sample.dtype).max
    else:
        sample = sample.astype(np.float32)
    if sample.ndim == 2:
        sample = sample[None, ...]
    if sample.ndim != 3 or sample.shape[0] != 1:
        raise ValueError(f"Expected [H,W] or [1,H,W], received {sample.shape}")
    return torch.from_numpy(np.array(sample, copy=True))


class DomainDataset(Dataset):
    """Memory-mapped NumPy dataset for one physical fiber state."""

    def __init__(
        self,
        root: str | Path,
        require_images: bool = False,
        require_labels: bool = False,
        require_ids: bool = False,
    ) -> None:
        self.root = Path(root)
        self.speckles = _find_array(
            self.root, ("speckles.npy", "speckles_sorted.npy"), required=True
        )
        self.images = _find_array(
            self.root, ("images.npy", "images_sorted.npy"), required=require_images
        )
        self.labels = _find_array(
            self.root, ("labels.npy", "labels_sorted.npy"), required=require_labels
        )
        self.ids = _find_array(
            self.root, ("ids.npy", "global_ids.npy", "pattern_ids.npy"), required=require_ids
        )
        for name, array in (
            ("images", self.images),
            ("labels", self.labels),
            ("ids", self.ids),
        ):
            if array is not None and len(array) != len(self.speckles):
                raise ValueError(f"{name} length does not match speckles in {self.root}")

    def __len__(self) -> int:
        return len(self.speckles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"speckle": _image_tensor(self.speckles[index])}
        if self.images is not None:
            item["image"] = _image_tensor(self.images[index])
        if self.labels is not None:
            item["label"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.ids is not None:
            item["id"] = torch.tensor(int(self.ids[index]), dtype=torch.long)
        return item

    def close(self) -> None:
        for name in ("speckles", "images", "labels", "ids"):
            array = getattr(self, name, None)
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __del__(self) -> None:
        self.close()


class PairedDomainDataset(Dataset):
    """Pair calibrated and target samples using stable DMD pattern IDs."""

    def __init__(self, source: DomainDataset, target: DomainDataset) -> None:
        if source.ids is None or target.ids is None:
            raise ValueError("Source and target datasets must both provide IDs")
        if target.labels is None:
            raise ValueError("Target labels are required for ULDA adaptation")
        source_index = {int(value): index for index, value in enumerate(source.ids)}
        target_index = {int(value): index for index, value in enumerate(target.ids)}
        shared = sorted(source_index.keys() & target_index.keys())
        if not shared:
            raise ValueError("No shared source/target pattern IDs were found")
        self.source = source
        self.target = target
        self.pairs = [(source_index[key], target_index[key]) for key in shared]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, dict[str, torch.Tensor]]:
        source_index, target_index = self.pairs[index]
        return {
            "source": self.source[source_index],
            "target": self.target[target_index],
        }
