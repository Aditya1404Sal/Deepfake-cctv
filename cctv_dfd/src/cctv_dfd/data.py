"""Dataset registry + ``CroppedFaceDataset``.

Order of operations applied to each item:
    load -> enhance(mode) -> degradation_profile (None -> identity) -> tensor.

Train splits: ``degradation_profile`` is ``None``, and a
``ProfileSampler`` is used inside the training loop to draw a profile per
batch.  Eval splits: ``degradation_profile`` is fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradations import ProfileName, apply_profile
from .enhance import EnhanceMode, enhance


LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_MAP = {"real": LABEL_REAL, "fake": LABEL_FAKE, "Real": LABEL_REAL, "Fake": LABEL_FAKE}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Dataset paths relative to the cctv_dfd/ directory. Override at call time.
DATASET_REGISTRY = {
    "local": "data/processed/local",
    "ffpp": "data/processed/ffpp",
    "celebdf": "data/processed/celebdf",
    "dfdc": "data/processed/dfdc",
    "scface": "data/processed/scface",
}


def resolve_dataset_dir(name: str, root: Optional[Path] = None) -> Path:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {list(DATASET_REGISTRY)}")
    base = Path(root) if root is not None else Path.cwd()
    return base / DATASET_REGISTRY[name]


# ---------------------------------------------------------------------------
# File discovery + splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    path: Path
    label: int  # 0=Real, 1=Fake


def list_items(dataset_dir: Path) -> List[Item]:
    items: List[Item] = []
    for sub in ("Real", "real"):
        d = dataset_dir / sub
        if d.is_dir():
            for p in sorted(d.rglob("*.jpg")):
                items.append(Item(p, LABEL_REAL))
            break
    for sub in ("Fake", "fake"):
        d = dataset_dir / sub
        if d.is_dir():
            for p in sorted(d.rglob("*.jpg")):
                items.append(Item(p, LABEL_FAKE))
            break
    return items


def train_val_test_split(
    items: Sequence[Item],
    *,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 1337,
) -> Tuple[List[Item], List[Item], List[Item]]:
    """Class-stratified split. Deterministic given ``seed``."""
    rng = np.random.RandomState(seed)
    real = [it for it in items if it.label == LABEL_REAL]
    fake = [it for it in items if it.label == LABEL_FAKE]

    def _split(group: List[Item]) -> Tuple[List[Item], List[Item], List[Item]]:
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(group)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_test - n_val
        return (
            [group[i] for i in idx[:n_train]],
            [group[i] for i in idx[n_train:n_train + n_val]],
            [group[i] for i in idx[n_train + n_val:]],
        )

    train_r, val_r, test_r = _split(real)
    train_f, val_f, test_f = _split(fake)
    return train_r + train_f, val_r + val_f, test_r + test_f


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CroppedFaceDataset(Dataset):
    """Yields ``(tensor[C,H,W] in [0,1], label, path_str)``.

    Apply order: file -> enhance -> degradation -> resize -> normalize.
    """

    def __init__(
        self,
        items: Sequence[Item],
        *,
        image_size: int = 224,
        enhance_mode: EnhanceMode = "none",
        degradation_profile: Optional[ProfileName] = None,
        normalize: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        return_path: bool = True,
    ):
        self.items = list(items)
        self.image_size = image_size
        self.enhance_mode = enhance_mode
        self.degradation_profile = degradation_profile
        self.normalize = normalize
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot read image {path}")
        return img

    def _to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        if img_bgr.shape[0] != self.image_size or img_bgr.shape[1] != self.image_size:
            img_bgr = cv2.resize(img_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        if self.normalize is not None:
            mean = np.asarray(self.normalize[0], dtype=np.float32)
            std = np.asarray(self.normalize[1], dtype=np.float32)
            arr = (arr - mean) / std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = self._load(it.path)
        if self.enhance_mode != "none":
            img = enhance(img, self.enhance_mode)
        if self.degradation_profile is not None and self.degradation_profile != "clean":
            img = apply_profile(img, self.degradation_profile)
        tensor = self._to_tensor(img)
        if self.return_path:
            return tensor, it.label, str(it.path)
        return tensor, it.label


def collate_with_paths(batch):
    """DataLoader collate that keeps file paths as a list of strings."""
    tensors = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    if len(batch[0]) == 3:
        paths = [b[2] for b in batch]
        return tensors, labels, paths
    return tensors, labels


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_loaders(
    dataset_name: str,
    *,
    batch_size: int = 64,
    image_size: int = 224,
    enhance_mode: EnhanceMode = "none",
    normalize: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    num_workers: int = 2,
    seed: int = 1337,
    root: Optional[Path] = None,
):
    """Build train/val/test DataLoaders with the same enhancement applied.

    Training DataLoader uses ``degradation_profile=None`` so the train loop can
    apply a sampled profile per batch (more flexible than baking it in here).
    """
    from torch.utils.data import DataLoader

    d = resolve_dataset_dir(dataset_name, root=root)
    items = list_items(d)
    train, val, test = train_val_test_split(items, seed=seed)

    common = dict(
        image_size=image_size,
        enhance_mode=enhance_mode,
        normalize=normalize,
    )
    train_ds = CroppedFaceDataset(train, degradation_profile=None, **common)
    val_ds = CroppedFaceDataset(val, degradation_profile="clean", **common)
    test_ds = CroppedFaceDataset(test, degradation_profile="clean", **common)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=collate_with_paths, drop_last=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate_with_paths),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, collate_fn=collate_with_paths),
        "items": {"train": train, "val": val, "test": test},
    }
