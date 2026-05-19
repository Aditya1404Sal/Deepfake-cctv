"""Stage-B training: freeze backbone, train classifier head.

Reads a YAML config; expects ``backbone.name``, ``head``, ``train``, and
``dataset`` sections. Train-time CCTV degradation is sampled via
:class:`degradations.ProfileSampler`. Early stops on val AUC.

Outputs:
    results/checkpoints/<tag>.pt   (best head weights + metadata)
    results/metrics/<tag>.json     (train/val curves, best epoch)
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .data import CroppedFaceDataset, collate_with_paths, list_items, resolve_dataset_dir, train_val_test_split
from .degradations import DEFAULT_SAMPLE_WEIGHTS, PROFILE_NAMES, ProfileSampler, apply_profile
from .features import BACKBONE_SPECS, Backbone
from .head import build_head
from .seed import set_determinism

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    backbone: str = "dinov2-small"
    head_kind: str = "mlp"
    head_hidden: int = 256
    head_dropout: float = 0.3
    enhance_mode: str = "none"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    num_workers: int = 2
    seed: int = 1337
    image_size: int = 224
    sample_weights: List[float] = None  # type: ignore
    sample_profiles: List[str] = None  # type: ignore

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        cfg = yaml.safe_load(Path(path).read_text())
        flat = {}
        flat.update(cfg.get("backbone", {}))  # {name: ...}
        if "name" in flat:
            flat["backbone"] = flat.pop("name")
        flat.update(cfg.get("head", {}))  # kind / hidden / dropout
        if "kind" in flat:
            flat["head_kind"] = flat.pop("kind")
        if "hidden" in flat:
            flat["head_hidden"] = flat.pop("hidden")
        if "dropout" in flat:
            flat["head_dropout"] = flat.pop("dropout")
        flat.update(cfg.get("train", {}))
        flat.update(cfg.get("enhance", {}))
        out = cls()
        for k, v in flat.items():
            if hasattr(out, k):
                setattr(out, k, v)
        if out.sample_profiles is None:
            out.sample_profiles = list(PROFILE_NAMES)
        if out.sample_weights is None:
            out.sample_weights = list(DEFAULT_SAMPLE_WEIGHTS)
        return out


# ---------------------------------------------------------------------------
# Train-time on-the-fly degradation
# ---------------------------------------------------------------------------


def _apply_degradation_to_batch_imgs(
    imgs: torch.Tensor, sampler: ProfileSampler
) -> torch.Tensor:
    """imgs is [B, 3, H, W] in [0,1] -- apply per-item degradation profile.

    Converts via numpy/uint8 BGR, runs profile, converts back.
    """
    out = torch.empty_like(imgs)
    for i in range(imgs.shape[0]):
        arr = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # tensors are RGB; degradations expect BGR
        bgr = arr[:, :, ::-1].copy()
        prof = sampler.sample()
        if prof != "clean":
            bgr = apply_profile(bgr, prof)
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        out[i] = torch.from_numpy(rgb).permute(2, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def _run_validation(backbone: Backbone, head: nn.Module, loader: DataLoader, device: torch.device):
    head.eval()
    losses, ys, ps = [], [], []
    pos_weight = None
    with torch.no_grad():
        for batch in loader:
            x, y, _paths = batch
            feats = backbone.embed(x).to(device)
            logits = head(feats)
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device))
            losses.append(float(loss.item()))
            ys.extend(y.tolist())
            ps.extend(torch.sigmoid(logits).cpu().tolist())
    head.train()
    try:
        auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {"loss": float(np.mean(losses)), "auc": float(auc), "n": len(ys)}


def train(
    cfg: TrainConfig,
    dataset_name: str,
    tag: str,
    *,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    project_root: Optional[Path] = None,
) -> Dict:
    set_determinism(cfg.seed)
    if epochs_override is not None:
        cfg.epochs = epochs_override
    if batch_size_override is not None:
        cfg.batch_size = batch_size_override

    project_root = Path(project_root or Path.cwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log.info("using device: %s", device)

    backbone = Backbone(cfg.backbone, device=device).load()
    spec = BACKBONE_SPECS[cfg.backbone]

    items = list_items(resolve_dataset_dir(dataset_name, root=project_root))
    if not items:
        raise RuntimeError(f"No images found for dataset {dataset_name!r}.")
    train_items, val_items, _ = train_val_test_split(items, seed=cfg.seed)
    _log.info("dataset %s: train=%d val=%d", dataset_name, len(train_items), len(val_items))

    train_ds = CroppedFaceDataset(
        train_items, image_size=spec.image_size,
        enhance_mode=cfg.enhance_mode, degradation_profile=None,
    )
    val_ds = CroppedFaceDataset(
        val_items, image_size=spec.image_size,
        enhance_mode=cfg.enhance_mode, degradation_profile="clean",
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_with_paths, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate_with_paths,
    )

    head = build_head(cfg.head_kind, in_dim=spec.feature_dim,
                      hidden=cfg.head_hidden, dropout=cfg.head_dropout).to(device)
    optim = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg.epochs))

    n_real = sum(1 for it in train_items if it.label == 0)
    n_fake = sum(1 for it in train_items if it.label == 1)
    pos_weight = torch.tensor([max(1.0, n_real / max(1, n_fake))], device=device)
    _log.info("class balance: real=%d fake=%d pos_weight=%.3f", n_real, n_fake, pos_weight.item())

    sampler = ProfileSampler(names=cfg.sample_profiles, weights=cfg.sample_weights,
                             rng=random.Random(cfg.seed))

    best = {"epoch": -1, "val_auc": -1.0, "val_loss": float("inf"), "state": None}
    history: List[Dict] = []
    patience = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        head.train()
        epoch_losses = []
        for x, y, _paths in train_loader:
            x = _apply_degradation_to_batch_imgs(x, sampler)
            with torch.no_grad():
                feats = backbone.embed(x).to(device)
            logits = head(feats)
            loss = F.binary_cross_entropy_with_logits(logits, y.to(device), pos_weight=pos_weight)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_losses.append(float(loss.item()))
        sched.step()

        val_metrics = _run_validation(backbone, head, val_loader, device)
        rec = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else float("nan"),
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "lr": sched.get_last_lr()[0],
        }
        history.append(rec)
        _log.info("epoch %d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f",
                  epoch, rec["train_loss"], rec["val_loss"], rec["val_auc"])

        if not np.isnan(rec["val_auc"]) and rec["val_auc"] > best["val_auc"]:
            best = {
                "epoch": epoch,
                "val_auc": rec["val_auc"],
                "val_loss": rec["val_loss"],
                "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                _log.info("early stop at epoch %d (no val_auc improvement for %d epochs)",
                          epoch, cfg.patience)
                break

    elapsed = time.time() - t0
    if best["state"] is None:
        best["state"] = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    ckpt_path = project_root / "results" / "checkpoints" / f"{tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best["state"],
        "config": asdict(cfg),
        "best_epoch": best["epoch"],
        "best_val_auc": best["val_auc"],
        "dataset": dataset_name,
        "backbone": cfg.backbone,
        "feature_dim": spec.feature_dim,
    }, ckpt_path)

    metrics_path = project_root / "results" / "metrics" / f"{tag}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "dataset": dataset_name,
        "config": asdict(cfg),
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_auc": best["val_auc"],
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    _log.info("saved checkpoint -> %s", ckpt_path)
    _log.info("saved metrics    -> %s", metrics_path)
    return payload


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train classifier head on frozen features")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = TrainConfig.from_yaml(args.config)
    train(cfg, args.dataset, args.tag,
          epochs_override=args.epochs, batch_size_override=args.batch_size)
    return 0


if __name__ == "__main__":
    main()
