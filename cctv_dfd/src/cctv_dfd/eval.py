"""Degradation-stratified evaluation.

Produces a metrics matrix indexed by ``(dataset, degradation_profile,
enhance_mode)``. Each cell reports accuracy, precision, recall, F1, AUC-ROC,
EER. Writes:

    results/metrics/<tag>_eval.json
    results/metrics/<tag>_eval.md   (markdown table for the report)
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .data import (
    CroppedFaceDataset,
    collate_with_paths,
    list_items,
    resolve_dataset_dir,
    train_val_test_split,
)
from .degradations import PROFILE_NAMES
from .enhance import ENHANCE_MODES
from .features import BACKBONE_SPECS, Backbone
from .head import build_head
from .seed import set_determinism
from .train import TrainConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _eer(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def _safe(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _binary_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = [1 if p >= threshold else 0 for p in y_score]
    return {
        "acc": _safe(accuracy_score, y_true, y_pred),
        "precision": _safe(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, zero_division=0),
        "f1": _safe(f1_score, y_true, y_pred, zero_division=0),
        "auc": _safe(roc_auc_score, y_true, y_score) if len(set(y_true)) > 1 else float("nan"),
        "eer": _eer(y_true, y_score),
        "n": len(y_true),
        "n_pos": int(sum(y_true)),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def _predict(backbone: Backbone, head, loader: DataLoader, device: torch.device):
    ys, ps = [], []
    for x, y, _paths in loader:
        feats = backbone.embed(x).to(device)
        logits = head(feats)
        ps.extend(torch.sigmoid(logits).cpu().tolist())
        ys.extend([int(v) for v in y.tolist()])
    return ys, ps


def evaluate(
    run_tag: str,
    *,
    datasets: Sequence[str],
    profiles: Sequence[str],
    enhance_modes: Sequence[str],
    project_root: Path = None,
) -> Dict:
    project_root = Path(project_root or Path.cwd())
    set_determinism(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = project_root / "results" / "checkpoints" / f"{run_tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    backbone_name = ckpt["backbone"]
    feat_dim = ckpt["feature_dim"]

    backbone = Backbone(backbone_name, device=device).load()
    head = build_head(
        cfg_dict.get("head_kind", "mlp"),
        in_dim=feat_dim,
        hidden=cfg_dict.get("head_hidden", 256),
        dropout=cfg_dict.get("head_dropout", 0.3),
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    spec = BACKBONE_SPECS[backbone_name]
    rows: List[Dict] = []
    for ds_name in datasets:
        try:
            items = list_items(resolve_dataset_dir(ds_name, root=project_root))
        except KeyError:
            _log.warning("skipping unknown dataset %s", ds_name)
            continue
        if not items:
            _log.warning("dataset %s has no items; skipping", ds_name)
            continue
        # Same deterministic split as training, so the test slice is held out.
        _, _, test_items = train_val_test_split(items, seed=cfg_dict.get("seed", 1337))
        if not test_items:
            _log.warning("dataset %s test split is empty; skipping", ds_name)
            continue
        for enh in enhance_modes:
            for prof in profiles:
                ds = CroppedFaceDataset(
                    test_items,
                    image_size=spec.image_size,
                    enhance_mode=enh,
                    degradation_profile=prof,
                )
                loader = DataLoader(ds, batch_size=cfg_dict.get("batch_size", 64),
                                    shuffle=False, num_workers=cfg_dict.get("num_workers", 2),
                                    collate_fn=collate_with_paths)
                ys, ps = _predict(backbone, head, loader, device)
                m = _binary_metrics(ys, ps)
                rows.append({
                    "dataset": ds_name,
                    "profile": prof,
                    "enhance": enh,
                    **m,
                })
                _log.info("eval %s | %s | %s -> auc=%.3f f1=%.3f acc=%.3f (n=%d)",
                          ds_name, prof, enh, m["auc"], m["f1"], m["acc"], m["n"])

    out_dir = project_root / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_tag": run_tag, "checkpoint": str(ckpt_path), "rows": rows}
    json_path = out_dir / f"{run_tag}_eval.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md_path = out_dir / f"{run_tag}_eval.md"
    md_path.write_text(_to_markdown(payload))
    _log.info("wrote %s and %s", json_path, md_path)
    return payload


def _to_markdown(payload: Dict) -> str:
    lines = [f"# Eval — {payload['run_tag']}", "", f"checkpoint: `{payload['checkpoint']}`", ""]
    by_ds: Dict[str, List[Dict]] = {}
    for r in payload["rows"]:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds_name, rows in by_ds.items():
        lines.append(f"## {ds_name}")
        lines.append("")
        lines.append("| profile | enhance | acc | precision | recall | f1 | auc | eer | n |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(
                f"| {r['profile']} | {r['enhance']} | {r['acc']:.3f} | "
                f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                f"{r['auc']:.3f} | {r['eer']:.3f} | {r['n']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Stratified evaluation")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--profiles", nargs="+", default=list(PROFILE_NAMES))
    p.add_argument("--enhance-modes", nargs="+", default=["none", "forensic"])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "all" in args.profiles:
        args.profiles = list(PROFILE_NAMES)
    evaluate(args.run_tag, datasets=args.datasets, profiles=args.profiles, enhance_modes=args.enhance_modes)
    return 0


if __name__ == "__main__":
    main()
