"""ViT attention rollout for DINOv2 + heatmap overlay PNGs.

Replaces Grad-CAM for transformer backbones. Reference:
    Abnar & Zuidema, 2020. "Quantifying Attention Flow in Transformers."

Workflow:
1. Forward the image with ``output_attentions=True``.
2. Average attention across heads, add identity, row-normalize.
3. Multiply per-layer matrices to compose the rollout.
4. Take the CLS row, drop the CLS index, reshape to the patch grid, upsample.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import LABEL_MAP
from .features import BACKBONE_SPECS, Backbone
from .head import build_head
from .seed import set_determinism

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _rollout(attentions) -> torch.Tensor:
    """attentions: tuple of [B, heads, T, T] tensors. Returns [B, T, T]."""
    rolled = None
    for attn in attentions:
        a = attn.mean(dim=1)  # [B, T, T]
        a = a + torch.eye(a.size(-1), device=a.device).unsqueeze(0)
        a = a / a.sum(dim=-1, keepdim=True)
        rolled = a if rolled is None else torch.bmm(a, rolled)
    return rolled  # [B, T, T]


def _cls_heatmap(rolled: torch.Tensor, grid: int) -> np.ndarray:
    cls_row = rolled[0, 0, 1:]  # drop CLS index
    h = cls_row.reshape(grid, grid).cpu().numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    return h


def _load_image_bgr(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Cannot read {path}")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _overlay(image_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm_u8 = (hm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(image_bgr, 1 - alpha, colored, alpha, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain_image(
    image_path: Path,
    *,
    run_tag: str,
    project_root: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> dict:
    set_determinism(1337)
    project_root = Path(project_root or Path.cwd())
    out_dir = Path(out_dir or project_root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = project_root / "results" / "checkpoints" / f"{run_tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_name = ckpt["backbone"]
    spec = BACKBONE_SPECS[backbone_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = Backbone(backbone_name, device=device).load()
    head = build_head(
        ckpt["config"].get("head_kind", "mlp"),
        in_dim=spec.feature_dim,
        hidden=ckpt["config"].get("head_hidden", 256),
        dropout=ckpt["config"].get("head_dropout", 0.3),
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    img_bgr = _load_image_bgr(image_path, spec.image_size)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    cls, patches, attentions = backbone.forward_patches(tensor)
    if not attentions:
        # Either: (a) backbone is CLIP via open_clip and doesn't return them,
        # or (b) HF model was loaded without attn_implementation="eager" and
        # the SDPA/flash kernels discarded them. We still save a placeholder
        # output so the predict pipeline doesn't break.
        _log.warning("backbone %s did not return attention maps; "
                     "saving raw prediction only (load with attn_implementation='eager' "
                     "for rollout heatmaps)", backbone_name)
        with torch.no_grad():
            logit = head(cls.to(device))
            prob = float(torch.sigmoid(logit).cpu().item())
        out_path = out_dir / f"{image_path.stem}_explain.png"
        cv2.imwrite(str(out_path), img_bgr)
        return {"prob_fake": prob, "explanation_path": str(out_path), "rollout": False}

    rolled = _rollout(attentions)
    # Determine patch grid: DINOv2 uses 14x14 for 224 / 16 = 14 (patch size 14 by name; verify).
    n_patches = rolled.shape[-1] - 1
    grid = int(round(n_patches ** 0.5))
    heat = _cls_heatmap(rolled, grid)
    overlay = _overlay(img_bgr, heat)

    with torch.no_grad():
        logit = head(cls.to(device))
        prob = float(torch.sigmoid(logit).cpu().item())

    out_path = out_dir / f"{image_path.stem}_explain.png"
    cv2.imwrite(str(out_path), overlay)
    return {
        "prob_fake": prob,
        "explanation_path": str(out_path),
        "rollout": True,
        "patch_grid": grid,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate attention-rollout explanation PNG")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = explain_image(args.image, run_tag=args.run_tag, out_dir=args.out_dir)
    _log.info("result: %s", result)
    return 0


if __name__ == "__main__":
    main()
