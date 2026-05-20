"""Frozen vision backbones for feature extraction.

Supported backbones (selected via config ``backbone.name``):

- ``dinov2-small``  (default) — ``facebook/dinov2-small`` via ``transformers``.
  CLS token, 384-d, Apache-2.0.
- ``clip-vit-b16``  — OpenCLIP ``ViT-B-16`` ``laion2b_s34b_b88k``. CLS token,
  512-d. Ablation backbone.

All forward passes are wrapped in ``torch.no_grad()`` and run in eval mode.
The dataset is expected to feed raw [0, 1] RGB tensors of size
``image_size``; we apply backbone-specific mean/std normalization inside
:func:`embed_batch` so the dataset stays backbone-agnostic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone descriptors
# ---------------------------------------------------------------------------


@dataclass
class BackboneSpec:
    name: str
    feature_dim: int
    image_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


BACKBONE_SPECS = {
    "dinov2-small": BackboneSpec(
        name="facebook/dinov2-small",
        feature_dim=384,
        image_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "clip-vit-b16": BackboneSpec(
        name="ViT-B-16/laion2b_s34b_b88k",
        feature_dim=512,
        image_size=224,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
}


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------


class Backbone:
    """Thin wrapper exposing ``embed(batch)``.

    Patch tokens are exposed via :meth:`forward_patches` for the
    attention-rollout explainer.
    """

    def __init__(self, key: str, device: str = "cpu"):
        if key not in BACKBONE_SPECS:
            raise KeyError(f"Unknown backbone {key!r}. Known: {list(BACKBONE_SPECS)}")
        self.key = key
        self.spec = BACKBONE_SPECS[key]
        self.device = torch.device(device)
        self.model = None
        self.kind = None  # "hf" | "openclip"

    def load(self) -> "Backbone":
        if self.key.startswith("dinov2"):
            from transformers import AutoModel  # type: ignore

            # attn_implementation="eager" is required for output_attentions
            # to actually populate (sdpa/flash-attn discard the maps).
            try:
                self.model = AutoModel.from_pretrained(
                    self.spec.name, attn_implementation="eager"
                )
            except TypeError:
                # transformers < 4.36 doesn't have the kwarg.
                self.model = AutoModel.from_pretrained(self.spec.name)
            self.model.eval().to(self.device)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.kind = "hf"
        elif self.key.startswith("clip"):
            import open_clip  # type: ignore

            arch, pretrained = self.spec.name.split("/")
            model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            model.eval().to(self.device)
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = model
            self.kind = "openclip"
        else:
            raise RuntimeError(f"Unsupported backbone key {self.key!r}")
        _log.info("loaded backbone %s (%s) on %s", self.key, self.kind, self.device)
        return self

    # ------------------------------------------------------------------ embed

    def _normalize(self, batch01: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.spec.mean, device=batch01.device).view(1, 3, 1, 1)
        std = torch.tensor(self.spec.std, device=batch01.device).view(1, 3, 1, 1)
        return (batch01 - mean) / std

    @torch.no_grad()
    def embed(self, batch01: torch.Tensor) -> torch.Tensor:
        """Embed a [B, 3, H, W] tensor in [0,1] -> [B, D] CLS features."""
        assert self.model is not None, "call .load() first"
        batch = self._normalize(batch01.to(self.device, non_blocking=True))
        if self.kind == "hf":
            out = self.model(pixel_values=batch)
            # DINOv2: out.last_hidden_state[:, 0] is the CLS token.
            cls = out.last_hidden_state[:, 0]
            return cls.detach().float().cpu()
        if self.kind == "openclip":
            feats = self.model.encode_image(batch)
            return feats.detach().float().cpu()
        raise RuntimeError("backbone not loaded")

    @torch.no_grad()
    def forward_patches(self, batch01: torch.Tensor):
        """Return (cls, patches) for explainability. patches shape [B, N, D]."""
        assert self.model is not None
        batch = self._normalize(batch01.to(self.device, non_blocking=True))
        if self.kind == "hf":
            out = self.model(pixel_values=batch, output_attentions=True)
            last = out.last_hidden_state  # [B, N+1, D]
            return last[:, 0], last[:, 1:], out.attentions
        if self.kind == "openclip":
            # OpenCLIP doesn't expose patch tokens conveniently; return None
            feats = self.model.encode_image(batch)
            return feats, None, None
        raise RuntimeError("backbone not loaded")


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------


def cache_features(
    backbone: Backbone,
    loader: Iterable,
    out_path: Path,
) -> dict:
    """Run the backbone over ``loader`` and save features + labels to disk."""
    feats: List[torch.Tensor] = []
    labels: List[int] = []
    paths: List[str] = []
    for batch in loader:
        if len(batch) == 3:
            x, y, p = batch
            paths.extend(p)
        else:
            x, y = batch
        emb = backbone.embed(x)
        feats.append(emb)
        labels.extend([int(v) for v in y.tolist()])
    feats_t = torch.cat(feats, dim=0) if feats else torch.empty(0)
    labels_t = torch.tensor(labels, dtype=torch.long)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": feats_t, "labels": labels_t, "paths": paths,
                "backbone": backbone.key}, out_path)
    return {"n": int(feats_t.shape[0]), "dim": int(feats_t.shape[1]) if feats_t.numel() else 0,
            "out": str(out_path)}


def load_cache(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)
