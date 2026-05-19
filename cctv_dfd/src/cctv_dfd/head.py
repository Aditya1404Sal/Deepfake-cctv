"""Classifier heads sitting on top of frozen backbone features.

Default: 2-layer MLP. Optional ``VideoMeanHead`` averages N CLS tokens per
clip before feeding the same MLP — keeps the PDF's sequence-modeling
narrative honest without a BiLSTM.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear (single logit)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]

    @property
    def in_dim(self) -> int:
        return self.net[0].normalized_shape[0]


class VideoMeanHead(nn.Module):
    """Mean-pool ``num_frames`` CLS tokens then apply MLP head."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3, num_frames: int = 8):
        super().__init__()
        self.num_frames = num_frames
        self.head = MLPHead(in_dim, hidden=hidden, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is either [B, num_frames, D] or [B, D]."""
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.head(x)


def build_head(kind: str, in_dim: int, **kwargs) -> nn.Module:
    if kind == "mlp":
        return MLPHead(in_dim, **kwargs)
    if kind == "video_mean":
        return VideoMeanHead(in_dim, **kwargs)
    raise ValueError(f"Unknown head kind {kind!r}")
