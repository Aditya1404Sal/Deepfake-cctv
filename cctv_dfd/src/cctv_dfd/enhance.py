"""Input enhancement for CCTV-grade footage.

Three modes:

- ``none``: identity.
- ``forensic`` (default for degraded profiles): deterministic, local filters.
  CLAHE on L channel + bilateral denoise + light unsharp mask. No learned
  prior, **no hallucination risk**.
- ``aggressive`` (opt-in, ablation only): neural restoration (Real-ESRGAN x2
  with NAFNet-deblur fallback). Known to *erase* deepfake artifacts on clean
  inputs or *fabricate* synthetic-looking detail on real faces; logged with
  warning, disabled by default. Use ``--enhance aggressive`` only when running
  the enhancement ablation.

API: ``enhance(img_bgr, mode) -> img_bgr`` accepting/returning uint8 BGR.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

import cv2
import numpy as np

EnhanceMode = Literal["none", "forensic", "aggressive"]
ENHANCE_MODES = ("none", "forensic", "aggressive")

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forensic mode (safe, deterministic)
# ---------------------------------------------------------------------------


def _forensic(img: np.ndarray) -> np.ndarray:
    """CLAHE(L) + bilateral denoise + light unsharp mask. Deterministic."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # CLAHE on L channel only — preserves chroma, lifts local contrast.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

    # Bilateral denoise — edge-preserving.
    img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # Light unsharp mask — radius=1.0, amount=0.5.
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


# ---------------------------------------------------------------------------
# Aggressive mode (opt-in, ablation only)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_realesrgan():
    """Lazy-load Real-ESRGAN x2. Returns ``None`` if unavailable."""
    try:
        from realesrgan import RealESRGANer  # type: ignore
        from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
    except Exception as e:
        _log.warning("aggressive enhancement requested but Real-ESRGAN unavailable: %s", e)
        return None
    try:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2,
            model_path=None,  # will try to download default weights on first use
            model=model,
            half=False,
        )
        return upsampler
    except Exception as e:
        _log.warning("Real-ESRGAN init failed: %s", e)
        return None


def _aggressive(img: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Real-ESRGAN x2 then resize back to ``target_hw``.

    Hallucination risk; for ablation use only. Falls back to ``_forensic`` if
    the upscaler cannot be loaded so the pipeline still runs.
    """
    upsampler = _load_realesrgan()
    if upsampler is None:
        _log.warning("falling back to forensic mode for this image")
        return _forensic(img)
    try:
        out, _ = upsampler.enhance(img, outscale=2)
        return cv2.resize(out, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    except Exception as e:
        _log.warning("Real-ESRGAN forward failed (%s); falling back to forensic", e)
        return _forensic(img)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_AGGRESSIVE_WARNED = False


def enhance(img: np.ndarray, mode: EnhanceMode = "none") -> np.ndarray:
    """Apply enhancement mode to a uint8 BGR image."""
    if mode == "none":
        return img
    if mode == "forensic":
        return _forensic(img)
    if mode == "aggressive":
        global _AGGRESSIVE_WARNED
        if not _AGGRESSIVE_WARNED:
            _log.warning(
                "enhancement mode 'aggressive' is enabled. Neural face restoration "
                "can ERASE deepfake artifacts on clean inputs and FABRICATE synthetic "
                "detail on real faces. Use for ablation only."
            )
            _AGGRESSIVE_WARNED = True
        return _aggressive(img, target_hw=img.shape[:2])
    raise ValueError(f"Unknown enhance mode: {mode!r}. Known: {ENHANCE_MODES}")
