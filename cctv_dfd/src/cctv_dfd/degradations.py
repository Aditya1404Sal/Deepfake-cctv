"""Five named CCTV degradation profiles.

Profiles
--------
- ``clean``           identity baseline
- ``light_cctv``      mild compression + downscale + noise
- ``heavy_cctv``      aggressive compression + downscale + motion blur + noise
- ``low_light_gray``  grayscale + brightness drop + noise
- ``ir_pseudo``       grayscale + CLAHE + slight blur + contrast bump

Used in two roles:
1. **Train-time augmentation**: sample a profile per item with configured
   weights (defaults: ``[0.4, 0.2, 0.2, 0.1, 0.1]``).
2. **Eval-time fixed transforms**: apply each profile deterministically to the
   held-out test set to produce stratified metrics.

Falls back to the OpenCV primitives in the legacy
``augment_fixed.RandomAugmentation`` class if ``albumentations`` cannot be
imported (preserves lineage with the existing repo code).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    import albumentations as A

    _HAS_ALB = True
except Exception:
    _HAS_ALB = False


ProfileName = str
PROFILE_NAMES: List[ProfileName] = [
    "clean",
    "light_cctv",
    "heavy_cctv",
    "low_light_gray",
    "ir_pseudo",
]
DEFAULT_SAMPLE_WEIGHTS: List[float] = [0.4, 0.2, 0.2, 0.1, 0.1]


# ---------------------------------------------------------------------------
# Albumentations-backed profiles (preferred)
# ---------------------------------------------------------------------------


def _compat_image_compression(q_low: int, q_high: int):
    """Albumentations renamed quality_lower/upper -> quality_range in 1.4+."""
    try:
        return A.ImageCompression(quality_range=(q_low, q_high), p=1.0)
    except TypeError:
        return A.ImageCompression(quality_lower=q_low, quality_upper=q_high, p=1.0)


def _compat_downscale(s_low: float, s_high: float):
    """Renamed scale_min/scale_max -> scale_range; interpolation -> interpolation_pair."""
    try:
        return A.Downscale(scale_range=(s_low, s_high), p=1.0)
    except TypeError:
        return A.Downscale(scale_min=s_low, scale_max=s_high,
                           interpolation=cv2.INTER_AREA, p=1.0)


def _compat_gauss_noise(var_low: float, var_high: float):
    """Renamed var_limit -> std_range (and units changed to std, not var)."""
    try:
        std_low = float(var_low) ** 0.5 / 255.0
        std_high = float(var_high) ** 0.5 / 255.0
        return A.GaussNoise(std_range=(std_low, std_high), p=1.0)
    except TypeError:
        return A.GaussNoise(var_limit=(var_low, var_high), p=1.0)


def _alb_clean() -> "A.Compose":
    return A.Compose([A.NoOp()])


def _alb_light_cctv() -> "A.Compose":
    return A.Compose(
        [
            _compat_image_compression(55, 75),
            _compat_downscale(0.5, 0.75),
            _compat_gauss_noise(10.0, 25.0),
        ]
    )


def _alb_heavy_cctv() -> "A.Compose":
    return A.Compose(
        [
            _compat_image_compression(30, 50),
            _compat_downscale(0.25, 0.4),
            A.MotionBlur(blur_limit=7, p=1.0),
            _compat_gauss_noise(20.0, 50.0),
        ]
    )


def _alb_low_light_gray() -> "A.Compose":
    return A.Compose(
        [
            A.ToGray(p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.4, -0.2), contrast_limit=(-0.1, 0.1), p=1.0),
            _compat_gauss_noise(15.0, 30.0),
        ]
    )


def _alb_ir_pseudo() -> "A.Compose":
    return A.Compose(
        [
            A.ToGray(p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(0.2, 0.4), p=1.0),
        ]
    )


_ALB_BUILDERS: Dict[ProfileName, Callable[[], "A.Compose"]] = {
    "clean": _alb_clean,
    "light_cctv": _alb_light_cctv,
    "heavy_cctv": _alb_heavy_cctv,
    "low_light_gray": _alb_low_light_gray,
    "ir_pseudo": _alb_ir_pseudo,
}


# ---------------------------------------------------------------------------
# OpenCV fallback (when albumentations missing)
# ---------------------------------------------------------------------------


def _cv_compress(img: np.ndarray, q_low: int, q_high: int) -> np.ndarray:
    q = random.randint(q_low, q_high)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def _cv_downscale(img: np.ndarray, s_low: float, s_high: float) -> np.ndarray:
    s = random.uniform(s_low, s_high)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _cv_gauss_noise(img: np.ndarray, var_low: float, var_high: float) -> np.ndarray:
    var = random.uniform(var_low, var_high)
    sigma = var**0.5
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _cv_motion_blur(img: np.ndarray, k: int = 7) -> np.ndarray:
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def _cv_to_gray_3c(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _cv_bright(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    delta = random.uniform(lo, hi)  # in [-1, 1] roughly
    beta = delta * 127.0
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def _cv_clahe(img: np.ndarray, clip: float = 4.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _cv_apply(profile: ProfileName, img: np.ndarray) -> np.ndarray:
    if profile == "clean":
        return img
    if profile == "light_cctv":
        img = _cv_compress(img, 55, 75)
        img = _cv_downscale(img, 0.5, 0.75)
        img = _cv_gauss_noise(img, 10.0, 25.0)
        return img
    if profile == "heavy_cctv":
        img = _cv_compress(img, 30, 50)
        img = _cv_downscale(img, 0.25, 0.4)
        img = _cv_motion_blur(img, 7)
        img = _cv_gauss_noise(img, 20.0, 50.0)
        return img
    if profile == "low_light_gray":
        img = _cv_to_gray_3c(img)
        img = _cv_bright(img, -0.4, -0.2)
        img = _cv_gauss_noise(img, 15.0, 30.0)
        return img
    if profile == "ir_pseudo":
        img = _cv_to_gray_3c(img)
        img = _cv_clahe(img, 4.0, 8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
        return img
    raise ValueError(f"Unknown profile: {profile!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_profile(img: np.ndarray, profile: ProfileName) -> np.ndarray:
    """Apply a single named degradation profile to a BGR uint8 image."""
    if profile not in _ALB_BUILDERS:
        raise ValueError(f"Unknown profile: {profile!r}. Known: {PROFILE_NAMES}")
    if not _HAS_ALB:
        return _cv_apply(profile, img)
    transform = _ALB_BUILDERS[profile]()
    out = transform(image=img)["image"]
    return out


class ProfileSampler:
    """Randomly select a profile name with configured weights (for training)."""

    def __init__(
        self,
        names: Sequence[ProfileName] = PROFILE_NAMES,
        weights: Optional[Sequence[float]] = None,
        rng: Optional[random.Random] = None,
    ):
        if weights is None:
            weights = DEFAULT_SAMPLE_WEIGHTS
        if len(names) != len(weights):
            raise ValueError("names and weights must be same length")
        self.names = list(names)
        self.weights = list(weights)
        self._rng = rng or random

    def sample(self) -> ProfileName:
        return self._rng.choices(self.names, weights=self.weights, k=1)[0]


def list_profiles() -> List[ProfileName]:
    return list(PROFILE_NAMES)
