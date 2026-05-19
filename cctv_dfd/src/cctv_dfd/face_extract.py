"""Face detection + cropping for video frames or pre-extracted images.

Two modes:
- ``--already-cropped``: inputs are already face crops (e.g. the existing
  ``Deepfake_Detection_Project/.../data/Real|Fake`` images). Only resize and
  re-save with a normalized filename. Idempotent.
- Default: run face detection (RetinaFace via insightface preferred,
  facenet-pytorch MTCNN fallback) on each input image, save the highest-score
  crop with 1.3x margin, resized to 224x224 JPEG-95.

Outputs to ``<output>/<label>/<id>.jpg``. Skips files that already exist so the
CLI is safe to re-run.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

_log = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
TARGET_SIZE = 224
JPEG_QUALITY = 95
MARGIN = 0.3  # 1.3x crop margin


# ---------------------------------------------------------------------------
# Detector loaders (lazy)
# ---------------------------------------------------------------------------


class _FaceDetector:
    """Wraps either insightface RetinaFace or facenet-pytorch MTCNN."""

    def __init__(self):
        self._backend = None
        self._impl = None

    def _load(self):
        if self._impl is not None:
            return
        # Try insightface first.
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._impl = app
            self._backend = "insightface"
            _log.info("face detector: insightface RetinaFace")
            return
        except Exception as e:
            _log.info("insightface unavailable (%s); trying facenet-pytorch", e)
        # Fallback to facenet-pytorch MTCNN.
        try:
            from facenet_pytorch import MTCNN  # type: ignore

            self._impl = MTCNN(keep_all=True, device="cpu")
            self._backend = "mtcnn"
            _log.info("face detector: facenet-pytorch MTCNN")
            return
        except Exception as e:
            raise RuntimeError(
                "No face detector available. Install `insightface` + `onnxruntime` "
                "or `facenet-pytorch`."
            ) from e

    def detect_best(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return (x1, y1, x2, y2) of the highest-confidence face, or None."""
        self._load()
        if self._backend == "insightface":
            faces = self._impl.get(img_bgr)
            if not faces:
                return None
            best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
            x1, y1, x2, y2 = best.bbox.astype(int).tolist()
            return x1, y1, x2, y2
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = self._impl.detect(rgb)
            if boxes is None or len(boxes) == 0:
                return None
            idx = int(np.argmax(probs))
            x1, y1, x2, y2 = [int(v) for v in boxes[idx]]
            return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def _expand_bbox(bbox: Tuple[int, int, int, int], img_hw: Tuple[int, int], margin: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = img_hw
    bw, bh = x2 - x1, y2 - y1
    cx, cy = x1 + bw / 2, y1 + bh / 2
    side = max(bw, bh) * (1.0 + margin)
    nx1 = int(max(0, cx - side / 2))
    ny1 = int(max(0, cy - side / 2))
    nx2 = int(min(w, cx + side / 2))
    ny2 = int(min(h, cy + side / 2))
    return nx1, ny1, nx2, ny2


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w == size:
        return img
    interp = cv2.INTER_AREA if max(h, w) > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def _save_jpeg(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def _iter_label_dirs(input_dir: Path) -> List[Tuple[str, Path]]:
    """Find label subdirectories. Supports {Real,Fake}, {real,fake}."""
    pairs: List[Tuple[str, Path]] = []
    for name in ("Real", "real", "REAL"):
        p = input_dir / name
        if p.is_dir():
            pairs.append(("Real", p))
            break
    for name in ("Fake", "fake", "FAKE"):
        p = input_dir / name
        if p.is_dir():
            pairs.append(("Fake", p))
            break
    if not pairs:
        # Allow flat directory with label inferred from filename prefix.
        pairs = [("", input_dir)]
    return pairs


def _iter_images(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            yield p


def _infer_label_from_filename(name: str) -> Optional[str]:
    low = name.lower()
    if low.startswith("real"):
        return "Real"
    if low.startswith("fake") or low.startswith("swapped") or "deepfake" in low:
        return "Fake"
    return None


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def extract(
    input_dir: Path,
    output_dir: Path,
    *,
    already_cropped: bool,
    limit: Optional[int] = None,
    real_only: bool = False,
) -> dict:
    detector = None if already_cropped else _FaceDetector()
    pairs = _iter_label_dirs(input_dir)
    stats = {"processed": 0, "skipped": 0, "no_face": 0, "errors": 0}
    written = 0
    for label, folder in pairs:
        for img_path in _iter_images(folder):
            if real_only and label and label.lower() != "real":
                continue
            effective_label = label or _infer_label_from_filename(img_path.name) or "Real"
            if real_only and effective_label.lower() != "real":
                continue
            out_path = output_dir / effective_label / f"{img_path.stem}.jpg"
            if out_path.exists():
                stats["skipped"] += 1
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    stats["errors"] += 1
                    continue
                if already_cropped:
                    crop = _resize_square(img, TARGET_SIZE)
                else:
                    bbox = detector.detect_best(img)
                    if bbox is None:
                        stats["no_face"] += 1
                        continue
                    nb = _expand_bbox(bbox, img.shape[:2], MARGIN)
                    crop = img[nb[1]:nb[3], nb[0]:nb[2]]
                    if crop.size == 0:
                        stats["no_face"] += 1
                        continue
                    crop = _resize_square(crop, TARGET_SIZE)
                _save_jpeg(crop, out_path)
                stats["processed"] += 1
                written += 1
                if limit is not None and written >= limit:
                    return stats
            except Exception as e:
                _log.warning("error on %s: %s", img_path, e)
                stats["errors"] += 1
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Face detection + cropping for cctv_dfd")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--already-cropped", action="store_true",
                   help="Skip face detection; inputs are already face crops")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N crops written")
    p.add_argument("--real-only", action="store_true",
                   help="Only emit Real-labeled crops")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = extract(
        args.input, args.output,
        already_cropped=args.already_cropped,
        limit=args.limit,
        real_only=args.real_only,
    )
    _log.info("done: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
