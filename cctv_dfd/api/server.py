"""FastAPI server for cctv_dfd inference.

Run:
    pip install fastapi uvicorn python-multipart
    cd cctv_dfd
    uvicorn api.server:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                 health + loaded model info
    POST /predict          multipart image -> JSON {prob_fake, prediction, ...}
    POST /predict/heatmap  multipart image -> PNG (attention rollout overlay)
    POST /predict/full     multipart image -> JSON with embedded base64 heatmap

Loads the checkpoint **once** at startup. CPU inference: ~1-1.5 s per image
on a modern laptop. CORS is wide-open by default for browser frontends; lock
down in production.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cctv_dfd.degradations import PROFILE_NAMES, apply_profile
from cctv_dfd.enhance import ENHANCE_MODES, enhance
from cctv_dfd.explain import _cls_heatmap, _overlay, _rollout
from cctv_dfd.features import BACKBONE_SPECS, Backbone
from cctv_dfd.forensic_log import render_html_report, sha256_file, write_record
from cctv_dfd.head import build_head
from cctv_dfd.seed import set_determinism

_log = logging.getLogger("cctv_dfd.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Config (override via env vars)
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path(os.environ.get(
    "CCTV_DFD_CHECKPOINT", "results/checkpoints/local.pt"
)).resolve()
DEVICE = os.environ.get("CCTV_DFD_DEVICE", "cpu")
HOST = os.environ.get("CCTV_DFD_HOST", "0.0.0.0")
PORT = int(os.environ.get("CCTV_DFD_PORT", "8000"))


# ---------------------------------------------------------------------------
# Model loader (called once at startup)
# ---------------------------------------------------------------------------


class _State:
    backbone: Optional[Backbone] = None
    head = None
    spec = None
    ckpt_path: Optional[Path] = None
    ckpt_sha: Optional[str] = None
    backbone_name: Optional[str] = None


STATE = _State()


def _load_model() -> None:
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Set CCTV_DFD_CHECKPOINT or place it at results/checkpoints/local.pt."
        )
    set_determinism(1337)
    _log.info("loading checkpoint %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    backbone_name = ckpt["backbone"]
    spec = BACKBONE_SPECS[backbone_name]
    device = torch.device(DEVICE)

    backbone = Backbone(backbone_name, device=device).load()
    head = build_head(
        ckpt["config"].get("head_kind", "mlp"),
        in_dim=spec.feature_dim,
        hidden=ckpt["config"].get("head_hidden", 256),
        dropout=ckpt["config"].get("head_dropout", 0.3),
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    STATE.backbone = backbone
    STATE.head = head
    STATE.spec = spec
    STATE.ckpt_path = CHECKPOINT_PATH
    STATE.ckpt_sha = sha256_file(CHECKPOINT_PATH)
    STATE.backbone_name = backbone_name
    _log.info("ready: backbone=%s dim=%d device=%s", backbone_name, spec.feature_dim, device)


# ---------------------------------------------------------------------------
# App + lifespan
# ---------------------------------------------------------------------------


app = FastAPI(
    title="cctv_dfd inference API",
    description="Forensic CCTV deepfake screening (frozen DINOv2 + MLP head).",
    version="0.1.0",
)

# Wide-open CORS for local demo / dev. Lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    _load_model()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    backbone: str
    feature_dim: int
    device: str
    checkpoint: str
    checkpoint_sha256: str
    profiles: list
    enhance_modes: list


class PredictResponse(BaseModel):
    prob_fake: float
    prediction: str
    backbone: str
    enhance_mode: str
    degradation_profile: str
    input_sha256: str
    model_sha256: str
    timestamp_utc: str


class PredictFullResponse(PredictResponse):
    heatmap_png_base64: Optional[str] = None


class PredictSimpleResponse(BaseModel):
    """React-app contract: {label, confidence}. Confidence is *always*
    the probability of the predicted class (so 'REAL' near 1.0 means very
    confident real, 'FAKE' near 1.0 means very confident fake)."""
    label: str
    confidence: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="cannot decode image")
    return img


def _preprocess(
    img_bgr: np.ndarray,
    enhance_mode: str,
    profile: str,
) -> tuple[np.ndarray, torch.Tensor]:
    spec = STATE.spec
    img = cv2.resize(img_bgr, (spec.image_size, spec.image_size), interpolation=cv2.INTER_AREA)
    if enhance_mode != "none":
        img = enhance(img, enhance_mode)
    if profile != "clean":
        img = apply_profile(img, profile)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return img, tensor


def _infer(tensor: torch.Tensor, want_attentions: bool = False):
    if want_attentions:
        cls, _patches, attentions = STATE.backbone.forward_patches(tensor)
    else:
        cls = STATE.backbone.embed(tensor)
        attentions = None
    with torch.no_grad():
        logit = STATE.head(cls.to(torch.device(DEVICE)))
        prob = float(torch.sigmoid(logit).cpu().item())
    return prob, attentions


def _heatmap_png(img_bgr: np.ndarray, attentions) -> Optional[bytes]:
    if not attentions:
        return None
    rolled = _rollout(attentions)
    n_patches = rolled.shape[-1] - 1
    grid = int(round(n_patches ** 0.5))
    heat = _cls_heatmap(rolled, grid)
    overlay = _overlay(img_bgr, heat)
    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        return None
    return buf.tobytes()


def _validate_modes(enhance_mode: str, profile: str) -> tuple[str, str]:
    if enhance_mode not in ENHANCE_MODES:
        raise HTTPException(status_code=400,
                            detail=f"enhance must be one of {list(ENHANCE_MODES)}")
    if profile not in PROFILE_NAMES:
        raise HTTPException(status_code=400,
                            detail=f"profile must be one of {list(PROFILE_NAMES)}")
    return enhance_mode, profile


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_PATH = _STATIC_DIR / "index.html"


@app.get("/health", response_model=HealthResponse)
def health():
    if STATE.backbone is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(
        status="ok",
        backbone=STATE.backbone_name or "",
        feature_dim=STATE.spec.feature_dim,
        device=DEVICE,
        checkpoint=str(STATE.ckpt_path),
        checkpoint_sha256=STATE.ckpt_sha or "",
        profiles=list(PROFILE_NAMES),
        enhance_modes=list(ENHANCE_MODES),
    )


# Browser landing page: serve index.html at root. JSON health is at /health.
@app.get("/")
def root():
    if _INDEX_PATH.exists():
        return FileResponse(_INDEX_PATH)
    return health()


if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.post("/predict", response_model=PredictSimpleResponse)
async def predict_simple(
    image: UploadFile = File(...),
    enhance_mode: str = "none",
    profile: str = "clean",
):
    """Minimal contract for browser frontends: {label, confidence}.

    Matches the shape the React frontend (``cctv_dfd/frontend/``) expects.
    Rich response (with SHA-256 hashes, backbone, etc.) is at
    ``POST /predict/v1``; heatmap variants at ``/predict/heatmap`` and
    ``/predict/full``.
    """
    enhance_mode, profile = _validate_modes(enhance_mode, profile)
    raw = await image.read()
    img_bgr = _read_image_from_upload(raw)
    _processed, tensor = _preprocess(img_bgr, enhance_mode, profile)
    prob, _ = _infer(tensor, want_attentions=False)
    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = float(prob if label == "FAKE" else 1.0 - prob)
    return PredictSimpleResponse(label=label, confidence=confidence)


@app.post("/predict/v1", response_model=PredictResponse)
async def predict_v1(
    image: UploadFile = File(...),
    enhance_mode: str = "none",
    profile: str = "clean",
):
    """Image -> rich JSON verdict (SHA-256 hashes, backbone, timestamps).
    No heatmap (faster than /predict/full)."""
    enhance_mode, profile = _validate_modes(enhance_mode, profile)
    raw = await image.read()
    img_bgr = _read_image_from_upload(raw)
    _processed, tensor = _preprocess(img_bgr, enhance_mode, profile)
    prob, _ = _infer(tensor, want_attentions=False)

    import hashlib
    import time

    rec = PredictResponse(
        prob_fake=prob,
        prediction="fake" if prob >= 0.5 else "real",
        backbone=STATE.backbone_name or "",
        enhance_mode=enhance_mode,
        degradation_profile=profile,
        input_sha256=hashlib.sha256(raw).hexdigest(),
        model_sha256=STATE.ckpt_sha or "",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    return rec


@app.post("/predict/heatmap")
async def predict_heatmap(
    image: UploadFile = File(...),
    enhance_mode: str = "none",
    profile: str = "clean",
):
    """Image -> PNG of the attention-rollout overlay only (no JSON)."""
    enhance_mode, profile = _validate_modes(enhance_mode, profile)
    raw = await image.read()
    img_bgr = _read_image_from_upload(raw)
    processed, tensor = _preprocess(img_bgr, enhance_mode, profile)
    _prob, attentions = _infer(tensor, want_attentions=True)
    png = _heatmap_png(processed, attentions)
    if png is None:
        # Fall back to the processed image so the frontend always has something.
        ok, buf = cv2.imencode(".png", processed)
        png = buf.tobytes() if ok else b""
    return Response(content=png, media_type="image/png")


@app.post("/predict/full", response_model=PredictFullResponse)
async def predict_full(
    image: UploadFile = File(...),
    enhance_mode: str = "none",
    profile: str = "clean",
):
    """Image -> JSON verdict + base64-encoded heatmap PNG inline."""
    enhance_mode, profile = _validate_modes(enhance_mode, profile)
    raw = await image.read()
    img_bgr = _read_image_from_upload(raw)
    processed, tensor = _preprocess(img_bgr, enhance_mode, profile)
    prob, attentions = _infer(tensor, want_attentions=True)
    png = _heatmap_png(processed, attentions)

    import hashlib
    import time

    payload = PredictFullResponse(
        prob_fake=prob,
        prediction="fake" if prob >= 0.5 else "real",
        backbone=STATE.backbone_name or "",
        enhance_mode=enhance_mode,
        degradation_profile=profile,
        input_sha256=hashlib.sha256(raw).hexdigest(),
        model_sha256=STATE.ckpt_sha or "",
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        heatmap_png_base64=base64.b64encode(png).decode("ascii") if png else None,
    )
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=False)
