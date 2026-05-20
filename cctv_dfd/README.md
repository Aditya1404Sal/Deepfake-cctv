# cctv_dfd — Foundation-Model Features for CCTV Deepfake Detection

Frozen DINOv2 ViT-S/14 + lightweight MLP head + CCTV degradation curriculum +
optional input-enhancement preprocessing + degradation-stratified evaluation.

## Quickstart

```bash
cd cctv_dfd
pip install -e .

# 100-image smoke test against existing extracted frames
python -m cctv_dfd.cli face-extract \
  --input ../Deepfake_Detection_Project/Deepfake_Detection_Project/data \
  --output data/processed/local \
  --limit 100 --already-cropped

python -m cctv_dfd.cli cache-features --config configs/dinov2s_mlp.yaml \
  --dataset local --split all

python -m cctv_dfd.cli train --config configs/dinov2s_mlp.yaml \
  --dataset local --epochs 1 --batch-size 32 --tag smoke

python -m cctv_dfd.cli eval --config configs/dinov2s_mlp.yaml \
  --run-tag smoke --dataset local --profiles all

python -m cctv_dfd.cli explain --run-tag smoke \
  --image data/processed/local/Fake/$(ls data/processed/local/Fake | head -n1)
```

## What this is, and what it is not

- **Frozen backbone (DINOv2 ViT-S/14, Apache-2.0).** A small MLP head learns to
  separate real from fake on top of cached features. Trains in minutes on a
  free Colab T4.
- **Five named CCTV degradation profiles** (`clean`, `light_cctv`,
  `heavy_cctv`, `low_light_gray`, `ir_pseudo`) used both as a training
  curriculum and as a stratified evaluation axis.
- **Three input-enhancement modes** (`none`, `forensic`, `aggressive`). The
  `forensic` default uses CLAHE + bilateral denoise + light unsharp mask —
  deterministic, no learned prior, cannot hallucinate. `aggressive` plugs in
  Real-ESRGAN x2 **for ablation only**, behind a CLI flag, with a logged
  warning: neural restoration is known to *erase* deepfake artifacts on clean
  inputs and to *fabricate* synthetic-looking ones in real faces.
- **Forensic-ready logging, not court-admissible.** Per-prediction SHA-256 of
  input + model, JSONL trace, optional HTML report with ViT attention rollout.

This is **offline forensic screening**, not a real-time detector. It is not
SOTA on FF++ in-domain accuracy — the headline result is robustness across the
five degradation profiles and across datasets.

## Files

| Module | Purpose |
| --- | --- |
| `seed.py` | one-call determinism |
| `degradations.py` | 5 named CCTV degradation profiles |
| `enhance.py` | `none` / `forensic` / `aggressive` input enhancement |
| `face_extract.py` | RetinaFace / MTCNN face crops |
| `data.py` | dataset registry, `CroppedFaceDataset` |
| `features.py` | frozen backbone loader + cache |
| `head.py` | MLP head, video mean-pool head |
| `train.py` | training loop |
| `eval.py` | per-profile × per-enhancement metrics table |
| `explain.py` | ViT attention rollout heatmap |
| `forensic_log.py` | JSONL + HTML per-prediction report |
| `cli.py` | unified CLI entry point |

Legacy reference scripts at the repo root (not modified, not imported except
where explicitly noted): `augment_fixed.py`, `finetune.py`, `Deepfake_Detector.py`.

## Running the inference API locally

After training (see Colab notebooks above), download `results/checkpoints/local.pt`
to this machine and start the FastAPI server:

```bash
cd cctv_dfd
pip install -e ".[api]"   # adds fastapi + uvicorn + python-multipart
# place the trained checkpoint at: results/checkpoints/local.pt

# start the server (CPU is fine; ~1-1.5 s per image)
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET  /` — **the web UI** (drag-drop face image → verdict + heatmap + JSON)
- `GET  /health` — JSON health blob: backbone, SHA-256, available profiles + enhancement modes
- `POST /predict` — multipart image upload → JSON verdict only (fastest)
- `POST /predict/heatmap` — multipart image → PNG of attention-rollout overlay
- `POST /predict/full` — multipart image → JSON with embedded base64 heatmap

For the web UI, just open `http://localhost:8000/` in any browser — there is
no separate frontend build step, the page is a static HTML file served by the
FastAPI app.

Query params accepted by all three POSTs:

- `enhance_mode` — `none` (default) / `forensic` / `aggressive`
- `profile` — `clean` (default) / `light_cctv` / `heavy_cctv` /
  `low_light_gray` / `ir_pseudo`

Example with curl:

```bash
# quick verdict
curl -X POST http://localhost:8000/predict \
  -F "image=@some_face.jpg" \
  -F "enhance_mode=forensic"

# get heatmap PNG directly
curl -X POST "http://localhost:8000/predict/heatmap?enhance_mode=forensic" \
  -F "image=@some_face.jpg" \
  --output heatmap.png
```

The interactive API docs (Swagger UI) are at `http://localhost:8000/docs` once
the server is running — drop an image into the form, hit "Execute", see the
JSON / PNG response.

### Configuration via env vars

- `CCTV_DFD_CHECKPOINT` — path to the head checkpoint (default `results/checkpoints/local.pt`)
- `CCTV_DFD_DEVICE` — `cpu` (default) or `cuda`
- `CCTV_DFD_PORT` — port to listen on (default 8000)
- `CCTV_DFD_HOST` — interface to bind (default `0.0.0.0`)

### CORS

Wide-open by default so any browser frontend can hit `localhost:8000`. Lock
down `allow_origins` in `api/server.py` before exposing publicly.
