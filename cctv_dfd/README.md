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
