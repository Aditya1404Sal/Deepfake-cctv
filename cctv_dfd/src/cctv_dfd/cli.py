"""Unified CLI: ``python -m cctv_dfd.cli <subcommand> [args]``.

Subcommands:
    face-extract     run face detection + cropping
    cache-features   precompute clean features for fast head training
    train            train the classifier head
    eval             degradation-stratified evaluation
    explain          attention rollout PNG for a single image
    predict          one-shot inference with forensic log + HTML report
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import data as data_mod
from . import eval as eval_mod
from . import explain as explain_mod
from . import face_extract as face_mod
from . import features as feat_mod
from . import forensic_log as flog
from . import head as head_mod
from . import train as train_mod
from .degradations import PROFILE_NAMES
from .enhance import ENHANCE_MODES
from .seed import set_determinism


def _add_face_extract(sub):
    p = sub.add_parser("face-extract", help="Detect + crop faces (or repack existing crops)")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--already-cropped", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--real-only", action="store_true")
    return p


def _add_cache_features(sub):
    p = sub.add_parser("cache-features", help="Cache backbone features to disk")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--enhance", default="none", choices=list(ENHANCE_MODES))
    return p


def _add_train(sub):
    p = sub.add_parser("train", help="Train classifier head on frozen features")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p


def _add_eval(sub):
    p = sub.add_parser("eval", help="Stratified evaluation")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--dataset", default=None, help="Single dataset shorthand for --datasets")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--profiles", nargs="+", default=["all"])
    p.add_argument("--enhance-modes", nargs="+", default=["none", "forensic"])
    return p


def _add_explain(sub):
    p = sub.add_parser("explain", help="Attention rollout PNG for a single image")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    return p


def _add_predict(sub):
    p = sub.add_parser("predict", help="One-shot inference with forensic log + HTML report")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--enhance", default="none", choices=list(ENHANCE_MODES))
    p.add_argument("--profile", default="clean", choices=list(PROFILE_NAMES))
    p.add_argument("--no-explain", action="store_true")
    return p


def cmd_cache_features(args) -> int:
    from torch.utils.data import DataLoader

    cfg = train_mod.TrainConfig.from_yaml(args.config)
    spec = feat_mod.BACKBONE_SPECS[cfg.backbone]
    backbone = feat_mod.Backbone(cfg.backbone).load()

    items = data_mod.list_items(data_mod.resolve_dataset_dir(args.dataset))
    if not items:
        logging.error("no items found for dataset %s", args.dataset)
        return 2
    train, val, test = data_mod.train_val_test_split(items, seed=cfg.seed)
    splits = {"train": train, "val": val, "test": test}
    if args.split != "all":
        splits = {args.split: splits[args.split]}

    out_root = Path("data/cache") / cfg.backbone / args.dataset
    enh = args.enhance
    for name, group in splits.items():
        ds = data_mod.CroppedFaceDataset(
            group, image_size=spec.image_size, enhance_mode=enh,
            degradation_profile="clean",
        )
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers,
                            collate_fn=data_mod.collate_with_paths)
        out_path = out_root / f"{name}_clean_{enh}.pt"
        info = feat_mod.cache_features(backbone, loader, out_path)
        logging.info("cached %s -> %s", name, info)
    return 0


def cmd_predict(args) -> int:
    import cv2
    import torch
    import numpy as np

    set_determinism(1337)
    project_root = Path.cwd()
    ckpt_path = project_root / "results" / "checkpoints" / f"{args.run_tag}.pt"
    if not ckpt_path.exists():
        logging.error("no checkpoint at %s", ckpt_path)
        return 2
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_name = ckpt["backbone"]
    spec = feat_mod.BACKBONE_SPECS[backbone_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = feat_mod.Backbone(backbone_name, device=device).load()
    head = head_mod.build_head(
        ckpt["config"].get("head_kind", "mlp"),
        in_dim=spec.feature_dim,
        hidden=ckpt["config"].get("head_hidden", 256),
        dropout=ckpt["config"].get("head_dropout", 0.3),
    ).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        logging.error("cannot read %s", args.image)
        return 2
    img_bgr = cv2.resize(img_bgr, (spec.image_size, spec.image_size), interpolation=cv2.INTER_AREA)
    from .enhance import enhance as _enh
    from .degradations import apply_profile

    if args.enhance != "none":
        img_bgr = _enh(img_bgr, args.enhance)
    if args.profile != "clean":
        img_bgr = apply_profile(img_bgr, args.profile)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        feat = backbone.embed(tensor).to(device)
        prob = float(torch.sigmoid(head(feat)).cpu().item())

    explanation_path = ""
    if not args.no_explain:
        info = explain_mod.explain_image(args.image, run_tag=args.run_tag)
        explanation_path = info["explanation_path"]

    rec = flog.write_record(
        args.image,
        ckpt_path,
        prob_fake=prob,
        degradation_profile=args.profile,
        enhance_mode=args.enhance,
        explanation_path=explanation_path,
    )
    html_path = flog.render_html_report(rec, Path("reports"))
    print(json.dumps({
        "prob_fake": prob,
        "prediction": rec.prediction,
        "report": str(html_path),
        "log": str(Path("results") / "forensic.jsonl"),
        "explanation": explanation_path,
    }, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cctv-dfd")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_face_extract(sub)
    _add_cache_features(sub)
    _add_train(sub)
    _add_eval(sub)
    _add_explain(sub)
    _add_predict(sub)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "face-extract":
        stats = face_mod.extract(
            args.input, args.output,
            already_cropped=args.already_cropped,
            limit=args.limit, real_only=args.real_only,
        )
        print(json.dumps(stats, indent=2))
        return 0
    if args.cmd == "cache-features":
        return cmd_cache_features(args)
    if args.cmd == "train":
        cfg = train_mod.TrainConfig.from_yaml(args.config)
        train_mod.train(cfg, args.dataset, args.tag,
                        epochs_override=args.epochs,
                        batch_size_override=args.batch_size)
        return 0
    if args.cmd == "eval":
        datasets = args.datasets or ([args.dataset] if args.dataset else None)
        if not datasets:
            logging.error("must pass --dataset or --datasets")
            return 2
        profiles = args.profiles
        if "all" in profiles:
            profiles = list(PROFILE_NAMES)
        eval_mod.evaluate(args.run_tag, datasets=datasets,
                          profiles=profiles, enhance_modes=args.enhance_modes)
        return 0
    if args.cmd == "explain":
        result = explain_mod.explain_image(
            args.image, run_tag=args.run_tag, out_dir=args.out_dir,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.cmd == "predict":
        return cmd_predict(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
