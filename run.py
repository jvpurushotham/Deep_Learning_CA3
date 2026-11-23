#!/usr/bin/env python
"""CSD402 Capstone Option A - CNN (+GAN) visual inspection. Command line entry point.

  python run.py inspect --data-root data/raw
  python run.py prepare --data-root data/raw --minority defect --minority-keep 0.10 [--label-map def_front=defect ok_front=ok]
  python run.py tune                      # scratch CNN vs ResNet-18, 3 LRs each, seed 0, validation macro-F1
  python run.py gan                       # DCGAN on TRAIN minority images only -> synthetic images
  python run.py arms --seeds 0 1 2        # scratch, resnet18, class-weights, oversampling, GAN-augmented  x seeds
  python run.py test                      # held-out TEST, run ONCE
  python run.py gradcam --arm gan         # Grad-CAM on test images
  python run.py export --arm gan          # copy chosen model to artifacts/
  python run.py predict --image some.png  # inference with the exported model
"""
import argparse, shutil
from pathlib import Path

from capstone_a.common import DEVICE, load_json, save_json
from capstone_a import data as D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "prepare", "tune", "gan", "arms", "test", "gradcam", "export", "predict"])
    ap.add_argument("--out", default="outputs"); ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--minority", default="defect"); ap.add_argument("--minority-keep", type=float, default=0.10)
    ap.add_argument("--label-map", nargs="*", help="folder=label pairs, e.g. def_front=defect ok_front=ok")
    ap.add_argument("--hash-size", type=int, default=16); ap.add_argument("--hash-thr", type=int, default=8, help="max Hamming distance (of 256 bits) for near-duplicates")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--freeze-epochs", type=int, default=3); ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="do not download ImageNet weights (smoke tests)")
    ap.add_argument("--degrade-res", type=int, default=0, help="control experiment: down/up-sample ALL images through this resolution")
    ap.add_argument("--gan-size", type=int, choices=[64, 128]); ap.add_argument("--gan-epochs", type=int)
    ap.add_argument("--n-synthetic", type=int, default=0, help="0 = enough to balance the classes (same count as oversampling)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--arms", nargs="+", default=["cw", "oversample", "gan"])
    ap.add_argument("--arm", default="gan"); ap.add_argument("--image", nargs="+"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); out = Path(a.out); print("device:", DEVICE)

    if a.cmd == "inspect":
        df = D.list_images(a.data_root, dict(kv.split("=") for kv in (a.label_map or [])))
        print("\nPer class:\n", df.label.value_counts()); print("\nPer original folder:\n", df.subtype.value_counts()); print(df.head()); from PIL import Image; print("example size:", Image.open(df.path[0]).size, Image.open(df.path[0]).mode); return
    if a.cmd == "prepare":
        df, classes = D.prepare(a, out)
        save_json({"classes": classes, "minority": a.minority, "minority_keep": a.minority_keep, "img_size": a.img_size, "batch_size": a.batch_size,
                   "epochs": a.epochs, "patience": a.patience, "freeze_epochs": a.freeze_epochs, "num_workers": a.num_workers, "seed": a.seed,
                   "pretrained": not a.no_pretrained, "degrade_res": a.degrade_res, "gan_size": a.gan_size or 64, "gan_epochs": a.gan_epochs or 300, "n_synthetic": a.n_synthetic}, out / "config.json")
        (out / "figures").mkdir(exist_ok=True); return
    if a.cmd == "predict":
        from capstone_a.predict import load_model, predict_paths
        m, meta = load_model(Path("artifacts") / "model"); P = predict_paths(m, meta, a.image)
        for p, pr in zip(a.image, P): print(p, {c: round(float(v), 4) for c, v in zip(meta["classes"], pr)}, "->", meta["classes"][int(pr.argmax())])
        return
    df, classes, cfg = D.load_split(out)
    from capstone_a import train as T, evaluate as E
    if a.cmd == "tune": T.tune(df, cfg, out)
    if a.cmd == "gan":
        import numpy as np
        from capstone_a.gan import train_gan
        tr = df[df.split == "train"]; cnt = tr.label.value_counts(); n = a.n_synthetic or int(cnt.max() - cnt[cfg["minority"]])
        train_gan(df, cfg["minority"], out, a.gan_size or cfg["gan_size"], a.gan_epochs or cfg["gan_epochs"], n, seed=0)
    if a.cmd == "arms": T.stage1_and_arms(df, cfg, out, a.seeds, a.arms)
    if a.cmd == "test": E.test_evaluation(df, cfg, out, a.arms, boot=a.boot)
    if a.cmd == "gradcam":
        from capstone_a.gradcam import run_gradcam; run_gradcam(df, cfg, out, a.arm)
    if a.cmd == "export":
        best = load_json(out / "best.json"); key = E._key(a.arm, best); s = E.seeds_of(out, key)
        import pandas as pd
        v = pd.read_csv(out / "validation_runs.csv"); v = v[v.arm == key].sort_values("macro_f1", ascending=False)   # seed chosen on VALIDATION
        s0 = int(v.iloc[0].seed); dst = Path("artifacts") / "model"; shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(T.run_dir(out, key, E._lr(a.arm, best), s0), dst, ignore=shutil.ignore_patterns("val_*.npy")); print(f"exported {a.arm} seed {s0} -> {dst}")


if __name__ == "__main__":
    main()
