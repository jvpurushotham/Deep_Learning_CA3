"""Image listing, NEAR-DUPLICATE grouping, leakage-safe group-aware stratified split, controlled minority reduction.

Why group-aware? Casting/defect datasets often contain augmented or near-identical copies of the same part. If copies
land in both train and test the score is inflated. Near-duplicates are found with a perceptual hash (dHash) and kept
together in one split.
"""
import random
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(root, label_map=None):
    label_map = label_map or {}
    rows = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() in IMG_EXT and not p.name.startswith("."):   # skip macOS "._xxx" resource files
            raw = p.parent.name.lower()
            rows.append({"path": str(p), "label": label_map.get(raw, raw), "subtype": raw})
    return pd.DataFrame(rows)


def dhash(path, size=16):
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()                       # size*size bits


def near_dup_groups(paths, size=16, thr=8):
    H = np.stack([dhash(p, size) for p in paths]).astype(np.uint8); n = len(H)
    parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n - 1):
        d = (H[i + 1:] != H[i]).sum(1)
        for j in np.nonzero(d <= thr)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b: parent[b] = a
    return np.array([find(i) for i in range(n)])


def prepare(args, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    lm = dict(kv.split("=") for kv in (args.label_map or []))
    df = list_images(args.data_root, {k.lower(): v for k, v in lm.items()})
    if df.empty: raise SystemExit(f"No images under {args.data_root}")
    classes = sorted(df.label.unique()); assert args.minority in classes, f"--minority must be one of {classes}"
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)})
    rep = {"n_images": len(df), "classes": classes, "per_class_total": df.label.value_counts().to_dict()}
    # a file must not have two labels
    df["group"] = near_dup_groups(df.path.tolist(), args.hash_size, args.hash_thr)
    gs = df.group.value_counts()
    rep.update(n_groups=int(len(gs)), n_images_in_multi_image_groups=int(gs[gs > 1].sum()), largest_group=int(gs.max()))
    if gs.max() > 0.05 * len(df): print("WARNING: one near-duplicate group holds >5% of the data; lower --hash-thr.")
    # 7 folds: 1 -> test (14.3%), 1 -> val (14.3%), 5 -> train (71.4%); duplicates never straddle splits, classes stratified.
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=args.seed)
    folds = np.zeros(len(df), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(df, df.y, df.group)): folds[te] = k
    df["split"] = np.where(folds == 0, "test", np.where(folds == 1, "val", "train"))
    assert df.groupby("group").split.nunique().max() == 1, "near-duplicates straddle splits"
    # controlled imbalance: reduce the minority class in TRAIN only
    rng = random.Random(args.seed); mi = classes.index(args.minority)
    tr_min = df.index[(df.split == "train") & (df.y == mi)].tolist()
    keep = set(rng.sample(tr_min, max(1, int(len(tr_min) * args.minority_keep))))
    drop = [i for i in tr_min if i not in keep]
    df.loc[drop, "split"] = "dropped_minority"
    df.to_csv(out / "split.csv", index=False)
    dist = df.pivot_table(index="label", columns="split", values="path", aggfunc="count").fillna(0).astype(int)
    dist.to_csv(out / "class_distribution.csv")
    rep["minority"] = args.minority; rep["minority_keep"] = args.minority_keep
    rep["train_imbalance_ratio_majority_over_minority"] = float(dist.loc[[c for c in classes if c != args.minority], "train"].max() / max(dist.loc[args.minority, "train"], 1))
    save_json_local(rep, out / "data_report.json"); print(pd.Series(rep).to_string()); print(dist)
    return df, classes


def save_json_local(o, p):
    from .common import save_json; save_json(o, p)


def load_split(out):
    df = pd.read_csv(Path(out) / "split.csv", keep_default_na=False, na_values=[])
    from .common import load_json
    cfg = load_json(Path(out) / "config.json")
    return df, cfg["classes"], cfg
