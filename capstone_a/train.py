"""Shared training function for every experiment arm (same optimiser, schedule, stopping rule, selection metric)."""
import copy, shutil, time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .common import DEVICE, MEAN, STD, set_seed, save_json, load_json
from .models import build_model, set_backbone_trainable

GRIDS = {"scratch": [3e-4, 1e-3, 3e-3], "resnet18": [1e-4, 3e-4, 1e-3]}


def tfs(size, degrade=0):
    pre = [transforms.Resize((degrade, degrade)), transforms.Resize((size, size))] if degrade else []
    train = transforms.Compose(pre + [transforms.RandomResizedCrop(size, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                               transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ev = transforms.Compose(pre + [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, ev


class ImgDS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]; return self.tf(Image.open(p).convert("RGB")), y


def make_loader(items, tf, cfg, shuffle, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ImgDS(items, tf), batch_size=cfg["batch_size"], shuffle=shuffle, num_workers=cfg["num_workers"], generator=g if shuffle else None)


@torch.no_grad()
def predict(model, dl):
    model.eval(); ys, ps = [], []
    for x, y in dl: ps.append(model(x.to(DEVICE)).float().softmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def summarise(y, p, mi):
    yp = p.argmax(1); oh = (y == mi).astype(int)
    return {"macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
            "minority_recall": float(recall_score(y == mi, yp == mi, zero_division=0)),
            "minority_precision": float(precision_score(y == mi, yp == mi, zero_division=0)),
            "minority_f1": float(f1_score(y == mi, yp == mi, zero_division=0)),
            "minority_pr_auc": float(average_precision_score(oh, p[:, mi])) if oh.sum() else float("nan"),
            "accuracy": float((y == yp).mean())}


def items_of(df, split): return [(r.path, int(r.y)) for r in df[df.split == split].itertuples()]


def arm_items(arm, df, cfg, out):
    """Training items for each experiment arm. Oversampling and GAN add the SAME number of minority samples (class balance)."""
    tr = items_of(df, "train"); mi = cfg["classes"].index(cfg["minority"])
    cnt = np.bincount([y for _, y in tr], minlength=len(cfg["classes"])); need = int(cnt.max() - cnt[mi])
    if arm == "oversample":
        mins = [t for t in tr if t[1] == mi]; rng = np.random.default_rng(cfg["seed"])
        return tr + [mins[i] for i in rng.integers(0, len(mins), need)]
    if arm == "gan":
        syn = sorted((Path(out) / "synthetic" / cfg["minority"]).glob("*.png"))
        if len(syn) < need: print(f"note: only {len(syn)} synthetic images for {need} needed")
        return tr + [(str(p), mi) for p in syn[:need]]
    return tr


def run_dir(out, arm, lr, seed): return Path(out) / "runs" / arm / f"lr{lr:g}_seed{seed}"


def train_run(arm, arch, lr, seed, df, cfg, out, quiet=False):
    """arm in {scratch, resnet18, baseline, cw, oversample, gan}; arch = backbone used."""
    key = arch if arm == "baseline" else arm
    rd = run_dir(out, key, lr, seed)
    if (rd / "val_result.json").exists(): return load_json(rd / "val_result.json")
    set_seed(seed); n = len(cfg["classes"]); mi = cfg["classes"].index(cfg["minority"])
    train_items = arm_items(arm if arm in ("oversample", "gan") else "plain", df, cfg, out)
    tr_tf, ev_tf = tfs(cfg["img_size"], cfg["degrade_res"])
    dl_tr = make_loader(train_items, tr_tf, cfg, True, seed); dl_va = make_loader(items_of(df, "val"), ev_tf, cfg, False)
    model = build_model(arch, n, cfg["pretrained"]).to(DEVICE)
    w = None
    if arm == "cw":
        cnt = np.bincount([y for _, y in train_items], minlength=n).astype(float)
        w = torch.tensor(cnt.sum() / (n * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    E = cfg["epochs"]; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    freeze = cfg["freeze_epochs"] if arch == "resnet18" and cfg["pretrained"] else 0
    hist, best, best_state, bad, best_ep, t0 = [], -1.0, None, 0, 0, time.time()
    for ep in range(1, E + 1):
        set_backbone_trainable(model, ep > freeze); model.train()
        if ep <= freeze:                                                  # frozen backbone: keep BatchNorm statistics fixed
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
        tl = c = 0
        for x, y in dl_tr:
            x, y = x.to(DEVICE), y.to(DEVICE); loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(y); c += len(y)
        sched.step()
        yv, pv = predict(model, dl_va); vl = F.cross_entropy(torch.log(torch.tensor(pv).clamp_min(1e-12)), torch.tensor(yv)).item()
        s = summarise(yv, pv, mi)
        hist.append({"epoch": ep, "train_loss": tl / c, "val_loss": vl, "val_macro_f1": s["macro_f1"], "val_min_recall": s["minority_recall"]})
        if not quiet: print(f"[{key} lr={lr:g} s={seed}] ep {ep:2d} train {tl/c:.4f} val {vl:.4f} macroF1 {s['macro_f1']:.4f} minRecall {s['minority_recall']:.4f}")
        if s["macro_f1"] > best: best, best_ep, bad, best_state = s["macro_f1"], ep, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    tt = time.time() - t0; model.load_state_dict(best_state)
    yv, pv = predict(model, dl_va); rd.mkdir(parents=True, exist_ok=True)
    res = {"arm": key, "arch": arch, "lr": lr, "seed": seed, **summarise(yv, pv, mi), "best_epoch": best_ep, "epochs_run": len(hist),
           "train_time_s": round(tt, 1), "parameters": sum(p.numel() for p in model.parameters()), "n_train_items": len(train_items)}
    torch.save(model.state_dict(), rd / "model.pt"); np.save(rd / "val_probs.npy", pv); np.save(rd / "val_y.npy", yv)
    save_json({"arch": arch, "classes": cfg["classes"], "minority": cfg["minority"], "img_size": cfg["img_size"], "degrade_res": cfg["degrade_res"]}, rd / "meta.json")
    save_json(hist, rd / "history.json"); save_json(res, rd / "val_result.json"); return res


def tune(df, cfg, out):
    """Stage-1 search: scratch vs ResNet-18, 3 LRs each on seed 0 (validation macro-F1). Picks best LR per arch and best arch overall."""
    rows = [train_run(a, a, lr, 0, df, cfg, out) for a in GRIDS for lr in GRIDS[a]]
    t = pd.DataFrame(rows); t.to_csv(Path(out) / "tuning_results.csv", index=False)
    per = {a: float(t[t.arch == a].loc[t[t.arch == a].macro_f1.idxmax()].lr) for a in GRIDS}
    arch = max(per, key=lambda a: t[(t.arch == a) & (t.lr == per[a])].macro_f1.iloc[0])
    for a in GRIDS:
        for lr in GRIDS[a]:
            if lr != per[a]: shutil.rmtree(run_dir(out, a, lr, 0), ignore_errors=True)
    save_json({"per_arch_lr": per, "best_arch": arch}, Path(out) / "best.json")
    print(t.pivot(index="arch", columns="lr", values="macro_f1").round(4)); print("Selected:", per, "best arch:", arch); return per, arch


def stage1_and_arms(df, cfg, out, seeds, arms):
    b = load_json(Path(out) / "best.json"); rows = []
    for a in GRIDS:                                                          # Experiment 1: scratch vs pretrained (all seeds)
        for s in seeds: rows.append(train_run(a, a, b["per_arch_lr"][a], s, df, cfg, out))
    for arm in arms:                                                        # Experiments 2-3: remedies for imbalance
        for s in seeds: rows.append(train_run(arm, b["best_arch"], b["per_arch_lr"][b["best_arch"]], s, df, cfg, out))
    t = pd.DataFrame(rows).drop_duplicates(["arm", "seed"]); t.to_csv(Path(out) / "validation_runs.csv", index=False)
    g = t.groupby("arm")[["macro_f1", "minority_recall", "minority_precision", "minority_pr_auc", "accuracy", "train_time_s"]].agg(["mean", "std"]).round(4)
    g.to_csv(Path(out) / "validation_comparison_mean_std.csv"); print(g.to_string()); return t
