"""Held-out TEST evaluation (run once): all arms, mean +- sd over seeds, paired bootstrap CI on minority recall, threshold tuning."""
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, recall_score

from .common import load_json, save_json, log_test_access, DEVICE
from .predict import load_model, predict_paths, latency_ms
from .train import run_dir, summarise, items_of

ARM_ORDER = ["baseline", "cw", "oversample", "gan"]      # arms that use the best architecture; "baseline" == the best-arch run


def _key(arm, best): return best["best_arch"] if arm == "baseline" else arm
def _lr(arm, best): return best["per_arch_lr"][best["best_arch"]] if arm in ARM_ORDER else best["per_arch_lr"][arm]
def seeds_of(out, key):
    return sorted(int(p.name.split("_seed")[1]) for p in (Path(out) / "runs" / key).glob("lr*_seed*") if (p / "val_result.json").exists())


def test_evaluation(df, cfg, out, arms, boot=1000, target_recall=0.95):
    out = Path(out); best = load_json(out / "best.json"); log_test_access(out, "evaluate.test_evaluation")
    te = df[df.split == "test"]; paths, y = te.path.tolist(), te.y.values; mi = cfg["classes"].index(cfg["minority"])
    base = best["best_arch"]; names = list(best["per_arch_lr"]) + arms; probs, vprobs, rows = {}, {}, []
    (out / "preds").mkdir(exist_ok=True)
    for a in names:
        key, lr = _key(a, best), _lr(a, best)
        for s in seeds_of(out, key):
            rd = run_dir(out, key, lr, s); m, meta = load_model(rd)
            p = predict_paths(m, meta, paths); probs[(a, s)] = p; np.save(out / "preds" / f"{a}_seed{s}_test_probs.npy", p)
            vprobs[(a, s)] = (np.load(rd / "val_probs.npy"), np.load(rd / "val_y.npy"))
            rows.append({"arm": a, "seed": s, **summarise(y, p, mi)})
    t = pd.DataFrame(rows); t.to_csv(out / "test_results_per_seed.csv", index=False)
    cols = ["macro_f1", "minority_recall", "minority_precision", "minority_f1", "minority_pr_auc", "accuracy"]
    agg = t.groupby("arm")[cols].agg(["mean", "std"]).round(4).reindex(names); agg.to_csv(out / "test_results_mean_std.csv")
    print("\nTEST (mean/std over seeds):"); print(agg.to_string())

    # paired bootstrap on test images: minority recall of each arm minus baseline (mean over seeds)
    rng = np.random.default_rng(0); diffs = {}
    for a in names:
        if a == base: continue
        common = [s for s in seeds_of(out, _key(a, best)) if (base, s) in probs and (a, s) in probs]
        if not common: continue
        d = []
        for _ in range(boot):
            idx = rng.integers(0, len(y), len(y)); yy = y[idx] == mi
            d.append(np.mean([recall_score(yy, probs[(a, s)][idx].argmax(1) == mi, zero_division=0) - recall_score(yy, probs[(base, s)][idx].argmax(1) == mi, zero_division=0) for s in common]))
        diffs[a] = {"vs": base, "mean_diff_minority_recall": float(np.mean(d)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "seed_sd_of_baseline_recall": float(t[t.arm == base].minority_recall.std())}
    save_json(diffs, out / "bootstrap_minority_recall_difference.json"); print(f"\nPaired bootstrap, minority recall difference vs baseline ({base}):"); print(diffs)

    # per-defect-type recall (needs the 'subtype' column = original folder name): which defect types does each arm find?
    if "subtype" in te:
        rr = []
        for (a, s), p in probs.items():
            hit = p.argmax(1) == mi
            for st in te[te.y == mi].subtype.unique():
                m = (te.subtype == st).values; rr.append({"arm": a, "subtype": st, "n_test": int(m.sum()), "recall": float(hit[m].mean())})
        sub = pd.DataFrame(rr); piv = sub.groupby(["subtype", "arm"]).recall.mean().unstack().round(3)
        piv.insert(0, "n_test", sub.groupby("subtype").n_test.first()); piv.to_csv(out / "test_recall_per_defect_type.csv")
        print("\nTest recall per original defect type (mean over seeds):"); print(piv.to_string())

    # decision-threshold tuning: threshold on P(minority) chosen on VALIDATION to reach target recall, then applied to TEST
    thr_rows = []
    for (a, s), pt in probs.items():
        pv, yv = vprobs[(a, s)]; pos = np.sort(pv[yv == mi, mi])
        thr = float(np.quantile(pos, 1 - target_recall)) if len(pos) else 0.5
        pred = pt[:, mi] >= thr; yt = y == mi
        thr_rows.append({"arm": a, "seed": s, "threshold": thr, "test_recall": recall_score(yt, pred, zero_division=0),
                         "test_precision": float((pred & yt).sum() / max(pred.sum(), 1)), "flagged_share": float(pred.mean())})
    tt = pd.DataFrame(thr_rows).groupby("arm")[["threshold", "test_recall", "test_precision", "flagged_share"]].mean().round(4).reindex(names)
    tt.to_csv(out / "threshold_tuning_target_recall.csv"); print(f"\nThreshold tuned on validation for recall>={target_recall}:"); print(tt)

    # cost (inference latency per image, one example) + confusion matrices + figures
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    cost = []
    for a in names:
        key = _key(a, best); s0 = seeds_of(out, key)[0]; m, meta = load_model(run_dir(out, key, _lr(a, best), s0))
        r = load_json(run_dir(out, key, _lr(a, best), s0) / "val_result.json")
        cost.append({"arm": a, "parameters": r["parameters"], "train_time_s": r["train_time_s"], "epochs_run": r["epochs_run"],
                     "latency_ms_cpu_per_image": latency_ms(m.cpu(), meta, paths[0], "cpu")})
    pd.DataFrame(cost).set_index("arm").round(2).to_csv(out / "cost_comparison.csv"); print(pd.DataFrame(cost).round(2).to_string(index=False))

    fig, ax = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.3), squeeze=False)
    for i, a in enumerate(names):
        s0 = seeds_of(out, _key(a, best))[0]; cm = confusion_matrix(y, probs[(a, s0)].argmax(1), labels=range(len(cfg["classes"])))
        ax[0, i].imshow(cm, cmap="Blues")
        for u in range(cm.shape[0]):
            for v in range(cm.shape[1]): ax[0, i].text(v, u, cm[u, v], ha="center", va="center")
        ax[0, i].set_xticks(range(len(cfg["classes"])), cfg["classes"], rotation=45); ax[0, i].set_yticks(range(len(cfg["classes"])), cfg["classes"])
        ax[0, i].set_title(a, fontsize=9); ax[0, i].set_xlabel("predicted"); ax[0, i].set_ylabel("true")
    plt.tight_layout(); plt.savefig(fig_dir / "confusion_matrices.png", dpi=150); plt.close()
    g = t.groupby("arm").minority_recall.agg(["mean", "std"]).reindex(names)
    plt.figure(figsize=(1.4 * len(names) + 2, 3.5)); plt.bar(names, g["mean"], yerr=g["std"].fillna(0), capsize=4); plt.ylabel("test minority recall")
    plt.title("Minority-class recall (mean ± sd over seeds)"); plt.xticks(rotation=20); plt.grid(alpha=.3, axis="y"); plt.tight_layout(); plt.savefig(fig_dir / "minority_recall.png", dpi=150); plt.close()
    main = "gan" if "gan" in names else names[-1]; s0 = seeds_of(out, _key(main, best))[0]
    (out / "classification_report_main.txt").write_text(classification_report(y, probs[(main, s0)].argmax(1), target_names=cfg["classes"], digits=4, zero_division=0))
    # learning curves
    fig, ax = plt.subplots(len(names), 2, figsize=(9, 2.6 * len(names)), squeeze=False)
    for i, a in enumerate(names):
        h = pd.DataFrame(load_json(run_dir(out, _key(a, best), _lr(a, best), seeds_of(out, _key(a, best))[0]) / "history.json"))
        ax[i, 0].plot(h.epoch, h.train_loss, label="train"); ax[i, 0].plot(h.epoch, h.val_loss, label="val"); ax[i, 0].set_title(f"{a}: loss"); ax[i, 0].legend()
        ax[i, 1].plot(h.epoch, h.val_macro_f1, label="val macro-F1"); ax[i, 1].plot(h.epoch, h.val_min_recall, label="val minority recall"); ax[i, 1].legend(); ax[i, 1].set_title(f"{a}: validation")
        for b in ax[i]: b.grid(alpha=.3); b.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(fig_dir / "learning_curves.png", dpi=150); plt.close()
    return t, agg, diffs
