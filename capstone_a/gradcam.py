"""Grad-CAM on test images (correct and incorrect), with a simple quantitative 'spurious cue' check: share of CAM energy on the image border."""
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .common import DEVICE, MEAN, STD, load_json
from .models import cam_layer
from .predict import load_model
from .train import tfs, run_dir
from .evaluate import _key, _lr, seeds_of


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.a, self.g = model, None, None
        layer.register_forward_hook(self._fwd)
    def _fwd(self, m, i, o):
        self.a = o
        if o.requires_grad: o.register_hook(lambda g: setattr(self, "g", g))        # tensor hook: safe with in-place ReLU
    def __call__(self, x, target=None):
        self.model.eval(); x = x.clone().requires_grad_(True); out = self.model(x)
        c = int(out.argmax(1)) if target is None else target
        self.model.zero_grad(); out[0, c].backward()
        w = self.g.mean(dim=(2, 3), keepdim=True); cam = F.relu((w * self.a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        o = out.detach(); return cam.cpu().numpy(), int(o.argmax(1)), float(o.softmax(1)[0, int(o.argmax(1))])


def border_share(cam, frac=0.15):
    h, w = cam.shape; b = int(frac * h); mask = np.ones_like(cam, bool); mask[b:h - b, b:w - b] = False
    return float(cam[mask].sum() / max(cam.sum(), 1e-8))


def run_gradcam(df, cfg, out, arm, seed=None, n_each=3):
    out = Path(out); best = load_json(out / "best.json"); key = _key(arm, best); seed = seeds_of(out, key)[0] if seed is None else seed
    m, meta = load_model(run_dir(out, key, _lr(arm, best), seed)); m = m.to(DEVICE); cam = GradCAM(m, cam_layer(m))
    mi = cfg["classes"].index(cfg["minority"]); te = df[df.split == "test"].reset_index(drop=True); _, ev = tfs(meta["img_size"], meta.get("degrade_res", 0))
    X = torch.stack([ev(Image.open(p).convert("RGB")) for p in te.path]); 
    with torch.no_grad(): P = torch.cat([m(X[i:i + 64].to(DEVICE)).softmax(1).cpu() for i in range(0, len(X), 64)]).numpy()
    pred, y = P.argmax(1), te.y.values
    groups = {"TP (defect found)": (y == mi) & (pred == mi), "FN (defect MISSED)": (y == mi) & (pred != mi),
              "FP (false alarm)": (y != mi) & (pred == mi), "TN (ok, correct)": (y != mi) & (pred != mi)}
    rng = np.random.default_rng(0); sel = []
    for name, mask in groups.items():
        idx = np.nonzero(mask)[0]; idx = rng.choice(idx, min(n_each, len(idx)), replace=False) if len(idx) else []
        sel += [(name, int(i)) for i in idx]
    rows = []; fig, axes = plt.subplots(2, len(sel), figsize=(2.4 * len(sel), 5.4), squeeze=False)
    for k, (name, i) in enumerate(sel):
        x = X[i:i + 1].to(DEVICE); heat, c, conf = cam(x); img = np.clip(X[i].permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
        bs = border_share(heat); rows.append({"group": name, "path": te.path[i], "true": cfg["classes"][y[i]], "pred": cfg["classes"][c], "confidence": conf, "cam_border_share": bs})
        axes[0, k].imshow(img); axes[0, k].set_title(f"{name}\ntrue: {cfg['classes'][y[i]]}", fontsize=7)
        axes[1, k].imshow(img); axes[1, k].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, k].set_title(f"pred: {cfg['classes'][c]} ({conf:.2f})\nborder share {bs:.2f}", fontsize=7, color="green" if c == y[i] else "red")
        axes[0, k].axis("off"); axes[1, k].axis("off")
    plt.tight_layout(); (out / "figures").mkdir(exist_ok=True); plt.savefig(out / "figures" / f"gradcam_{arm}.png", dpi=130); plt.close()
    r = pd.DataFrame(rows); r.to_csv(out / f"gradcam_{arm}_examples.csv", index=False)
    uniform = 1 - (1 - 2 * 0.15) ** 2
    print(r.round(3).to_string(index=False)); print(f"\nBorder-share reference for a UNIFORM map: {uniform:.2f}. Much higher => the model may rely on the border/background.")
    return r
