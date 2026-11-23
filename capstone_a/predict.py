from pathlib import Path
import time, numpy as np, torch
from PIL import Image

from .common import DEVICE, load_json
from .models import build_model
from .train import tfs


def load_model(run_dir, device=DEVICE):
    run_dir = Path(run_dir); meta = load_json(run_dir / "meta.json")
    m = build_model(meta["arch"], len(meta["classes"]), pretrained=False)     # weights come from model.pt, no download needed
    m.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu")); m.to(device).eval()
    return m, meta


@torch.no_grad()
def predict_paths(model, meta, paths, device=DEVICE, bs=32):
    _, ev = tfs(meta["img_size"], meta.get("degrade_res", 0)); out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([ev(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device)
        out.append(model(x).float().softmax(1).cpu().numpy())
    return np.concatenate(out)


def latency_ms(model, meta, path, device, n=30):
    _, ev = tfs(meta["img_size"], meta.get("degrade_res", 0)); ts = []
    m = model.to(device).eval()
    for _ in range(n + 3):
        s = time.perf_counter()
        with torch.no_grad(): m(ev(Image.open(path).convert("RGB")).unsqueeze(0).to(device))
        if device == "cuda": torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - s))
    return float(np.median(ts[3:]))
