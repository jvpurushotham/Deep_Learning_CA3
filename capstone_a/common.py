import json, random, time
from pathlib import Path
import numpy as np, torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=float)


def load_json(path):
    with open(path) as f: return json.load(f)


def log_test_access(out, what):
    with open(Path(out) / "test_access.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  test set evaluated by: {what}\n")
