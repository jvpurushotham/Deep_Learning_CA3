"""Offline end-to-end smoke test: synthetic 'casting' images (ok = clean disc, defect = disc with dark blemishes), no downloads.
Verifies prepare -> tune -> gan -> arms -> test -> gradcam -> export -> predict. Scores are meaningless; this only proves the wiring.
    python tests/smoke_test.py
"""
import random, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]; tmp = Path(tempfile.mkdtemp(prefix="capA_smoke_")); rng = np.random.default_rng(0); random.seed(0)
def make(defect, seed):
    r = np.random.default_rng(seed); im = Image.new("L", (96, 96), 40); d = ImageDraw.Draw(im); d.ellipse((8, 8, 88, 88), fill=int(r.integers(120, 170)))
    if defect:
        for _ in range(int(r.integers(1, 4))):
            x, y, s = r.integers(25, 70), r.integers(25, 70), r.integers(3, 8); d.ellipse((x - s, y - s, x + s, y + s), fill=20)
    a = np.asarray(im, dtype=np.float32) + r.normal(0, 6, (96, 96)); return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
for cls, flag, n in [("ok_front", False, 300), ("def_front", True, 300)]:
    (tmp / "raw" / cls).mkdir(parents=True)
    for i in range(n):
        im = make(flag, i + (1000 if flag else 0)); im.save(tmp / "raw" / cls / f"{i}.png")
        if i % 10 == 0: im.rotate(0).save(tmp / "raw" / cls / f"{i}_copy.png")        # exact duplicates -> must be grouped
out = tmp / "out"; C = ["--out", str(out)]
def run(*a):
    cmd = [sys.executable, "run.py", *a]; print("\n$", " ".join(cmd[1:])); subprocess.run(cmd, cwd=ROOT, check=True)
common = ["--data-root", str(tmp / "raw"), "--label-map", "def_front=defect", "ok_front=ok"]
run("inspect", *common)
run("prepare", *C, *common, "--minority", "defect", "--minority-keep", "0.25", "--img-size", "64", "--epochs", "3", "--patience", "2", "--freeze-epochs", "1",
    "--no-pretrained", "--num-workers", "0", "--batch-size", "32", "--gan-epochs", "20")
run("tune", *C); run("gan", *C)
run("arms", *C, "--seeds", "0", "1"); run("test", *C, "--boot", "30"); run("gradcam", *C, "--arm", "gan"); run("export", *C, "--arm", "gan")
run("predict", *C, "--image", str(tmp / "raw" / "def_front" / "3.png"), str(tmp / "raw" / "ok_front" / "3.png"))
shutil.rmtree(ROOT / "artifacts", ignore_errors=True); print("\nSMOKE TEST PASSED. Outputs in", out)
