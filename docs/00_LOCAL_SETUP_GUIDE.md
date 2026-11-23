# Run Option A on your own computer — from scratch (Mac; Windows notes at the end)

Time needed: about 30 minutes of setup, then a few hours of mostly unattended computing on a CPU (a GPU / Colab is much faster). I could not test on your real dataset, so the first run is a *rehearsal*.

## Step 1 — Check Python
```bash
python3 --version        # want 3.10, 3.11 or 3.12
```
If it is missing or older, install from https://www.python.org/downloads/ (or `brew install python@3.12`).

## Step 2 — Unzip the project and create a virtual environment
```bash
cd ~/Desktop                           # or wherever you saved the zip
unzip CSD402_Capstone_OptionA.zip
cd CSD402_Capstone_OptionA
python3 -m venv venv                   # skip if you already have a venv folder here
source venv/bin/activate               # your prompt now starts with (venv)
pip install --upgrade pip
pip install -r requirements.txt        # downloads PyTorch etc. (several hundred MB)
python -c "import torch, torchvision; print('torch', torch.__version__)"
```
Every time you open a new Terminal window: `cd` into the project and run `source venv/bin/activate` again. **Always run commands from the project folder** (the one containing `run.py`).
In VS Code, open the folder `CSD402_Capstone_OptionA` and use *Terminal → New Terminal* — it starts in the right place.

## Step 3 — Prove the code works (no dataset needed)
```bash
python tests/smoke_test.py
```
It builds fake images, runs every stage and ends with `SMOKE TEST PASSED` (about 5–10 minutes on CPU). If this fails, fix it before anything else (see Troubleshooting).

## Step 4 — Put the dataset in the right place
1. Download from Kaggle (log in, click **Download**): https://www.kaggle.com/datasets/simmoshaikh/casting-defect-detection
2. Unzip it. You get a folder `output/` containing `defect_dataset/`, `hand_overlays/`, `visualizations/`.
3. Move those three folders into the project so you have exactly:
```
CSD402_Capstone_OptionA/data/raw/defect_dataset/ok/...
CSD402_Capstone_OptionA/data/raw/defect_dataset/crack/...        (and cold_shut, dent, flash, inclusion, mixed_defects,
                                                                  pinhole, porosity, scratch, shrinkage, surface_roughness)
CSD402_Capstone_OptionA/data/raw/defect_dataset/dataset_stats.json
```
**Only `defect_dataset` is used.** `hand_overlays` and `visualizations` are pictures derived from the same images; using them would leak data between train and test. (The scripts never look there.) Delete any `__MACOSX` folder if present (harmless, ignored).
4. Open `dataset_stats.json` and read the Kaggle "About" text: write down how many images there are and how the dataset was made (real photos, generated or augmented?). You must state this in the report.

## Step 5 — Look at the data
```bash
python run.py inspect --data-root data/raw/defect_dataset \
  --label-map cold_shut=defect crack=defect dent=defect flash=defect inclusion=defect mixed_defects=defect pinhole=defect porosity=defect scratch=defect shrinkage=defect surface_roughness=defect ok=ok
```
It prints images per class (ok vs defect), images per original folder, and the image size. Write the numbers down. Two decisions follow:
* If a defect folder has very few images, the GAN will have very little to learn from. After the controlled reduction (`KEEP`), aim for **at least ~60–100 defect images in the training split**. Raise `KEEP` (0.25 → 0.5) if you have fewer; the assignment only requires the defect class to be *under-represented*.
* `mixed_defects` is merged into `defect`; say so in the report.

## Step 6 — Rehearsal run (≈ 20–40 minutes on CPU; checks everything on the real data)
```bash
IMG=64 EPOCHS=4 GAN_EPOCHS=40 SEEDS="0" bash scripts/01_prepare_and_train.sh
```
(The first run downloads the ResNet-18 ImageNet weights, ~45 MB, so you need internet.) Check:
* `outputs/data_report.json` — `n_images`, `largest_group` (should be a small number; if one group holds > 5 % of the data you will see a WARNING → add `EXTRA="--hash-thr 4"`), the final train imbalance ratio.
* `outputs/figures/gan_samples.png` opens and shows images.
Then delete the rehearsal results: `rm -rf outputs artifacts`. **Do not run the test stage during the rehearsal.**

## Step 7 — The real run (Stage A: everything except the test set)
```bash
KEEP=0.25 IMG=128 SEEDS="0 1 2" bash scripts/01_prepare_and_train.sh
```
On a CPU this can take several hours (I could not measure it on your machine). Leave the laptop plugged in and awake. If it is too slow: `SEEDS="0 1"`, `EPOCHS=12`, `GAN_EPOCHS=150`, or use Google Colab (T4 GPU) with `notebooks/Colab_Runner.ipynb`. If you stop it, run the same command again — finished runs are skipped.
It runs: `inspect → prepare → tune → gan → arms` (see README for what each does). When it ends, look at, in this order:
1. `outputs/tuning_results.csv` — which backbone/learning rate won.
2. `outputs/figures/gan_samples.png` and `gan_losses.png` — do the fake defects look real and varied? Any mode collapse? Save a note.
3. `outputs/validation_comparison_mean_std.csv` — defect recall per arm on validation.
**Make any design change now** (e.g. a different `KEEP`, GAN epochs), because after the next step the test set must not influence decisions.

## Step 8 — Stage B: the held-out test (run this ONCE)
```bash
bash scripts/02_test_and_explain.sh
```
Produces the test tables, bootstrap confidence intervals, per-defect-type recall (`test_recall_per_defect_type.csv`), threshold tuning, Grad-CAM figure, and exports the model to `artifacts/model`. Then:
```bash
python run.py predict --image data/raw/defect_dataset/crack/<any image>.jpg
```

## Step 9 — Collect material for the report
| Report section | Take from `outputs/` |
|---|---|
| Data | `data_report.json`, `class_distribution.csv` |
| Stage 1 comparison | `tuning_results.csv`, `validation_comparison_mean_std.csv`, `figures/learning_curves.png` |
| GAN | `gan/gan_info.json`, `figures/gan_samples.png`, `figures/gan_losses.png` |
| Final results | `test_results_mean_std.csv`, `bootstrap_minority_recall_difference.json`, `threshold_tuning_target_recall.csv`, `test_recall_per_defect_type.csv`, `cost_comparison.csv`, `figures/minority_recall.png`, `figures/confusion_matrices.png` |
| Grad-CAM | `figures/gradcam_gan.png`, `gradcam_gan_examples.csv` |
Then follow `docs/03_TECHNICAL_REPORT_TEMPLATE.md` and `viva/VIVA_PREP.md`.

## Start over from scratch
`rm -rf outputs artifacts` (keeps your data and code), then Step 7 again.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `python3: command not found` / wrong version | Step 1 |
| `ModuleNotFoundError: capstone_a` | You are not in the project folder; `cd` to the folder with `run.py` |
| `ModuleNotFoundError: torch` | venv not active: `source venv/bin/activate` |
| `SSL: CERTIFICATE_VERIFY_FAILED` while downloading ResNet weights (Mac) | run `/Applications/Python\ 3.12/Install\ Certificates.command` (use your version number), retry |
| `No images found under ...` | wrong `DATA_ROOT`; check `ls data/raw/defect_dataset` |
| `--minority must be one of [...]` | the `--label-map` did not match the folder names; re-check spelling in `inspect` output |
| `Killed` / out of memory | `IMG=96`, add `EXTRA="--batch-size 16"` to the first script call, close other apps |
| Multiprocessing / DataLoader worker error | add `EXTRA="--num-workers 0"` |
| Very slow | `IMG=64` or `96`, fewer seeds/epochs (say so in the report), or Colab |
| WARNING: one near-duplicate group holds > 5 % | `EXTRA="--hash-thr 4"` |
| `Permission denied` on a script | run it with `bash scripts/01_prepare_and_train.sh` |
| Stage seems to restart | normal: finished runs are skipped, unfinished ones restart |

## Windows
Install Python 3.10–3.12 (tick "Add to PATH"). Use `python -m venv venv` then `venv\Scripts\activate`. The `.sh` scripts need **Git Bash** or **WSL**; otherwise paste the commands from the scripts (replace `$VARS` by values) into PowerShell. Everything else is identical (`python run.py ...`).
