# CSD402 Capstone (CA3) — Option A: Multi-Stage Visual Quality System (CNN + GAN)

**Core question (from the class notes):** *Does GAN-generated data really improve recall on the rare class (defect)?* — answered on a **real, held-out test set that the GAN never saw**.

| Stage | What | Where |
|---|---|---|
| 1 | CNN classifier: **from scratch vs pre-trained ResNet-18** (transfer learning), augmentation, dropout, early stopping, proper train/val/test | `tune`, `arms` |
| 2 | **DCGAN** trained only on minority images of the train split → synthetic minority images | `gan` |
| Controlled comparison | baseline vs **class-weighted loss** vs **random oversampling** vs **GAN augmentation** (same number of added minority samples as oversampling), 3 seeds, paired bootstrap | `arms`, `test` |
| Unit VI | **Grad-CAM** on test images (TP / FN / FP / TN) + border-share check for spurious cues | `gradcam` |

> Status: smoke-tested end-to-end on synthetic images with no downloads. **No real result is included — every number in your report must come from your own run.**

## Folder map
```
run.py                 command line entry point
capstone_a/            data.py models.py gan.py train.py evaluate.py gradcam.py predict.py common.py
tests/smoke_test.py    offline end-to-end test (synthetic images)
notebooks/             Colab_Runner.ipynb
data/README.md         how to get the Kaggle casting dataset and arrange folders
scripts/               two shell scripts: train stage / test stage
docs/                  00_LOCAL_SETUP_GUIDE (start here), step-by-step guide, proposal, report template, presentation plan
viva/                  viva preparation from scratch
reference_notebooks/   instructor's Option A template + original brief and class notes
```

## Reproduce (full from-scratch instructions: `docs/00_LOCAL_SETUP_GUIDE.md`)
Dataset: Kaggle *Casting Defect Detection*; use only `data/raw/defect_dataset/` (ok + 11 defect folders). See `data/README.md`.
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python tests/smoke_test.py                                   # offline check, no data needed
IMG=64 EPOCHS=4 GAN_EPOCHS=40 SEEDS="0" bash scripts/01_prepare_and_train.sh   # rehearsal, then: rm -rf outputs artifacts
KEEP=0.25 IMG=128 SEEDS="0 1 2" bash scripts/01_prepare_and_train.sh            # real run: inspect, prepare, tune, gan, arms
bash scripts/02_test_and_explain.sh                          # held-out TEST (once) + Grad-CAM + export
python run.py predict --image data/raw/defect_dataset/crack/<file>.jpg
```
The scripts wrap these commands (use them directly if you want finer control):
```bash
python run.py inspect --data-root data/raw/defect_dataset --label-map cold_shut=defect crack=defect dent=defect flash=defect inclusion=defect mixed_defects=defect pinhole=defect porosity=defect scratch=defect shrinkage=defect surface_roughness=defect ok=ok
python run.py prepare --data-root data/raw/defect_dataset --minority defect --minority-keep 0.25 --img-size 128 --label-map <same as above>
python run.py tune ; python run.py gan ; python run.py arms --seeds 0 1 2
python run.py test ; python run.py gradcam --arm gan ; python run.py export --arm gan
```
Rehearsal on CPU: `--img-size 128 --epochs 5 --gan-epochs 50 --seeds 0`.

### Latest changes
* Dataset-specific instructions (real folder structure, label map for the 11 defect types); `hand_overlays`/`visualizations` must not be used.
* New output `test_recall_per_defect_type.csv` (which defect types each arm finds); `subtype` column in `split.csv`.
* macOS `._*` resource files are ignored; `inspect` also lists images per original folder.
* `scripts/01_prepare_and_train.sh`, `scripts/02_test_and_explain.sh` (test stage separated to keep "test once" discipline); `docs/00_LOCAL_SETUP_GUIDE.md`.

## Where each reported number comes from (`outputs/`)
| File | Content |
|---|---|
| `data_report.json`, `class_distribution.csv` | counts, near-duplicate groups, imbalance ratio after the controlled reduction |
| `tuning_results.csv`, `best.json` | scratch vs ResNet-18 LR search; selected architecture and LRs |
| `validation_runs.csv`, `validation_comparison_mean_std.csv` | validation metrics, all arms |
| `gan/gan_info.json`, `figures/gan_losses.png`, `figures/gan_samples.png` | GAN diagnostics (uncurated samples, diversity ratio → mode collapse indicator) |
| `test_results_mean_std.csv`, `test_results_per_seed.csv` | final TEST: macro-F1, minority recall/precision/F1/PR-AUC, accuracy |
| `test_recall_per_defect_type.csv` | defect recall per original defect type and arm |
| `bootstrap_minority_recall_difference.json` | each arm minus baseline, 95 % CI, plus baseline seed sd |
| `threshold_tuning_target_recall.csv` | decision threshold tuned on validation for ≥ 95 % recall, applied to test |
| `cost_comparison.csv` | parameters, training time, CPU latency per image |
| `figures/` | learning curves, confusion matrices, minority recall bars, Grad-CAM |
| `gradcam_<arm>_examples.csv` | per-example Grad-CAM table with border share |
| `test_access.log` | audit of test-set access |

## Design decisions worth defending (details in `viva/`)
* **Near-duplicate-safe split:** perceptual-hash (dHash) groups are kept inside one split (`StratifiedGroupKFold`), because industrial image sets often contain augmented copies; assertion checks no group straddles splits.
* **Controlled imbalance:** the minority is reduced in the **train split only** (`--minority-keep`); validation/test keep the natural distribution. Dropped images are not reused anywhere.
* **GAN leakage control:** the GAN function asserts it only receives minority images of the train split.
* **Fair arms:** oversampling and GAN add the *same number* of minority samples; same optimiser, schedule, epochs, early-stopping rule, learning rate and selection metric for every arm.
* **Two-phase transfer learning:** frozen backbone (BatchNorm kept in eval mode) for 3 epochs, then full fine-tuning with cosine LR.
* **Resolution-confound control:** the DCGAN works at 64×64 (or 128). Synthetic images are therefore blurrier than real ones; a classifier could learn "blurry = defect". `--degrade-res N` pushes *all* images through the same resolution as a control, `--gan-size 128` reduces the gap.
* **Test once:** `test` is a separate command and logs each access. Threshold tuning uses validation only.

## Using the trained model
```python
from capstone_a.predict import load_model, predict_paths
model, meta = load_model("artifacts/model")
probs = predict_paths(model, meta, ["part_001.png"]); print(meta["classes"], probs)
```

## Citations / integrity (edit before submission)
ResNet-18 (He et al., 2016) with ImageNet weights from torchvision; DCGAN (Radford et al., 2016); Grad-CAM (Selvaraju et al., 2017); dataset: <Kaggle dataset + author>; PyTorch, torchvision, scikit-learn, pandas, matplotlib. Structure adapted from the course's Option A template. Code/documents drafted with generative-AI assistance (Claude); declare your own review honestly.
