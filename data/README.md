# Data (not included in the zip)

Dataset named in your class notes: **Casting Defect Detection** — https://www.kaggle.com/datasets/simmoshaikh/casting-defect-detection

## Real structure (checked from your screenshots)
```
output/
  defect_dataset/   <- THE ONLY FOLDER USED: ok/ + 11 defect-type folders + dataset_stats.json
      ok, cold_shut, crack, dent, flash, inclusion, mixed_defects, pinhole, porosity, scratch, shrinkage, surface_roughness
  hand_overlays/    <- casting0_hand0 ... casting5_hand0 (derived pictures: do NOT use)
  visualizations/   <- augmentation_samples, defect_samples, intensity_samples (derived pictures: do NOT use)
```
Place the three folders under `data/raw/` so the images are at `data/raw/defect_dataset/<class>/<image>`.
Option A is binary: **ok vs defect**. All 11 defect folders (including `mixed_defects`) are merged into `defect` with `--label-map`; the original folder name is kept as `subtype` for the per-defect-type recall table.

## Commands
Use `scripts/01_prepare_and_train.sh` and `scripts/02_test_and_explain.sh` (see `docs/00_LOCAL_SETUP_GUIDE.md`), or the full commands in `README.md`.

## Check after `inspect` / `prepare`
1. **Counts:** images per class and per folder. Aim for >= ~60-100 defect images in the *training* split after the reduction (`--minority-keep`), otherwise raise it.
2. **Provenance:** read the Kaggle description + `dataset_stats.json`. The folder names (augmentation, overlays, intensity samples) suggest generated/augmented images; state what the data really is in your report and treat it as a limitation.
3. **Near-duplicates:** `data_report.json` -> `largest_group`; if large, lower `--hash-thr`.
