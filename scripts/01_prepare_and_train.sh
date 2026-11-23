#!/usr/bin/env bash
# Stage A: inspect -> prepare -> tune -> gan -> arms.  Does NOT touch the test set.
# Run from the project root:   bash scripts/01_prepare_and_train.sh
# Override settings with environment variables, e.g.:  KEEP=0.10 IMG=224 SEEDS="0 1 2" bash scripts/01_prepare_and_train.sh
set -euo pipefail
DATA_ROOT="${DATA_ROOT:-data/raw/defect_dataset}"   # ONLY the defect_dataset folder (not hand_overlays / visualizations)
KEEP="${KEEP:-0.25}"            # fraction of TRAIN defect images kept (controlled imbalance)
IMG="${IMG:-128}"               # 128 = fast on CPU; use 224 on a GPU
EPOCHS="${EPOCHS:-20}"; PATIENCE="${PATIENCE:-5}"; GAN_EPOCHS="${GAN_EPOCHS:-300}"
SEEDS="${SEEDS:-0 1 2}"; OUT="${OUT:-outputs}"; EXTRA="${EXTRA:-}"
LABELMAP="cold_shut=defect crack=defect dent=defect flash=defect inclusion=defect mixed_defects=defect pinhole=defect porosity=defect scratch=defect shrinkage=defect surface_roughness=defect ok=ok"

python run.py inspect --data-root "$DATA_ROOT" --label-map $LABELMAP
python run.py prepare --out "$OUT" --data-root "$DATA_ROOT" --minority defect --minority-keep "$KEEP" --img-size "$IMG" \
       --epochs "$EPOCHS" --patience "$PATIENCE" --gan-epochs "$GAN_EPOCHS" --label-map $LABELMAP $EXTRA
python run.py tune --out "$OUT"
python run.py gan  --out "$OUT"
python run.py arms --out "$OUT" --seeds $SEEDS
echo
echo "Stage A finished. Now LOOK at: $OUT/data_report.json, $OUT/tuning_results.csv, $OUT/figures/gan_samples.png, $OUT/figures/gan_losses.png, $OUT/figures/learning_curves.png (after test)."
echo "Only when every design decision is final, run:  bash scripts/02_test_and_explain.sh"
