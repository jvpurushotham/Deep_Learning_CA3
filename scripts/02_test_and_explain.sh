#!/usr/bin/env bash
# Stage B: HELD-OUT TEST (run once!) -> Grad-CAM -> export model.
# Run from the project root:   bash scripts/02_test_and_explain.sh
set -euo pipefail
OUT="${OUT:-outputs}"; ARM="${ARM:-gan}"; BOOT="${BOOT:-1000}"
python run.py test    --out "$OUT" --boot "$BOOT"
python run.py gradcam --out "$OUT" --arm "$ARM"
python run.py export  --out "$OUT" --arm "$ARM"
echo "Done. Results are in $OUT/ (tables .csv/.json, figures/) and the model is in artifacts/model"
