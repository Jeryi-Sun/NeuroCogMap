#!/usr/bin/env bash
set -euo pipefail

POS_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_0.5_8b"
NEG_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_-1.0_8b"
OUT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate_intervention_8b"

SCRIPT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/code"

python3 "$SCRIPT_DIR/merge_intervention_scores.py" \
  --pos_dir "$POS_DIR" \
  --neg_dir "$NEG_DIR" \
  --output_dir "$OUT_DIR" \
  --score_field "logprob_diff_avg" \
  --pos_strength 0.5 \
  --neg_strength -1.0 \
  --norm minmax

echo "完成：输出在 $OUT_DIR"


