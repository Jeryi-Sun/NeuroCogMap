#!/usr/bin/env bash
set -euo pipefail

PY="python3"
SCRIPT="/path/to/project_root/neural_area/connect_cap_parcel/code/extract_reverse_activation_parcels.py"

POS_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_0.5"
NEG_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_-1.0"
OUT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/reverse_activation_parcels"

$PY "$SCRIPT" \
  --pos_dir "$POS_DIR" \
  --neg_dir "$NEG_DIR" \
  --pos_strength 0.5 \
  --neg_strength -1.0 \
  --score_field logprob_diff_avg \
  --capability_stats \
  "/path/to/project_root/neural_area/capability_data_v2/data_stastic/final_merged_capability_dataset_stats.json" \
  --output_dir "$OUT_DIR"

echo "完成：结果已输出到 $OUT_DIR"
echo "主要输出文件："
echo "  - capability_reverse_activation_parcels_by_type.json (按能力分组的反向激活parcels)"
echo "  - reverse_activation_parcels_all_datasets.json (所有数据集的并集)"
echo "  - reverse_activation_index.json (统计索引)"


