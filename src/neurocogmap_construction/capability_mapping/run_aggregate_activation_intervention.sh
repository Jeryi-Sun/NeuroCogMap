#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="python"
SCRIPT_PATH="/path/to/project_root/neural_area/connect_cap_parcel/code/aggregate_activation_intervention.py"

# 运行模式：all=处理所有数据集，single=只处理指定数据集
MODE="all"             # all | single
DATASET="adversarial"  # MODE=single 时有效

# 聚合参数
K=50
ALPHA=0.5
PREFER_OPTIMIZED=1      # 1=开启，0=关闭
STRENGTH_MODE="mean"   # mean | at | max_abs
AT_STRENGTH=-1.0
L_ABS=1                 # 1=对 L 取绝对值，0=不取

# 文件路径
ACTIVATION_FILE="/path/to/project_root/neural_area/connect_cap_parcel/results/rank_activation_8b/parcel_activation_rankings.json"
INTERVENTION_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate_intervention_8b"  # 使用合并后的干预结果
OUTPUT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate_data_parcel_8b"

# 构建基础命令
CMD=("$PYTHON_BIN" "$SCRIPT_PATH" \
  --K "$K" \
  --alpha "$ALPHA" \
  --score_field logprob_diff_avg \
  --strength_mode "$STRENGTH_MODE" \
  --at_strength "$AT_STRENGTH" \
  --activation_file "$ACTIVATION_FILE" \
  --intervention_dir "$INTERVENTION_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --norm "minmax" \
  --intervention_mode merged \
  --per_dataset_norm \
  --report_stats)

# 单数据集模式：添加数据集过滤
if [[ "$MODE" == "single" ]]; then
  CMD+=(--dataset_filter "$DATASET")
fi

# 可选参数
# merged 模式下无需 --prefer_optimized

# merged 模式下通常不再对 L 取绝对值，这里保持默认

echo "> Running: ${CMD[*]}" >&2
exec "${CMD[@]}"


