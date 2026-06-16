#!/usr/bin/env bash
# 功能：运行 sycophancy 干预结果聚合脚本。
# 输入：results/intervention/strength_* 目录下的 *_summary.json
# 输出：results/intervention/aggregate/intervention_sycophancy_table_strengths.csv/.json
# 用法：在任意目录执行
#   bash run_aggregate_intervention_effect.sh
#   bash run_aggregate_intervention_effect.sh --intervention_base_dir <DIR> --output_dir <DIR>

set -euo pipefail

BASE_DIR="/path/to/project_root"
SYCOPHANCY_DIR="$BASE_DIR/safety_explanation/sycophancy"
INTERVENTION_ROOT="$SYCOPHANCY_DIR/results/intervention"
OUTPUT_DIR="$SYCOPHANCY_DIR/results/intervention/aggregate"
PYTHON_SCRIPT="$SYCOPHANCY_DIR/code/detection/aggregate_intervention_sycophancy_table.py"
PYTHON_BIN="python"

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
  echo "[ERROR] 找不到聚合脚本: $PYTHON_SCRIPT"
  exit 1
fi

if [[ ! -d "$INTERVENTION_ROOT" ]]; then
  echo "[ERROR] 找不到干预结果目录: $INTERVENTION_ROOT"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[WARNING] 指定 Python 不可执行，回退到 python3: $PYTHON_BIN"
  PYTHON_BIN="python3"
fi

echo "[INFO] 使用 Python: $PYTHON_BIN"
echo "[INFO] 聚合脚本: $PYTHON_SCRIPT"
echo "[INFO] 输入目录: $INTERVENTION_ROOT"
echo "[INFO] 输出目录: $OUTPUT_DIR"

"$PYTHON_BIN" "$PYTHON_SCRIPT" \
  --intervention_base_dir "$INTERVENTION_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  "$@"

echo "[INFO] 聚合完成。"
