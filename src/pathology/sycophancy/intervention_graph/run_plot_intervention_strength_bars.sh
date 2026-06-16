#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   读取 aggregate_intervention_sycophancy_table 生成的汇总 CSV
#   （intervention_sycophancy_table_strengths.csv），绘制谄媚指标柱状图：
#   baseline 柱：1 - baseline_accuracy；干预柱：1 - intervention_accuracy（CSV 中 acc 为非谄媚率）
#   （CSV 中 accuracy 列为非谄媚率，见 code/detection 下聚合脚本说明）
#
# 用法示例：
#   bash run_plot_intervention_strength_bars.sh
#   CSV_PATH=/path/to/intervention_sycophancy_table_strengths.csv OUT_DIR=/path/to/out bash run_plot_intervention_strength_bars.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
CSV_PATH="${CSV_PATH:-$BASE_DIR/safety_explanation/sycophancy/results/intervention/aggregate/intervention_sycophancy_table_strengths.csv}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/sycophancy/intervention_graph/output}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/sycophancy/intervention_graph/plot_intervention_sycophancy_strength_bars.py"

python3 "$SCRIPT_PATH" \
    --csv-path "$CSV_PATH" \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "$BASE_DIR/safety_explanation/sycophancy/intervention_graph/run_plot_intervention_strength_bars.log"
