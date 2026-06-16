#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 jailbreak 结果绘制干预强度柱状图，复用 hallucination
#   下的 plot_intervention_strength_bars.py。
#
# 用法示例：
#   bash run_plot_intervention_strength_bars.sh
#   CSV_PATH=/path/to/data.csv OUT_DIR=/path/to/output bash run_plot_intervention_strength_bars.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
CSV_PATH="${CSV_PATH:-$BASE_DIR/safety_explanation/jailbreak/results/intervention/aggregate/intervention_accuracy_table_strength_0.1_0.3_0.5.csv}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/jailbreak/intervention_graph/output}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/hallucination/intervention_graph/plot_intervention_strength_bars.py"

python3 "$SCRIPT_PATH" \
    --csv-path "$CSV_PATH" \
    --out-dir "$OUT_DIR" \
    --project-type jailbreak \
    2>&1 | tee "$BASE_DIR/safety_explanation/jailbreak/intervention_graph/run_plot_intervention_strength_bars.log"
