#!/usr/bin/env bash
# ============================================================
# 功能介绍：为 jailbreak 数据绘制分组柱状检测性能图。
#   使用本目录下 plot_fig3_bar.py，输入 all_metrics_plot_auroc.csv，
#   输出 fig3_bar_*.pdf/png/svg 到 detection_graphs/output。
#
# 用法示例：
#   bash safety_explanation/jailbreak/detection_graphs/run_plot_fig3_bar.sh
#   METRIC=auroc bash safety_explanation/jailbreak/detection_graphs/run_plot_fig3_bar.sh
#   CSV_PATH=/path/to/data.csv OUT_DIR=/path/to/output bash .../run_plot_fig3_bar.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
METRIC="${METRIC:-auroc}"
CSV_PATH="${CSV_PATH:-$BASE_DIR/safety_explanation/jailbreak/results/detection/all_results/all_metrics_plot_auroc.csv}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/jailbreak/detection_graphs/output}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/jailbreak/detection_graphs/plot_fig3_bar.py"
LOG_PATH="$BASE_DIR/safety_explanation/jailbreak/detection_graphs/run_plot_fig3_bar.log"

python3 "$SCRIPT_PATH" \
    --csv-path "$CSV_PATH" \
    --out-dir "$OUT_DIR" \
    --metric "$METRIC" \
    2>&1 | tee "$LOG_PATH"
