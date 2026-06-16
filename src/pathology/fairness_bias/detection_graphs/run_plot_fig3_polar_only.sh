#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 fairness_bias 结果绘制极坐标检测性能图，复用 hallucination
#   下的 plot_fig3_polar_only.py。
#
# 用法示例：
#   bash safety_explanation/fairness_bias/detection_graphs/run_plot_fig3_polar_only.sh
#   METRIC=auprc bash safety_explanation/fairness_bias/detection_graphs/run_plot_fig3_polar_only.sh
#   CSV_PATH=/path/to/data.csv OUT_DIR=/path/to/output bash .../run_plot_fig3_polar_only.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
METRIC="${METRIC:-auroc}"
CSV_PATH="${CSV_PATH:-$BASE_DIR/safety_explanation/fairness_bias/results/detection/all_results/all_metrics_plot_auroc.csv}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/fairness_bias/detection_graphs/output}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/hallucination/detection_graphs/plot_fig3_polar_only.py"

python3 "$SCRIPT_PATH" \
    --csv-path "$CSV_PATH" \
    --out-dir "$OUT_DIR" \
    --metric "$METRIC" \
    2>&1 | tee "$BASE_DIR/safety_explanation/fairness_bias/detection_graphs/run_plot_fig3_polar_only.log"
