#!/usr/bin/env bash
# ============================================================
# 功能介绍：为 sycophancy 数据绘制 1/2 圆极坐标检测性能图（减小占地）。
#   使用本目录下 plot_fig3_polar_half.py，包含 Answer/Feedback 数据集，
#   输出 fig3_polar_half_*.pdf/png/svg。
#
# 用法示例：
#   bash safety_explanation/sycophancy/detection_graphs/run_plot_fig3_polar_half.sh
#   METRIC=auprc bash safety_explanation/sycophancy/detection_graphs/run_plot_fig3_polar_half.sh
#   SKIP_EXISTING=1 bash safety_explanation/sycophancy/detection_graphs/run_plot_fig3_polar_half.sh
#   CSV_PATH=/path/to/data.csv OUT_DIR=/path/to/output bash .../run_plot_fig3_polar_half.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
METRIC="${METRIC:-auroc}"
CSV_PATH="${CSV_PATH:-$BASE_DIR/safety_explanation/sycophancy/results/detection/all_results/all_metrics_plot_auroc.csv}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/sycophancy/detection_graphs/output}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/sycophancy/detection_graphs/plot_fig3_polar_half.py"

if [[ "$SKIP_EXISTING" == "1" ]]; then
    python3 "$SCRIPT_PATH" \
        --csv-path "$CSV_PATH" \
        --out-dir "$OUT_DIR" \
        --metric "$METRIC" \
        --skip-existing \
        2>&1 | tee "$BASE_DIR/safety_explanation/sycophancy/detection_graphs/run_plot_fig3_polar_half.log"
else
    python3 "$SCRIPT_PATH" \
        --csv-path "$CSV_PATH" \
        --out-dir "$OUT_DIR" \
        --metric "$METRIC" \
        2>&1 | tee "$BASE_DIR/safety_explanation/sycophancy/detection_graphs/run_plot_fig3_polar_half.log"
fi

