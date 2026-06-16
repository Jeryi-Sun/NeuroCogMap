#!/usr/bin/env bash
# ============================================================
# 功能介绍：为 sycophancy 数据绘制常规柱状检测性能图（Nature 风格）。
#   使用本目录下 plot_fig3_bar_only.py，包含 Answer/Feedback 数据集，
#   输出 fig3_bar_only_*.pdf/png/svg。
#
# 用法示例：
#   bash safety_explanation/sycophancy/detection_graphs/run_plot_fig3_bar_only.sh
#   METRIC=auprc bash safety_explanation/sycophancy/detection_graphs/run_plot_fig3_bar_only.sh
#   SKIP_EXISTING=1 bash safety_explanation/sycophancy/detection_graphs/run_plot_fig3_bar_only.sh
#   CSV_PATH=/path/to/data.csv OUT_DIR=/path/to/output bash .../run_plot_fig3_bar_only.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
METRIC="${METRIC:-auroc}"
CSV_PATH="${CSV_PATH:-$BASE_DIR/safety_explanation/sycophancy/results/detection/all_results/all_metrics_plot_auroc.csv}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/sycophancy/detection_graphs/output}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/sycophancy/detection_graphs/plot_fig3_bar_only.py"

if [[ "$SKIP_EXISTING" == "1" ]]; then
    python3 "$SCRIPT_PATH" \
        --csv-path "$CSV_PATH" \
        --out-dir "$OUT_DIR" \
        --metric "$METRIC" \
        --skip-existing \
        2>&1 | tee "$BASE_DIR/safety_explanation/sycophancy/detection_graphs/run_plot_fig3_bar_only.log"
else
    python3 "$SCRIPT_PATH" \
        --csv-path "$CSV_PATH" \
        --out-dir "$OUT_DIR" \
        --metric "$METRIC" \
        2>&1 | tee "$BASE_DIR/safety_explanation/sycophancy/detection_graphs/run_plot_fig3_bar_only.log"
fi

