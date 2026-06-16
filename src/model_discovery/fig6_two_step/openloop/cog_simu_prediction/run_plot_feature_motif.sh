#!/usr/bin/env bash
# 功能：调用 plot_feature_motif.py 生成 Task × Motif mean PCC heatmap（并输出 CSV / PNG / PDF / SVG）。
#
# 用法示例：
#   bash run_plot_feature_motif.sh
#   TOP_K=20 bash run_plot_feature_motif.sh
#   SKIP_EXISTING=0 bash run_plot_feature_motif.sh
#   NEAR_ZERO_ABS_THRESHOLD=0.05 SKIP_EXISTING=0 bash run_plot_feature_motif.sh
#
# 可调参数（环境变量）：
#   TOP_K=10                          近零桶/正桶/负桶各自取 Top-K
#   NEAR_ZERO_ABS_THRESHOLD=0.1      近零桶阈值：|pcc_with_train_aic| < 阈值
#   SKIP_EXISTING=1                   若为 1，则添加 --skip-existing（输出已存在则跳过写入）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PLOT="${SCRIPT_DIR}/plot_feature_motif.py"

TOP_K="${TOP_K:-10}"
NEAR_ZERO_ABS_THRESHOLD="${NEAR_ZERO_ABS_THRESHOLD:-0.0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

ARGS=(--top-k "${TOP_K}" --near-zero-abs-threshold "${NEAR_ZERO_ABS_THRESHOLD}")
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

python3 "${SCRIPT_PLOT}" "${ARGS[@]}"

