#!/usr/bin/env bash
# 功能：从 results/feature_analysis 生成“不同实验(dataset_key) 的 PCC”柱状图（Step1 top1-feature PCC + Step2B overall PCC），并输出为单独 PDF。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 默认使用：openloop/cog_simu_prediction/results/feature_analysis（脚本移动位置后仍可用）
FEATURE_ANALYSIS_DIR="${1:-${SCRIPT_DIR}/results/feature_analysis}"
OUT_DIR="${2:-${FEATURE_ANALYSIS_DIR}/plots}"
METRIC="${3:-aic}"          # aic 或 nll
USE_ABS="${4:-1}"           # 1 表示使用 |PCC|，0 表示使用有符号 PCC
SKIP_EXISTING="${5:-1}"     # 1 表示若输出存在则跳过

# plot_experiment_pcc_bars.py 与本脚本放在同一目录下
SCRIPT="${SCRIPT_DIR}/plot_experiment_pcc_bars.py"

ARGS=(--feature-analysis-dir "${FEATURE_ANALYSIS_DIR}" --out-dir "${OUT_DIR}" --metric "${METRIC}")
if [[ "${USE_ABS}" == "1" ]]; then
  ARGS+=(--use-abs)
fi
if [[ "${SKIP_EXISTING}" == "0" ]]; then
  ARGS+=(--skip-existing)
fi

python "${SCRIPT}" "${ARGS[@]}"

