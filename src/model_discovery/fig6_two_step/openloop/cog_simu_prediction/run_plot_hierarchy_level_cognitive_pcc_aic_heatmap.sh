#!/usr/bin/env bash
# 功能：绘制 hierarchy_level(A/B/C/D) × cognitive data 的 heatmap（只取 pcc_with_train_aic 列）
#
# 输出：
# - results/feature_analysis/plots/hierarchy_level_x_cognitive_pcc_with_train_aic_heatmap.png/pdf/svg
# - results/feature_analysis/plots/hierarchy_level_x_cognitive_pcc_with_train_aic_heatmap.csv
#
# 用法示例：
#   bash run_plot_hierarchy_level_cognitive_pcc_aic_heatmap.sh
#   SKIP_EXISTING=1 bash run_plot_hierarchy_level_cognitive_pcc_aic_heatmap.sh
#
# 可调参数（环境变量）：
#   SKIP_EXISTING=0|1
#   EXPERIMENT_DIRS：逗号分隔的实验目录名（位于 results/feature_analysis/ 下）

set -euo pipefail

BASE_DIR="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction"
SCRIPT_PATH="$BASE_DIR/plot_hierarchy_level_cognitive_pcc_aic_heatmap.py"

SKIP_EXISTING="${SKIP_EXISTING:-0}"
EXPERIMENT_DIRS="${EXPERIMENT_DIRS:-badham2017deficits_exp1_csv,bahrami2020four_exp_csv,collsiöö2023MCPL_exp1_csv,hilbig2014generalized_exp1_csv,popov2023intent_exp1_csv,ruggeri2022globalizability_exp1_csv}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "[ERROR] 找不到绘图脚本: $SCRIPT_PATH" >&2
  exit 1
fi

ARGS=(--experiment-dirs "$EXPERIMENT_DIRS")
if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

python3 "$SCRIPT_PATH" "${ARGS[@]}"

