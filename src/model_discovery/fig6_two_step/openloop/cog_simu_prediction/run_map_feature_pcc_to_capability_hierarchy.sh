#!/usr/bin/env bash
# 功能：批量将 step1_train_feature_pcc_ranked_by_aic.csv 的 Parcel 相关性映射到 Capability 与 Hierarchy(A/B/C/D)。
# 说明：
# - 输入文件：每个实验目录下 results/feature_analysis/<exp_dir>/step1_train_feature_pcc_ranked_by_aic.csv
# - 输出文件：写回各实验目录（capability CSV / hierarchy CSV / stats JSON）
#
# 用法示例：
#   bash run_map_feature_pcc_to_capability_hierarchy.sh
#   SKIP_EXISTING=1 bash run_map_feature_pcc_to_capability_hierarchy.sh
#
# 可调参数（环境变量）：
#   SKIP_EXISTING=0|1        若为 1 且输出已存在则跳过对应实验目录
#   TOP_K_MAX=<int>          capability 聚合时正向极值选择个数（默认 10）
#   TOP_K_MIN=<int>          capability 聚合时负向极值选择个数（默认 10）
#                            当 TOP_K_MAX=0 且 TOP_K_MIN=0 时，改为使用该 capability 的全部候选 parcel
#   DISABLE_ABS_RECOMPUTE=0|1  若为 1 则关闭 abs_* 指标重算（默认开启重算，保证 abs 与原值一致）

set -euo pipefail

BASE_DIR="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction"
FEATURE_DIR="$BASE_DIR/results/feature_analysis"
SCRIPT_PATH="$BASE_DIR/map_feature_pcc_to_capability_hierarchy.py"

SKIP_EXISTING="${SKIP_EXISTING:-0}"
TOP_K_MAX="${TOP_K_MAX:-0}"
TOP_K_MIN="${TOP_K_MIN:-0}"
DISABLE_ABS_RECOMPUTE="${DISABLE_ABS_RECOMPUTE:-0}"

EXPERIMENT_DIRS=(
  "badham2017deficits_exp1_csv"
  "bahrami2020four_exp_csv"
  "collsiöö2023MCPL_exp1_csv"
  "hilbig2014generalized_exp1_csv"
  "popov2023intent_exp1_csv"
  "ruggeri2022globalizability_exp1_csv"
)

print_info() {
  echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_error() {
  echo -e "\033[0;31m[ERROR]\033[0m $1"
}

if [[ ! -f "$SCRIPT_PATH" ]]; then
  print_error "找不到映射脚本: $SCRIPT_PATH"
  exit 1
fi

for EXP_DIR in "${EXPERIMENT_DIRS[@]}"; do
  EXP_PATH="$FEATURE_DIR/$EXP_DIR"
  INPUT_CSV="$EXP_PATH/step1_train_feature_pcc_ranked_by_aic.csv"

  if [[ ! -f "$INPUT_CSV" ]]; then
    print_error "缺少输入文件，跳过: $INPUT_CSV"
    continue
  fi

  ARGS=(--input-csv "$INPUT_CSV")
  ARGS+=(--top-k-max "$TOP_K_MAX" --top-k-min "$TOP_K_MIN")
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip-existing)
  fi
  if [[ "$DISABLE_ABS_RECOMPUTE" == "1" ]]; then
    ARGS+=(--disable-abs-recompute)
  fi

  print_info "开始处理: $EXP_DIR"
  python3 "$SCRIPT_PATH" "${ARGS[@]}"
done

print_info "全部处理完成"
