#!/usr/bin/env bash
# 功能：将多个模型的 capability-level activation_diff 聚合到 A/B/C/D 四个认知层级，输出 hierarchy-level JSON。
# 前置：请先在 sycophancy 任务上运行对应的整体分析脚本（如 run_analysis_sycophancy.sh 或 analysis_capability_level.py），生成：
#   <BASE_DIR>/safety_explanation/sycophancy/results/analysis_output/<MODEL_DATA>/capability_level/capability_level_analysis_complete.json
#
# 用法示例：
#   bash run_analysis_hierarchy_level.sh
#   SKIP_EXISTING=1 bash run_analysis_hierarchy_level.sh
#   DISABLE_ORDER_VALIDATION=1 bash run_analysis_hierarchy_level.sh
#   USE_ABS_FOR_MEAN=1 bash run_analysis_hierarchy_level.sh   # 先取绝对值再做层级均值与排序

set -euo pipefail

# ==================== 配置区域 ====================
# 基础路径
BASE_DIR="/path/to/project_root"

# 模型数据列表 - 与 sycophancy 下的 run_analysis_sycophancy.sh 对齐
MODEL_DATA_LIST=(
    "answer_gemma-2-2b"
    "feedback_gemma-2-2b"
)

SKIP_EXISTING="${SKIP_EXISTING:-0}"
DISABLE_ORDER_VALIDATION="${DISABLE_ORDER_VALIDATION:-0}"
USE_ABS_FOR_MEAN="${USE_ABS_FOR_MEAN:-0}"

# 仍然复用 hallucination 目录下的 Python 脚本
SCRIPT_PATH="$BASE_DIR/safety_explanation/hallucination/code/analysis/analysis_hierarchy_level.py"

print_info() {
  echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_error() {
  echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# ==================== 主流程 ====================

for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
  MODEL_OUTPUT_DIR="$BASE_DIR/safety_explanation/sycophancy/results/analysis_output/$MODEL_DATA"
  INPUT_JSON="$MODEL_OUTPUT_DIR/capability_level/capability_level_analysis_complete.json"

  if [[ ! -f "$INPUT_JSON" ]]; then
    print_error "缺少 capability-level 结果，跳过模型: $MODEL_DATA"
    print_error "  期望文件: $INPUT_JSON"
    continue
  fi

  print_info "处理模型: $MODEL_DATA"

  ARGS=(--model-output-dir "$MODEL_OUTPUT_DIR")

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    ARGS+=(--skip-existing)
  fi

  if [[ "${DISABLE_ORDER_VALIDATION}" == "1" ]]; then
    ARGS+=(--disable-order-validation)
  fi

  if [[ "${USE_ABS_FOR_MEAN}" == "1" ]]; then
    ARGS+=(--use-abs-for-mean)
  fi

  python3 "$SCRIPT_PATH" "${ARGS[@]}"
done

