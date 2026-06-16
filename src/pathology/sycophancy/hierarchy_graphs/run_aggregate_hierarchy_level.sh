#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   汇总多个模型的 hierarchy_level_activation_diff.json 到一个总的 JSON 文件，
#   存放在 hierarchy_graphs/data 目录下，供绘图脚本直接使用。
#
# 前置：先运行 run_analysis_hierarchy_level.sh，确保各模型的
#       hierarchy_level_activation_diff.json 已经生成。
#
# 用法示例：
#   bash safety_explanation/sycophancy/hierarchy_graphs/run_aggregate_hierarchy_level.sh
#   SKIP_EXISTING=1 bash .../run_aggregate_hierarchy_level.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/sycophancy/hierarchy_graphs/aggregate_hierarchy_level.py"

# sycophancy 数据集列表
MODEL_DATA_LIST=(
  "answer_gemma-2-2b"
  "answer_gemma-2-9b-it"
  "feedback_gemma-2-2b"
)

ARGS=(--base-dir "$BASE_DIR" --model-data-list "${MODEL_DATA_LIST[@]}" --output-json "$BASE_DIR/safety_explanation/sycophancy/hierarchy_graphs/data/hierarchy_level_all_models.json")

if [[ "$SKIP_EXISTING" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

python3 "$SCRIPT_PATH" "${ARGS[@]}"
