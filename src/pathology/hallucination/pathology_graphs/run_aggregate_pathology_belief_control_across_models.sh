#!/usr/bin/env bash

# 功能：
# 1）对同一模型系列（gemma-2-2b / gemma-2-9b-it）跨多个数据集聚合
#     Belief-related / Control-related 的 capability + parcel activation difference（先跨任务取平均），
# 2）分别绘制一张聚合后的小提琴图：
#     - gemma-2-2b_cap_parcel_belief_vs_control_violin_agg.pdf
#     - gemma-2-9b-it_cap_parcel_belief_vs_control_violin_agg.pdf
#
# 用法示例：
#   bash run_aggregate_pathology_belief_control_across_models.sh
#   SKIP_EXISTING=1 bash run_aggregate_pathology_belief_control_across_models.sh
#   USE_ABS=0 bash run_aggregate_pathology_belief_control_across_models.sh   # 不取绝对值（默认取绝对值）

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
USE_ABS="${USE_ABS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/aggregate_pathology_belief_control_across_models.py"

CAPABILITY_CLS_V2="$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/capability_descriptions_run2_pathology_classification_v3.json"
PARCEL_CLS_V2="$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/latent_parcel_topsamples_functionality_summary_pathology_classification_9b_it_v3.json"

OUTPUT_DIR="$BASE_DIR/safety_explanation/hallucination/pathology_graphs/graph"
mkdir -p "$OUTPUT_DIR"

run_agg() {
  local AGG_NAME="$1"; shift
  # 剩下的是 model_data_list
  local MODELS=("$@")

  echo "=== 聚合模型系列: ${AGG_NAME} ==="

  local ARGS=(
    --aggregated_model_name "${AGG_NAME}"
    --model_data_list "${MODELS[@]}"
    --base_dir "${BASE_DIR}"
    --capability_cls_v2 "${CAPABILITY_CLS_V2}"
    --parcel_cls_v2 "${PARCEL_CLS_V2}"
    --output_dir "${OUTPUT_DIR}"
  )

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip_existing)
  fi
  if [[ "$USE_ABS" == "0" ]]; then
    ARGS+=(--no_abs)
  fi

  python "${PY_SCRIPT}" "${ARGS[@]}"
}

# 1) gemma-2-9b-it 系列（与 hierarchy_level 聚合脚本一致的 7 个任务）
MODEL_DATA_LIST_9B=(
  "MedHallu_gemma-2-9b-it"
  "HaluEval_gemma-2-9b-it"
  "dolly_close_gemma-2-9b-it"
  "nq_open_gemma-2-9b-it"
  "sciq_gemma-2-9b-it"
  # "triviaqa_gemma-2-9b-it"
  "truthfulqa_gemma-2-9b-it"
)
run_agg "gemma-2-9b-it" "${MODEL_DATA_LIST_9B[@]}"
PARCEL_CLS_V2="$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/latent_parcel_topsamples_functionality_summary_pathology_classification_v3.json"

# 2) gemma-2-2b 系列（如果你的 2b 任务命名不同，可在此处按需调整列表）
MODEL_DATA_LIST_2B=(
  "MedHallu_gemma-2-2b"
  "HaluEval_gemma-2-2b"
  "dolly_close_gemma-2-2b"
  "nq_open_gemma-2-2b"
  "sciq_gemma-2-2b"
  # "triviaqa_gemma-2-2b"
  "truthfulqa_gemma-2-2b"
)
run_agg "gemma-2-2b" "${MODEL_DATA_LIST_2B[@]}"

echo "[DONE] 聚合小提琴图已生成，目录：${OUTPUT_DIR}"

