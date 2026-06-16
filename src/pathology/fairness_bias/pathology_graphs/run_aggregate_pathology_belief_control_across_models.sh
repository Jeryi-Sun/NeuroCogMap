#!/usr/bin/env bash
# ============================================================
# 功能：
#   1）对同一模型系列（gemma-2-2b / gemma-2-9b-it）跨多个数据集聚合
#      Belief-related / Control-related 的 capability + parcel activation difference
#      （先跨任务取平均），
#   2）分别绘制一张聚合后的小提琴图：
#      - gemma-2-2b_cap_parcel_belief_vs_control_violin_agg.pdf
#      - gemma-2-9b-it_cap_parcel_belief_vs_control_violin_agg.pdf
#
# 用法示例：
#   bash run_aggregate_pathology_belief_control_across_models.sh
#   SKIP_EXISTING=1 bash run_aggregate_pathology_belief_control_across_models.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$BASE_DIR/safety_explanation/hallucination/pathology_graphs/aggregate_pathology_belief_control_across_models.py"

# fairness_bias 的病理分类文件路径（如果存在，否则使用 hallucination 的）
CAPABILITY_CLS_V2="${CAPABILITY_CLS_V2:-$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/capability_descriptions_run2_pathology_classification_v3.json}"
PARCEL_CLS_V2_9B="${PARCEL_CLS_V2_9B:-$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/latent_parcel_topsamples_functionality_summary_pathology_classification_9b_it_v3.json}"
PARCEL_CLS_V2_2B="${PARCEL_CLS_V2_2B:-$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/latent_parcel_topsamples_functionality_summary_pathology_classification_v3.json}"

OUTPUT_DIR="$BASE_DIR/safety_explanation/fairness_bias/pathology_graphs/graph"
mkdir -p "$OUTPUT_DIR"

run_agg() {
  local AGG_NAME="$1"; shift
  local PARCEL_CLS="$1"; shift
  # 剩下的是 model_data_list
  local MODELS=("$@")

  echo "=== 聚合模型系列: ${AGG_NAME} ==="

  local ARGS=(
    --aggregated_model_name "${AGG_NAME}"
    --model_data_list "${MODELS[@]}"
    --base_dir "${BASE_DIR}"
    --project_type "fairness_bias"
    --capability_cls_v2 "${CAPABILITY_CLS_V2}"
    --parcel_cls_v2 "${PARCEL_CLS}"
    --output_dir "${OUTPUT_DIR}"
  )

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip_existing)
  fi

  python "${PY_SCRIPT}" "${ARGS[@]}"
}

# 1) gemma-2-9b-it 系列
MODEL_DATA_LIST_9B=(
  "bbq_age_gemma-2-9b-it"
  "bbq_nationality_gemma-2-9b-it"
  "bbq_gender_identity_gemma-2-9b-it"
  "bbq_disability_status_gemma-2-9b-it"
)
run_agg "gemma-2-9b-it" "${PARCEL_CLS_V2_9B}" "${MODEL_DATA_LIST_9B[@]}"

# 2) gemma-2-2b 系列
MODEL_DATA_LIST_2B=(
  # "bbq_age_gemma-2-2b"
  "bbq_nationality_gemma-2-2b"
  # "bbq_gender_identity_gemma-2-2b"
  # "bbq_disability_status_gemma-2-2b"
)
run_agg "gemma-2-2b" "${PARCEL_CLS_V2_2B}" "${MODEL_DATA_LIST_2B[@]}"

echo "[DONE] 聚合小提琴图已生成，目录：${OUTPUT_DIR}"
