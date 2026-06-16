#!/usr/bin/env bash

# 功能：基于病理分类（Belief-related vs Control-related），
# 从各模型的 capability/parcel level analysis complete JSON 取出 activation_diff，
# 并为每个模型绘制两张独立的小提琴图（capability 一张，parcel 一张），保存为 PDF。
#
# 输出目录：$BASE_DIR/safety_explanation/hallucination/pathology_graphs/graph
#
# 用法示例：
#   bash run_plot_pathology_belief_control_violins.sh
#   SKIP_EXISTING=1 bash run_plot_pathology_belief_control_violins.sh

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/plot_pathology_belief_control_violins.py"

CAPABILITY_CLS_V2="$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/capability_descriptions_run2_pathology_classification_v2.json"
PARCEL_CLS_V2="$BASE_DIR/safety_explanation/hallucination/code/pathology_analysis/data/latent_parcel_topsamples_functionality_summary_pathology_classification_v2.json"

OUTPUT_DIR="$BASE_DIR/safety_explanation/hallucination/pathology_graphs/graph"

# 与 run_aggregate_hierarchy_level.sh 风格对齐的模型列表（可按需要增减）
MODEL_DATA_LIST=(
  "MedHallu_gemma-2-9b-it"
  "HaluEval_gemma-2-9b-it"
  "dolly_close_gemma-2-9b-it"
  "nq_open_gemma-2-9b-it"
  "sciq_gemma-2-9b-it"
  "triviaqa_gemma-2-9b-it"
  "truthfulqa_gemma-2-9b-it"
  "MedHallu_gemma-2-2b"
  "HaluEval_gemma-2-2b"
  "dolly_close_gemma-2-2b"
  "nq_open_gemma-2-2b"
  "sciq_gemma-2-2b"
  "triviaqa_gemma-2-2b"
  "truthfulqa_gemma-2-2b"
)

mkdir -p "$OUTPUT_DIR"

for MODEL_NAME in "${MODEL_DATA_LIST[@]}"; do
  echo "=== 处理模型: ${MODEL_NAME} ==="

  CAPABILITY_ANALYSIS_COMPLETE="$BASE_DIR/safety_explanation/hallucination/results/analysis_output/${MODEL_NAME}/capability_level/capability_level_analysis_complete.json"
  PARCEL_ANALYSIS_COMPLETE="$BASE_DIR/safety_explanation/hallucination/results/analysis_output/${MODEL_NAME}/parcel_level/parcel_level_analysis_complete.json"

  if [[ ! -f "$CAPABILITY_ANALYSIS_COMPLETE" || ! -f "$PARCEL_ANALYSIS_COMPLETE" ]]; then
    echo "[WARN] 找不到分析结果文件，跳过模型: ${MODEL_NAME}"
    continue
  fi

  ARGS=(
    --model_name "${MODEL_NAME}"
    --capability_cls_v2 "${CAPABILITY_CLS_V2}"
    --parcel_cls_v2 "${PARCEL_CLS_V2}"
    --capability_analysis_complete "${CAPABILITY_ANALYSIS_COMPLETE}"
    --parcel_analysis_complete "${PARCEL_ANALYSIS_COMPLETE}"
    --output_dir "${OUTPUT_DIR}"
  )

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip_existing)
  fi

  python "${PY_SCRIPT}" "${ARGS[@]}"
done

echo "[DONE] 所有模型绘图完成，输出目录：${OUTPUT_DIR}"

