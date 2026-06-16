#!/bin/bash
# 案例分析运行脚本 - 简化版

set -e

# ==================== 配置区域 ====================
# 基础路径
BASE_DIR="/path/to/project_root"

# 模型列表（按需调整）
MODEL_DATA_LIST=(
  "truthfulqa_gemma-2-2b"
  "MedHallu_gemma-2-2b"
  "HaluEval_gemma-2-2b"
  "dolly_close_gemma-2-2b"
  "nq_open_gemma-2-2b"
  "sciq_gemma-2-2b"
  "triviaqa_gemma-2-2b"
)

# 数据文件路径（MAPPING/PARCEL/CAP描述固定；correct/incorrect/对比/输出按模型动态设置）
CORRECT_JSONL=""
INCORRECT_JSONL=""
MAPPING_JSON="$BASE_DIR/neural_area/connect_cap_parcel/results/aggrate_final/final_capability_parcel_all.json"
PARCEL_DESC="$BASE_DIR/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
CAP_DESC="$BASE_DIR/capability_analysis/data/capability_descriptions/capability_descriptions_run2.json"

# 输出目录（按模型动态设置）
OUTPUT_BASE=""

# 分析参数
WINDOW_SIZE=5
TOP_K=5

# 是否跳过已存在结果 (true/false)，存在则跳过该样本（python层）
SKIP_EXISTING=false

# 是否跳过已完成模型 (true/false)，存在非空输出则跳过该模型（shell层）
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-true}

# 运行模式 (check/analyze/pairs)
RUN_MODE="analyze"

# 对比选择结果 JSON（按模型动态设置，来自 question_contrastive_selector.py 输出）
CONTRASTIVE_JSON=""

# 要分析的案例列表（如留空则由对比对选择文件驱动；若保留则对所有模型同一批次）
CASE_LIST=( )

# ==================== 简化执行：直接调用 case_study.py 批量模式 ====================

print_info() { echo -e "\033[0;32m[INFO]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

check_inputs() {
  local files=("$CORRECT_JSONL" "$INCORRECT_JSONL" "$MAPPING_JSON" "$PARCEL_DESC" "$CAP_DESC")
  # CONTRASTIVE_JSON 可选：若设置则检查
  if [ -n "$CONTRASTIVE_JSON" ]; then
    files+=("$CONTRASTIVE_JSON")
  fi
  for f in "${files[@]}"; do
    if [ ! -f "$f" ]; then
      print_error "文件不存在: $f"
      exit 1
    fi
  done
}

# 可选分数阈值（为空则不加）
MIN_OVERALL_SCORE=""

run_one_model() {
  local model_name="$1"
  print_info "==================== 开始处理模型: $model_name ===================="
  CORRECT_JSONL="$BASE_DIR/safety_explanation/hallucination/results/$model_name/correct.jsonl"
  INCORRECT_JSONL="$BASE_DIR/safety_explanation/hallucination/results/$model_name/incorrect.jsonl"
  OUTPUT_BASE="$BASE_DIR/safety_explanation/hallucination/results/analysis_output/$model_name/case_analysis"
  CONTRASTIVE_JSON="$BASE_DIR/safety_explanation/hallucination/results/analysis_output/$model_name/contrastive_selector/question_contrastive_pairs.json"

  check_inputs

  if [ "$SKIP_IF_EXISTS" = true ] && [ -d "$OUTPUT_BASE" ] && [ "$(ls -A \"$OUTPUT_BASE\")" ]; then
    print_info "模型输出已存在且非空，跳过: $OUTPUT_BASE"
    print_info "==================== 完成模型(跳过): $model_name ===================="
    return 0
  fi

  local cmd="python3 case_study.py \
    --correct_jsonl \"$CORRECT_JSONL\" \
    --incorrect_jsonl \"$INCORRECT_JSONL\" \
    --mapping_json \"$MAPPING_JSON\" \
    --parcel_desc \"$PARCEL_DESC\" \
    --cap_desc \"$CAP_DESC\" \
    --output_base \"$OUTPUT_BASE\" \
    --window $WINDOW_SIZE \
    --topk $TOP_K"

  if [ -n "$CONTRASTIVE_JSON" ]; then
    cmd="$cmd \
    --pairs_json \"$CONTRASTIVE_JSON\""
  fi

  if [ "$SKIP_EXISTING" = true ]; then
    cmd="$cmd --skip_existing"
  fi

  if [ -n "$MIN_OVERALL_SCORE" ]; then
    cmd="$cmd --min_overall_score $MIN_OVERALL_SCORE"
  fi

  print_info "执行: $cmd"
  eval $cmd
  print_info "==================== 完成模型: $model_name ===================="
}

# 主循环
for MODEL_NAME in "${MODEL_DATA_LIST[@]}"; do
  run_one_model "$MODEL_NAME"
done