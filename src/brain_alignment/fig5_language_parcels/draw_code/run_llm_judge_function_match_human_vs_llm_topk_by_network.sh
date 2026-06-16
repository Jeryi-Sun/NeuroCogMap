#!/bin/bash
# 功能：
#   在各个 Yeo7 network 内，用 LLM judge：
#   每个 Human parcel 的 function_description 是否能被其 top-k LLM parcel 的功能（llm_function）匹配/覆盖。
#
# 输入：
#   默认读取 export_top_human_matches.py 生成的结果：
#     ${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}/top_human_parcels_per_llm.csv
#
# 输出：
#   写入：
#     ${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}/llm_judge_match_by_network/
#       - judgements.jsonl
#       - parcel_summary.csv
#       - network_summary.csv
#
# 使用方式：
#   1) 推荐把 key 放到环境变量里（避免写进脚本）：
#        export OPENAI_API_KEY="..."
#      或 export VLLM_API_KEY="..."
#   2) 然后运行：
#        bash run_llm_judge_function_match_human_vs_llm_topk_by_network.sh
#
# 可选参数（通过修改下面变量）：
#   - UST_IDS / STORY_NAME / METHODS / TOP_K / SELECTION_SUFFIX
#   - VLLM_URL / MODEL
#   - SKIP_EXISTING：是否启用“断点续跑”（跳过已完成 parcel，继续跑剩余）
#   - OVERWRITE：强制重跑覆盖（删除旧结果并全量重算）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基本配置（直接在这里改）
# =========================
UST_IDS=("uts02" "uts03")
STORY_NAME="whereisthesmoke"
METHODS=("saeact")

TOP_K=10
SELECTION_TYPE="top"

SELECTION_SUFFIX="_bottom10_bottomH10"

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs"

# LLM 配置（参考 openloop/main_method 的调用方式）
VLLM_URL="https://api2.aigcbest.top/v1"
MODEL="gpt-5.2-2025-12-11"

# LLM parcel 长描述文件（用于 prompt 中与 Human 长描述对齐比较）
LLM_PARCEL_DESCRIPTIONS_JSON="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"

# 输出控制
# 断点续跑默认开启：若 judgements.jsonl 已存在，会跳过已完成的 parcel，继续跑剩余的。
SKIP_EXISTING=true
OVERWRITE=false

OVERALL_EXIT_CODE=0

for UST_ID in "${UST_IDS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "LLM judge: Human vs Top-${TOP_K} LLM parcels"
    echo "UST_ID: ${UST_ID}"
    echo "STORY_NAME: ${STORY_NAME}"
    echo "METHOD: ${METHOD}"
    echo "Selection suffix: ${SELECTION_SUFFIX}"
    echo "=========================================="

    CURRENT_RESULT_DIR="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}"
    INPUT_CSV="${CURRENT_RESULT_DIR}/top_human_parcels_per_llm.csv"
    OUTPUT_DIR="${CURRENT_RESULT_DIR}/llm_judge_match_by_network"

    echo "输入 CSV: ${INPUT_CSV}"
    echo "输出目录: ${OUTPUT_DIR}"

    if [ ! -f "$INPUT_CSV" ]; then
      echo "错误: 找不到输入 CSV 文件: $INPUT_CSV"
      OVERALL_EXIT_CODE=1
      continue
    fi

    mkdir -p "$OUTPUT_DIR"

    CMD=(python3 "${SCRIPT_DIR}/llm_judge_function_match_human_vs_llm_topk_by_network.py"
      --input-csv "$INPUT_CSV"
      --output-dir "$OUTPUT_DIR"
      --top-k "$TOP_K"
      --selection-type "$SELECTION_TYPE"
      --llm-parcel-descriptions "$LLM_PARCEL_DESCRIPTIONS_JSON"
      --vllm-url "$VLLM_URL"
      --model "$MODEL"
      --max-tokens 10000
      --human-desc-max-chars 0
      --llm-desc-max-chars 0
      --reasoning-effort "low"
    )

    # 注意：这里不要传 --skip-existing。
    # Python 脚本已实现“断点续跑”（读取 judgements.jsonl，逐 parcel 跳过已完成项并继续追加）。
    # 传 --skip-existing 会导致只要 judgements.jsonl 存在就整次直接退出，无法续跑。
    if [ "$OVERWRITE" = true ]; then
      CMD+=("--overwrite")
    fi

    echo "执行命令:"
    printf '  %q' "${CMD[@]}"
    echo ""

    "${CMD[@]}"

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
      echo "✓ 完成: ${OUTPUT_DIR}"
    else
      echo "✗ 失败: UST_ID=${UST_ID}, METHOD=${METHOD}，退出码: $exit_code"
      OVERALL_EXIT_CODE=$exit_code
    fi
  done
done

exit $OVERALL_EXIT_CODE

