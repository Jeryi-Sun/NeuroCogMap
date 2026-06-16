#!/bin/bash
# 功能：
#   基于 export_top_human_matches.py 生成的 top/bottom 匹配结果 CSV，
#   对多个 UST_ID 和 METHOD 批量绘制：
#     - 按 Yeo7 network 聚合的 semantic_similarity Top vs Bottom 柱状图（带 parcel 散点）。
#
# 说明：
#   - 默认假定 top/bottom 结果保存在：
#       ${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}/top_human_parcels_per_llm.csv
#   - 其中 SELECTION_SUFFIX 可以用来区分是否包含 bottom / random 等，例如 "_bottom10_bottomH10"。
#   - 若路径结构不同，可以在下面直接修改 CURRENT_RESULT_DIR 或 INPUT_CSV 构造方式。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基本配置（直接在这里改）
# =========================
# UST 被试 ID 列表
UST_IDS=("uts02" "uts03")
# 是否额外生成「多被试融合」结果（将 UST_IDS 中全部被试合并）
RUN_MERGED_SUBJECTS=true
# 融合结果目录标签
MERGED_SUBJECTS_TAG="merged_uts02_uts03"
# 数据名称 / 故事名称
STORY_NAME="adollshouse"
# 方法名称列表（目前支持：sae / saeact）
METHODS=("saeact")

# top / bottom K 仅用于图例和输出文件名标注，需与生成 CSV 时保持一致
TOP_K=10
BOTTOM_K=10

# 若运行 run_top_human_LLM_matches.sh 时开启了 bottom/bottom-human，
# 并在输出目录后自动追加了后缀（例如 "_bottom10_bottomH10"），
# 请在这里设置相同的后缀；若没有后缀，则留空即可。
SELECTION_SUFFIX="_bottom10_bottomH10"

# 根目录
PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs"

OVERALL_EXIT_CODE=0

for UST_ID in "${UST_IDS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "按 7Networks 绘制 semantic_similarity Top vs Bottom"
    echo "UST_ID: ${UST_ID}"
    echo "STORY_NAME: ${STORY_NAME}"
    echo "METHOD: ${METHOD}"
    echo "Top-K: ${TOP_K}, Bottom-K: ${BOTTOM_K}"
    echo "Selection suffix: ${SELECTION_SUFFIX}"
    echo "=========================================="

    CURRENT_RESULT_DIR="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}"
    INPUT_CSV="${CURRENT_RESULT_DIR}/top_human_parcels_per_llm.csv"
    OUTPUT_DIR="${CURRENT_RESULT_DIR}/semantic_similarity_by_network"

    echo "结果目录: ${CURRENT_RESULT_DIR}"
    echo "输入 CSV: ${INPUT_CSV}"
    echo "输出目录: ${OUTPUT_DIR}"

    if [ ! -d "$CURRENT_RESULT_DIR" ]; then
        echo "错误: 结果目录不存在: $CURRENT_RESULT_DIR"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ ! -f "$INPUT_CSV" ]; then
        echo "错误: 找不到输入 CSV 文件: $INPUT_CSV"
        OVERALL_EXIT_CODE=1
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    python plot_semantic_similarity_top_bottom_by_network.py \
        --input-csv "$INPUT_CSV" \
        --output-dir "$OUTPUT_DIR" \
        --top-k "$TOP_K" \
        --bottom-k "$BOTTOM_K" \
        --overwrite 

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ 组合 UST_ID=${UST_ID}, METHOD=${METHOD} semantic_similarity 网络图绘制完成！"
        echo "结果保存在: $OUTPUT_DIR"
        echo "  - semantic_similarity_by_network_top${TOP_K}_bottom${BOTTOM_K}.png / .svg"
        echo "=========================================="
    else
        echo ""
        echo "✗ 组合 UST_ID=${UST_ID}, METHOD=${METHOD} 绘制失败，退出码: $exit_code"
        OVERALL_EXIT_CODE=$exit_code
    fi

  done
done

# 额外：融合 UST_IDS 全部被试，按 network 统计并绘图
if [ "$RUN_MERGED_SUBJECTS" = true ]; then
  for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "融合多被试绘图（Top vs Bottom + Violin）"
    echo "UST_IDS: ${UST_IDS[*]}"
    echo "STORY_NAME: ${STORY_NAME}"
    echo "METHOD: ${METHOD}"
    echo "Top-K: ${TOP_K}, Bottom-K: ${BOTTOM_K}"
    echo "=========================================="

    INPUT_CSVS=()
    MISSING_INPUT=0
    for UST_ID in "${UST_IDS[@]}"; do
      CURRENT_RESULT_DIR="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}"
      INPUT_CSV="${CURRENT_RESULT_DIR}/top_human_parcels_per_llm.csv"
      if [ ! -f "$INPUT_CSV" ]; then
        echo "错误: 找不到输入 CSV 文件: $INPUT_CSV"
        MISSING_INPUT=1
      else
        INPUT_CSVS+=("$INPUT_CSV")
      fi
    done

    if [ $MISSING_INPUT -ne 0 ]; then
      OVERALL_EXIT_CODE=1
      continue
    fi

    OUTPUT_DIR="${PROJECT_ROOT}/draw_result/${MERGED_SUBJECTS_TAG}/${STORY_NAME}/${METHOD}${SELECTION_SUFFIX}/semantic_similarity_by_network"
    mkdir -p "$OUTPUT_DIR"
    echo "融合输出目录: ${OUTPUT_DIR}"

    python plot_semantic_similarity_top_bottom_by_network.py \
      --input-csvs "${INPUT_CSVS[@]}" \
      --output-dir "$OUTPUT_DIR" \
      --top-k "$TOP_K" \
      --bottom-k "$BOTTOM_K" \
      --overwrite

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
      echo ""
      echo "✓ METHOD=${METHOD} 多被试融合绘制完成！"
      echo "结果保存在: $OUTPUT_DIR"
      echo "  - semantic_similarity_by_network_top${TOP_K}_bottom${BOTTOM_K}.png/.svg/.pdf"
      echo "  - semantic_similarity_violin_by_network_top${TOP_K}_bottom${BOTTOM_K}.png/.svg/.pdf"
      echo "  - semantic_similarity_network_mean_variance_top${TOP_K}_bottom${BOTTOM_K}.csv"
      echo "=========================================="
    else
      echo ""
      echo "✗ METHOD=${METHOD} 多被试融合绘制失败，退出码: $exit_code"
      OVERALL_EXIT_CODE=$exit_code
    fi
  done
fi

exit $OVERALL_EXIT_CODE

