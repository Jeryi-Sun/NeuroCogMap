#!/bin/bash
# 运行脚本：导出所有 LLM parcel 和 human parcel 的 top-k 配对关系
# 适配新的输出格式：从 run_build_prediction_matrix_from_results.sh 生成的预测矩阵

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基本配置（直接在这里改）
# =========================
# UST 被试 ID 列表（如：uts02 / uts03），会依次循环运行
UST_IDS=("uts02" "uts03")
# 数据名称 / 故事名称（如：whereisthesmoke / adollshouse）
STORY_NAME="adollshouse"
# 方法名称列表（目前支持：sae / saeact），会依次循环运行
METHODS=("saeact")

# 根目录
PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs"

# 默认数据与结果目录（仅用于帮助信息展示，真正运行时会在循环中按 UST_ID/METHOD 重新计算）
DEFAULT_UST_ID="${UST_IDS[0]}"
DEFAULT_METHOD="${METHODS[0]}"
# 注意：这里不要直接给 DATA_DIR / OUTPUT_DIR 赋默认值，否则会导致循环中所有组合
# 都复用同一个（通常是 uts02）的目录。真正运行时，这两个变量应默认为空，
# 仅当用户通过命令行参数显式指定时才生效。
DATA_DIR=""
OUTPUT_DIR=""
# 用于 help 信息展示的示例默认路径
DEFAULT_DATA_DIR_EXAMPLE="${PROJECT_ROOT}/data4draw/${DEFAULT_UST_ID}/${STORY_NAME}/${DEFAULT_METHOD}"
DEFAULT_OUTPUT_DIR_EXAMPLE="${PROJECT_ROOT}/draw_result/${DEFAULT_UST_ID}/${STORY_NAME}/${DEFAULT_METHOD}"
# 全局共享语义矩阵（固定路径）
GLOBAL_DATA_DIR="${PROJECT_ROOT}/data4draw"
GLOBAL_SEMANTIC_MATRIX="${GLOBAL_DATA_DIR}/semantic_matrix_gemma2_2b.csv"

# =========================
# 其他参数（如无特殊需求可不改）
# =========================
PREDICTION_MATRIX=""
SEMANTIC_MATRIX=""
MAPPING_FILE=""
RESULTS_BASENAME=""  # 用于自动构建文件路径（例如：results_sae_parcel_level_20251122_133937）
OUTPUT_FILE=""
TOP_K=10
TOP_K_HUMAN=10
BOTTOM_K=0
RANDOM_K=0
BOTTOM_K_HUMAN=0
RANDOM_K_HUMAN=0
OVERWRITE="--overwrite"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --story_name)
            STORY_NAME="$2"
            DATA_DIR="${PROJECT_ROOT}/data4draw/${UST_ID}/${STORY_NAME}/${METHOD}"
            OUTPUT_DIR="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --results_basename)
            RESULTS_BASENAME="$2"
            shift 2
            ;;
        --prediction-matrix)
            PREDICTION_MATRIX="$2"
            shift 2
            ;;
        --semantic-matrix)
            SEMANTIC_MATRIX="$2"
            shift 2
            ;;
        --mapping-file)
            MAPPING_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --top-k-human)
            TOP_K_HUMAN="$2"
            shift 2
            ;;
        --bottom-k)
            BOTTOM_K="$2"
            shift 2
            ;;
        --random-k)
            RANDOM_K="$2"
            shift 2
            ;;
        --bottom-k-human)
            BOTTOM_K_HUMAN="$2"
            shift 2
            ;;
        --random-k-human)
            RANDOM_K_HUMAN="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --story_name NAME         Story 名称（用于构建默认路径）"
            echo "  --data_dir DIR           数据目录路径（默认: ${DEFAULT_DATA_DIR_EXAMPLE}，批量模式下一般按 UST_ID/METHOD 自动构建）"
            echo "  --results_basename NAME  结果文件夹名称（用于自动构建文件路径，例如: results_sae_parcel_level_20251122_133937）"
            echo "  --prediction-matrix FILE 预测矩阵 CSV 文件路径（如果指定，将覆盖自动构建的路径）"
            echo "  --semantic-matrix FILE   语义矩阵 CSV 文件路径（如果指定，将覆盖自动构建的路径）"
            echo "  --mapping-file FILE      映射文件 JSON 路径（如果指定，将覆盖自动构建的路径）"
            echo "  --output FILE            输出 CSV 文件路径（默认: \${OUTPUT_DIR}/fig2_top_llm_parcels_per_human.csv）"
            echo "  --output_dir DIR         输出目录路径（默认: \${PROJECT_ROOT}/draw_result/\${UST_ID}/\${STORY_NAME}/\${METHOD}，例如: ${DEFAULT_OUTPUT_DIR_EXAMPLE}）"
            echo "  --top-k K                每个 LLM parcel 保留的 human parcels 数目（默认: $TOP_K）"
            echo "  --top-k-human K          每个 human parcel 保留的 LLM parcels 数目（默认: $TOP_K_HUMAN）"
            echo "  --bottom-k K             每个 LLM parcel 额外保留的 bottom-k human parcels 数目（默认: $BOTTOM_K）"
            echo "  --random-k K             从非 top/bottom 的 human parcels 中随机采样的数量（默认: $RANDOM_K）"
            echo "  --bottom-k-human K       每个 human parcel 额外保留的 bottom-k LLM parcels 数目（默认: $BOTTOM_K_HUMAN）"
            echo "  --random-k-human K       从非 top/bottom 的 LLM parcels 中为每个 human parcel 随机采样的数量（默认: $RANDOM_K_HUMAN）"
            echo "  --overwrite               覆盖已存在的输出文件"
            echo "  -h, --help                显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # 使用 results_basename 自动构建文件路径"
            echo "  $0 --story_name adollshouse --results_basename results_sae_parcel_level_20251122_133937"
            echo "  # 直接指定文件路径"
            echo "  $0 --prediction-matrix /path/to/prediction_matrix.csv --semantic-matrix /path/to/semantic_matrix.csv --mapping-file /path/to/mapping.json"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 总体退出码（用于记录是否有某些组合失败）
OVERALL_EXIT_CODE=0

for UST_ID in "${UST_IDS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    echo "============================================"
    echo "当前组合: UST_ID=${UST_ID}, STORY_NAME=${STORY_NAME}, METHOD=${METHOD}"
    echo "============================================"

    # 针对当前 UST_ID / METHOD 构建数据与输出目录（如果用户没有通过参数显式指定）
    if [[ -z "$DATA_DIR" ]]; then
        CURRENT_DATA_DIR="${PROJECT_ROOT}/data4draw/${UST_ID}/${STORY_NAME}/${METHOD}"
    else
        CURRENT_DATA_DIR="$DATA_DIR"
    fi

    if [[ -z "$OUTPUT_DIR" ]]; then
        CURRENT_OUTPUT_DIR="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}"
    else
        CURRENT_OUTPUT_DIR="$OUTPUT_DIR"
    fi

    # 如果 bottom / random 相关参数被开启，则在输出路径上附加这些信息
    OUTPUT_SUFFIX=""
    # 注意：只要对应的 k > 0，就认为是“开启”
    if [[ "$BOTTOM_K" -gt 0 || "$RANDOM_K" -gt 0 || "$BOTTOM_K_HUMAN" -gt 0 || "$RANDOM_K_HUMAN" -gt 0 ]]; then
        # 逐项拼接，保证只把实际开启的部分写进路径
        SUFFIX_PARTS=()
        if [[ "$BOTTOM_K" -gt 0 ]]; then
            SUFFIX_PARTS+=("bottom${BOTTOM_K}")
        fi
        if [[ "$RANDOM_K" -gt 0 ]]; then
            SUFFIX_PARTS+=("random${RANDOM_K}")
        fi
        if [[ "$BOTTOM_K_HUMAN" -gt 0 ]]; then
            SUFFIX_PARTS+=("bottomH${BOTTOM_K_HUMAN}")
        fi
        if [[ "$RANDOM_K_HUMAN" -gt 0 ]]; then
            SUFFIX_PARTS+=("randomH${RANDOM_K_HUMAN}")
        fi

        OUTPUT_SUFFIX="_$(IFS=_; echo "${SUFFIX_PARTS[*]}")"
        CURRENT_OUTPUT_DIR="${CURRENT_OUTPUT_DIR}${OUTPUT_SUFFIX}"
    fi

    # 为当前组合拷贝一份局部变量，避免不同组合之间相互污染
    CURRENT_PREDICTION_MATRIX="$PREDICTION_MATRIX"
    CURRENT_SEMANTIC_MATRIX="$SEMANTIC_MATRIX"
    CURRENT_MAPPING_FILE="$MAPPING_FILE"
    CURRENT_OUTPUT_FILE="$OUTPUT_FILE"

    # 如果没有显式指定 PREDICTION_MATRIX，则在 CURRENT_DATA_DIR 中自动寻找最新的 prediction_matrix_*.csv
    if [ -z "$CURRENT_PREDICTION_MATRIX" ]; then
        if [ ! -d "$CURRENT_DATA_DIR" ]; then
            echo "错误: 数据目录不存在: $CURRENT_DATA_DIR"
            echo "请检查 UST_ID / STORY_NAME / METHOD 是否配置正确。"
            OVERALL_EXIT_CODE=1
            continue
        fi
        LATEST_PRED=$(ls -1 "${CURRENT_DATA_DIR}"/prediction_matrix_*.csv 2>/dev/null | sort | tail -n 1)
        if [ -z "$LATEST_PRED" ]; then
            echo "错误: 在 ${CURRENT_DATA_DIR} 中未找到 prediction_matrix_*.csv"
            echo "请先运行 run_build_prediction_matrix_from_results.sh 生成预测矩阵，或手动指定 --prediction-matrix。"
            OVERALL_EXIT_CODE=1
            continue
        fi
        CURRENT_PREDICTION_MATRIX="$LATEST_PRED"
    fi

    # 如果没有显式指定 MAPPING_FILE，则根据预测矩阵自动推断
    if [ -z "$CURRENT_MAPPING_FILE" ]; then
        PRED_BASENAME="$(basename "$CURRENT_PREDICTION_MATRIX")"
        PRED_PREFIX="${PRED_BASENAME%.csv}"
        CURRENT_MAPPING_FILE="${CURRENT_DATA_DIR}/${PRED_PREFIX}_parcel_id_to_function_name.json"
    fi

    # 如果没有显式指定 SEMANTIC_MATRIX，则直接使用全局共享语义矩阵
    if [ -z "$CURRENT_SEMANTIC_MATRIX" ]; then
        CURRENT_SEMANTIC_MATRIX="$GLOBAL_SEMANTIC_MATRIX"
    fi

    # 检查必要的文件是否存在
    if [ -z "$CURRENT_PREDICTION_MATRIX" ]; then
        echo "错误: 无法确定预测矩阵文件路径，请检查配置或手动指定 --prediction-matrix"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ ! -f "$CURRENT_PREDICTION_MATRIX" ]; then
        echo "错误: 找不到预测矩阵文件: $CURRENT_PREDICTION_MATRIX"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ -z "$CURRENT_SEMANTIC_MATRIX" ]; then
        echo "错误: 无法确定语义矩阵文件路径，请检查配置或手动指定 --semantic-matrix"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ ! -f "$CURRENT_SEMANTIC_MATRIX" ]; then
        echo "错误: 找不到语义矩阵文件: $CURRENT_SEMANTIC_MATRIX"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ -z "$CURRENT_MAPPING_FILE" ]; then
        echo "错误: 无法确定映射文件路径，请检查配置或手动指定 --mapping-file"
        OVERALL_EXIT_CODE=1
        continue
    fi

    if [ ! -f "$CURRENT_MAPPING_FILE" ]; then
        echo "错误: 找不到映射文件: $CURRENT_MAPPING_FILE"
        OVERALL_EXIT_CODE=1
        continue
    fi

    # 设置输出文件路径
    if [ -z "$CURRENT_OUTPUT_FILE" ]; then
        mkdir -p "$CURRENT_OUTPUT_DIR"
        CURRENT_OUTPUT_FILE="${CURRENT_OUTPUT_DIR}/"
    fi

    # 运行 Python 脚本
    echo "开始导出 top-k 配对关系..."
    echo "预测矩阵: $CURRENT_PREDICTION_MATRIX"
    echo "语义矩阵: $CURRENT_SEMANTIC_MATRIX"
    echo "映射文件: $CURRENT_MAPPING_FILE"
    echo "输出文件: $CURRENT_OUTPUT_FILE"
    echo ""

    python export_top_human_matches.py \
      --prediction-matrix "$CURRENT_PREDICTION_MATRIX" \
      --semantic-matrix "$CURRENT_SEMANTIC_MATRIX" \
      --mapping-file "$CURRENT_MAPPING_FILE" \
      --top-k "$TOP_K" \
      --top-k-human "$TOP_K_HUMAN" \
      --output "$CURRENT_OUTPUT_FILE" \
      --bottom-k "$BOTTOM_K" \
      --random-k "$RANDOM_K" \
      --bottom-k-human "$BOTTOM_K_HUMAN" \
      --random-k-human "$RANDOM_K_HUMAN" \
      $OVERWRITE

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ 当前组合完成：UST_ID=${UST_ID}, STORY_NAME=${STORY_NAME}, METHOD=${METHOD}"
        echo "结果已保存到: $CURRENT_OUTPUT_FILE"
    else
        echo ""
        echo "✗ 当前组合执行失败，退出码: $exit_code"
        OVERALL_EXIT_CODE=$exit_code
    fi

  done
done

if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ 所有 UST_ID 与 METHOD 组合均已成功完成！"
else
    echo ""
    echo "✗ 部分组合执行失败，最后一个失败的退出码: $OVERALL_EXIT_CODE"
    exit $OVERALL_EXIT_CODE
fi