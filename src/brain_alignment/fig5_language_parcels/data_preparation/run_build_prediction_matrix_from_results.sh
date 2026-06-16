#!/bin/bash
# 运行脚本：从原始结果文件夹构建 LLM Parcel 对 Human Parcel 的预测结果矩阵
# 功能：直接从 results_* 文件夹读取数据，构建预测矩阵，去除第 50 和 101 个 Human Parcel

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基本配置（直接在这里改）
# =========================
# UST 被试 ID（如：uts02 / uts03）
UST_ID="uts03"
# 数据名称 / 故事名称（如：whereisthesmoke / adollshouse）
STORY_NAME="adollshouse"
# 方法名称（目前支持：sae / saeact）
METHOD="saeact"

# 结果根目录与数据根目录（一般不需要改）
PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core"
RESULTS_ROOT="${PROJECT_ROOT}/results/${UST_ID}"
DATA4DRAW_ROOT="${PROJECT_ROOT}/data_analysis/draw_graphs/data4draw/${UST_ID}/${STORY_NAME}/${METHOD}"

# =========================
# 其他参数（如无特殊需求可不改）
# =========================
#/path/to/project_root/Human_LLM_align/litcoder_core/results/uts02/results_saeact_singleparcel_test25_eval0_mode_single_20260312_215326
RESULTS_DIR="/path/to/project_root/Human_LLM_align/litcoder_core/results/uts03/results_saeact_singleparcel_test25_eval0_mode_single_20260311_110819"   # 如果留空，将自动在 RESULTS_ROOT 中寻找最新的 results_${METHOD}_parcel_level_* 目录
OUTPUT_DIR="${DATA4DRAW_ROOT}"
HUMAN_PARCEL_FILE="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
LLM_PARCEL_FILE="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
ID_KEY="layer_idx"
SKIP_EXISTING=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --human_parcel_file)
            HUMAN_PARCEL_FILE="$2"
            shift 2
            ;;
        --llm_parcel_file)
            LLM_PARCEL_FILE="$2"
            shift 2
            ;;
        --id_key)
            ID_KEY="$2"
            shift 2
            ;;
        --skip_existing)
            SKIP_EXISTING=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --results_dir DIR         原始结果文件夹路径（包含多个 run_* 子目录）"
            echo "                            (默认: 自动在 \${RESULTS_ROOT} 中查找最新的 results_\${METHOD}_parcel_level_*）"
            echo "  --output_dir DIR          输出目录路径（用于自动生成输出文件名）"
            echo "                            (默认: $OUTPUT_DIR)"
            echo "  --output_file FILE         输出 CSV 文件路径（如果指定，将覆盖 --output_dir）"
            echo "  --human_parcel_file FILE   Human Parcel descriptions JSON 文件路径 (默认: $HUMAN_PARCEL_FILE)"
            echo "  --llm_parcel_file FILE     LLM Parcel functionality summary JSON 文件路径 (默认: $LLM_PARCEL_FILE)"
            echo "  --id_key KEY               hyperparams.json 中用于标识 ID 的字段名 (默认: $ID_KEY)"
            echo "  --skip_existing           如果输出文件已存在，则跳过处理"
            echo "  -h, --help                显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --results_dir /path/to/results_sae_parcel_level_20251122_133937"
            echo "  $0 --results_dir /path/to/results_dir --output_file /path/to/output.csv"
            echo "  $0 --results_dir /path/to/results_dir --skip_existing"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查 Python 脚本是否存在
PYTHON_SCRIPT="$SCRIPT_DIR/build_prediction_matrix_from_results.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 如果没有显式指定 RESULTS_DIR，则在 RESULTS_ROOT 中自动寻找最新的 results_${METHOD}_parcel_level_* 目录
if [ -z "$RESULTS_DIR" ]; then
    if [ ! -d "$RESULTS_ROOT" ]; then
        echo "错误: 找不到结果根目录: $RESULTS_ROOT"
        exit 1
    fi
    LATEST_RESULTS_DIR=$(ls -d "${RESULTS_ROOT}/results_${METHOD}_parcel_level_"* 2>/dev/null | sort | tail -n 1)
    if [ -z "$LATEST_RESULTS_DIR" ]; then
        echo "错误: 在 ${RESULTS_ROOT} 中未找到 results_${METHOD}_parcel_level_* 目录，请检查 UST_ID/METHOD 是否正确"
        exit 1
    fi
    RESULTS_DIR="$LATEST_RESULTS_DIR"
fi

# 检查输入文件夹是否存在
if [ ! -d "$RESULTS_DIR" ]; then
    echo "错误: 找不到输入文件夹: $RESULTS_DIR"
    exit 1
fi

# 如果没有指定输出文件，则根据输入文件夹名称自动生成
if [ -z "$OUTPUT_FILE" ]; then
    # 从结果文件夹名称提取标识符
    RESULTS_BASENAME=$(basename "$RESULTS_DIR")
    OUTPUT_NAME="prediction_matrix_${RESULTS_BASENAME}.csv"
    OUTPUT_FILE="$OUTPUT_DIR/$OUTPUT_NAME"
fi

# 运行 Python 脚本（使用 conda 的 lit 环境）
echo "开始从原始结果文件夹构建预测结果矩阵..."
echo "输入文件夹: $RESULTS_DIR"
echo "输出文件: $OUTPUT_FILE"
echo "ID 字段: $ID_KEY"
if [ "$SKIP_EXISTING" = true ]; then
    echo "跳过已存在文件: 是"
fi
echo ""

# 构建命令
CMD="conda run -n lit python \"$PYTHON_SCRIPT\" \
    --results_dir \"$RESULTS_DIR\" \
    --output_file \"$OUTPUT_FILE\" \
    --id_key \"$ID_KEY\""

if [ -n "$HUMAN_PARCEL_FILE" ] && [ -f "$HUMAN_PARCEL_FILE" ]; then
    CMD="$CMD --human_parcel_file \"$HUMAN_PARCEL_FILE\""
fi

if [ -n "$LLM_PARCEL_FILE" ] && [ -f "$LLM_PARCEL_FILE" ]; then
    CMD="$CMD --llm_parcel_file \"$LLM_PARCEL_FILE\""
fi

if [ "$SKIP_EXISTING" = true ]; then
    CMD="$CMD --skip_existing"
fi

# 执行命令
eval $CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ 所有任务完成！"
    echo "结果已保存到: $OUTPUT_FILE"
else
    echo ""
    echo "✗ 任务执行失败，退出码: $exit_code"
    exit $exit_code
fi
