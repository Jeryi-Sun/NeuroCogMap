#!/bin/bash
# 功能介绍：运行 generate_improvement_suggestions_cogneuromap.py 脚本，基于 Parcel 和 Capability 激活信息生成认知模型改进建议
# 功能说明：
#   - 读取 parcel_activation_summary.json 中的神经激活模式
#   - 使用 LLM 分析不同条件下的 Top Parcel 和 Top Capability
#   - 生成针对认知模型改进的建议
#   - 重点关注差值分析中的认知脑区
# 使用方法：
#   bash run_generate_improvement_suggestions_cogneuromap.sh [--summary-file PATH] [--force] [--steps step1 step2 reward] [--simple]
#   参数说明：
#     --summary-file PATH: 指定激活汇总文件路径（默认使用 parcel_activation_summary.json）
#     --force: 强制运行，即使输出文件已存在也会覆盖
#     --steps: 指定要处理的步骤（默认: step1 step2 reward）
#     --skip-existing: 如果输出文件已存在则跳过（默认行为）
#     --simple: 使用 simple 模式，只包含用户实验数据，不包含 cogNeuromap 信息

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/path/to/project_root"
PYTHON_SCRIPT="${SCRIPT_DIR}/generate_improvement_suggestions_cogneuromap.py"
DEFAULT_SUMMARY_FILE="${PROJECT_ROOT}/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/results/comparison_results/parcel_activation_summary.json"
OUTPUT_DIR="${PROJECT_ROOT}/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_suggestion_results"

# 默认参数
FORCE_RUN=true
SKIP_IF_EXISTS=true
SUMMARY_FILE="$DEFAULT_SUMMARY_FILE"
STEPS="step1 step2 reward"
SIMPLE_MODE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_RUN=true
            SKIP_IF_EXISTS=false
            shift
            ;;
        --skip-existing)
            SKIP_IF_EXISTS=true
            shift
            ;;
        --summary-file)
            SUMMARY_FILE="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --simple)
            SIMPLE_MODE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [--summary-file PATH] [--force] [--skip-existing] [--steps \"step1 step2 reward\"] [--simple]"
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$PROJECT_ROOT" || {
    echo "错误: 无法切换到项目根目录: $PROJECT_ROOT"
    exit 1
}

# 检查输入文件是否存在
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "错误: 激活汇总文件不存在: $SUMMARY_FILE"
    exit 1
fi

# 从文件名提取输出文件名
DATASET_NAME=$(basename "$SUMMARY_FILE")
if [[ "$DATASET_NAME" == *_summary.json ]]; then
    DATASET_NAME="${DATASET_NAME%_summary.json}"
elif [[ "$DATASET_NAME" == *.json ]]; then
    DATASET_NAME="${DATASET_NAME%.json}"
fi
# 根据模式确定输出文件名
if [ "$SIMPLE_MODE" = true ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_cogneuromap_improvement_suggestions_simple.json"
else
    OUTPUT_FILE="${OUTPUT_DIR}/${DATASET_NAME}_cogneuromap_improvement_suggestions.json"
fi

# 检查输出文件是否存在
if [ -f "$OUTPUT_FILE" ]; then
    if [ "$SKIP_IF_EXISTS" = true ] && [ "$FORCE_RUN" = false ]; then
        echo "=========================================="
        echo "输出文件已存在: $OUTPUT_FILE"
        echo "跳过运行（使用 --force 可强制覆盖）"
        echo "=========================================="
        exit 0
    elif [ "$FORCE_RUN" = true ]; then
        echo "=========================================="
        echo "输出文件已存在，将覆盖: $OUTPUT_FILE"
        echo "=========================================="
    fi
fi

# 检查 Python 脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: Python 脚本不存在: $PYTHON_SCRIPT"
    exit 1
fi

# 运行 Python 脚本
echo "=========================================="
echo "开始运行认知模型改进建议生成..."
echo "=========================================="
echo "Python 脚本: $PYTHON_SCRIPT"
echo "输入文件: $SUMMARY_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "处理步骤: $STEPS"
if [ "$SIMPLE_MODE" = true ]; then
    echo "模式: Simple（只包含实验数据，不包含 cogNeuromap 信息）"
else
    echo "模式: cogNeuromap（包含 Parcel 和 Capability 激活信息）"
fi
echo "=========================================="

# 构建 Python 命令
PYTHON_CMD="python3 \"$PYTHON_SCRIPT\" --summary_file \"$SUMMARY_FILE\" --output_dir \"$OUTPUT_DIR\" --steps $STEPS --vllm_url \"https://api2.aigcbest.top/v1\" --api_key \"sk-liaIJiVfNSuOIZzmypbfVhDKJkTbD7boA1QZNimDegW0PJ4I\" --model \"gpt-5.2-2025-12-11\""
if [ "$FORCE_RUN" = true ]; then
    # 注意：Python 脚本使用 --skip_existing，所以 force 模式下不传这个参数
    PYTHON_CMD="$PYTHON_CMD"
else
    PYTHON_CMD="$PYTHON_CMD --skip_existing"
fi
if [ "$SIMPLE_MODE" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --simple"
fi

eval "$PYTHON_CMD"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "运行成功！"
    echo "结果已保存到: $OUTPUT_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "运行失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

