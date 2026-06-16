#!/bin/bash
# 功能介绍：运行 summarize_parcel_activations.py 脚本，汇总 two-step 认知实验中 LLM parcel 激活数据
# 功能说明：
#   - 读取 participant_parcel_activations.json 中的激活数据
#   - 结合 parcel 功能描述文件，生成 Top-K parcel 汇总结果
#   - 如果提供 Capability 映射文件，同时计算 Capability 级别的 Top-K
#   - 支持检测输出文件是否已存在，避免重复运行
# 使用方法：
#   bash run_summarize_parcel_activations.sh [--force] [--top-k K] [--capability-mapping PATH]
#   参数说明：
#     --force: 强制运行，即使输出文件已存在也会覆盖
#     --top-k K: 指定 Top-K 数量（默认 5）
#     --capability-mapping PATH: 指定 Capability-Parcel 映射 JSON 文件路径（可选）
#     --skip-if-exists: 如果输出文件已存在则跳过（默认行为）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/path/to/project_root"
PYTHON_SCRIPT="${SCRIPT_DIR}/summarize_parcel_activations.py"
OUTPUT_FILE="${PROJECT_ROOT}/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/results/comparison_results/parcel_activation_summary.json"

# 默认参数
FORCE_RUN=true
SKIP_IF_EXISTS=false
TOP_K=10
CAPABILITY_MAPPING=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_RUN=true
            SKIP_IF_EXISTS=false
            shift
            ;;
        --skip-if-exists)
            SKIP_IF_EXISTS=true
            shift
            ;;
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --capability-mapping)
            CAPABILITY_MAPPING="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "使用方法: $0 [--force] [--skip-if-exists] [--top-k K] [--capability-mapping PATH]"
            exit 1
            ;;
    esac
done

# 切换到项目根目录
cd "$PROJECT_ROOT" || {
    echo "错误: 无法切换到项目根目录: $PROJECT_ROOT"
    exit 1
}

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
echo "开始运行 parcel 激活汇总分析..."
echo "=========================================="
echo "Python 脚本: $PYTHON_SCRIPT"
echo "输出文件: $OUTPUT_FILE"
echo "Top-K: $TOP_K"
if [ -n "$CAPABILITY_MAPPING" ]; then
    echo "Capability 映射文件: $CAPABILITY_MAPPING"
fi
echo "=========================================="

# 构建 Python 命令
PYTHON_CMD="python3 \"$PYTHON_SCRIPT\" --top-k \"$TOP_K\""
if [ "$FORCE_RUN" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --overwrite"
fi
if [ -n "$CAPABILITY_MAPPING" ]; then
    PYTHON_CMD="$PYTHON_CMD --capability-mapping \"$CAPABILITY_MAPPING\""
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

