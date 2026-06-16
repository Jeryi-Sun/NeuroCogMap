#!/bin/bash

# Jailbreak 生成和评估运行脚本（支持多数据集循环）
# 使用方法: ./run_jailbreak.sh [模型名称] [是否使用in-context] [最大样本数] [是否跳过已存在] [数据集列表]
#
# 使用示例:
# 1. 处理所有数据集: ./run_jailbreak.sh
# 2. 只处理JBB数据集: ./run_jailbreak.sh google/gemma-2-2b true 100 false jbb
# 3. 只处理AdvBench数据集: ./run_jailbreak.sh google/gemma-2-2b true 100 false advbench
# 4. 处理多个指定数据集: ./run_jailbreak.sh google/gemma-2-2b true 100 false jbb,advbench
# 5. 跳过已存在的结果: ./run_jailbreak.sh google/gemma-2-2b true 100 true all

set -e  # 遇到错误立即退出

# 默认配置
DEFAULT_MODEL="meta-llama/Llama-3.1-8B"
DEFAULT_USE_INCONTEXT="true"
DEFAULT_MAX_SAMPLES="0"
DEFAULT_MODE="both"
DEFAULT_SKIP_EXISTING="false"

# 检查帮助参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Jailbreak 生成和评估运行脚本（支持多数据集循环）"
    echo ""
    echo "使用方法: ./run_jailbreak.sh [模型名称] [是否使用in-context] [最大样本数] [是否跳过已存在] [数据集列表]"
    echo ""
    echo "参数说明:"
    echo "  模型名称         - 要使用的模型ID (默认: $DEFAULT_MODEL)"
    echo "  是否使用in-context - true/false (默认: $DEFAULT_USE_INCONTEXT)"
    echo "  最大样本数       - 每个数据集处理的最大样本数 (默认: $DEFAULT_MAX_SAMPLES)"
    echo "  是否跳过已存在   - true/false (默认: $DEFAULT_SKIP_EXISTING)"
    echo "  数据集列表       - 要处理的数据集，用逗号分隔 (默认: all)"
    echo ""
    echo "使用示例:"
    echo "  1. 处理所有数据集: ./run_jailbreak.sh"
    echo "  2. 只处理JBB数据集: ./run_jailbreak.sh google/gemma-2-2b true 100 false jbb"
    echo "  3. 只处理AdvBench数据集: ./run_jailbreak.sh google/gemma-2-2b true 100 false advbench"
    echo "  4. 处理多个指定数据集: ./run_jailbreak.sh google/gemma-2-2b true 100 false jbb,advbench"
    echo "  5. 跳过已存在的结果: ./run_jailbreak.sh google/gemma-2-2b true 100 true all"
    echo ""
    echo "可用的数据集: jbb, advbench"
    exit 0
fi

# 获取参数
MODEL_ID=${1:-$DEFAULT_MODEL}
USE_INCONTEXT=${2:-$DEFAULT_USE_INCONTEXT}
MAX_SAMPLES=${3:-$DEFAULT_MAX_SAMPLES}
SKIP_EXISTING=${4:-$DEFAULT_SKIP_EXISTING}
DATASET_LIST=${5:-"all"}
MODE=$DEFAULT_MODE

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="/path/to/project_root/safety_explanation/jailbreak/results"

# 数据集配置
declare -A DATASETS
DATASETS["jbb"]="/path/to/project_root/safety_explanation/jailbreak/dataset/JBB-Behaviors.csv"
DATASETS["advbench"]="/path/to/project_root/safety_explanation/jailbreak/dataset/AdvBench.csv"

# vLLM配置
VLLM_URL="http://0.0.0.0:8001/v1"
API_KEY="abcabc"

# 确定要处理的数据集列表
if [ "$DATASET_LIST" = "all" ]; then
    DATASET_KEYS=("jbb" "advbench")
else
    # 解析逗号分隔的数据集列表
    IFS=',' read -ra DATASET_KEYS <<< "$DATASET_LIST"
fi

# 检查数据集文件是否存在
for dataset_key in "${DATASET_KEYS[@]}"; do
    if [ ! -v DATASETS["$dataset_key"] ]; then
        echo "错误: 未知的数据集 '$dataset_key'"
        echo "可用的数据集: jbb, advbench"
        exit 1
    fi
    
    csv_path="${DATASETS[$dataset_key]}"
    if [ ! -f "$csv_path" ]; then
        echo "错误: CSV文件不存在: $csv_path"
        exit 1
    fi
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 显示配置信息
echo "=========================================="
echo "Jailbreak 生成和评估脚本（多数据集循环）"
echo "=========================================="
echo "模型ID: $MODEL_ID"
echo "使用in-context learning: $USE_INCONTEXT"
echo "最大样本数: $MAX_SAMPLES"
echo "运行模式: $MODE"
echo "跳过已存在: $SKIP_EXISTING"
echo "数据集列表: ${DATASET_KEYS[*]}"
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# 记录总开始时间
TOTAL_START_TIME=$(date +%s)
echo "总开始时间: $(date)"
echo ""

# 循环处理每个数据集
for dataset_key in "${DATASET_KEYS[@]}"; do
    csv_path="${DATASETS[$dataset_key]}"
    
    echo "=========================================="
    echo "开始处理数据集: $dataset_key"
    echo "CSV文件: $csv_path"
    echo "=========================================="
    
    # 记录单个数据集开始时间
    DATASET_START_TIME=$(date +%s)
    echo "数据集开始时间: $(date)"
    
    # 构建命令
    CMD="python $SCRIPT_DIR/jailbreak_generate_and_eval.py"
    CMD="$CMD --mode $MODE"
    CMD="$CMD --model_id $MODEL_ID"
    CMD="$CMD --csv_path $csv_path"
    CMD="$CMD --output_dir $OUTPUT_DIR"
    CMD="$CMD --max_samples $MAX_SAMPLES"
    CMD="$CMD --max_new_tokens 512"
    CMD="$CMD --temperature 0.0"
    CMD="$CMD --vllm_url $VLLM_URL"
    CMD="$CMD --api_key $API_KEY"
    
    # 添加跳过已存在参数
    if [ "$SKIP_EXISTING" = "true" ]; then
        CMD="$CMD --skip_existing"
    fi
    
    # 添加in-context learning参数
    if [ "$USE_INCONTEXT" = "true" ]; then
        CMD="$CMD --use_incontext"
    fi
    
    # 执行命令
    echo "执行命令: $CMD"
    echo ""
    eval $CMD
    
    # 记录单个数据集结束时间
    DATASET_END_TIME=$(date +%s)
    DATASET_DURATION=$((DATASET_END_TIME - DATASET_START_TIME))
    echo ""
    echo "数据集 $dataset_key 完成!"
    echo "数据集耗时: ${DATASET_DURATION}秒"
    echo ""
done

# 记录总结束时间
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
echo "=========================================="
echo "所有数据集处理完成!"
echo "总结束时间: $(date)"
echo "总耗时: ${TOTAL_DURATION}秒"
echo "=========================================="
