#!/bin/bash
# 使用 LLM 预测人类认知行为的批处理脚本
# 遍历不同的模型和max_tokens组合
#
# 使用方法:
#   1. 不传参数：自动扫描所有 *_reformatted.jsonl 文件并处理
#      ./run_llm_prediction.sh
#
#   2. 传入指定数据集：只处理指定的数据集
#      ./run_llm_prediction.sh kool2016when_exp2 kool2017cost_exp2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/../dataset"
OUTPUT_DIR="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/predictions"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 模型列表：模型全名和简写名称的映射
declare -A MODELS
MODELS["google/gemma-2-2b"]="gemma2b"
# MODELS["google/gemma-2-9b-it"]="gemma2-9b-it"

# max_tokens 列表
MAX_TOKENS_LIST=(1024)

# 数据集列表
# 如果通过命令行参数传入，使用传入的数据集；否则自动扫描 train 和 test 目录
if [ $# -gt 0 ]; then
    # 从命令行参数读取数据集列表（同时检查 train 和 test 目录）
    declare -a TRAIN_DATASETS
    declare -a TEST_DATASETS
    
    for DATASET in "$@"; do
        if [ -f "$DATASET_DIR/train/${DATASET}_reformatted.jsonl" ]; then
            TRAIN_DATASETS+=("$DATASET")
        fi
        if [ -f "$DATASET_DIR/test/${DATASET}_reformatted.jsonl" ]; then
            TEST_DATASETS+=("$DATASET")
        fi
    done
    
    echo "使用命令行参数指定的数据集: $@"
    echo "train 数据集: ${TRAIN_DATASETS[@]}"
    echo "test 数据集: ${TEST_DATASETS[@]}"
else
    # 自动扫描 train 和 test 目录，找到所有 *_reformatted.jsonl 文件
    echo "未指定数据集，自动扫描 train 和 test 目录..."
    
    # 扫描 train 目录
    declare -A TRAIN_DATASET_SET
    if [ -d "$DATASET_DIR/train" ]; then
        while IFS= read -r file; do
            dataset_name=$(basename "$file" | sed 's/_reformatted\.jsonl$//')
            if [ -n "$dataset_name" ]; then
                TRAIN_DATASET_SET["$dataset_name"]=1
            fi
        done < <(find "$DATASET_DIR/train" -name "*_reformatted.jsonl" -type f | sort)
    fi
    
    # 扫描 test 目录
    declare -A TEST_DATASET_SET
    if [ -d "$DATASET_DIR/test" ]; then
        while IFS= read -r file; do
            dataset_name=$(basename "$file" | sed 's/_reformatted\.jsonl$//')
            if [ -n "$dataset_name" ]; then
                TEST_DATASET_SET["$dataset_name"]=1
            fi
        done < <(find "$DATASET_DIR/test" -name "*_reformatted.jsonl" -type f | sort)
    fi
    
    # 将关联数组的键转换为数组（已排序）
    TRAIN_DATASETS=($(printf '%s\n' "${!TRAIN_DATASET_SET[@]}" | sort))
    TEST_DATASETS=($(printf '%s\n' "${!TEST_DATASET_SET[@]}" | sort))
    
    echo "找到 ${#TRAIN_DATASETS[@]} 个 train 数据集: ${TRAIN_DATASETS[@]}"
    echo "找到 ${#TEST_DATASETS[@]} 个 test 数据集: ${TEST_DATASETS[@]}"
fi

DEVICE="cuda"  # 或 "cpu"

# 设置固定的 max_tokens 值（使用列表中的第一个值）
MAX_TOKENS=${MAX_TOKENS_LIST[0]}

# 总任务数（train + test）
TOTAL_DATASETS=$((${#TRAIN_DATASETS[@]} + ${#TEST_DATASETS[@]}))
TOTAL_TASKS=$((${#MODELS[@]} * $TOTAL_DATASETS))
CURRENT_TASK=0

echo "=========================================="
echo "开始批量处理"
echo "模型数量: ${#MODELS[@]}"
echo "max_tokens: $MAX_TOKENS"
echo "train 数据集数量: ${#TRAIN_DATASETS[@]}"
echo "test 数据集数量: ${#TEST_DATASETS[@]}"
echo "总数据集数量: $TOTAL_DATASETS"
echo "总任务数: $TOTAL_TASKS"
echo "=========================================="
echo ""

# 遍历所有模型
for MODEL_FULL in "${!MODELS[@]}"; do
    MODEL_SHORT="${MODELS[$MODEL_FULL]}"
    
    # # 先处理 train 目录的数据集
    # for DATASET in "${TRAIN_DATASETS[@]}"; do
    #     CURRENT_TASK=$((CURRENT_TASK + 1))
        
    #     INPUT_FILE="$DATASET_DIR/train/${DATASET}_reformatted.jsonl"
        
    #     if [ ! -f "$INPUT_FILE" ]; then
    #         echo "  ✗ 跳过: 未找到数据集文件 $INPUT_FILE"
    #         continue
    #     fi
        
    #     OUTPUT_FILE="$OUTPUT_DIR/llm_prediction_${MODEL_SHORT}_max${MAX_TOKENS}_train_${DATASET}_filtered.csv"
        
    #     echo "[$CURRENT_TASK/$TOTAL_TASKS] 处理: $MODEL_SHORT, max_tokens=$MAX_TOKENS, dataset=train/$DATASET"
    #     echo "  输入文件: $INPUT_FILE"
    #     echo "  输出文件: $OUTPUT_FILE"
        
    #     python3 "$SCRIPT_DIR/llm_cognitive_prediction.py" \
    #         --model_name "$MODEL_FULL" \
    #         --max_tokens "$MAX_TOKENS" \
    #         --input_file "$INPUT_FILE" \
    #         --output_file "$OUTPUT_FILE" \
    #         --device "$DEVICE"
        
    #     if [ $? -eq 0 ]; then
    #         echo "  ✓ 完成"
    #     else
    #         echo "  ✗ 失败"
    #     fi
    #     echo ""
    # done
    
    # 再处理 test 目录的数据集
    for DATASET in "${TEST_DATASETS[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        
        INPUT_FILE="$DATASET_DIR/test/${DATASET}_reformatted.jsonl"
        
        if [ ! -f "$INPUT_FILE" ]; then
            echo "  ✗ 跳过: 未找到数据集文件 $INPUT_FILE"
            continue
        fi
        
        OUTPUT_FILE="$OUTPUT_DIR/test/llm_prediction_${MODEL_SHORT}_max${MAX_TOKENS}_${DATASET}_filtered.csv"
        
        echo "[$CURRENT_TASK/$TOTAL_TASKS] 处理: $MODEL_SHORT, max_tokens=$MAX_TOKENS, dataset=test/$DATASET"
        echo "  输入文件: $INPUT_FILE"
        echo "  输出文件: $OUTPUT_FILE"
        
        python3 "$SCRIPT_DIR/llm_cognitive_prediction.py" \
            --model_name "$MODEL_FULL" \
            --max_tokens "$MAX_TOKENS" \
            --input_file "$INPUT_FILE" \
            --output_file "$OUTPUT_FILE" \
            --device "$DEVICE"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ 完成"
        else
            echo "  ✗ 失败"
        fi
        echo ""
    done
done

echo "=========================================="
echo "所有处理完成！"
echo "=========================================="

