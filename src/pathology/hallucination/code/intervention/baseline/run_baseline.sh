#!/bin/bash

# Baseline 干预实验批量运行脚本
# 流程：
#   1) 先从 correct / incorrect 数据中提取 steer vector 并保存
#   2) 再加载保存的 steer vector 进行干预评估
# 评测方案与上级目录的 intervention 保持一致

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_baseline.py"
EXTRACT_SCRIPT="$SCRIPT_DIR/extract_steer_vector.py"
CONFIG_FILE="$SCRIPT_DIR/../config.json"

# STRENGTH 列表：可以修改为需要测试的强度值
if [[ -n "${INTERVENTION_STRENGTHS:-}" ]]; then
    read -r -a STRENGTH_LIST <<< "$INTERVENTION_STRENGTHS"
else
    STRENGTH_LIST=(paper)
fi

# 配置参数（内部设置，无需外部传入）
RESULTS_BASE_DIR="/path/to/project_root/safety_explanation/hallucination/results"
DATASET_DIR="/path/to/project_root/safety_explanation/hallucination/dataset"
STEER_DIR="$SCRIPT_DIR/steer_vectors"
MAX_SAMPLES="1500"                            # 最大样本数（控制计算量）0表示处理所有样本
ENABLE_EVAL="yes"                             # 是否启用评估（需要vLLM服务）
VLLM_URL="http://0.0.0.0:8001/v1"            # vLLM服务地址
MODEL_NAME="gemma-2-2b"                      # 模型名称（用于文件命名）
SKIP_EXISTING="yes"                           # 是否跳过已存在的结果文件

# 获取数据集列表
get_dataset_list() {
    local datasets=()
    for csv_file in "$DATASET_DIR"/*.csv; do
        if [[ -f "$csv_file" ]]; then
            local basename=$(basename "$csv_file" .csv)
            datasets+=("$basename")
        fi
    done
    echo "${datasets[@]}"
}

# 主函数
main() {
    echo "=========================================="
    echo "Baseline 干预实验批量运行"
    echo "=========================================="
    echo "配置参数:"
    echo "  - 数据集目录: $DATASET_DIR"
    echo "  - 结果目录: $RESULTS_BASE_DIR"
    echo "  - 最大样本数: $MAX_SAMPLES"
    echo "  - 启用评估: $ENABLE_EVAL"
    echo "  - vLLM URL: $VLLM_URL"
    echo "  - 模型名称: $MODEL_NAME"
    echo "  - 跳过已存在: $SKIP_EXISTING"
    echo "  - 干预强度列表: ${STRENGTH_LIST[@]}"
    echo ""
    
    mkdir -p "$STEER_DIR"
    
    # 获取数据集列表
    local datasets=($(get_dataset_list))
    
    if [[ ${#datasets[@]} -eq 0 ]]; then
        echo "[ERROR] 未找到任何数据集文件" >&2
        exit 1
    fi
    
    echo "[INFO] 找到 ${#datasets[@]} 个数据集: ${datasets[@]}"
    echo ""
    
    # 遍历每个数据集
    for dataset_name in "${datasets[@]}"; do
        echo "=========================================="
        echo "处理数据集: $dataset_name"
        echo "=========================================="
        
        # 构建文件路径
        local correct_jsonl="$RESULTS_BASE_DIR/${dataset_name}_${MODEL_NAME}/correct.jsonl"
        local incorrect_jsonl="$RESULTS_BASE_DIR/${dataset_name}_${MODEL_NAME}/incorrect.jsonl"
        local test_dataset="$DATASET_DIR/${dataset_name}.csv"
        
        # 检查文件是否存在
        if [[ ! -f "$correct_jsonl" ]]; then
            echo "[WARNING] correct 数据文件不存在: $correct_jsonl，跳过" >&2
            continue
        fi
        
        if [[ ! -f "$incorrect_jsonl" ]]; then
            echo "[WARNING] incorrect 数据文件不存在: $incorrect_jsonl，跳过" >&2
            continue
        fi
        
        if [[ ! -f "$test_dataset" ]]; then
            echo "[WARNING] 测试数据集文件不存在: $test_dataset，跳过" >&2
            continue
        fi
        
        echo "[INFO] correct 数据: $correct_jsonl"
        echo "[INFO] incorrect 数据: $incorrect_jsonl"
        echo "[INFO] 测试数据集: $test_dataset"
        echo ""
        
        # 生成 steer vector
        local steer_vector_file="$STEER_DIR/${dataset_name}_${MODEL_NAME}_steer.pt"
        if [[ "$SKIP_EXISTING" == "yes" && -f "$steer_vector_file" ]]; then
            echo "[INFO] Steer vector 文件已存在，跳过提取: $steer_vector_file"
        else
            echo "[INFO] 开始提取 steer vector -> $steer_vector_file"
            local extractor_cmd="python"
            if "$extractor_cmd" "$EXTRACT_SCRIPT" \
                --config "$CONFIG_FILE" \
                --correct_jsonl "$correct_jsonl" \
                --incorrect_jsonl "$incorrect_jsonl" \
                --output_file "$steer_vector_file"; then
                echo "[SUCCESS] Steer vector 提取完成"
            else
                echo "[ERROR] Steer vector 提取失败，跳过该数据集" >&2
                continue
            fi
        fi
        
        # 遍历每个干预强度
        for strength in "${STRENGTH_LIST[@]}"; do
            echo "----------------------------------------"
            echo "测试干预强度: $strength"
            echo "----------------------------------------"
            
            # 构建输出文件路径
            local output_file="$RESULTS_BASE_DIR/intervention/baseline/${dataset_name}_${MODEL_NAME}_strength${strength}_baseline.json"
            
            # 检查是否跳过已存在的文件
            if [[ "$SKIP_EXISTING" == "yes" && -f "$output_file" ]]; then
                echo "[INFO] 输出文件已存在，跳过: $output_file"
                continue
            fi
            
            # 构建 Python 命令
            local python_cmd="python"
            
            local cmd_args=(
                "$PY_SCRIPT"
                --config "$CONFIG_FILE"
                --steer_vector_file "$steer_vector_file"
                --test_dataset "$test_dataset"
                --intervention_strength "$strength"
                --max_samples "$MAX_SAMPLES"
                --output_file "$output_file"
            )
            
            # 添加评估选项
            if [[ "$ENABLE_EVAL" == "yes" ]]; then
                cmd_args+=(--enable_evaluation)
            fi
            
            # 添加 vLLM URL
            if [[ -n "$VLLM_URL" ]]; then
                cmd_args+=(--vllm_url "$VLLM_URL")
            fi
            
            # 添加跳过选项
            if [[ "$SKIP_EXISTING" == "yes" ]]; then
                cmd_args+=(--skip_if_exists)
            fi
            
            # 执行命令
            echo "[INFO] 执行命令:"
            echo "  $python_cmd ${cmd_args[@]}"
            echo ""
            
            if "$python_cmd" "${cmd_args[@]}"; then
                echo "[SUCCESS] 强度 $strength 实验完成"
            else
                echo "[ERROR] 强度 $strength 实验失败" >&2
                # 继续处理下一个强度，不中断整个流程
            fi
            
            echo ""
        done
        
        echo "[INFO] 数据集 $dataset_name 处理完成"
        echo ""
    done
    
    echo "=========================================="
    echo "所有实验完成"
    echo "=========================================="
}

# 运行主函数
main "$@"

