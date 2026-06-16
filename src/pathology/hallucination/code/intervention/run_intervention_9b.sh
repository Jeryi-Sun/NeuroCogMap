#!/bin/bash

# 幻觉干预实验批量运行脚本
# 自动循环处理所有CSV数据集，保存模型生成结果和评估结果


# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_intervention.py"
CONFIG_FILE="$SCRIPT_DIR/config_9b.json"

# 配置参数（内部设置，无需外部传入）
# Default release mode uses the paper-selected strength for each dataset.
# Set INTERVENTION_STRENGTHS="0.1 0.3 0.5" to run an optional sensitivity sweep.
if [[ -n "${INTERVENTION_STRENGTHS:-}" ]]; then
    read -r -a STRENGTH_LIST <<< "$INTERVENTION_STRENGTHS"
else
    STRENGTH_LIST=(paper)
fi
DATASET_DIR="/path/to/project_root/safety_explanation/hallucination/dataset"
ANALYSIS_DIR="/path/to/project_root/safety_explanation/hallucination/results/analysis_output"
PARCEL_IDS=""                                 # 自动从异常分析文件加载
MAX_SAMPLES="1500"                            # 最大样本数（控制计算量）0表示处理所有样本
ENABLE_EVAL="yes"                           # 是否启用评估（需要vLLM服务）
VLLM_URL="http://0.0.0.0:8001/v1"            # vLLM服务地址
MODEL_NAME="gemma-2-9b-it"                      # 模型名称（用于文件命名）
SKIP_EXISTING="no"                          # 是否跳过已存在的结果文件

# 自动提取parcel ID

get_paper_strength() {
    local model_name="$1"
    local dataset_name="$2"
    case "${model_name}|${dataset_name}" in
        "Llama-3.1-8B|MedHallu") echo "0.3" ;;
        "Llama-3.1-8B|nq_open") echo "0.1" ;;
        "Llama-3.1-8B|truthfulqa") echo "0.3" ;;
        "gemma-2-2b|MedHallu") echo "0.5" ;;
        "gemma-2-2b|nq_open") echo "0.3" ;;
        "gemma-2-2b|truthfulqa") echo "0.5" ;;
        "gemma-2-9b-it|MedHallu") echo "0.5" ;;
        "gemma-2-9b-it|nq_open") echo "0.1" ;;
        "gemma-2-9b-it|truthfulqa") echo "0.5" ;;
        *) echo "0.3" ;;
    esac
}

extract_parcel_ids() {
    local dataset_name="$1"
    local model_name="gemma-2-9b-it"  # 从模型名称转换而来
    local analysis_file="$ANALYSIS_DIR/${dataset_name}_${model_name}/parcel_level/top_anomalous_parcels.json"
    
    if [[ ! -f "$analysis_file" ]]; then
        echo "[WARNING] 未找到异常分析文件: $analysis_file" >&2
        echo "[INFO] 使用默认parcel IDs: 233 89 156" >&2
        echo "233 89 156"
        return
    fi
    
    echo "[INFO] 从异常分析文件提取parcel IDs: $analysis_file" >&2
    
    # 使用独立的Python脚本提取parcel IDs
    local python_cmd="python"
    local extract_script="$SCRIPT_DIR/extract_parcel_ids.py"
    
    # 执行Python脚本提取parcel IDs
    local parcel_ids=$($python_cmd "$extract_script" "$analysis_file" 2>/dev/null)
    
    if [[ -z "$parcel_ids" || "$parcel_ids" =~ ^\[.*\]$ ]]; then
        echo "[WARNING] 提取parcel IDs失败，使用默认值" >&2
        echo "233 89 156"
    else
        echo "$parcel_ids"
    fi
}

# 显示配置信息
show_config() {
    local strength="$1"
    local results_dir="$2"
    echo "🧠 幻觉干预批量实验配置"
    echo "============================================================"
    echo "数据集目录: $DATASET_DIR"
    echo "异常分析目录: $ANALYSIS_DIR"
    echo "结果保存目录: $results_dir"
    echo "模型名称: $MODEL_NAME"
    echo "Parcel IDs: 自动从异常分析文件提取"
    echo "当前干预强度: $strength"
    echo "最大样本数: $MAX_SAMPLES"
    echo "启用评估: $ENABLE_EVAL"
    echo "vLLM地址: $VLLM_URL"
    echo "跳过已存在文件: $SKIP_EXISTING"
    echo "============================================================"
}

# 检查依赖
check_dependencies() {
    echo "[INFO] 检查依赖..."
    
    # 使用conda sae环境
    local python_cmd="python"
    
    if [[ ! -f "$python_cmd" ]]; then
        echo "[ERROR] 找不到conda sae环境的Python: $python_cmd"
        exit 1
    fi
    
    $python_cmd -c "import torch, transformers, sae_lens" 2>/dev/null || {
        echo "[ERROR] conda sae环境中缺少必要的Python包"
        exit 1
    }
    
    echo "[SUCCESS] 依赖检查通过 (使用conda sae环境)"
}

# 检查vLLM服务是否可用
check_vllm_service() {
    if [[ "$ENABLE_EVAL" == "yes" ]]; then
        echo "[INFO] 检查vLLM服务是否可用..."
        
        # 提取主机和端口
        local base_url="${VLLM_URL%/v1}"
        
        # 尝试连接vLLM健康检查端点
        if command -v curl &> /dev/null; then
            if curl -s --connect-timeout 2 "${base_url}/health" > /dev/null 2>&1; then
                echo "[SUCCESS] vLLM服务可用: $VLLM_URL"
                return 0
            else
                echo "[WARNING] vLLM服务不可用: $VLLM_URL"
                echo "[INFO] 评估功能将被禁用或使用备用方法"
                return 1
            fi
        else
            echo "[WARNING] 未找到curl命令，跳过vLLM服务检查"
            return 0
        fi
    else
        echo "[INFO] 评估功能已禁用，跳过vLLM服务检查"
        return 0
    fi
}

# 检查数据集目录
check_dataset_dir() {
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "[ERROR] 数据集目录不存在: $DATASET_DIR"
        exit 1
    fi
    
    # 查找CSV文件
    local csv_files=($(find "$DATASET_DIR" -name "*.csv" -type f))
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        echo "[ERROR] 数据集目录中没有找到CSV文件: $DATASET_DIR"
        exit 1
    fi
    
    echo "[INFO] 找到 ${#csv_files[@]} 个CSV文件"
    for file in "${csv_files[@]}"; do
        echo "  - $(basename "$file")"
    done
}

# 检查并创建结果目录
check_results_dir() {
    local results_dir="$1"
    if [[ ! -d "$results_dir" ]]; then
        echo "[INFO] 创建结果目录: $results_dir"
        mkdir -p "$results_dir"
    fi
    
    echo "[INFO] 结果将保存到: $results_dir"
}

# 运行单个数据集的干预实验
run_single_experiment() {
    local dataset="$1"
    local strength="$2"
    local results_dir="$3"
    local dataset_name=$(basename "$dataset" .csv)
    
    echo ""
    echo "🚀 开始处理数据集: $dataset_name (强度: $strength)"
    echo "文件路径: $dataset"
    echo "--------------------------------------------------"
    
    # 检查是否跳过已存在的结果文件
    if [[ "$SKIP_EXISTING" == "yes" ]]; then
        local result_file="$results_dir/${MODEL_NAME}_${dataset_name}_intervention.json"
        if [[ -f "$result_file" ]]; then
            echo "⏭️  跳过已存在的结果文件: $(basename "$result_file")"
            return 0
        fi
    fi
    
    # 为当前数据集提取parcel IDs
    echo "[INFO] 为数据集 $dataset_name 提取parcel IDs..."
    local parcel_ids=$(extract_parcel_ids "$dataset_name")
    
    if [[ -z "$parcel_ids" ]]; then
        echo "[ERROR] 无法为数据集 $dataset_name 提取parcel IDs"
        return 1
    fi
    
    echo "[INFO] 提取到的parcel IDs: $parcel_ids"
    
    local result_file="$results_dir/${MODEL_NAME}_${dataset_name}_intervention.json"
    
    # 构建Python命令（使用conda sae环境，添加-u参数强制无缓冲输出）
    local python_cmd="python -u"
    local cmd="$python_cmd /path/to/project_root/safety_explanation/hallucination/code/intervention/run_intervention.py --mode single --config $CONFIG_FILE"
    cmd="$cmd --dataset $dataset --parcel_ids $parcel_ids"
    cmd="$cmd --intervention_strength $strength --max_samples $MAX_SAMPLES"
    cmd="$cmd --output_file $result_file"
    if [[ "$ENABLE_EVAL" == "yes" ]]; then
        cmd="$cmd --enable_evaluation --vllm_url $VLLM_URL"
    fi
    
    echo "[INFO] 执行命令: $cmd"
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 执行命令
    echo "[INFO] 开始执行Python脚本..."
    set +e
    eval $cmd
    python_exit_code=$?
    
    if [[ $python_exit_code -eq 0 ]]; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "✅ 数据集 $dataset_name 处理完成 (耗时: ${duration}s)"
        
        if [[ -f "$result_file" ]]; then
            echo "[SUCCESS] 结果已保存到: $result_file"
        else
            echo "[WARNING] Python 脚本执行成功，但未找到结果文件: $result_file"
        fi
        return 0
    else
        echo "❌ 数据集 $dataset_name 处理失败 (退出码: $python_exit_code)"
        return 1
    fi
}

# 运行所有数据集的干预实验
run_all_experiments() {
    local strength="$1"
    local results_dir="$2"
    echo ""
    echo "🔄 开始循环处理所有数据集 (强度: $strength)..."
    echo "============================================================"
    
    # 获取所有CSV文件，过滤包含 nq_open、truthfulqa、MedHallu 的数据集
    local csv_files=($(find "$DATASET_DIR" -name "*.csv" -type f 2>/dev/null | grep -E "(truthfulqa|MedHallu|nq_open)" 2>/dev/null | sort || true))
    local total_files=${#csv_files[@]}
    local success_count=0
    local fail_count=0
    
    echo "总共需要处理 $total_files 个数据集（仅包含: nq_open, truthfulqa, MedHallu）"
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        echo "[WARNING] 未找到匹配的数据集文件"
        return 1
    fi
    echo ""
    
    # 循环处理每个数据集
    for i in "${!csv_files[@]}"; do
        dataset="${csv_files[$i]}"
        dataset_name=$(basename "$dataset" .csv)
        selected_strength="$strength"
        if [[ "$strength" == "paper" ]]; then
            selected_strength="$(get_paper_strength "$MODEL_NAME" "$dataset_name")"
        fi
        
        echo "📊 处理进度: $((i+1))/$total_files - $dataset_name (strength: $selected_strength)"
        
        if run_single_experiment "$dataset" "$selected_strength" "$results_dir"; then
            ((success_count++))
        else
            ((fail_count++))
        fi
        
        # 在数据集之间添加分隔线
        if [[ $i -lt $((total_files-1)) ]]; then
            echo ""
            echo "============================================================"
        fi
    done
    
    # 显示最终统计
    echo ""
    echo "🎉 所有数据集处理完成 (强度: $strength)！"
    echo "============================================================"
    echo "总数据集数: $total_files"
    echo "成功处理: $success_count"
    echo "处理失败: $fail_count"
    echo "成功率: $(( success_count * 100 / total_files ))%"
    
    # 显示生成的结果文件
    echo ""
    echo "📁 生成的结果文件:"
    local result_files=($(find "$results_dir" -name "${MODEL_NAME}_*_intervention.json" -type f | sort))
    if [[ ${#result_files[@]} -gt 0 ]]; then
        for file in "${result_files[@]}"; do
            echo "  - $(basename "$file")"
        done
        echo ""
        echo "结果文件保存在: $results_dir"
    else
        echo "  ⚠️ 未找到结果文件"
    fi
}

# 主函数
main() {
    echo "🧠 幻觉干预实验自动运行脚本"
    echo "=========================================="
    echo "强度列表: ${STRENGTH_LIST[@]}"
    echo ""
    
    # 检查必要文件
    if [[ ! -f "$PY_SCRIPT" ]]; then
        echo "[ERROR] 找不到Python脚本: $PY_SCRIPT"
        exit 1
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "[ERROR] 找不到配置文件: $CONFIG_FILE"
        exit 1
    fi
    
    # 检查依赖
    check_dependencies
    
    # 检查vLLM服务
    check_vllm_service
    
    # 检查数据集目录
    check_dataset_dir
    
    # 遍历所有强度值
    for STRENGTH in "${STRENGTH_LIST[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "🔬 开始处理强度: $STRENGTH"
        echo "════════════════════════════════════════════════════════════"
        
        RESULTS_DIR="/path/to/project_root/safety_explanation/hallucination/results/intervention/strength_${STRENGTH}"
        show_config "$STRENGTH" "$RESULTS_DIR"
        check_results_dir "$RESULTS_DIR"
        run_all_experiments "$STRENGTH" "$RESULTS_DIR"
        
        echo ""
        echo "✅ 强度 $STRENGTH 处理完成"
        echo ""
    done
    
    echo "🎉 所有强度值处理完成！"
}

# 运行主函数
main "$@"
