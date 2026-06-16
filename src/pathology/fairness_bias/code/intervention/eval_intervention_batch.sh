#!/bin/bash

# 批量评测干预结果脚本
# 功能：批量评测不同强度下的干预结果

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_intervention.py"

# 配置参数
INTERVENTION_BASE_DIR="/path/to/project_root/safety_explanation/fairness_bias/results/intervention"
DATASET_DIR="/path/to/project_root/safety_explanation/fairness_bias/dataset"
# 注意：评测结果将保存到对应的干预结果文件所在目录，不再使用统一的 OUTPUT_DIR
VLLM_URL="http://0.0.0.0:8001/v1"
API_KEY="abcabc"
SKIP_EXISTING="yes"

# 强度列表：可以修改为需要评测的强度值
if [[ -n "${INTERVENTION_STRENGTHS:-}" ]]; then
    read -r -a STRENGTH_LIST <<< "$INTERVENTION_STRENGTHS"
else
    STRENGTH_LIST=(paper)
fi

show_config() {
    local strength="$1"
    echo "📊 干预结果批量评测配置"
    echo "------------------------------------------------------------"
    echo "干预结果目录: $INTERVENTION_BASE_DIR"
    echo "数据集目录:   $DATASET_DIR"
    echo "输出方式:     保存到对应的干预结果文件所在目录"
    echo "当前强度:     $strength"
    echo "vLLM 地址:    $VLLM_URL"
    echo "跳过已存在:   $SKIP_EXISTING"
    echo "------------------------------------------------------------"
}

check_dependencies() {
    local python_cmd="python"
    if [[ ! -f "$python_cmd" ]]; then
        echo "[ERROR] 找不到 conda sae 环境: $python_cmd"
        exit 1
    fi
    $python_cmd -c "import requests" 2>/dev/null || {
        echo "[ERROR] Python 环境缺少 requests 库"
        exit 1
    }
}

check_directories() {
    if [[ ! -d "$INTERVENTION_BASE_DIR" ]]; then
        echo "[ERROR] 干预结果目录不存在: $INTERVENTION_BASE_DIR"
        exit 1
    fi
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "[ERROR] 数据集目录不存在: $DATASET_DIR"
        exit 1
    fi
}

run_single_eval() {
    local intervention_json="$1"
    local csv_path="$2"
    local strength="$3"
    
    local dataset_name="$(basename "$csv_path" .csv)"
    local model_name="$(basename "$intervention_json" _intervention.json | sed 's/.*_//')"
    
    # 评测结果保存到干预结果文件所在的目录
    local output_dir="$(dirname "$intervention_json")"
    
    echo ""
    echo "🔍 评测: $intervention_json"
    echo "   数据集: $dataset_name"
    echo "   强度: $strength"
    echo "   输出目录: $output_dir"
    
    local python_cmd="python -u"
    local cmd="$python_cmd $PY_SCRIPT"
    cmd="$cmd --intervention_json $intervention_json"
    cmd="$cmd --csv_path $csv_path"
    cmd="$cmd --output_dir $output_dir"
    cmd="$cmd --vllm_url $VLLM_URL"
    cmd="$cmd --api_key $API_KEY"
    cmd="$cmd --intervention_strength $strength"
    if [[ "$SKIP_EXISTING" == "yes" ]]; then
        cmd="$cmd --skip_existing"
    fi
    
    echo "[INFO] 执行命令: $cmd"
    set +e
    eval $cmd
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        echo "[SUCCESS] 评测完成，结果保存至: $output_dir"
    else
        echo "[ERROR] 评测失败 (code=$exit_code)"
    fi
    set -e
}

run_all_eval_for_strength() {
    local strength="$1"
    local strength_dir="$INTERVENTION_BASE_DIR/strength_$strength"
    
    if [[ ! -d "$strength_dir" ]]; then
        echo "[WARN] 强度目录不存在: $strength_dir，跳过"
        return
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "🔬 开始评测强度: $strength"
    echo "════════════════════════════════════════════════════════════"
    
    # 查找所有干预结果JSON文件
    local json_files=($(find "$strength_dir" -name "*_intervention.json" -type f 2>/dev/null | sort || true))
    
    if [[ ${#json_files[@]} -eq 0 ]]; then
        echo "[WARN] 未找到干预结果文件，跳过"
        return
    fi
    
    for json_file in "${json_files[@]}"; do
        # 从文件名提取数据集名称
        # 例如: gemma-2-2b_bbq_age_intervention.json -> bbq_age
        # 例如: gemma-2-2b_bbq_disability_status_intervention.json -> bbq_disability_status
        local basename_file="$(basename "$json_file" .json)"
        # 先去掉 _intervention 后缀，再提取 bbq_ 到末尾（支持多段如 disability_status、gender_identity）
        local dataset_pattern=$(echo "$basename_file" | sed 's/_intervention$//' | grep -o 'bbq_.*')
        
        if [[ -z "$dataset_pattern" ]]; then
            echo "[WARN] 无法从文件名提取数据集名称: $basename_file，跳过"
            continue
        fi
        
        local csv_path="$DATASET_DIR/${dataset_pattern}.csv"
        if [[ ! -f "$csv_path" ]]; then
            echo "[WARN] 数据集文件不存在: $csv_path，跳过"
            continue
        fi
        
        run_single_eval "$json_file" "$csv_path" "$strength"
    done
    
    echo ""
    echo "✅ 强度 $strength 评测完成"
}

main() {
    echo "📊 干预结果批量评测脚本"
    echo "强度列表: ${STRENGTH_LIST[@]}"
    echo ""
    
    [[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] 找不到 Python 脚本: $PY_SCRIPT"; exit 1; }
    
    check_dependencies
    check_directories
    
    for STRENGTH in "${STRENGTH_LIST[@]}"; do
        show_config "$STRENGTH"
        run_all_eval_for_strength "$STRENGTH"
        echo ""
    done
    
    echo "🎉 所有强度值评测完成！"
    echo "评测结果已保存到对应的干预结果文件所在目录"
}

set -e
main "$@"
