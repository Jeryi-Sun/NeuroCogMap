#!/bin/bash

# Jailbreak 干预结果批量评测脚本
# 功能：批量评测不同强度下的干预结果

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/eval_intervention.py"

# 配置参数
INTERVENTION_BASE_DIR="/path/to/project_root/safety_explanation/jailbreak/results/intervention"
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
    echo "📊 Jailbreak 干预结果批量评测配置"
    echo "------------------------------------------------------------"
    echo "干预结果目录: $INTERVENTION_BASE_DIR"
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
}

run_single_eval() {
    local intervention_json="$1"
    local strength="$2"
    
    local dataset_name="$(basename "$intervention_json" _intervention.json)"
    # 提取模型名称（例如：gemma-2-2b_JBB-Behaviors -> gemma-2-2b）
    local model_name=$(echo "$dataset_name" | cut -d'_' -f1)
    # 提取数据集名称（例如：gemma-2-2b_JBB-Behaviors -> JBB-Behaviors）
    local dataset_only=$(echo "$dataset_name" | sed "s/^${model_name}_//")
    
    # 评测结果保存到干预结果文件所在的目录，文件名格式：{model_name}_{dataset_name}_baseline_eval.jsonl
    local output_dir="$(dirname "$intervention_json")"
    local output_file="${output_dir}/${dataset_name}_eval.jsonl"
    
    echo ""
    echo "🔍 评测: $(basename "$intervention_json")"
    echo "   数据集: $dataset_only"
    echo "   模型: $model_name"
    echo "   强度: $strength"
    echo "   输出文件: $output_file"
    
    # 注意：即使输出文件存在，也会传递给评测脚本，由评测脚本内部的 --skip_existing 逻辑处理
    # 这样可以支持断点续评，跳过已评测的记录
    
    local python_cmd="python -u"
    local cmd="$python_cmd $PY_SCRIPT"
    cmd="$cmd --intervention_file $intervention_json"
    cmd="$cmd --output_file $output_file"
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
        if [[ -f "$output_file" ]]; then
            local line_count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
            echo "[SUCCESS] 评测完成，结果保存至: $output_file (共 $line_count 条记录)"
        else
            echo "[WARNING] Python 脚本执行成功，但未找到输出文件: $output_file"
        fi
    else
        echo "[ERROR] 评测失败 (code=$exit_code)"
    fi
    set -e
    return $exit_code
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
    
    echo "[INFO] 找到 ${#json_files[@]} 个干预结果文件"
    
    local success=0
    local fail=0
    
    # 使用 success=$((success+1)) 避免 ((success++)) 在 success=0 时退出码为 1 触发 set -e 导致脚本提前退出
    for json_file in "${json_files[@]}"; do
        if run_single_eval "$json_file" "$strength"; then
            success=$((success + 1))
        else
            fail=$((fail + 1))
        fi
    done
    
    echo ""
    echo "✅ 强度 $strength 评测完成 (成功: $success, 失败: $fail)"
}

main() {
    echo "📊 Jailbreak 干预结果批量评测脚本"
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
