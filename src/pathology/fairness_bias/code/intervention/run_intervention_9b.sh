#!/bin/bash

# Fairness/Bias 干预批量运行脚本（Gemma-2-9b-it）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_intervention.py"
CONFIG_FILE="$SCRIPT_DIR/config_9b.json"
# Default release mode uses the paper-selected strength for each dataset.
# Set INTERVENTION_STRENGTHS="0.1 0.3 0.5" to run an optional sensitivity sweep.
if [[ -n "${INTERVENTION_STRENGTHS:-}" ]]; then
    read -r -a STRENGTH_LIST <<< "$INTERVENTION_STRENGTHS"
else
    STRENGTH_LIST=(paper)
fi

DATASET_DIR="/path/to/project_root/safety_explanation/fairness_bias/dataset"
ANALYSIS_DIR="/path/to/project_root/safety_explanation/fairness_bias/results/analysis_output"
MODEL_NAME="gemma-2-9b-it"
MAX_SAMPLES="1500"
ENABLE_EVAL="no"
VLLM_URL="http://0.0.0.0:8001/v1"
SKIP_EXISTING="no"


get_paper_strength() {
    local model_name="$1"
    local dataset_name="$2"
    case "${model_name}|${dataset_name}" in
        "gemma-2-2b|bbq_age") echo "0.5" ;;
        "gemma-2-2b|bbq_disability_status") echo "0.3" ;;
        "gemma-2-2b|bbq_gender_identity") echo "0.3" ;;
        "gemma-2-2b|bbq_nationality") echo "0.1" ;;
        "gemma-2-9b-it|bbq_age") echo "0.1" ;;
        "gemma-2-9b-it|bbq_disability_status") echo "0.3" ;;
        "gemma-2-9b-it|bbq_gender_identity") echo "0.3" ;;
        "gemma-2-9b-it|bbq_nationality") echo "0.5" ;;
        *) echo "0.3" ;;
    esac
}

extract_parcel_ids() {
    local dataset_name="$1"
    local analysis_file="$ANALYSIS_DIR/${dataset_name}_${MODEL_NAME}/parcel_level/top_anomalous_parcels.json"
    if [[ ! -f "$analysis_file" ]]; then
        echo "[WARNING] 未找到异常分析文件: $analysis_file" >&2
        echo "233 89 156"
        return
    fi
    local python_cmd="python"
    local extract_script="$SCRIPT_DIR/extract_parcel_ids.py"
    local parcel_ids=$($python_cmd "$extract_script" "$analysis_file" 2>/dev/null)
    if [[ -z "$parcel_ids" ]]; then
        echo "233 89 156"
    else
        echo "$parcel_ids"
    fi
}

show_config() {
    local strength="$1"
    local results_dir="$2"
    echo "🛡️ Fairness/Bias 干预批量实验配置 (9B-IT)"
    echo "------------------------------------------------------------"
    echo "数据集目录: $DATASET_DIR"
    echo "异常分析目录: $ANALYSIS_DIR"
    echo "结果目录:     $results_dir"
    echo "模型名称:     $MODEL_NAME"
    echo "当前强度:     $strength"
    echo "最大样本数:   $MAX_SAMPLES"
    echo "启用评估:     $ENABLE_EVAL"
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
    $python_cmd -c "import torch, transformers, sae_lens" 2>/dev/null || {
        echo "[ERROR] sae 环境缺少必要依赖"
        exit 1
    }
}

check_dataset_dir() {
    if [[ ! -d "$DATASET_DIR" ]]; then
        echo "[ERROR] 数据集目录不存在: $DATASET_DIR"
        exit 1
    fi
    local csv_files=($(find "$DATASET_DIR" -name "*.csv" -type f))
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        echo "[ERROR] 数据集目录下未找到 CSV 文件"
        exit 1
    fi
}

check_results_dir() {
    local results_dir="$1"
    mkdir -p "$results_dir"
}

run_single_experiment() {
    local dataset="$1"
    local strength="$2"
    local results_dir="$3"
    local dataset_name="$(basename "$dataset" .csv)"

    echo ""
    echo "🚀 处理数据集: $dataset_name (强度: $strength)"

    if [[ "$SKIP_EXISTING" == "yes" ]]; then
        local result_file="$results_dir/${MODEL_NAME}_${dataset_name}_intervention.json"
        if [[ -f "$result_file" ]]; then
            echo "⏭️  已存在结果，跳过"
            return 0
        fi
    fi

    local parcel_ids=$(extract_parcel_ids "$dataset_name")
    echo "[INFO] 使用 parcel IDs: $parcel_ids"

    local result_file="$results_dir/${MODEL_NAME}_${dataset_name}_intervention.json"
    local python_cmd="python -u"
    local cmd="$python_cmd $PY_SCRIPT --mode single --config $CONFIG_FILE"
    cmd="$cmd --dataset $dataset --parcel_ids $parcel_ids"
    cmd="$cmd --intervention_strength $strength --max_samples $MAX_SAMPLES"
    cmd="$cmd --output_file $result_file"
    if [[ "$ENABLE_EVAL" == "yes" ]]; then
        cmd="$cmd --enable_evaluation"
    fi

    echo "[INFO] 执行命令: $cmd"
    set +e
    eval $cmd
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        if [[ -f "$result_file" ]]; then
            echo "[SUCCESS] 结果保存至: $result_file"
        else
            echo "[WARNING] Python 脚本执行成功，但未找到结果文件: $result_file"
        fi
    else
        echo "[ERROR] 干预执行失败 (code=$exit_code)"
    fi
}

run_all_experiments() {
    local strength="$1"
    local results_dir="$2"
    local csv_files=($(find "$DATASET_DIR" -name "*.csv" -type f 2>/dev/null | grep -E "(bbq_)" 2>/dev/null | sort || true))
    if [[ ${#csv_files[@]} -eq 0 ]]; then
        echo "[ERROR] 未找到符合条件的数据集"
        exit 1
    fi
    for dataset in "${csv_files[@]}"; do
        local dataset_name="$(basename "$dataset" .csv)"
        local selected_strength="$strength"
        if [[ "$strength" == "paper" ]]; then
            selected_strength="$(get_paper_strength "$MODEL_NAME" "$dataset_name")"
        fi
        run_single_experiment "$dataset" "$selected_strength" "$results_dir"
    done
    echo "\n🎉 9B-IT 干预完成 (强度: $strength)，结果目录: $results_dir"
}

main() {
    echo "🛡️ Fairness/Bias 干预批量运行脚本 (9B-IT)"
    echo "强度列表: ${STRENGTH_LIST[@]}"
    echo ""

    [[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] 找不到 Python 脚本"; exit 1; }
    [[ -f "$CONFIG_FILE" ]] || { echo "[ERROR] 找不到配置文件"; exit 1; }

    check_dependencies
    check_dataset_dir

    for STRENGTH in "${STRENGTH_LIST[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "🔬 开始处理强度: $STRENGTH"
        echo "════════════════════════════════════════════════════════════"

        RESULTS_DIR="/path/to/project_root/safety_explanation/fairness_bias/results/intervention/strength_${STRENGTH}"
        show_config "$STRENGTH" "$RESULTS_DIR"
        check_results_dir "$RESULTS_DIR"
        run_all_experiments "$STRENGTH" "$RESULTS_DIR"

        echo ""
        echo "✅ 强度 $STRENGTH 处理完成"
        echo ""
    done

    echo "🎉 所有强度值处理完成！"
}

set -e
main "$@"


