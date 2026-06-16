#!/bin/bash

# Sycophancy 干预实验批量运行脚本（Gemma-2-2b）
# 同时处理 answer 与 feedback 两类数据，参考 fairness_bias 的 run_intervention.sh 结构
# 干预输入：原始 gen_eval 文件（非 correct/incorrect 拆分）
# 输出格式：与 fairness_bias 一致，即 ${MODEL_NAME}_${dataset_name}_intervention.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_intervention.py"
CONFIG_FILE="$SCRIPT_DIR/config.json"

# Default release mode uses the paper-selected strength for each dataset.
# Set INTERVENTION_STRENGTHS="0.1 0.3 0.5" to run an optional sensitivity sweep.
if [[ -n "${INTERVENTION_STRENGTHS:-}" ]]; then
    read -r -a STRENGTH_LIST <<< "$INTERVENTION_STRENGTHS"
else
    STRENGTH_LIST=(paper)
fi

# 数据源类型：answer 与 feedback 均做干预
DATA_TYPES=(answer feedback)
RESULTS_ROOT="/path/to/project_root/safety_explanation/sycophancy/results"
# 原始数据目录：
# - answer 使用 answer_with_groups_${MODEL_NAME}_gen_eval.jsonl
# - feedback 使用 feedback_with_groups_${MODEL_NAME}_gen.jsonl（保留中性模板）
DATASET_ROOT="/path/to/project_root/safety_explanation/sycophancy/dataset/sycophancy-eval/results"
ANALYSIS_DIR="$RESULTS_ROOT/analysis_output"
MODEL_NAME="gemma-2-2b"
MAX_SAMPLES="1500"
ENABLE_EVAL="yes"
SKIP_EXISTING="no"


get_paper_strength() {
    local model_name="$1"
    local data_type="$2"
    case "${model_name}|${data_type}" in
        "gemma-2-2b|answer") echo "0.1" ;;
        "gemma-2-2b|feedback") echo "0.3" ;;
        "gemma-2-9b-it|answer") echo "0.5" ;;
        "gemma-2-9b-it|feedback") echo "0.5" ;;
        *) echo "0.3" ;;
    esac
}

extract_parcel_ids() {
    local analysis_key="$1"
    local analysis_file="$ANALYSIS_DIR/${analysis_key}/parcel_level/top_anomalous_parcels.json"
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
    local data_type="$3"
    local input_file="$4"
    echo "🧩 Sycophancy 干预批量实验配置 (2B)"
    echo "------------------------------------------------------------"
    echo "数据源类型:   $data_type"
    echo "输入文件:     $input_file"
    echo "异常分析目录: $ANALYSIS_DIR"
    echo "结果目录:     $results_dir"
    echo "模型名称:     $MODEL_NAME"
    echo "当前强度:     $strength"
    echo "最大样本数:   $MAX_SAMPLES"
    echo "启用评估:     $ENABLE_EVAL"
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

check_input_file() {
    local input_file="$1"
    if [[ ! -f "$input_file" ]]; then
        echo "[WARNING] 原始输入文件不存在，跳过: $input_file"
        return 1
    fi
    echo "[INFO] 找到原始输入文件: $input_file"
    return 0
}

check_results_dir() {
    local results_dir="$1"
    mkdir -p "$results_dir"
}

get_input_file_for_data_type() {
    local data_type="$1"
    # feedback 必须使用 *_gen.jsonl，确保包含中性模板(prompt_template_type="")
    if [[ "$data_type" == "feedback" ]]; then
        echo "${DATASET_ROOT}/${data_type}_with_groups_${MODEL_NAME}_gen.jsonl"
    else
        # answer 继续使用评测过的 gen_eval 文件，保持现有流程不变
        echo "${DATASET_ROOT}/${data_type}_with_groups_${MODEL_NAME}_gen_eval.jsonl"
    fi
}

run_single_experiment() {
    local dataset="$1"
    local strength="$2"
    local results_dir="$3"
    local analysis_key="$4"
    local data_type="$5"
    # 与 fairness_bias 一致：${MODEL_NAME}_${dataset_name}_intervention.json
    local dataset_name="$data_type"

    echo ""
    echo "🚀 处理数据集: [$data_type] (强度: $strength)"

    local result_file="$results_dir/${MODEL_NAME}_${dataset_name}_intervention.json"
    if [[ "$SKIP_EXISTING" == "yes" ]]; then
        if [[ -f "$result_file" ]]; then
            echo "⏭️  已存在结果，跳过"
            return 0
        fi
    fi

    local parcel_ids
    parcel_ids=$(extract_parcel_ids "$analysis_key")
    echo "[INFO] 使用 parcel IDs: $parcel_ids"
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
    local analysis_key="$3"
    local data_type="$4"
    local input_file
    input_file="$(get_input_file_for_data_type "$data_type")"
    if [[ ! -f "$input_file" ]]; then
        echo "[ERROR] 未找到原始输入文件: $input_file"
        return 1
    fi
    local selected_strength="$strength"
    if [[ "$strength" == "paper" ]]; then
        selected_strength="$(get_paper_strength "$MODEL_NAME" "$data_type")"
    fi
    run_single_experiment "$input_file" "$selected_strength" "$results_dir" "$analysis_key" "$data_type"
    echo ""
    echo "🎉 2B Sycophancy [$data_type] 干预完成 (强度: $strength)，结果目录: $results_dir"
    return 0
}

main() {
    echo "🧩 Sycophancy 干预实验自动运行脚本 (2B) — answer + feedback"
    echo "强度列表: ${STRENGTH_LIST[@]}"
    echo "数据源:   ${DATA_TYPES[@]}"
    echo ""

    [[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] 找不到 Python 脚本: $PY_SCRIPT"; exit 1; }
    [[ -f "$CONFIG_FILE" ]] || { echo "[ERROR] 找不到配置文件: $CONFIG_FILE"; exit 1; }
    [[ -d "$RESULTS_ROOT" ]] || { echo "[ERROR] 结果根目录不存在: $RESULTS_ROOT"; exit 1; }

    check_dependencies

    for STRENGTH in "${STRENGTH_LIST[@]}"; do
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "🔬 开始处理强度: $STRENGTH"
        echo "════════════════════════════════════════════════════════════"
        # 与 9B-IT 保持一致：同一强度的不同模型结果放在同一个 strength_$STRENGTH 目录下，
        # 通过文件名中的 MODEL_NAME 区分，避免 strength_0.1 / strength_0.1_2b 这种拆分。
        RESULTS_DIR="$RESULTS_ROOT/intervention/strength_${STRENGTH}"
        check_results_dir "$RESULTS_DIR"

        for DATA_TYPE in "${DATA_TYPES[@]}"; do
            INPUT_FILE="$(get_input_file_for_data_type "$DATA_TYPE")"
            ANALYSIS_KEY="${DATA_TYPE}_${MODEL_NAME}"
            show_config "$STRENGTH" "$RESULTS_DIR" "$DATA_TYPE" "$INPUT_FILE"
            check_input_file "$INPUT_FILE" || continue
            run_all_experiments "$STRENGTH" "$RESULTS_DIR" "$ANALYSIS_KEY" "$DATA_TYPE" || true
        done
        echo ""
        echo "✅ 强度 $STRENGTH 处理完成"
        echo ""
    done

    echo "🎉 所有强度值处理完成！"
}

set -e
main "$@"

