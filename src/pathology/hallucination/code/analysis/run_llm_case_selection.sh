#!/bin/bash
# LLM增强案例选择运行脚本 - 简化版

set -e

# ==================== 配置区域 ====================
# 基础路径
BASE_DIR="/path/to/project_root"
RESULTS_DIR="$BASE_DIR/safety_explanation/hallucination/results"
MODEL_DATA_LIST=(
    "truthfulqa_gemma-2-2b"
    "MedHallu_gemma-2-2b"
    "HaluEval_gemma-2-2b"
    "dolly_close_gemma-2-2b"
    "nq_open_gemma-2-2b"
    "sciq_gemma-2-2b"
    "triviaqa_gemma-2-2b"
)

# LLM配置
VLLM_URL="http://0.0.0.0:8001/v1"
API_KEY="abcabc"

# 选择参数
NUM_CASES=20
MIN_SCORE=7.0

# 结果存在则跳过（true/false），可用环境变量覆盖
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-true}

# 运行模式 (check/select)
RUN_MODE="select"

# ==================== 功能函数 ====================

print_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

print_llm() {
    echo -e "\033[0;35m[LLM]\033[0m $1"
}

# 检查数据文件
check_data_files() {
    print_info "检查数据文件..."
    
    local files=("$CORRECT_JSONL" "$INCORRECT_JSONL" "$PARCEL_ANALYSIS" "$CAPABILITY_ANALYSIS")
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "数据文件不存在: $file"
            print_info "请先运行Parcel和Capability级别分析"
            exit 1
        fi
    done
    
    print_info "数据文件检查通过"
}

# 检查LLM服务
check_llm_service() {
    print_llm "检查LLM服务..."
    
    if curl -s --connect-timeout 5 "$VLLM_URL/health" > /dev/null 2>&1; then
        print_llm "LLM服务运行正常"
        return 0
    else
        print_info "LLM服务不可用，将使用传统方法"
        return 1
    fi
}

# 运行LLM案例选择
run_llm_selection() {
    print_llm "运行LLM案例选择..."
    print_llm "目标案例数: $NUM_CASES, 最低分数: $MIN_SCORE"
    
    local output_dir="$OUTPUT_BASE"
    mkdir -p "$output_dir"
    if [ "$SKIP_IF_EXISTS" = "true" ] && [ -f "$output_dir/selected_cases.json" ]; then
        print_llm "结果已存在，跳过: $output_dir/selected_cases.json"
        return 0
    fi
    
    python3 case_selector.py \
        --correct_jsonl "$CORRECT_JSONL" \
        --incorrect_jsonl "$INCORRECT_JSONL" \
        --parcel_analysis "$PARCEL_ANALYSIS" \
        --capability_analysis "$CAPABILITY_ANALYSIS" \
        --strategy comprehensive \
        --num_cases "$NUM_CASES" \
        --output "$output_dir/selected_cases.json" \
        --vllm_url "$VLLM_URL" \
        --api_key "$API_KEY" \
        --min_score "$MIN_SCORE" \
        --use_llm
    
    print_llm "LLM案例选择完成！结果保存在: $output_dir"
}

# 运行传统案例选择（无LLM）
run_traditional_selection() {
    print_info "运行传统案例选择（无LLM）..."
    print_info "目标案例数: $NUM_CASES"
    
    local output_dir="$OUTPUT_BASE"
    mkdir -p "$output_dir"
    if [ "$SKIP_IF_EXISTS" = "true" ] && [ -f "$output_dir/selected_cases.json" ]; then
        print_info "结果已存在，跳过: $output_dir/selected_cases.json"
        return 0
    fi
    
    python3 case_selector.py \
        --correct_jsonl "$CORRECT_JSONL" \
        --incorrect_jsonl "$INCORRECT_JSONL" \
        --parcel_analysis "$PARCEL_ANALYSIS" \
        --capability_analysis "$CAPABILITY_ANALYSIS" \
        --strategy comprehensive \
        --num_cases "$NUM_CASES" \
        --output "$output_dir/selected_cases.json" \
        --vllm_url "$VLLM_URL" \
        --api_key "$API_KEY"
    
    print_info "传统案例选择完成！结果保存在: $output_dir"
}

# ==================== 主程序 ====================

main() {
    case "$RUN_MODE" in
        "check")
            # 针对列表中的第一个模型做一次环境检查
            local first_model="${MODEL_DATA_LIST[0]}"
            print_info "环境检查 - 模型: $first_model"
            CORRECT_JSONL="$RESULTS_DIR/$first_model/correct.jsonl"
            INCORRECT_JSONL="$RESULTS_DIR/$first_model/incorrect.jsonl"
            PARCEL_ANALYSIS="$RESULTS_DIR/analysis_output/truthfulqa_gemma-2-2b/parcel_level/top_anomalous_parcels.json"
            CAPABILITY_ANALYSIS="$RESULTS_DIR/analysis_output/truthfulqa_gemma-2-2b/capability_level/top_anomalous_capabilities.json"
            OUTPUT_BASE="$RESULTS_DIR/analysis_output/truthfulqa_gemma-2-2b/llm_case_selection/"
            check_data_files
            check_llm_service
            print_info "环境检查完成"
            ;;
        "select"|*)
            for MODEL_NAME in "${MODEL_DATA_LIST[@]}"; do
                print_info "==================== 开始处理模型: $MODEL_NAME ===================="
                # 按模型设置路径
                CORRECT_JSONL="$RESULTS_DIR/$MODEL_NAME/correct.jsonl"
                INCORRECT_JSONL="$RESULTS_DIR/$MODEL_NAME/incorrect.jsonl"
                PARCEL_ANALYSIS="$RESULTS_DIR/analysis_output/$MODEL_NAME/parcel_level/top_anomalous_parcels.json"
                CAPABILITY_ANALYSIS="$RESULTS_DIR/analysis_output/$MODEL_NAME/capability_level/top_anomalous_capabilities.json"
                OUTPUT_BASE="$RESULTS_DIR/analysis_output/$MODEL_NAME/llm_case_selection/"

                check_data_files
                if check_llm_service; then
                    run_llm_selection
                else
                    print_info "LLM服务不可用，使用传统方法"
                    run_traditional_selection
                fi
                print_info "==================== 完成模型: $MODEL_NAME ===================="
            done
            ;;
    esac
}

# 运行主程序
main