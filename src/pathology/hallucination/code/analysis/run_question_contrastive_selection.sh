#!/bin/bash
# Question Contrastive Selector 运行脚本

set -e

# ==================== 配置区域 ====================
# 基础路径
BASE_DIR="/path/to/project_root"
RESULTS_DIR="$BASE_DIR/safety_explanation/hallucination/results"
ANALYSIS_DIR="$BASE_DIR/safety_explanation/hallucination/code/analysis"

# 模型列表（按需调整）
MODEL_DATA_LIST=(
  "truthfulqa_gemma-2-2b"
  "MedHallu_gemma-2-2b"
  "HaluEval_gemma-2-2b"
  "dolly_close_gemma-2-2b"
  "nq_open_gemma-2-2b"
  "sciq_gemma-2-2b"
  "triviaqa_gemma-2-2b"
)

# 数据文件路径（按模型动态设置）
CORRECT_JSONL=""
INCORRECT_JSONL=""

# 输出目录（按模型动态设置）
OUTPUT_DIR=""
OUTPUT_JSON=""
EMBED_MODEL="/path/to/local_models/Qwen3-Embedding-8B/"

# LLM配置
VLLM_URL="http://0.0.0.0:8001/v1"
API_KEY="abcabc"

# 选择参数（可调整）
MIN_CATEGORIES=5
MAX_CATEGORIES=10
MIN_WORDS=10
MAX_WORDS=100
MIN_CONFIDENCE=0.5
PAIRS_PER_CATEGORY=5
REFINED_PAIRS_TOPK=30

# 运行模式 (check/select)
RUN_MODE="select"

# 是否跳过已存在输出（传给python）
SKIP_EXISTING=0

# 是否跳过已完成模型（shell层，true/false；默认true）
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-true}

# ==================== 工具函数 ====================
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
    local files=("$CORRECT_JSONL" "$INCORRECT_JSONL")
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "数据文件不存在: $file"
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
        print_info "LLM服务不可用，将使用非LLM模式"
        return 1
    fi
}

# 运行选择流程
run_selection() {
    mkdir -p "$OUTPUT_DIR"

    # 模型级跳过：若输出json已存在且文件非空
    if [ "$SKIP_IF_EXISTS" = "true" ] && [ -s "$OUTPUT_JSON" ]; then
        print_info "结果已存在，跳过: $OUTPUT_JSON"
        return 0
    fi

    local common_args=(
        "${ANALYSIS_DIR}/question_contrastive_selector.py"
        --correct_jsonl "$CORRECT_JSONL"
        --incorrect_jsonl "$INCORRECT_JSONL"
        --output "$OUTPUT_JSON"
        --vllm_url "$VLLM_URL"
        --api_key "$API_KEY"
        --min_categories "$MIN_CATEGORIES"
        --max_categories "$MAX_CATEGORIES"
        --min_words "$MIN_WORDS"
        --max_words "$MAX_WORDS"
        --min_confidence "$MIN_CONFIDENCE"
        --pairs_per_category "$PAIRS_PER_CATEGORY"
        --refined_pairs_topk "$REFINED_PAIRS_TOPK"
        --embed_model "$EMBED_MODEL"
    )

    if [ "$SKIP_EXISTING" = "1" ]; then
        common_args+=(--skip_existing)
    fi

    if check_llm_service; then
        print_llm "启用 LLM 分类与复筛"
        python3 "${common_args[@]}" --use_llm
    else
        print_info "使用非LLM回退分类（首词分桶）"
        python3 "${common_args[@]}"
    fi

    print_info "完成！结果保存在: $OUTPUT_JSON"
}

# ==================== 主程序 ====================
main() {
    case "$RUN_MODE" in
        "check")
            # 使用第一个模型进行一次检查
            local first_model="${MODEL_DATA_LIST[0]}"
            print_info "环境检查 - 模型: $first_model"
            CORRECT_JSONL="$RESULTS_DIR/$first_model/correct.jsonl"
            INCORRECT_JSONL="$RESULTS_DIR/$first_model/incorrect.jsonl"
            OUTPUT_DIR="$RESULTS_DIR/analysis_output/$first_model/contrastive_selector"
            OUTPUT_JSON="$OUTPUT_DIR/question_contrastive_pairs.json"
            check_data_files
            check_llm_service || true
            print_info "环境检查完成"
            ;;
        "select"|*)
            for MODEL_NAME in "${MODEL_DATA_LIST[@]}"; do
                print_info "==================== 开始处理模型: $MODEL_NAME ===================="
                CORRECT_JSONL="$RESULTS_DIR/$MODEL_NAME/correct.jsonl"
                INCORRECT_JSONL="$RESULTS_DIR/$MODEL_NAME/incorrect.jsonl"
                OUTPUT_DIR="$RESULTS_DIR/analysis_output/$MODEL_NAME/contrastive_selector"
                OUTPUT_JSON="$OUTPUT_DIR/question_contrastive_pairs.json"
                check_data_files
                run_selection
                print_info "==================== 完成模型: $MODEL_NAME ===================="
            done
            ;;
    esac
}

# 运行主程序
main
