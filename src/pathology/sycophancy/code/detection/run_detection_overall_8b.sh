#!/bin/bash
# 谄媚检测（5折交叉验证）批量运行脚本（Llama-3.1-8B 版本）
#
# 用法示例：
#   bash run_detection_overall_8b.sh          # 批量对 MODEL_DATA_LIST 中的模型进行训练与评估
#   bash run_detection_overall_8b.sh check    # 仅做环境与数据检查
#
# 特性：
# - 批量遍历多个模型数据
# - 自动检查输入文件是否存在
# - 支持跳过已存在结果（--skip_existing）

set -e

# ==================== 配置区域 ====================
# 基础路径
BASE_DIR="/path/to/project_root"

# 模型数据列表（与分析脚本一致，可按需增删，8B 版本）
MODEL_DATA_LIST=(
"answer_Llama-3.1-8B"
"feedback_Llama-3.1-8B"
)

# 能力-Parcel 映射文件（8B 版本）
MAPPING_JSON="$BASE_DIR/neural_area/connect_cap_parcel/results/aggrate_final_8b/final_capability_parcel_all.json"

# 结果输出根目录
DETECT_OUT_DIR="$BASE_DIR/safety_explanation/sycophancy/results/detection"
# 分析输出根目录（用于自动构建指示器配置）
ANALYSIS_OUT_ROOT="$BASE_DIR/safety_explanation/sycophancy/results/analysis_output"

# 交叉验证参数
FOLDS=5
RANDOM_STATE=42

# 检测模型配置
MODEL_TYPE="${MODEL_TYPE:-lr}"
TUNE_HYPERPARAMS="${TUNE_HYPERPARAMS:-false}"
LR_PENALTY="${LR_PENALTY:-l2}"
LR_SOLVER="${LR_SOLVER:-lbfgs}"
LR_C="${LR_C:-1.0}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"

# 批量处理控制
SKIP_EXISTING=false

# ==================== 工具函数 ====================
print_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

check_environment() {
    print_info "检查环境..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi

    if [ ! -f "$MAPPING_JSON" ]; then
        print_error "映射文件不存在: $MAPPING_JSON"
        exit 1
    fi

    # 检查至少有一个模型数据存在
    local has_model_data=false
    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        local correct_file="$BASE_DIR/safety_explanation/sycophancy/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
        local incorrect_file="$BASE_DIR/safety_explanation/sycophancy/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        if [ -f "$correct_file" ] && [ -f "$incorrect_file" ]; then
            has_model_data=true
            print_info "找到模型数据: $MODEL_DATA"
            break
        fi
    done

    if [ "$has_model_data" = false ]; then
        print_error "未找到任何可用的模型数据文件"
        for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
            print_error "  - $BASE_DIR/safety_explanation/sycophancy/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
            print_error "  - $BASE_DIR/safety_explanation/sycophancy/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        done
        exit 1
    fi

    print_info "环境检查通过"
}

run_detection() {
    print_info "开始批量训练幻觉检测器..."

    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        print_info "处理模型: $MODEL_DATA"

        CORRECT_JSONL="$BASE_DIR/safety_explanation/sycophancy/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
        INCORRECT_JSONL="$BASE_DIR/safety_explanation/sycophancy/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        OUT_DIR="$DETECT_OUT_DIR/$MODEL_DATA"
        ANALYSIS_OUTPUT_ROOT="$ANALYSIS_OUT_ROOT/$MODEL_DATA"

        if [ ! -f "$CORRECT_JSONL" ] || [ ! -f "$INCORRECT_JSONL" ]; then
            print_error "数据缺失，跳过: $MODEL_DATA"
            print_error "  - 正确样本: $CORRECT_JSONL"
            print_error "  - 幻觉样本: $INCORRECT_JSONL"
            continue
        fi

        mkdir -p "$OUT_DIR"

        CMD=(python3 "$BASE_DIR/safety_explanation/hallucination/code/detection/train_cv.py" \
            --correct "$CORRECT_JSONL" \
            --incorrect "$INCORRECT_JSONL" \
            --mapping_json "$MAPPING_JSON" \
            --out_dir "$OUT_DIR" \
            --analysis_output_root "$ANALYSIS_OUTPUT_ROOT" \
            --folds $FOLDS \
            --random_state $RANDOM_STATE \
            --model_type "$MODEL_TYPE" \
            --lr_penalty "$LR_PENALTY" \
            --lr_solver "$LR_SOLVER" \
            --lr_C "$LR_C" \
            --class_weight "$CLASS_WEIGHT")

        if [ "$TUNE_HYPERPARAMS" = true ]; then
            CMD+=(--tune_hyperparams)
        fi

        if [ "$SKIP_EXISTING" = true ]; then
            CMD+=(--skip_existing)
        fi

        print_info "运行命令: ${CMD[*]}"
        "${CMD[@]}"

        if [ $? -eq 0 ]; then
            print_info "模型 $MODEL_DATA 训练完成。输出目录: $OUT_DIR"
            print_info "- $OUT_DIR/cv_metrics.json"
            print_info "- $OUT_DIR/sycophancy_detector.joblib"
        else
            print_error "模型 $MODEL_DATA 训练失败，继续下一个。"
        fi
    done

    print_info "批量训练完成。"
}

# ==================== 主程序 ====================
main() {
    case "$1" in
        "check")
            check_environment
            ;;
        "run"|"" )
            check_environment
            run_detection
            ;;
        * )
            echo "用法: bash run_detection_overall_8b.sh [check|run]"
            exit 1
            ;;
    esac
}

main "$@"


