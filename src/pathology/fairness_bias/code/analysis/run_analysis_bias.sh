#!/bin/bash
# 幻觉机制分析运行脚本
# 
# 使用方法:
#   bash run_analysis_overall.sh                    # 运行完整分析（所有模型数据）
#   bash run_analysis_overall.sh check              # 检查环境
#   bash run_analysis_overall.sh parcel             # 运行Parcel级别分析（所有模型数据）
#   bash run_analysis_overall.sh capability         # 运行Capability级别分析（所有模型数据）
#   bash run_analysis_overall.sh llm                # 运行LLM报告生成（所有模型数据）
#
# 分析功能:
#   - 支持批量处理多个模型数据（MedHallu_gemma-2-2b, HaluEval_gemma-2-2b等）
#   - Parcel级别: 包含Parcel功能名称和描述信息，分析异常连接关系
#   - Capability级别: 包含Capability功能描述信息，分析异常连接关系
#   - 支持跳过已存在的结果文件（通过SKIP_EXISTING参数控制）
#   - 自动检测数据文件是否存在，跳过缺失的模型数据
#   - 提供更详细的分析结果用于人工查看
#   - 可视化参数可调节: 修改TOP_K_EDGES变量控制显示的连接数量
#   - 结构性连接mask功能: 使用结构性连接矩阵作为mask优化功能连接计算
#     * 设置USE_STRUCTURAL_MASK=true启用mask功能
#     * 设置STRUCTURAL_MATRIX_PATH指定结构性连接矩阵文件路径
#     * 设置STRUCTURAL_THRESHOLD控制mask阈值
#     * 设置MASK_TYPE选择mask类型（binary或weighted）
#   - PCA连接性计算功能: 拼接所有token后PCA降维计算连接性
#     * 设置USE_PCA_CONNECTIVITY=true启用PCA连接性计算
#     * 设置PCA_EXPLAINED_VARIANCE控制PCA保留的可解释方差比例（默认0.8）
#   - 连接矩阵保存功能: 自动保存correct和incorrect整体连接矩阵
#     * 保存高连接信息（前100个最强连接）
#     * 保存异常连接详细信息（正负异常分别保存）

set -e

# ==================== 配置区域 ====================
# 基础路径
BASE_DIR="/path/to/project_root"

# 模型数据列表 - 支持批量处理多个模型
MODEL_DATA_LIST=(
"bbq_age_gemma-2-2b"
"bbq_disability_status_gemma-2-2b"
"bbq_gender_identity_gemma-2-2b"
"bbq_nationality_gemma-2-2b"
)

# 数据文件路径（全局共享文件）
MAPPING_JSON="$BASE_DIR/neural_area/connect_cap_parcel/results/aggrate_final/final_capability_parcel_all.json"
PARCEL_DESC="$BASE_DIR/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
CAP_DESC="$BASE_DIR/capability_analysis/data/capability_descriptions/capability_descriptions_run2.json"

# 运行模式 (full/parcel/capability/llm)
RUN_MODE="full"

# 可视化参数
TOP_K_EDGES=5000  # 显示前k个最强的连接

# 批量处理控制参数
SKIP_EXISTING=true  # 是否跳过已存在的结果文件

# 结构性连接mask参数
USE_STRUCTURAL_MASK=true  # 是否使用结构性连接矩阵作为mask
STRUCTURAL_MATRIX_PATH="/path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv"
STRUCTURAL_THRESHOLD=0.0  # 结构性连接阈值
MASK_TYPE="binary"  # mask类型: binary 或 weighted

# PCA连接性计算参数
USE_PCA_CONNECTIVITY=true  # 是否使用PCA方法计算连接性（拼接所有token后PCA降维）
PCA_EXPLAINED_VARIANCE=0.8  # PCA保留的可解释方差比例

# Token限制参数
MAX_TOKENS=  # 最大token数量，如果指定则只分析前max_tokens个token（默认空表示使用所有token）

# ==================== 功能函数 ====================

print_info() {
    echo -e "\033[0;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# 检查环境和数据
check_environment() {
    print_info "检查环境..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查全局共享数据文件
    local files=("$MAPPING_JSON" "$PARCEL_DESC" "$CAP_DESC")
    for file in "${files[@]}"; do
        if [ ! -f "$file" ]; then
            print_error "全局数据文件不存在: $file"
            exit 1
        fi
    done
    
    # 检查至少有一个模型数据存在
    local has_model_data=false
    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        local correct_file="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
        local incorrect_file="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        if [ -f "$correct_file" ] && [ -f "$incorrect_file" ]; then
            has_model_data=true
            print_info "找到模型数据: $MODEL_DATA"
            break
        fi
    done
    
    if [ "$has_model_data" = false ]; then
        print_error "未找到任何可用的模型数据文件"
        print_error "请检查以下路径是否存在："
        for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
            print_error "  - $BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
            print_error "  - $BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        done
        exit 1
    fi
    
    print_info "环境检查通过"
}

# 运行完整分析
run_full_analysis() {
    print_info "运行完整幻觉机制分析..."
    
    # 遍历所有模型数据
    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        print_info "正在处理模型数据: $MODEL_DATA"
        
        # 为每个模型数据设置路径
        CORRECT_ACTIVATIONS="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
        INCORRECT_ACTIVATIONS="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        OUTPUT_DIR="$BASE_DIR/safety_explanation/fairness_bias/results/analysis_output/$MODEL_DATA"
        
        # 检查数据文件是否存在
        if [ ! -f "$CORRECT_ACTIVATIONS" ] || [ ! -f "$INCORRECT_ACTIVATIONS" ]; then
            print_error "模型数据文件不存在，跳过: $MODEL_DATA"
            print_error "  - 正确样本文件: $CORRECT_ACTIVATIONS"
            print_error "  - 错误样本文件: $INCORRECT_ACTIVATIONS"
            continue
        fi
        
        mkdir -p "$OUTPUT_DIR"
        
        python3 /path/to/project_root/safety_explanation/hallucination/code/analysis/main.py \
            --correct_jsonl "$CORRECT_ACTIVATIONS" \
            --incorrect_jsonl "$INCORRECT_ACTIVATIONS" \
            --mapping_json "$MAPPING_JSON" \
            --parcel_desc "$PARCEL_DESC" \
            --cap_desc "$CAP_DESC" \
            --output_dir "$OUTPUT_DIR" \
            --verbose
        
        print_info "模型 $MODEL_DATA 分析完成！结果保存在: $OUTPUT_DIR"
    done
    
    print_info "所有模型数据分析完成！"
}

# 运行Parcel级别分析（增强版，包含功能描述和异常连接）
run_parcel_analysis() {
    print_info "运行Parcel级别分析（增强版，包含功能描述和异常连接）..."
    print_info "可视化参数: TOP_K_EDGES=$TOP_K_EDGES"
    
    # 遍历所有模型数据
    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        print_info "正在处理模型数据: $MODEL_DATA"
        
        # 为每个模型数据设置路径
        CORRECT_ACTIVATIONS="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
        INCORRECT_ACTIVATIONS="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        OUTPUT_DIR="$BASE_DIR/safety_explanation/fairness_bias/results/analysis_output/$MODEL_DATA"
        
        # 检查数据文件是否存在
        if [ ! -f "$CORRECT_ACTIVATIONS" ] || [ ! -f "$INCORRECT_ACTIVATIONS" ]; then
            print_error "模型数据文件不存在，跳过: $MODEL_DATA"
            print_error "  - 正确样本文件: $CORRECT_ACTIVATIONS"
            print_error "  - 错误样本文件: $INCORRECT_ACTIVATIONS"
            continue
        fi
        
        # 构建Parcel级别分析命令
        PARCEL_CMD="python3 /path/to/project_root/safety_explanation/hallucination/code/analysis/analysis_parcel_level.py \
            --correct_jsonl \"$CORRECT_ACTIVATIONS\" \
            --incorrect_jsonl \"$INCORRECT_ACTIVATIONS\" \
            --out_dir \"$OUTPUT_DIR/parcel_level\" \
            --parcel_info \"$PARCEL_DESC\" \
            --epsilon 1e-8 \
            --significance_threshold 0.05 \
            --skip_existing \
            --top_k_edges \"$TOP_K_EDGES\""
        
        # 添加结构性连接mask参数
        if [ "$USE_STRUCTURAL_MASK" = true ]; then
            print_info "启用结构性连接mask功能"
            PARCEL_CMD="$PARCEL_CMD --use_structural_mask --structural_matrix_path \"$STRUCTURAL_MATRIX_PATH\" --structural_threshold \"$STRUCTURAL_THRESHOLD\" --mask_type \"$MASK_TYPE\""
        fi
        
        # 添加PCA连接性计算参数
        if [ "$USE_PCA_CONNECTIVITY" = true ]; then
            print_info "启用PCA连接性计算功能"
            PARCEL_CMD="$PARCEL_CMD --use_pca_connectivity --pca_explained_variance \"$PCA_EXPLAINED_VARIANCE\""
        fi
        
        # 添加max_tokens参数
        if [ -n "$MAX_TOKENS" ]; then
            print_info "限制token数量: $MAX_TOKENS"
            PARCEL_CMD="$PARCEL_CMD --max_tokens \"$MAX_TOKENS\""
        fi
        
        # 执行Parcel级别分析
        eval $PARCEL_CMD
        
        print_info "模型 $MODEL_DATA Parcel级别分析完成！结果文件："
        print_info "- parcel_activation_diff.json: 激活异常分析结果（包含Parcel功能描述）"
        print_info "- top_anomalous_parcels.json: 异常Parcel排名（包含功能信息）"
        print_info "- anomalous_connections.json: 异常连接关系（包含Parcel功能描述）"
        print_info "- parcel_connectivity_diff.npy: 连接差异矩阵"
        print_info "- connectivity_matrices/: 连接矩阵和高连接、异常连接信息"
        print_info "  * correct_connectivity_matrix.npy: 正确样本整体连接矩阵"
        print_info "  * incorrect_connectivity_matrix.npy: 幻觉样本整体连接矩阵"
        print_info "  * connectivity_difference_matrix.npy: 连接差异矩阵"
        print_info "  * high_connections.json: 高连接信息（前100个最强连接）"
        print_info "  * anomaly_connections_detailed.json: 异常连接详细信息"
        print_info "- connectivity_visualizations/: 连接关系可视化图（3张HTML图）"
        print_info "  * correct_parcel_connectivity.html: 正确样本连接关系图（交互式）"
        print_info "  * incorrect_parcel_connectivity.html: 幻觉样本连接关系图（交互式）"
        print_info "  * parcel_connectivity_diff.html: 连接差异图（交互式）"
        print_info "- parcel_level_analysis_complete.json: 完整分析结果"
    done
    
    print_info "所有模型数据Parcel级别分析完成！"
}

# 运行Capability级别分析（增强版，包含功能描述和异常连接）
run_capability_analysis() {
    print_info "运行Capability级别分析（增强版，包含功能描述和异常连接）..."
    print_info "可视化参数: TOP_K_EDGES=$TOP_K_EDGES"
    
    # 遍历所有模型数据
    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        print_info "正在处理模型数据: $MODEL_DATA"
        
        # 为每个模型数据设置路径
        CORRECT_ACTIVATIONS="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/correct/token_parcels.jsonl"
        INCORRECT_ACTIVATIONS="$BASE_DIR/safety_explanation/fairness_bias/results/$MODEL_DATA/parcels_token_acts/incorrect/token_parcels.jsonl"
        OUTPUT_DIR="$BASE_DIR/safety_explanation/fairness_bias/results/analysis_output/$MODEL_DATA"
        
        # 检查数据文件是否存在
        if [ ! -f "$CORRECT_ACTIVATIONS" ] || [ ! -f "$INCORRECT_ACTIVATIONS" ]; then
            print_error "模型数据文件不存在，跳过: $MODEL_DATA"
            print_error "  - 正确样本文件: $CORRECT_ACTIVATIONS"
            print_error "  - 错误样本文件: $INCORRECT_ACTIVATIONS"
            continue
        fi
        
        # 构建Capability级别分析命令
        CAPABILITY_CMD="python3 /path/to/project_root/safety_explanation/hallucination/code/analysis/analysis_capability_level.py \
            --mapping_json \"$MAPPING_JSON\" \
            --correct_jsonl \"$CORRECT_ACTIVATIONS\" \
            --incorrect_jsonl \"$INCORRECT_ACTIVATIONS\" \
            --out_dir \"$OUTPUT_DIR/capability_level\" \
            --capability_desc \"$CAP_DESC\" \
            --epsilon 1e-8 \
            --significance_threshold 0.05 \
            --skip_existing \
            --top_k_edges \"$TOP_K_EDGES\""
        
        # 添加结构性连接mask参数
        if [ "$USE_STRUCTURAL_MASK" = true ]; then
            print_info "启用结构性连接mask功能"
            CAPABILITY_CMD="$CAPABILITY_CMD --use_structural_mask --structural_matrix_path \"$STRUCTURAL_MATRIX_PATH\" --structural_threshold \"$STRUCTURAL_THRESHOLD\" --mask_type \"$MASK_TYPE\""
        fi
        
        # 添加PCA连接性计算参数
        if [ "$USE_PCA_CONNECTIVITY" = true ]; then
            print_info "启用PCA连接性计算功能"
            CAPABILITY_CMD="$CAPABILITY_CMD --use_pca_connectivity --pca_explained_variance \"$PCA_EXPLAINED_VARIANCE\""
        fi
        
        # 添加max_tokens参数
        if [ -n "$MAX_TOKENS" ]; then
            print_info "限制token数量: $MAX_TOKENS"
            CAPABILITY_CMD="$CAPABILITY_CMD --max_tokens \"$MAX_TOKENS\""
        fi
        
        # 执行Capability级别分析
        eval $CAPABILITY_CMD
        
        print_info "模型 $MODEL_DATA Capability级别分析完成！结果文件："
        print_info "- capability_activation_diff.json: 激活异常分析结果（包含Capability功能描述）"
        print_info "- top_anomalous_capabilities.json: 异常Capability排名（包含功能信息）"
        print_info "- anomalous_capability_connections.json: 异常Capability连接关系（包含功能描述）"
        print_info "- capability_connectivity_diff.npy: 连接差异矩阵"
        print_info "- connectivity_matrices/: 连接矩阵和高连接、异常连接信息"
        print_info "  * correct_capability_connectivity_matrix.npy: 正确样本整体连接矩阵"
        print_info "  * incorrect_capability_connectivity_matrix.npy: 幻觉样本整体连接矩阵"
        print_info "  * capability_connectivity_difference_matrix.npy: 连接差异矩阵"
        print_info "  * high_capability_connections.json: 高连接信息（前100个最强连接）"
        print_info "  * anomaly_capability_connections_detailed.json: 异常连接详细信息"
        print_info "- connectivity_visualizations/: 连接关系可视化图（3张HTML图）"
        print_info "  * correct_capability_connectivity.html: 正确样本连接关系图（交互式）"
        print_info "  * incorrect_capability_connectivity.html: 幻觉样本连接关系图（交互式）"
        print_info "  * capability_connectivity_diff.html: 连接差异图（交互式）"
        print_info "- capability_level_analysis_complete.json: 完整分析结果"
    done
    
    print_info "所有模型数据Capability级别分析完成！"
}

# 运行LLM报告生成
run_llm_report() {
    print_info "运行LLM报告生成..."
    
    # 遍历所有模型数据
    for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
        print_info "正在为模型数据生成报告: $MODEL_DATA"
        
        # 为每个模型数据设置路径
        OUTPUT_DIR="$BASE_DIR/safety_explanation/fairness_bias/results/analysis_output/$MODEL_DATA"
        PARCEL_DIFF_FILE="$OUTPUT_DIR/parcel_level/top_anomalous_parcels.json"
        CAP_DIFF_FILE="$OUTPUT_DIR/capability_level/top_anomalous_capabilities.json"
        
        # 检查分析结果文件是否存在
        if [ ! -f "$PARCEL_DIFF_FILE" ] || [ ! -f "$CAP_DIFF_FILE" ]; then
            print_error "分析结果文件不存在，跳过: $MODEL_DATA"
            print_error "  - Parcel分析文件: $PARCEL_DIFF_FILE"
            print_error "  - Capability分析文件: $CAP_DIFF_FILE"
            continue
        fi
        
        python3 /path/to/project_root/safety_explanation/hallucination/code/analysis/analysis_llm_summary.py \
            --parcel_desc "$PARCEL_DESC" \
            --cap_desc "$CAP_DESC" \
            --parcel_diff "$PARCEL_DIFF_FILE" \
            --cap_diff "$CAP_DIFF_FILE" \
            --out "$OUTPUT_DIR/nature_style_report.md"
        
        print_info "模型 $MODEL_DATA LLM报告生成完成！报告保存在: $OUTPUT_DIR/nature_style_report.md"
    done
    
    print_info "所有模型数据LLM报告生成完成！"
}

# ==================== 主程序 ====================

main() {
    case "$RUN_MODE" in
        "check")
            check_environment
            ;;
        "parcel")
            check_environment
            run_parcel_analysis
            ;;
        "capability")
            check_environment
            run_capability_analysis
            ;;
        "llm")
            check_environment
            run_llm_report
            ;;
        "full"|*)
            check_environment
            run_full_analysis
            ;;
    esac
}

# 运行主程序
main
