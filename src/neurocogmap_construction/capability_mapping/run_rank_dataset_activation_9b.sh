#!/bin/bash

# Capability-Parcel Connection Analysis Runner
# 能力-分区连接分析运行脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo "Capability-Parcel Connection Analysis Runner"
    echo "=========================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -d, --data-level        数据级别 (sentence|example|dataset) [默认: sentence]"
    echo "  -n, --normalize         归一化方法 (l2|zscore|none) [默认: l2]"
    echo "  -k, --top-k            数据驱动分析中的top_k参数 [默认: 10]"
    echo "  -o, --output           结果输出目录 [默认: ./results]"
    echo "  --capability-data      能力数据集统计文件路径"
    echo "  --parcel-assignments   parcel分配结果文件路径"
    echo "  --sae-output           SAE激活数据目录路径"
    echo ""
    echo "示例:"
    echo "  $0 -d sentence -n l2 -k 15"
    echo "  $0 --data-level example --normalize zscore --top-k 20"
    echo ""
}

# 默认参数
DATA_LEVEL="sentence"
NORMALIZE_METHOD="l2"
TOP_K=50
RESULTS_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/rank_activation_9b"
CAPABILITY_DATA="/path/to/project_root/neural_area/capability_data_v2/data_stastic/final_merged_capability_dataset_stats.json"
PARCEL_ASSIGNMENTS="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_9b_it/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json"
SAE_OUTPUT="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_9b_it"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--data-level)
            DATA_LEVEL="$2"
            shift 2
            ;;
        -n|--normalize)
            NORMALIZE_METHOD="$2"
            shift 2
            ;;
        -k|--top-k)
            TOP_K="$2"
            shift 2
            ;;
        -o|--output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --capability-data)
            CAPABILITY_DATA="$2"
            shift 2
            ;;
        --parcel-assignments)
            PARCEL_ASSIGNMENTS="$2"
            shift 2
            ;;
        --sae-output)
            SAE_OUTPUT="$2"
            shift 2
            ;;
        *)
            print_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 验证参数
validate_params() {
    print_info "验证参数..."
    
    # 检查数据级别
    if [[ ! "$DATA_LEVEL" =~ ^(sentence|example|dataset)$ ]]; then
        print_error "无效的数据级别: $DATA_LEVEL"
        print_error "有效选项: sentence, example, dataset"
        exit 1
    fi
    
    # 检查归一化方法
    if [[ ! "$NORMALIZE_METHOD" =~ ^(l2|zscore|none)$ ]]; then
        print_error "无效的归一化方法: $NORMALIZE_METHOD"
        print_error "有效选项: l2, zscore, none"
        exit 1
    fi
    
    # 检查top_k
    if ! [[ "$TOP_K" =~ ^[0-9]+$ ]] || [ "$TOP_K" -lt 1 ]; then
        print_error "无效的top_k值: $TOP_K"
        print_error "必须是正整数"
        exit 1
    fi
    
    # 检查输入文件是否存在
    if [[ ! -f "$CAPABILITY_DATA" ]]; then
        print_error "能力数据集文件不存在: $CAPABILITY_DATA"
        exit 1
    fi
    
    if [[ ! -f "$PARCEL_ASSIGNMENTS" ]]; then
        print_error "Parcel分配文件不存在: $PARCEL_ASSIGNMENTS"
        exit 1
    fi
    
    if [[ ! -d "$SAE_OUTPUT" ]]; then
        print_error "SAE输出目录不存在: $SAE_OUTPUT"
        exit 1
    fi
    
    print_success "参数验证通过"
}

# 检查Python环境
check_python_env() {
    print_info "检查Python环境..."
    
    # 检查Python版本
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_info "Python版本: $PYTHON_VERSION"
    
    # 检查必要的Python包
    REQUIRED_PACKAGES=("numpy" "pandas" "scipy" "matplotlib" "seaborn" "sklearn" "tqdm")
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import $package" 2>/dev/null; then
            print_error "缺少Python包: $package"
            print_info "请运行: pip3 install $package"
            exit 1
        fi
    done
    
    print_success "Python环境检查通过"
}

# 创建输出目录
create_output_dir() {
    print_info "创建输出目录..."
    
    if [[ ! -d "$RESULTS_DIR" ]]; then
        mkdir -p "$RESULTS_DIR"
        print_info "创建目录: $RESULTS_DIR"
    else
        print_info "输出目录已存在: $RESULTS_DIR"
    fi
}

# 运行分析
run_analysis() {
    print_info "开始运行能力-分区连接分析..."
    print_info "参数配置:"
    print_info "  数据级别: $DATA_LEVEL"
    print_info "  归一化方法: $NORMALIZE_METHOD"
    print_info "  Top-K: $TOP_K"
    print_info "  输出目录: $RESULTS_DIR"
    
    # 构建Python命令
    PYTHON_SCRIPT="rank_dataset_activation.py"
    
    if [[ ! -f "$PYTHON_SCRIPT" ]]; then
        print_error "Python脚本不存在: $PYTHON_SCRIPT"
        exit 1
    fi
    
    # 运行Python分析脚本
    python3 "$PYTHON_SCRIPT" \
        --capability_data "$CAPABILITY_DATA" \
        --parcel_assignments "$PARCEL_ASSIGNMENTS" \
        --sae_output "$SAE_OUTPUT" \
        --results_dir "$RESULTS_DIR" \
        --data_level "$DATA_LEVEL" \
        --normalize_method "$NORMALIZE_METHOD" \
        --top_k "$TOP_K" \
        --normalize_method "positive"
    
    if [[ $? -eq 0 ]]; then
        print_success "分析完成！"
    else
        print_error "分析失败！"
        exit 1
    fi
}

# 显示结果摘要
show_results_summary() {
    print_info "分析结果摘要..."
    
    if [[ -d "$RESULTS_DIR" ]]; then
        echo ""
        echo "生成的文件:"
        ls -la "$RESULTS_DIR" | grep -E "\.(json|csv|png)$" | while read line; do
            echo "  $line"
        done
        
        # 检查关键结果文件
        KEY_FILES=(
            "analysis_summary.json"
            "parcel_activation_rankings.json"
            "data_driven_results.json"
            "knowledge_driven_results.json"
        )
        
        echo ""
        echo "关键结果文件状态:"
        for file in "${KEY_FILES[@]}"; do
            if [[ -f "$RESULTS_DIR/$file" ]]; then
                print_success "✓ $file"
            else
                print_warning "✗ $file (未找到)"
            fi
        done
    else
        print_error "结果目录不存在: $RESULTS_DIR"
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "Capability-Parcel Connection Analysis"
    echo "能力-分区连接分析系统"
    echo "=========================================="
    echo ""
    
    # 执行各个步骤
    validate_params
    check_python_env
    create_output_dir
    run_analysis
    show_results_summary
    
    echo ""
    echo "=========================================="
    print_success "所有任务完成！"
    echo "=========================================="
}

# 运行主函数
main "$@" 