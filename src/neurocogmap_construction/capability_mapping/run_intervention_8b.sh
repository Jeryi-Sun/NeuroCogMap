#!/bin/bash

# Parcel Intervention System 运行脚本（Llama-3.1-8B Base + LXR-8x）
# 用于执行第三步的因果干预实验（8B 版本）

TARGET_PID=${TARGET_PID:-0}

# 改进的进程检测函数
check_process_exists() {
    local pid=$1
    
    # 方法1: 使用 kill -0 (需要权限)
    if kill -0 $pid 2>/dev/null; then
        return 0
    fi
    
    # 方法2: 使用 ps 命令检查
    if ps -p $pid >/dev/null 2>&1; then
        return 0
    fi
    
    # 方法3: 使用 pgrep 命令检查
    if pgrep -f "^$pid$" >/dev/null 2>&1; then
        return 0
    fi
    
    # 方法4: 检查 /proc 文件系统
    if [[ -d "/proc/$pid" ]]; then
        return 0
    fi
    
    return 1
}

# 获取进程信息
get_process_info() {
    local pid=$1
    
    # 尝试获取进程详细信息
    if ps -p $pid >/dev/null 2>&1; then
        local process_info=$(ps -p $pid -o pid,ppid,user,comm,etime --no-headers 2>/dev/null)
        if [[ -n "$process_info" ]]; then
            echo "进程信息: $process_info"
            return
        fi
    fi
    
    # 尝试从 /proc 获取信息
    if [[ -d "/proc/$pid" ]]; then
        local cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ')
        local status=$(cat "/proc/$pid/status" 2>/dev/null | grep "State:" | cut -f2)
        echo "命令行: $cmdline"
        echo "状态: $status"
        return
    fi
    
    echo "无法获取进程详细信息"
}

# 检查进程是否还存在，存在就等待
if [ "$TARGET_PID" -ne 0 ]; then
    echo "开始检测进程 $TARGET_PID..."
    
    # 首先检查进程是否存在
    if ! check_process_exists $TARGET_PID; then
        echo "进程 $TARGET_PID 不存在，直接开始执行"
    else
        echo "进程 $TARGET_PID 正在运行，等待其结束..."
        get_process_info $TARGET_PID
        
        # 等待进程结束
        while check_process_exists $TARGET_PID; do
            echo "Process $TARGET_PID is still running..."
            sleep 5
        done
        
        echo "Process $TARGET_PID is no longer running"
    fi
fi

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

# 检查必要的文件是否存在
check_dependencies() {
    print_info "检查依赖文件..."
    
    # 检查数据驱动结果文件
    if [ ! -f "$DATA_DRIVEN_RESULTS" ]; then
        print_error "数据驱动结果文件不存在: $DATA_DRIVEN_RESULTS"
        exit 1
    fi
    
    # 检查latent-parcel分配文件
    if [ ! -f "$LATENT_PARCEL_ASSIGNMENTS" ]; then
        print_error "Latent-parcel分配文件不存在: $LATENT_PARCEL_ASSIGNMENTS"
        exit 1
    fi
    
    # 检查最大激活目录
    if [ ! -d "$MAX_ACTIVATION_DIR" ]; then
        print_error "最大激活目录不存在: $MAX_ACTIVATION_DIR"
        exit 1
    fi
    
    # 检查合并后的最大激活文件（8B 专用）
    if [ ! -f "$MAX_ACTIVATION_DIR/merged_max_activations_8b.pkl" ]; then
        print_error "最大激活目录中没有找到合并后的激活文件: merged_max_activations_8b.pkl"
        exit 1
    fi
    
    print_success "所有依赖文件检查通过"
}

# 创建输出目录
create_output_dirs() {
    print_info "创建输出目录..."
    
    mkdir -p "$RESULTS_DIR"
    mkdir -p "$LOG_DIR"
    
    print_success "输出目录创建完成"
}

# 运行单个数据集的干预实验
run_single_dataset() {
    local dataset_name=$1
    local top_k=$2
    
    print_info "开始处理数据集: $dataset_name (top_k=$top_k)"
    
    local log_file="$LOG_DIR/${dataset_name}_intervention_8b.log"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    echo "=== 数据集 $dataset_name 干预实验开始 ($timestamp) ===" > "$log_file"
    
    python "$SCRIPT_DIR/parcel_intervention.py" \
        --data_driven_results "$DATA_DRIVEN_RESULTS" \
        --latent_parcel_assignments "$LATENT_PARCEL_ASSIGNMENTS" \
        --max_activation_dir "$MAX_ACTIVATION_DIR" \
        --results_dir "$RESULTS_DIR" \
        --model_name "$MODEL_NAME" \
        --sae_release "$SAE_RELEASE" \
        --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
        --sae_paths "$SAE_PATHS" \
        --top_k "$top_k" \
        --dataset "$dataset_name" \
        --use_optimized \
        --batch_size 1 \
        --sample_limit 128 \
        --use_file_lock \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_success "数据集 $dataset_name 处理完成"
        echo "=== 数据集 $dataset_name 干预实验完成 ($(date '+%Y%m%d_%H%M%S')) ===" >> "$log_file"
    else
        print_error "数据集 $dataset_name 处理失败，退出码: $exit_code"
        echo "=== 数据集 $dataset_name 干预实验失败 ($(date '+%Y%m%d_%H%M%S')) ===" >> "$log_file"
        return $exit_code
    fi
}

# 运行所有数据集的干预实验
run_all_datasets() {
    local top_k=$1
    
    print_info "开始处理所有数据集 (top_k=$top_k)"
    
    local log_file="$LOG_DIR/all_datasets_intervention_8b.log"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    
    echo "=== 所有数据集干预实验开始 ($timestamp) ===" > "$log_file"
    
    python "$SCRIPT_DIR/parcel_intervention.py" \
        --data_driven_results "$DATA_DRIVEN_RESULTS" \
        --latent_parcel_assignments "$LATENT_PARCEL_ASSIGNMENTS" \
        --max_activation_dir "$MAX_ACTIVATION_DIR" \
        --results_dir "$RESULTS_DIR" \
        --model_name "$MODEL_NAME" \
        --sae_release "$SAE_RELEASE" \
        --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
        --sae_paths "$SAE_PATHS" \
        --top_k "$top_k" \
        --use_optimized \
        --batch_size 1 \
        --sample_limit 128 \
        --use_file_lock \
        2>&1 | tee -a "$log_file"
    
    local exit_code=${PIPESTATUS[0]}
    
    if [ $exit_code -eq 0 ]; then
        print_success "所有数据集处理完成"
        echo "=== 所有数据集干预实验完成 ($(date '+%Y%m%d_%H%M%S')) ===" >> "$log_file"
    else
        print_error "所有数据集处理失败，退出码: $exit_code"
        echo "=== 所有数据集干预实验失败 ($(date '+%Y%m%d_%H%M%S')) ===" >> "$log_file"
        return $exit_code
    fi
}

# 显示帮助信息
show_help() {
    echo "Parcel Intervention System 运行脚本（Llama-3.1-8B Base + LXR-8x）"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -d, --dataset DATASET   指定单个数据集进行干预"
    echo "  -k, --top-k NUM         指定每个数据集取前k个parcel (默认: 5)"
    echo "  -a, --all               处理所有数据集"
    echo "  -c, --check             只检查依赖文件，不运行实验"
    echo "  -l, --list              列出可用的数据集"
    echo ""
    echo "示例:"
    echo "  $0 --dataset piqa --top-k 10    # 处理 piqa 数据集，取前 10 个 parcel"
    echo "  $0 --all --top-k 5              # 处理所有数据集，每个取前 5 个 parcel"
    echo "  $0 --check                      # 只检查依赖文件"
    echo ""
}

# 列出可用的数据集
list_datasets() {
    print_info "可用的数据集:"
    
    if [ -f "$DATA_DRIVEN_RESULTS" ]; then
        python -c "
import json
with open('$DATA_DRIVEN_RESULTS', 'r') as f:
    data = json.load(f)
datasets = list(data.get('top_parcels_by_dataset', {}).keys())
for i, dataset in enumerate(sorted(datasets)):
    print(f'  {i+1:2d}: {dataset}')
print(f'\\n共 {len(datasets)} 个数据集')
"
    else
        print_error "无法读取数据驱动结果文件"
        return 1
    fi
}

# 主函数
main() {
    # 默认参数
    DATASET=""
    TOP_K=5
    RUN_ALL=false
    CHECK_ONLY=false
    LIST_DATASETS=false
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dataset)
                DATASET="$2"
                shift 2
                ;;
            -k|--top-k)
                TOP_K="$2"
                shift 2
                ;;
            -a|--all)
                RUN_ALL=true
                shift
                ;;
            -c|--check)
                CHECK_ONLY=true
                shift
                ;;
            -l|--list)
                LIST_DATASETS=true
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置路径
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    # 数据驱动结果与 8B parcel 聚类结果路径
    # 目前复用通用的 rank_activation 结果，如需区分 8B 可自行改为 rank_activation_8b
    DATA_DRIVEN_RESULTS="$PROJECT_ROOT/results/rank_activation_8b/parcel_activation_rankings.json"
    LATENT_PARCEL_ASSIGNMENTS="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_llama_8b_pt/clustering_results_sentence_prep0.01_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels240/latent_parcel_assignments.json"
    MAX_ACTIVATION_DIR="$PROJECT_ROOT/results/steer_activation"
    RESULTS_DIR="$PROJECT_ROOT/results/intervention/strength_-1.0_8b"
    LOG_DIR="$PROJECT_ROOT/results/intervention/logs"
    
    # Llama-3.1-8B 模型与 SAE 配置
    MODEL_NAME="meta-llama/Llama-3.1-8B"
    SAE_RELEASE="llama_scope_lxr_8x"
    SAE_LOCAL_BASE_DIR="/path/to/local_models/Llama3_1-8B-Base-LXR-8x"
    
    # 32 层 LXR-8x SAE 路径（与 run_extract_max_activation_8b.sh 一致）
    SAE_PATHS="l0r_8x,l1r_8x,l2r_8x,l3r_8x,l4r_8x,l5r_8x,l6r_8x,l7r_8x,l8r_8x,l9r_8x,l10r_8x,l11r_8x,l12r_8x,l13r_8x,l14r_8x,l15r_8x,l16r_8x,l17r_8x,l18r_8x,l19r_8x,l20r_8x,l21r_8x,l22r_8x,l23r_8x,l24r_8x,l25r_8x,l26r_8x,l27r_8x,l28r_8x,l29r_8x,l30r_8x,l31r_8x"
    
    print_info "Parcel Intervention System 启动（Llama-3.1-8B Base + LXR-8x）"
    print_info "项目根目录: $PROJECT_ROOT"
    print_info "结果目录: $RESULTS_DIR"
    print_info "日志目录: $LOG_DIR"
    
    # 检查依赖
    check_dependencies
    
    if [ "$CHECK_ONLY" = true ]; then
        print_success "依赖检查完成，退出"
        exit 0
    fi
    
    if [ "$LIST_DATASETS" = true ]; then
        list_datasets
        exit 0
    fi
    
    # 创建输出目录
    create_output_dirs
    
    # 运行干预实验
    if [ "$RUN_ALL" = true ]; then
        print_info "运行所有数据集的干预实验"
        run_all_datasets "$TOP_K"
    elif [ -n "$DATASET" ]; then
        print_info "运行指定数据集的干预实验: $DATASET"
        run_single_dataset "$DATASET" "$TOP_K"
    else
        print_error "请指定 --dataset 或 --all 参数"
        show_help
        exit 1
    fi
    
    print_success "干预实验完成！"
    print_info "结果保存在: $RESULTS_DIR"
    print_info "日志保存在: $LOG_DIR"
}

# 运行主函数
main "$@"

