#!/usr/bin/env bash
# 这行命令用于设置bash脚本的执行选项：
# -e: 一旦脚本中的命令返回非零（出错）就立即退出脚本
# -u: 使用未定义的变量会导致脚本报错并退出
# -o pipefail: 只要管道中的任一命令失败，整个管道的返回值就是失败

# 要等待的进程 PID
TARGET_PID=2663761

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


echo "Process $TARGET_PID finished. Starting vLLM server..."


set -euo pipefail

# # 项目与数据路径（绝对路径）
# ROOT="/path/to/local_workspace"
# PROJ="$ROOT/code/Capability_relation_tree/neural_area/divide_area_by_sae_act"
# OUT_ROOT="$PROJ/cluster_output_9b_it"
# DATA_ROOT="$PROJ/qa_sae_output_9b_it"


# cd "$PROJ"

# python "$PROJ/gwMRF_latent_clustering_optimized.py" \
#   --output_root "$DATA_ROOT" \
#   --out_dir "$OUT_ROOT" \
#   --data_level sentence \
#   --auto_n_parcels \
#   --n_parcels_range 10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300 \
#   --n_iter 50 \
#   --spatial_weight 0.01 \
#   --use_preprocessing \
#   --min_activation_rate 0.03 \
#   --gini_keep_quantile 0.8 \
#   --drop_row_low_sum_quantile 10.0 \
#   --use_svd_reduction \
#   --svd_target_variance 0.80 \
#   --svd_use_sparse \
#   --svd_load_cached \
#   --pairwise_weight 1.0 \
  #--svd_load_cached \
  #--svd_cache_dir /path/to/project_root/neural_area/divide_area_by_sae_act/pca_output/

# --------------------------------------random_cluster_baseline--------------------------------------

ROOT="/path/to/local_workspace"
PROJ="$ROOT/code/Capability_relation_tree/neural_area/divide_area_by_sae_act"
OUT_ROOT="$PROJ/cluster_output_9b_it_random_cluster_baseline"
DATA_ROOT="$PROJ/qa_sae_output_9b_it"


cd "$PROJ"

python "$PROJ/gwMRF_latent_clustering_optimized.py" \
  --output_root "$DATA_ROOT" \
  --out_dir "$OUT_ROOT" \
  --data_level sentence \
  --auto_n_parcels \
  --n_parcels_range 270 \
  --n_iter 1 \
  --spatial_weight 0.01 \
  --use_preprocessing \
  --min_activation_rate 0.03 \
  --gini_keep_quantile 0.8 \
  --drop_row_low_sum_quantile 10.0 \
  --use_svd_reduction \
  --svd_target_variance 0.80 \
  --svd_use_sparse \
  --svd_load_cached \
  --pairwise_weight 1.0 \
  --svd_load_cached \
  --svd_cache_dir /path/to/project_root/neural_area/divide_area_by_sae_act/pca_output \
  --random_cluster_baseline



# --------------------------------------neural--------------------------------------

# ROOT="/path/to/local_workspace"
# PROJ="$ROOT/code/Capability_relation_tree/neural_area/divide_area_by_sae_act"
# OUT_ROOT="$PROJ/cluster_output_9b_it_neural"
# DATA_ROOT="$PROJ/qa_sae_output_9b_it_neural"

# python "$PROJ/gwMRF_latent_clustering_optimized.py" \
#   --output_root "$DATA_ROOT" \
#   --out_dir "$OUT_ROOT" \
#   --data_level sentence \
#   --auto_n_parcels \
#   --n_parcels_range 195 \
#   --n_iter 50 \
#   --spatial_weight 0.01 \
#   --use_preprocessing \
#   --min_activation_rate 0.03 \
#   --gini_keep_quantile 0.8 \
#   --drop_row_low_sum_quantile 10.0 \
#   --use_svd_reduction \
#   --svd_target_variance 0.80 \
#   --svd_use_sparse \
#   --svd_load_cached \
#   --pairwise_weight 1.0 \
#   --random_cluster_baseline \
#   --sae_dim 3584 \
#   --svd_load_cached \
#   --svd_cache_dir /path/to/project_root/neural_area/divide_area_by_sae_act/pca_output/



  #--svd_load_cached \
  #--svd_cache_dir /path/to/project_root/neural_area/divide_area_by_sae_act/pca_output/ \
