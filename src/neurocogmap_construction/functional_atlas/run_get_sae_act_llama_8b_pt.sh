#!/usr/bin/env bash
set -euo pipefail

TARGET_PID=2597836

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

# 基本路径
PYTHON_BIN="python"
SCRIPT_PATH="/path/to/project_root/neural_area/divide_area_by_sae_act/get_sae_act.py"

# 模型与SAE配置（可按需修改）
MODEL_NAME="meta-llama/Llama-3.1-8B"
SAE_RELEASE="llama_scope_lxr_8x"
SAE_LOCAL_BASE_DIR="/path/to/local_models/Llama3_1-8B-Base-LXR-8x"
# 只示例3层，如需更多层可自行补充
SAE_PATHS="l0r_8x,l1r_8x,l2r_8x,l3r_8x,l4r_8x,l5r_8x,l6r_8x,l7r_8x,l8r_8x,l9r_8x,l10r_8x,l11r_8x,l12r_8x,l13r_8x,l14r_8x,l15r_8x,l16r_8x,l17r_8x,l18r_8x,l19r_8x,l20r_8x,l21r_8x,l22r_8x,l23r_8x,l24r_8x,l25r_8x,l26r_8x,l27r_8x,l28r_8x,l29r_8x,l30r_8x,l31r_8x,l31r_8x"

##############test mode neural########################
# 数据与输出配置
DATA_DIR="/path/to/project_root/neural_area/capability_data_v2/test_dataset"
OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_llama_8b_pt_test_neural"

# 处理范围配置
START_IDX=0
END_IDX=-1
LAYERS_PER_BATCH=1  # 每次只处理1层，减少显存使用
LAYER_START=0
LAYER_END=31

# 显存优化配置
# 如果显存不足，可以尝试以下设置：
# 1. 减少每批处理的层数（已设置为1）
# 2. 减少每批处理的样本数（在get_sae_act.py中添加batch_size参数）
# 3. 使用梯度检查点（gradient checkpointing）
# 4. 使用混合精度（fp16）

# 如果只想先列出数据集，打开下面一行
# "$PYTHON_BIN" "$SCRIPT_PATH" --data_dir "$DATA_DIR" --list_datasets --model_name "$MODEL_NAME" --sae_release "$SAE_RELEASE" --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" --sae_paths "$SAE_PATHS"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  --data_dir "$DATA_DIR" \
  --start "$START_IDX" \
  --end "$END_IDX" \
  --layers_per_batch "$LAYERS_PER_BATCH" \
  --layer_start "$LAYER_START" \
  --layer_end "$LAYER_END" \
  --output_base_dir "$OUTPUT_BASE_DIR" \
  --model_name "$MODEL_NAME" \
  --sae_release "$SAE_RELEASE" \
  --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
  --sae_paths "$SAE_PATHS" \
  --n_devices 2  \
  --test_mode \
  --baseline_direct_neural

##############test mode main########################
# 数据与输出配置
DATA_DIR="/path/to/project_root/neural_area/capability_data_v2/test_dataset"
OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_llama_8b_pt_test"


"$PYTHON_BIN" "$SCRIPT_PATH" \
  --data_dir "$DATA_DIR" \
  --start "$START_IDX" \
  --end "$END_IDX" \
  --layers_per_batch "$LAYERS_PER_BATCH" \
  --layer_start "$LAYER_START" \
  --layer_end "$LAYER_END" \
  --output_base_dir "$OUTPUT_BASE_DIR" \
  --model_name "$MODEL_NAME" \
  --sae_release "$SAE_RELEASE" \
  --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
  --sae_paths "$SAE_PATHS" \
  --n_devices 2  \
  --test_mode 

##############main mode########################
# 数据与输出配置
DATA_DIR="/path/to/project_root/neural_area/capability_data_v2"
OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_llama_8b_pt"


"$PYTHON_BIN" "$SCRIPT_PATH" \
  --data_dir "$DATA_DIR" \
  --start "$START_IDX" \
  --end "$END_IDX" \
  --layers_per_batch "$LAYERS_PER_BATCH" \
  --layer_start "$LAYER_START" \
  --layer_end "$LAYER_END" \
  --output_base_dir "$OUTPUT_BASE_DIR" \
  --model_name "$MODEL_NAME" \
  --sae_release "$SAE_RELEASE" \
  --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
  --sae_paths "$SAE_PATHS" \
  --n_devices 2  \

##############main mode baseline direct neural########################
# 数据与输出配置
DATA_DIR="/path/to/project_root/neural_area/capability_data_v2"
OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_llama_8b_pt_neural"


"$PYTHON_BIN" "$SCRIPT_PATH" \
  --data_dir "$DATA_DIR" \
  --start "$START_IDX" \
  --end "$END_IDX" \
  --layers_per_batch "$LAYERS_PER_BATCH" \
  --layer_start "$LAYER_START" \
  --layer_end "$LAYER_END" \
  --output_base_dir "$OUTPUT_BASE_DIR" \
  --model_name "$MODEL_NAME" \
  --sae_release "$SAE_RELEASE" \
  --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
  --sae_paths "$SAE_PATHS" \
  --n_devices 2  \
  --baseline_direct_neural