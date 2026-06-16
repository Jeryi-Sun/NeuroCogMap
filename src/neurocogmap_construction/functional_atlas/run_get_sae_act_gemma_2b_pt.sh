#!/usr/bin/env bash
set -euo pipefail


# 基本路径
PYTHON_BIN="python"
SCRIPT_PATH="/path/to/project_root/neural_area/divide_area_by_sae_act/get_sae_act.py"

# 模型与SAE配置（保持原始2B-PT）
MODEL_NAME="google/gemma-2-2b"
SAE_RELEASE="gemma-scope-2b-pt-res"
SAE_LOCAL_BASE_DIR="/path/to/local_models/gemma-scope-2b-pt-res"

# 数据与输出配置
DATA_DIR="/path/to/project_root/neural_area/capability_data_v2/"
OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_2b_pt"

# 处理范围配置（参考原始默认）
START_IDX=0
END_IDX=-1
LAYERS_PER_BATCH=8
LAYER_START=0
LAYER_END=25

# 如果只想先列出数据集，打开下面一行（会读取默认SAE层列表）
# "$PYTHON_BIN" "$SCRIPT_PATH" --data_dir "$DATA_DIR" --list_datasets --model_name "$MODEL_NAME" --sae_release "$SAE_RELEASE" --sae_local_base_dir "$SAE_LOCAL_BASE_DIR"

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
  --n_devices 1


# Test mode
# TARGET_PID=3889621

# # 检查进程是否还存在，存在就等待
# while kill -0 $TARGET_PID 2>/dev/null; do
#     echo "Process $TARGET_PID is still running..."
#     sleep 5
# done

# # 基本路径
PYTHON_BIN="python"
SCRIPT_PATH="/path/to/project_root/neural_area/divide_area_by_sae_act/get_sae_act_test.py"

# 模型与SAE配置（保持原始2B-PT）
MODEL_NAME="google/gemma-2-2b"
SAE_RELEASE="gemma-scope-2b-pt-res"
SAE_LOCAL_BASE_DIR="/path/to/local_models/gemma-scope-2b-pt-res"

# 数据与输出配置
DATA_DIR="/path/to/project_root/neural_area/capability_data_v2/test_dataset"
OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_2b_pt_test"

# 处理范围配置（参考原始默认）
START_IDX=0
END_IDX=-1
LAYERS_PER_BATCH=8
LAYER_START=0
LAYER_END=25

# 如果只想先列出数据集，打开下面一行（会读取默认SAE层列表）
# "$PYTHON_BIN" "$SCRIPT_PATH" --data_dir "$DATA_DIR" --list_datasets --model_name "$MODEL_NAME" --sae_release "$SAE_RELEASE" --sae_local_base_dir "$SAE_LOCAL_BASE_DIR"

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
  --n_devices 1

# #################### baseline direct neural
# # TARGET_PID=3889621

# # # 检查进程是否还存在，存在就等待
# # while kill -0 $TARGET_PID 2>/dev/null; do
# #     echo "Process $TARGET_PID is still running..."
# #     sleep 5
# # done

# # # 基本路径 
# PYTHON_BIN="python"
# SCRIPT_PATH="/path/to/project_root/neural_area/divide_area_by_sae_act/get_sae_act.py"

# # 模型与SAE配置（保持原始2B-PT）
# MODEL_NAME="google/gemma-2-2b"
# SAE_RELEASE="gemma-scope-2b-pt-res"
# SAE_LOCAL_BASE_DIR="/path/to/local_models/gemma-scope-2b-pt-res"

# # 数据与输出配置
# DATA_DIR="/path/to/project_root/neural_area/capability_data/"
# OUTPUT_BASE_DIR="/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_2b_pt_neural"

# # 处理范围配置（参考原始默认）
# START_IDX=0
# END_IDX=-1
# LAYERS_PER_BATCH=8
# LAYER_START=0
# LAYER_END=25

# # 如果只想先列出数据集，打开下面一行（会读取默认SAE层列表）
# # "$PYTHON_BIN" "$SCRIPT_PATH" --data_dir "$DATA_DIR" --list_datasets --model_name "$MODEL_NAME" --sae_release "$SAE_RELEASE" --sae_local_base_dir "$SAE_LOCAL_BASE_DIR"

# "$PYTHON_BIN" "$SCRIPT_PATH" \
#   --data_dir "$DATA_DIR" \
#   --start "$START_IDX" \
#   --end "$END_IDX" \
#   --layers_per_batch "$LAYERS_PER_BATCH" \
#   --layer_start "$LAYER_START" \
#   --layer_end "$LAYER_END" \
#   --output_base_dir "$OUTPUT_BASE_DIR" \
#   --model_name "$MODEL_NAME" \
#   --sae_release "$SAE_RELEASE" \
#   --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
#   --baseline_direct_neural