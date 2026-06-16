#!/bin/bash

# 运行SAE最大激活提取脚本（9b-it + instruct）
# 使用方法: ./run_extract_max_activation_9b.sh [dataset_start] [dataset_end]

# 设置默认参数
DATASET_START=${1:-0}
DATASET_END=${2:--1}

# 设置环境变量（按需修改 GPU）

# 设置输出目录
OUTPUT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation/dataset_9b"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始提取SAE最大激活状态（9b-it）..."
echo "数据集范围: $DATASET_START 到 $DATASET_END"
echo "输出目录: $OUTPUT_DIR"

# 9b 模型与 SAE 配置
MODEL_NAME="google/gemma-2-9b-it"
SAE_RELEASE="gemma-scope-9b-it-res"
SAE_LOCAL_BASE_DIR="/path/to/local_models/gemma-scope-9b-it-res"

# 如需限制层，可指定 SAE_PATHS（逗号分隔）。示例三层：
SAE_PATHS="layer_9/width_16k/average_l0_88,layer_20/width_16k/average_l0_91,layer_31/width_16k/average_l0_76"

# 运行Python脚本
python extract_max_activation.py \
    --start $DATASET_START \
    --end $DATASET_END \
    --layers_per_batch 1 \
    --layer_start 0 \
    --layer_end 40 \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --sae_release "$SAE_RELEASE" \
    --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
    --sae_paths "$SAE_PATHS" \
    --n_devices 1 \
    --is_instruct

echo "最大激活提取完成（9b-it）！"
echo "结果保存在: $OUTPUT_DIR"

