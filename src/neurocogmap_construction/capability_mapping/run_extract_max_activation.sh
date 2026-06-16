#!/bin/bash

# 运行SAE最大激活提取脚本
# 使用方法: ./run_extract_max_activation.sh [dataset_start] [dataset_end]

# 设置默认参数
DATASET_START=${1:-0}
DATASET_END=${2:--1}

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置输出目录
OUTPUT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始提取SAE最大激活状态..."
echo "数据集范围: $DATASET_START 到 $DATASET_END"
echo "输出目录: $OUTPUT_DIR"

# 运行Python脚本
python extract_max_activation.py \
    --start $DATASET_START \
    --end $DATASET_END \
    --layers_per_batch 4 \
    --layer_start 0 \
    --layer_end 25 \
    --output_dir "$OUTPUT_DIR" \
    --model_name "google/gemma-2-2b" \
    --sae_release "gemma-scope-2b-pt-res" \
    --sae_local_base_dir "/path/to/local_models/gemma-scope-2b-pt-res" \
    --n_devices 1

echo "最大激活提取完成！"
echo "结果保存在: $OUTPUT_DIR" 