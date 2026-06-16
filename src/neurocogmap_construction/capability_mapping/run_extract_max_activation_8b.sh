#!/bin/bash

# 运行 SAE 最大激活提取脚本（Llama-3.1-8B Base + LXR-8x SAE）
# 参考 run_extract_max_activation_9b.sh 流程，模型与 SAE 配置参考 run_get_sae_act_llama_8b_pt.sh
# 使用方法: ./run_extract_max_activation_8b.sh [dataset_start] [dataset_end]

# 设置默认参数
DATASET_START=${1:-0}
DATASET_END=${2:--1}

# 设置输出目录
OUTPUT_DIR="/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation/dataset_8b"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始提取 SAE 最大激活状态（Llama-3.1-8B Base）..."
echo "数据集范围: $DATASET_START 到 $DATASET_END"
echo "输出目录: $OUTPUT_DIR"

# Llama 8B 模型与 SAE 配置（与 run_get_sae_act_llama_8b_pt.sh 一致）
MODEL_NAME="meta-llama/Llama-3.1-8B"
SAE_RELEASE="llama_scope_lxr_8x"
SAE_LOCAL_BASE_DIR="/path/to/local_models/Llama3_1-8B-Base-LXR-8x"

# 32 层 SAE 路径（layer 0–31）
SAE_PATHS="l0r_8x,l1r_8x,l2r_8x,l3r_8x,l4r_8x,l5r_8x,l6r_8x,l7r_8x,l8r_8x,l9r_8x,l10r_8x,l11r_8x,l12r_8x,l13r_8x,l14r_8x,l15r_8x,l16r_8x,l17r_8x,l18r_8x,l19r_8x,l20r_8x,l21r_8x,l22r_8x,l23r_8x,l24r_8x,l25r_8x,l26r_8x,l27r_8x,l28r_8x,l29r_8x,l30r_8x,l31r_8x"

# 运行 Python 脚本（Base 模型，不加 --is_instruct）
python extract_max_activation.py \
    --start $DATASET_START \
    --end $DATASET_END \
    --layers_per_batch 1 \
    --layer_start 0 \
    --layer_end 31 \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --sae_release "$SAE_RELEASE" \
    --sae_local_base_dir "$SAE_LOCAL_BASE_DIR" \
    --sae_paths "$SAE_PATHS" \
    --n_devices 1

echo "最大激活提取完成（Llama-3.1-8B）！"
echo "结果保存在: $OUTPUT_DIR"
