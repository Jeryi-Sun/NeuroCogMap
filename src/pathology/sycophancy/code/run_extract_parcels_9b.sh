#!/usr/bin/env bash

# 仅定义变量并执行，不做任何参数解析

RESULTS_ROOT=/path/to/project_root/safety_explanation/sycophancy/results
COMBO_NAMES=("feedback_gemma-2-9b-it") # 
MODEL_NAME="google/gemma-2-9b-it"
PARCEL_MAPPING=/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_9b_it/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json

# 循环处理每个 COMBO_NAME
for COMBO_NAME in "${COMBO_NAMES[@]}"; do
  echo "正在处理: $COMBO_NAME"
  
  python /path/to/project_root/safety_explanation/hallucination/code/extract_parcel_token_activations.py \
    --results-root "$RESULTS_ROOT" \
    --combo-name "$COMBO_NAME" \
    --parcel-mapping "$PARCEL_MAPPING" \
    --model-name "$MODEL_NAME" \
    --layers-per-batch 1 \
    --sae_paths "layer_9/width_16k/average_l0_88,layer_20/width_16k/average_l0_91,layer_31/width_16k/average_l0_76" \
    --is_instruct \
    --sae-release "gemma-scope-9b-it-res" \
    --sae-local-base-dir "/path/to/local_models/gemma-scope-9b-it-res" \
    --n-devices 2 \
  
  echo "完成处理: $COMBO_NAME"
  echo "----------------------------------------"
done

echo "所有 COMBO_NAME 处理完成！"

