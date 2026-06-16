#!/usr/bin/env bash

# 仅定义变量并执行，不做任何参数解析

RESULTS_ROOT=/path/to/project_root/safety_explanation/jailbreak/results
COMBO_NAMES=("AdvBench_gemma-2-9b-it" "JBB-Behaviors_gemma-2-9b-it")
MODEL_NAME="google/gemma-2-9b-it"
PARCEL_MAPPING=/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json

# 循环处理每个 COMBO_NAME
for COMBO_NAME in "${COMBO_NAMES[@]}"; do
  echo "正在处理: $COMBO_NAME"
  
  python /path/to/project_root/safety_explanation/hallucination/code/extract_parcel_token_activations.py \
    --results-root "$RESULTS_ROOT" \
    --combo-name "$COMBO_NAME" \
    --parcel-mapping "$PARCEL_MAPPING" \
    --model-name "$MODEL_NAME" \
    --layers-per-batch 2 \
    --skip-existing \

  
  echo "完成处理: $COMBO_NAME"
  echo "----------------------------------------"
done

echo "所有 COMBO_NAME 处理完成！"

