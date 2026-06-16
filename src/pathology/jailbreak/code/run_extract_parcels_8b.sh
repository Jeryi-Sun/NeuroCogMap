#!/usr/bin/env bash

# 功能: 针对 Llama-3.1-8B 的 Jailbreak 结果，抽取 parcel-level token activations（按 combo 循环）。
# 说明: 该脚本仅定义变量并执行，不做参数解析；可通过 SKIP_EXISTING 控制是否跳过已存在结果。

set -euo pipefail

RESULTS_ROOT="/path/to/project_root/safety_explanation/jailbreak/results"
COMBO_NAMES=("AdvBench_Llama-3.1-8B" "JBB-Behaviors_Llama-3.1-8B")
MODEL_NAME="meta-llama/Llama-3.1-8B"

# Llama-3.1-8B 对应的 parcel 映射（与 hallucination/8b 脚本保持一致）
PARCEL_MAPPING="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_llama_8b_pt/clustering_results_sentence_prep0.01_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels240/latent_parcel_assignments.json"

# 是否跳过已存在的结果: 1=跳过, 0=不跳过
SKIP_EXISTING=1

for COMBO_NAME in "${COMBO_NAMES[@]}"; do
  echo "正在处理: $COMBO_NAME"

  ARGS=(
    --results-root "$RESULTS_ROOT"
    --combo-name "$COMBO_NAME"
    --parcel-mapping "$PARCEL_MAPPING"
    --model-name "$MODEL_NAME"
    --layers-per-batch 1
    --sae_paths "l0r_8x,l1r_8x,l2r_8x,l3r_8x,l4r_8x,l5r_8x,l6r_8x,l7r_8x,l8r_8x,l9r_8x,l10r_8x,l11r_8x,l12r_8x,l13r_8x,l14r_8x,l15r_8x,l16r_8x,l17r_8x,l18r_8x,l19r_8x,l20r_8x,l21r_8x,l22r_8x,l23r_8x,l24r_8x,l25r_8x,l26r_8x,l27r_8x,l28r_8x,l29r_8x,l30r_8x,l31r_8x"
    --sae-release "llama_scope_lxr_8x"
    --sae-local-base-dir "/path/to/local_models/Llama3_1-8B-Base-LXR-8x"
    --n-devices 2
  )

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip-existing)
  fi

  python3 /path/to/project_root/safety_explanation/hallucination/code/extract_parcel_token_activations.py \
    "${ARGS[@]}"

  echo "完成处理: $COMBO_NAME"
  echo "----------------------------------------"
done

echo "所有 COMBO_NAME 处理完成！"

