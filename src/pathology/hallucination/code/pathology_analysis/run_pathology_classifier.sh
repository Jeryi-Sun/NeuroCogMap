#!/bin/bash

# 功能：调用 pathology_classifier.py
# 使用本地 vLLM 模型，对 capability 描述 (definition_refined)
# 和 SAE parcel 功能描述 (function_description) 进行病理类型分类：
# - Belief-related
# - Control-related
# - Mixed
# - Neutral
#
# 运行前请确保 vLLM 服务已在 --vllm_url 指定的地址上启动。

set -e

echo "启动 pathology 分类任务..."

# 可根据需要修改的公共参数
VLLM_URL="https://api2.aigcbest.top/v1"
API_KEY="sk-liaIJiVfNSuOIZzmypbfVhDKJkTbD7boA1QZNimDegW0PJ4I"
DELAY=1.0
SKIP_EXISTING=""  # 如需强制重跑，可设为 "" 覆盖

PYTHON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# 1) capability 描述分类
# -----------------------------
CAPABILITY_INPUT="/path/to/project_root/capability_analysis/data/capability_descriptions/capability_descriptions_run2.json"
CAPABILITY_OUTPUT="/path/to/project_root/safety_explanation/hallucination/code/pathology_analysis/data/capability_descriptions_run2_pathology_classification_v3.json"

echo "对 capability 描述进行病理分类..."
# python "${PYTHON_SCRIPT_DIR}/pathology_classifier.py" \
#   --mode "capability" \
#   --input_file "${CAPABILITY_INPUT}" \
#   --output_file "${CAPABILITY_OUTPUT}" \
#   --vllm_url "${VLLM_URL}" \
#   --api_key "${API_KEY}" \
#   --delay "${DELAY}" \
#   ${SKIP_EXISTING}

# -----------------------------
# 2) SAE parcel 功能描述分类
# -----------------------------
PARCEL_INPUT="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_9b_it/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
PARCEL_OUTPUT="/path/to/project_root/safety_explanation/hallucination/code/pathology_analysis/data/latent_parcel_topsamples_functionality_summary_pathology_classification_9b_it_v3.json"

echo "对 SAE parcels 功能描述进行病理分类..."
python "${PYTHON_SCRIPT_DIR}/pathology_classifier.py" \
  --mode "parcel" \
  --input_file "${PARCEL_INPUT}" \
  --output_file "${PARCEL_OUTPUT}" \
  --vllm_url "${VLLM_URL}" \
  --api_key "${API_KEY}" \
  --delay "${DELAY}" \
  ${SKIP_EXISTING}

echo "pathology 分类任务完成。"

