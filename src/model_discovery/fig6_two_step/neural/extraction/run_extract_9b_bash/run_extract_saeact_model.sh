#!/usr/bin/env bash
# 批量提取 SAE Activation 特征，一次性提取所有 Parcel (0-269)
# 使用 --extract_all_parcels 选项大幅提升性能（每个实验只运行一次模型前向传播）
# 支持跳过已存在的结果文件
set -euo pipefail

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural"
SCRIPT="${PROJECT_ROOT}/extract_with_token_limit.py"
LOG_DIR="${PROJECT_ROOT}/logs_9b/extract_saeact_model_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${PROJECT_ROOT}/results_9b/cache_saeact_model_pre_nozero_patch"

# 控制是否跳过已存在的结果文件，设置为 1 表示跳过，0 表示不跳过
SKIP_EXISTING=${SKIP_EXISTING:-1}

# 模型和输入文件路径
MODEL_PATH="${MODEL_PATH:-google/gemma-2-9b-it}"
INPUT_FILE="${INPUT_FILE:-feher2023rethinking/prompts_reformatted.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# SAE 相关路径
PARCEL_MAPPING_PATH="${PARCEL_MAPPING_PATH:-/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_9b_it/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json}"
SAE_LOCAL_BASE_DIR="${SAE_LOCAL_BASE_DIR:-/path/to/local_models/gemma-scope-9b-it-res}"
SAE_RELEASE="${SAE_RELEASE:-gemma-scope-9b-it-res}"

# SAE 路径列表
SAE_PATHS="layer_9/width_16k/average_l0_88,layer_20/width_16k/average_l0_91,layer_31/width_16k/average_l0_76"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

log_file="${LOG_DIR}/extract_all_parcels.log"

if [ "${SKIP_EXISTING}" -eq 1 ]; then
  # 检查是否所有 Parcel 的结果文件都已存在
  # 查找所有参与者文件，检查是否都已存在
  found_all=true
    for parcel_id in {0..269}; do
      for participant_id in 0 1; do
        output_file="${RESULTS_DIR}/parcel_${parcel_id}/model=${MODEL_PATH//\//-}_extractor=saeact_model_parcel_${parcel_id}_participant=${participant_id}.pth"
      if [ ! -f "${output_file}" ]; then
        found_all=false
        break 2
      fi
    done
  done
  
  if [ "$found_all" = true ]; then
    echo "跳过所有 Parcel，结果文件已存在" | tee -a "${LOG_DIR}/batch_run.log"
    exit 0
  fi
fi

echo "开始一次性提取所有 Parcel (0-269) 的特征, 日志文件: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
echo "注意: 使用 --extract_all_parcels 选项，每个实验只运行一次模型前向传播，性能大幅提升" | tee -a "${LOG_DIR}/batch_run.log"

python "${SCRIPT}" \
  --model "${MODEL_PATH}" \
  --input "${INPUT_FILE}" \
  --extractor_type "saeact_model" \
  --extract_all_parcels \
  --parcel_mapping_path "${PARCEL_MAPPING_PATH}" \
  --sae_release "${SAE_RELEASE}" \
  --sae_local_base_dir "${SAE_LOCAL_BASE_DIR}" \
  --sae_paths "${SAE_PATHS}" \
  --max_tokens "${MAX_TOKENS}" \
  --skip_existing \
  > "${log_file}" 2>&1

if [ $? -eq 0 ]; then
  echo "所有 Parcel (0-269) 处理完成" | tee -a "${LOG_DIR}/batch_run.log"
else
  echo "处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
  exit 1
fi

echo "所有 Parcel (0-269) 处理完成" | tee -a "${LOG_DIR}/batch_run.log"

