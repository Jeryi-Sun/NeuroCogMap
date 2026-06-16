#!/usr/bin/env bash
# 批量提取 Language Model 特征，一次性提取所有层 (0-25)
# 使用 --layers 参数一次性提取所有层，每个实验只运行一次模型前向传播，性能大幅提升
# 支持跳过已存在的结果文件
set -euo pipefail

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural"
SCRIPT="${PROJECT_ROOT}/extract_with_token_limit.py"
LOG_DIR="${PROJECT_ROOT}/logs_9b/extract_language_model_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${PROJECT_ROOT}/results_9b/cache_language_model"

# 控制是否跳过已存在的结果文件，设置为 1 表示跳过，0 表示不跳过
SKIP_EXISTING=${SKIP_EXISTING:-1}

# 模型和输入文件路径
MODEL_PATH="${MODEL_PATH:-google/gemma-2-9b-it}"
INPUT_FILE="${INPUT_FILE:-feher2023rethinking/prompts_reformatted.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# 要提取的所有层（0-25）
LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

log_file="${LOG_DIR}/extract_all_layers.log"

if [ "${SKIP_EXISTING}" -eq 1 ]; then
  # 检查是否所有层的结果文件都已存在
  # 查找所有参与者文件，检查是否都已存在
  found_all=true
    for layer_idx in {0..25}; do
      for participant_id in 0 1; do
        output_file="${RESULTS_DIR}/layer_${layer_idx}/model=${MODEL_PATH//\//-}_extractor=language_model_layer_${layer_idx}_participant=${participant_id}.pth"
      if [ ! -f "${output_file}" ]; then
        found_all=false
        break 2
      fi
    done
  done
  
  if [ "$found_all" = true ]; then
    echo "跳过所有层，结果文件已存在" | tee -a "${LOG_DIR}/batch_run.log"
    exit 0
  fi
fi

echo "开始一次性提取所有层 (0-25) 的特征, 日志文件: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
echo "注意: 使用 --layers 参数，每个实验只运行一次模型前向传播，性能大幅提升" | tee -a "${LOG_DIR}/batch_run.log"

python "${SCRIPT}" \
  --model "${MODEL_PATH}" \
  --input "${INPUT_FILE}" \
  --extractor_type "language_model" \
  --layers "${LAYERS}" \
  --max_tokens "${MAX_TOKENS}" \
  --skip_existing \
  > "${log_file}" 2>&1

if [ $? -eq 0 ]; then
  echo "所有层 (0-25) 处理完成" | tee -a "${LOG_DIR}/batch_run.log"
else
  echo "处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
  exit 1
fi

echo "所有层 (0-25) 处理完成" | tee -a "${LOG_DIR}/batch_run.log"

