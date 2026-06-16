#!/usr/bin/env bash
# 统一批量提取特征脚本，支持：
# - language_model（residual hidden state）
# - bert_model（复用 language_model 提取流程，单独写入 cache_bert_model）
# - language_model_attention（attention 输出，hook_attn_out）
# - embeddings（静态词向量，按实验聚合为一条特征）
set -euo pipefail

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural"
SCRIPT="${PROJECT_ROOT}/extract_with_token_limit.py"

EXTRACTOR_TYPE="${EXTRACTOR_TYPE:-language_model}"  # language_model / bert_model / language_model_attention / embeddings
MODEL_PATH="${MODEL_PATH:-google/gemma-2-2b}"
INPUT_FILE="${INPUT_FILE:-feher2023rethinking/prompts_reformatted.jsonl}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
REVERSE_ORDER="${REVERSE_ORDER:-0}"
LAYERS="${LAYERS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}"
PARTICIPANT_IDS="${PARTICIPANT_IDS:-0 1}"
VECTOR_PATH="${VECTOR_PATH:-/path/to/local_models/word2vec/nlwiki_20180420_300d.txt}"
EMBEDDING_LOWERCASE="${EMBEDDING_LOWERCASE:-1}"
EMBEDDING_OOV_HANDLING="${EMBEDDING_OOV_HANDLING:-copy_prev}"
EMBEDDING_POOLING="${EMBEDDING_POOLING:-mean}"

case "${EXTRACTOR_TYPE}" in
  language_model)
    : "${LAYERS:=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}"
    ;;
  bert_model)
    : "${MODEL_PATH:=google-bert/bert-base-uncased}"
    : "${LAYERS:=1,2,3,4,5,6,7,8,9,10,11,12}"
    ;;
  language_model_attention)
    : "${MODEL_PATH:=google/gemma-2-2b}"
    : "${LAYERS:=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25}"
    ;;
  embeddings)
    : "${MODEL_PATH:=word2vec}"
    ;;
  *)
    echo "[ERROR] 不支持的 EXTRACTOR_TYPE: ${EXTRACTOR_TYPE}"
    echo "[ERROR] 仅支持: language_model / bert_model / language_model_attention / embeddings"
    exit 1
    ;;
esac

RESULTS_DIR="${PROJECT_ROOT}/results/cache_${EXTRACTOR_TYPE}"
LOG_DIR="${PROJECT_ROOT}/logs/extract_${EXTRACTOR_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"

log_file="${LOG_DIR}/extract.log"
echo "配置: EXTRACTOR_TYPE=${EXTRACTOR_TYPE}, MODEL_PATH=${MODEL_PATH}"
echo "配置: INPUT_FILE=${INPUT_FILE}, MAX_TOKENS=${MAX_TOKENS}, SKIP_EXISTING=${SKIP_EXISTING}, REVERSE_ORDER=${REVERSE_ORDER}"
if [[ "${EXTRACTOR_TYPE}" == "embeddings" ]]; then
  echo "配置: VECTOR_PATH=${VECTOR_PATH}, EMBEDDING_OOV_HANDLING=${EMBEDDING_OOV_HANDLING}, EMBEDDING_POOLING=${EMBEDDING_POOLING}"
else
  echo "配置: LAYERS=${LAYERS}"
fi

if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
  if python - "${RESULTS_DIR}" "${MODEL_PATH}" "${EXTRACTOR_TYPE}" "${LAYERS}" "${PARTICIPANT_IDS}" << 'PY'
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
model_safe = sys.argv[2].replace("/", "-")
extractor = sys.argv[3]
layers = [x.strip() for x in sys.argv[4].split(",") if x.strip()]
participant_ids = sys.argv[5].split()

found_all = True
if extractor == "embeddings":
    for pid in participant_ids:
        output_file = (
            results_dir
            / "general"
            / f"model={model_safe}_extractor={extractor}_general_participant={pid}.pth"
        )
        if not output_file.exists():
            found_all = False
            break
else:
    for layer in layers:
        for pid in participant_ids:
            output_file = (
                results_dir
                / f"layer_{layer}"
                / f"model={model_safe}_extractor={extractor}_layer_{layer}_participant={pid}.pth"
            )
            if not output_file.exists():
                found_all = False
                break
        if not found_all:
            break

sys.exit(0 if found_all else 1)
PY
  then
    echo "跳过提取：目标结果文件已存在" | tee -a "${LOG_DIR}/batch_run.log"
    exit 0
  fi
fi

cmd=(python "${SCRIPT}"
  --model "${MODEL_PATH}"
  --input "${INPUT_FILE}"
  --extractor_type "${EXTRACTOR_TYPE}"
  --max_tokens "${MAX_TOKENS}")

if [[ "${EXTRACTOR_TYPE}" == "embeddings" ]]; then
  cmd+=(--vector_path "${VECTOR_PATH}")
  cmd+=(--embedding_lowercase "${EMBEDDING_LOWERCASE}")
  cmd+=(--embedding_oov_handling "${EMBEDDING_OOV_HANDLING}")
  cmd+=(--embedding_pooling "${EMBEDDING_POOLING}")
else
  cmd+=(--layers "${LAYERS}")
fi

if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip_existing)
fi
if [[ "${REVERSE_ORDER}" == "1" ]]; then
  cmd+=(--reverse_order)
fi

if "${cmd[@]}" > "${log_file}" 2>&1; then
  echo "提取完成：${EXTRACTOR_TYPE}" | tee -a "${LOG_DIR}/batch_run.log"
else
  echo "提取失败：${EXTRACTOR_TYPE}，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
  exit 1
fi

