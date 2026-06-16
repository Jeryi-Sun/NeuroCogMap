#!/usr/bin/env bash
# 统一批量运行 fit.py，支持三类 extractor：
# - bert_model（按 layer 循环）
# - language_model_attention（按 layer 循环）
# - embeddings（单次运行）
# 可通过 EXTRACTOR_TYPE 切换，支持 SKIP_EXISTING 与常用超参数。
set -euo pipefail

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural"
SCRIPT="${PROJECT_ROOT}/fit.py"

EXTRACTOR_TYPE="${EXTRACTOR_TYPE:-bert_model}"  # bert_model / language_model_attention / embeddings
MODEL_NAME="${MODEL_NAME:-google-bert/bert-base-uncased}"
PARTICIPANT_NUM="${PARTICIPANT_NUM:-}"
ALPHA_FDR="${ALPHA_FDR:-0.05}"
ROI="${ROI:-}"
TEST_MODE="${TEST_MODE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

START_LAYER="${START_LAYER:-1}"
END_LAYER="${END_LAYER:-12}"

case "${EXTRACTOR_TYPE}" in
  bert_model)
    : "${MODEL_NAME:=google-bert/bert-base-uncased}"
    : "${START_LAYER:=11}"
    : "${END_LAYER:=12}"
    ;;
  language_model_attention)
    : "${MODEL_NAME:=google/gemma-2-2b}"
    : "${START_LAYER:=8}"
    : "${END_LAYER:=14}"
    ;;
  embeddings)
    : "${MODEL_NAME:=word2vec}"
    ;;
  *)
    echo "不支持的 EXTRACTOR_TYPE: ${EXTRACTOR_TYPE}"
    echo "仅支持: bert_model / language_model_attention / embeddings"
    exit 1
    ;;
esac

LOG_DIR="${PROJECT_ROOT}/logs/fit_${EXTRACTOR_TYPE}_$(date +%Y%m%d_%H%M%S)"
FIT_DIR="${PROJECT_ROOT}/fits/${EXTRACTOR_TYPE}"
mkdir -p "${LOG_DIR}" "${FIT_DIR}"

echo "配置: EXTRACTOR_TYPE=${EXTRACTOR_TYPE}, MODEL_NAME=${MODEL_NAME}, PARTICIPANT_NUM=${PARTICIPANT_NUM}"
echo "配置: ALPHA_FDR=${ALPHA_FDR}, TEST_MODE=${TEST_MODE}, SKIP_EXISTING=${SKIP_EXISTING}, ROI='${ROI}'"
if [[ "${EXTRACTOR_TYPE}" != "embeddings" ]]; then
  echo "配置: layer 范围=[${START_LAYER}-${END_LAYER}]"
fi

check_existing_layer_result() {
  local layer="$1"
  python - "${FIT_DIR}" "${MODEL_NAME}" "${EXTRACTOR_TYPE}" "${layer}" << 'PY'
import sys
from pathlib import Path

fits_dir = Path(sys.argv[1])
model_safe = sys.argv[2].replace("/", "-")
extractor = sys.argv[3]
layer = sys.argv[4]
pattern = f"model={model_safe}_extractor={extractor}_layer={layer}_roi=*.json"
found = any(fits_dir.glob(pattern))
sys.exit(0 if found else 1)
PY
}

check_existing_general_result() {
  python - "${FIT_DIR}" "${MODEL_NAME}" "${EXTRACTOR_TYPE}" << 'PY'
import sys
from pathlib import Path

fits_dir = Path(sys.argv[1])
model_safe = sys.argv[2].replace("/", "-")
extractor = sys.argv[3]
pattern = f"model={model_safe}_extractor={extractor}_general_roi=*.json"
found = any(fits_dir.glob(pattern))
sys.exit(0 if found else 1)
PY
}

run_fit_once() {
  local log_file="$1"
  shift
  local cmd=(python "${SCRIPT}"
    --model "${MODEL_NAME}"
    --extractor_type "${EXTRACTOR_TYPE}"
    --alpha_fdr "${ALPHA_FDR}")
  cmd+=("$@")

  if [[ -n "${ROI}" ]]; then
    cmd+=(--roi "${ROI}")
  fi
  if [[ "${TEST_MODE}" == "1" ]]; then
    cmd+=(--test)
  fi

  if "${cmd[@]}" > "${log_file}" 2>&1; then
    return 0
  fi
  return 1
}

if [[ "${EXTRACTOR_TYPE}" == "embeddings" ]]; then
  log_file="${LOG_DIR}/general.log"
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && check_existing_general_result; then
    echo "跳过 embeddings：结果已存在" | tee -a "${LOG_DIR}/batch_run.log"
    exit 0
  fi

  if run_fit_once "${log_file}"; then
    echo "embeddings 处理完成" | tee -a "${LOG_DIR}/batch_run.log"
  else
    echo "embeddings 处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
    exit 1
  fi
  exit 0
fi

for layer in $(seq "${START_LAYER}" "${END_LAYER}"); do
  log_file="${LOG_DIR}/layer_${layer}.log"
  if [[ "${SKIP_EXISTING}" -eq 1 ]] && check_existing_layer_result "${layer}"; then
    echo "跳过 layer=${layer}：结果已存在" | tee -a "${LOG_DIR}/batch_run.log"
    continue
  fi

  if run_fit_once "${log_file}" --layer "${layer}"; then
    echo "layer=${layer} 处理完成" | tee -a "${LOG_DIR}/batch_run.log"
  else
    echo "layer=${layer} 处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
  fi
done

echo "${EXTRACTOR_TYPE} 所有任务处理完成" | tee -a "${LOG_DIR}/batch_run.log"

