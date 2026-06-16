#!/usr/bin/env bash
set -euo pipefail

# 运行 bias_generate_and_eval.py 的批处理脚本
# 可通过环境变量或命令行参数覆盖默认值
#
# 用法示例：
#   bash run_bias.sh
#   SKIP_EXISTING=1 USE_INCONTEXT=1 bash run_bias.sh
#   MODE=both MODEL_ID="google/gemma-2-9b-it" VLLM_URL="http://0.0.0.0:8001/v1" API_KEY="abcabc" bash run_bias.sh
#
# 可用环境变量（带默认值）：
: "${MODE:=both}"                     # generate/eval/both
: "${MODEL_ID:=google/gemma-2-2b}" # 用于生成的模型
: "${VLLM_URL:=http://0.0.0.0:8001/v1}"
: "${API_KEY:=abcabc}"
: "${USE_INCONTEXT:=1}"               # 1=启用 in-context learning, 0=关闭
: "${SKIP_EXISTING:=0}"               # 1=跳过已存在，0=覆盖
: "${MAX_SAMPLES:=0}"                 # 0=全量
: "${MAX_NEW_TOKENS:=128}"
: "${TEMPERATURE:=0.0}"

ROOT_DIR="/path/to/project_root"
SCRIPT_PATH="${ROOT_DIR}/safety_explanation/fairness_bias/code/bias_generate_and_eval.py"
DATA_DIR="${ROOT_DIR}/safety_explanation/fairness_bias/dataset"
OUTPUT_DIR="${ROOT_DIR}/safety_explanation/fairness_bias/results"

# 数据集列表（按需增减）
DATASETS=(
  "bbq_age.csv"
  "bbq_disability_status.csv"
  "bbq_gender_identity.csv"
  "bbq_nationality.csv"
)

mkdir -p "${OUTPUT_DIR}"

echo "MODE=${MODE}"
echo "MODEL_ID=${MODEL_ID}"
echo "VLLM_URL=${VLLM_URL}"
echo "USE_INCONTEXT=${USE_INCONTEXT}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "TEMPERATURE=${TEMPERATURE}"
echo

for ds in "${DATASETS[@]}"; do
  CSV_PATH="${DATA_DIR}/${ds}"
  if [[ ! -f "${CSV_PATH}" ]]; then
    echo "[WARN] 数据集不存在，跳过: ${CSV_PATH}"
    continue
  fi

  echo "============================================================"
  echo "开始处理数据集: ${ds}"
  echo "============================================================"

  ARGS=(
    "--mode" "${MODE}"
    "--model_id" "${MODEL_ID}"
    "--csv_path" "${CSV_PATH}"
    "--output_dir" "${OUTPUT_DIR}"
    "--vllm_url" "${VLLM_URL}"
    "--api_key" "${API_KEY}"
    "--max_samples" "${MAX_SAMPLES}"
    "--max_new_tokens" "${MAX_NEW_TOKENS}"
    "--temperature" "${TEMPERATURE}"
  )

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    ARGS+=("--skip_existing")
  fi

  if [[ "${USE_INCONTEXT}" == "1" ]]; then
    ARGS+=("--use_incontext")
  fi

  python "${SCRIPT_PATH}" "${ARGS[@]}"
done

echo
echo "[DONE] 全部数据集处理完成。"


