#!/usr/bin/env bash
# 批量训练 wordrate model，parcel level
# 结果会保存到独立的 log 文件中，不开启 wandb
# 支持跳过已存在的结果文件
set -euo pipefail
subject_id="uts249"

data_name="narratives"

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core"
SCRIPT="${PROJECT_ROOT}/train_simple.py"
LOG_DIR="/path/to/project_root/Human_LLM_align/litcoder_core/logs/${subject_id}/wordrate_parcel_level_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${PROJECT_ROOT}/results/${subject_id}/results_wordrate_parcel_level_$(date +%Y%m%d_%H%M%S)"

# 控制是否跳过已存在的结果文件，设置为 1 表示跳过，0 表示不跳过
SKIP_EXISTING=${SKIP_EXISTING:-0}

if [[ -z "${data_name}" ]]; then
  ASSEMBLY_PATH="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/assembly_lebel_${subject_id}.pkl"
else
  ASSEMBLY_PATH="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/assembly_${data_name}_${subject_id}.pkl"
fi

mkdir -p "${PROJECT_ROOT}/cache${data_name}" "${RESULTS_DIR}" "${LOG_DIR}"

log_file="${LOG_DIR}/wordrate_parcel.log"

if [ "${SKIP_EXISTING}" -eq 1 ]; then
  # 检查是否存在结果目录，通过检查是否有 run_* 目录来判断
  found_existing=false
  if [ -d "${RESULTS_DIR}" ]; then
    for result_dir in "${RESULTS_DIR}"/run_*; do
      if [ -d "${result_dir}" ] && [ -f "${result_dir}/hyperparams.json" ]; then
        # 检查是否是 lebel 数据集的结果（wordrate 训练使用 lebel 数据集）
        if grep -q "\"dataset_type\": \"lebel\"" "${result_dir}/hyperparams.json" 2>/dev/null; then
          found_existing=true
          break
        fi
      fi
    done
  fi
  
  if [ "$found_existing" = true ]; then
    echo "跳过 wordrate parcel level 训练，结果文件已存在" | tee -a "${LOG_DIR}/batch_run_wordrate_parcel.log"
    exit 0
  fi
fi

echo "开始处理 wordrate parcel level 训练, 日志文件: ${log_file}" | tee -a "${LOG_DIR}/batch_run_wordrate_parcel.log"

python "${SCRIPT}" \
  --assembly_path "${ASSEMBLY_PATH}" \
  --cache_dir "${PROJECT_ROOT}/cache${data_name}" \
  --results_dir "${RESULTS_DIR}" \
  --logger_backend "none" \
  --parcel_level \
  > "${log_file}" 2>&1

if [ $? -eq 0 ]; then
  echo "wordrate parcel level 训练完成" | tee -a "${LOG_DIR}/batch_run_wordrate_parcel.log"
else
  echo "wordrate parcel level 训练失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run_wordrate_parcel.log"
  exit 1
fi

echo "wordrate parcel level 处理完成" | tee -a "${LOG_DIR}/batch_run_wordrate_parcel.log"

