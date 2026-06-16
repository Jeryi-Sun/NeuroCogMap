#!/usr/bin/env bash
# 批量训练 BERT model parcel-level（narratives 数据集）
# 功能：
# 1) 循环处理 layer_idx（默认 1..12，对应 BERT 的 12 个 transformer 层）
# 2) 支持 train/test split，并可选 evaluation split
# 3) 支持按 layer_idx 和已有结果跳过任务
set -euo pipefail
subject_id="${SUBJECT_ID:-uts02}"

data_name="${DATA_NAME:-lebel}"
dataset_type="${DATASET_TYPE:-lebel}"

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core"
SCRIPT="${PROJECT_ROOT}/train_simple_bert.py"
LOG_DIR="${PROJECT_ROOT}/logs/${subject_id}/bert_parcel_level_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR_DEFAULT="${PROJECT_ROOT}/results/${subject_id}/results_bert_parcel_level_$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULT_DIR_OVERRIDE:-${RESULTS_DIR_DEFAULT}}"

# 控制是否跳过已存在的结果文件，设置为 1 表示跳过，0 表示不跳过
SKIP_EXISTING=${SKIP_EXISTING:-1}

# ========== 可配置超参数 ==========
# BERT-base 有 12 个 transformer 层（1-12），layer 0 是 embedding 层，-1 是 pooler 输出
START_LAYER_IDX="${START_LAYER_IDX:-1}"
END_LAYER_IDX="${END_LAYER_IDX:-12}"
USE_TRAIN_TEST_SPLIT="${USE_TRAIN_TEST_SPLIT:-1}"
TEST_STORY_FROM_END="${TEST_STORY_FROM_END:-1}"
EVAL_STORY_FROM_END="${EVAL_STORY_FROM_END:-0}"

if [[ -z "${data_name}" ]]; then
  ASSEMBLY_PATH="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/assembly_lebel_${subject_id}.pkl"
else
  ASSEMBLY_PATH="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/assembly_${data_name}_${subject_id}.pkl"
fi

mkdir -p "${PROJECT_ROOT}/cache_bert_model${data_name}" "${RESULTS_DIR}" "${LOG_DIR}"

echo "配置: subject_id=${subject_id}, dataset_type=${dataset_type}, RESULTS_DIR=${RESULTS_DIR}"
echo "配置: layer_idx 范围 [${START_LAYER_IDX}-${END_LAYER_IDX}], SKIP_EXISTING=${SKIP_EXISTING}"
echo "配置: use_train_test_split=${USE_TRAIN_TEST_SPLIT}, test_story_from_end=${TEST_STORY_FROM_END}, eval_story_from_end=${EVAL_STORY_FROM_END}"
if [[ "${dataset_type}" == "narratives" ]]; then
  echo "注意: narratives 数据集只有一个故事，将不使用 train_test_split"
fi

# 循环处理 layer_idx
for layer_idx in $(seq "${START_LAYER_IDX}" "${END_LAYER_IDX}"); do
  log_file="${LOG_DIR}/${layer_idx}.log"
  
  if [ "${SKIP_EXISTING}" -eq 1 ]; then
    # 检查是否存在包含该 layer_idx 的结果（包括 evaluation_metrics）
    if python - "${RESULTS_DIR}" "${layer_idx}" << 'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
layer_idx = int(sys.argv[2])
found = False
for hp in results_dir.glob("run_*/hyperparams.json"):
    try:
        with open(hp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("layer_idx")) == layer_idx:
            metrics_path = hp.parent / "metrics.pkl"
            eval_metrics_path = hp.parent / "evaluation_metrics.pkl"
            if metrics_path.exists() or eval_metrics_path.exists():
                found = True
                break
    except Exception as exc:
        print(f"[WARN] 读取 {hp} 失败: {exc}")
        continue

sys.exit(0 if found else 1)
PY
    then
      echo "跳过 layer_idx=${layer_idx}，结果文件已存在" | tee -a "${LOG_DIR}/batch_run_bert_parcel.log"
      continue
    fi
  fi
  
  echo "开始处理 layer_idx=${layer_idx}, 日志文件: ${log_file}" | tee -a "${LOG_DIR}/batch_run_bert_parcel.log"
  
  cmd=(python "${SCRIPT}" \
    --assembly_path "${ASSEMBLY_PATH}" \
    --layer_idx "${layer_idx}" \
    --cache_dir "${PROJECT_ROOT}/cache_bert_model${data_name}" \
    --results_dir "${RESULTS_DIR}" \
    --logger_backend "none" \
    --parcel_level \
    --lookback 512 \
    --dataset_type "${dataset_type}")

  # narratives 数据集只有一个故事，不使用 train_test_split
  if [[ "${USE_TRAIN_TEST_SPLIT}" == "1" && "${dataset_type}" != "narratives" ]]; then
    cmd+=(--use_train_test_split --test_story_from_end "${TEST_STORY_FROM_END}")
    if [[ "${EVAL_STORY_FROM_END}" != "0" ]]; then
      cmd+=(--eval_story_from_end "${EVAL_STORY_FROM_END}")
    fi
  fi

  "${cmd[@]}" > "${log_file}" 2>&1

  
  if [ $? -eq 0 ]; then
    echo "layer_idx=${layer_idx} 处理完成" | tee -a "${LOG_DIR}/batch_run_bert_parcel.log"
  else
    echo "layer_idx=${layer_idx} 处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run_bert_parcel.log"
  fi
done

echo "所有 layer_idx (${START_LAYER_IDX}-${END_LAYER_IDX}) 处理完成" | tee -a "${LOG_DIR}/batch_run_bert_parcel.log"
