#!/usr/bin/env bash
# 批量训练 SAE activation 模型（支持双模式）
# 模式说明：
# 1) RUN_MODE=all（默认）：一次性拼接全部 Parcel 特征，仅运行一次训练
# 2) RUN_MODE=single：按 parcel_id 范围循环训练（兼容旧流程）
# 3) SKIP_EXISTING 可控跳过：all 模式按整次运行跳过，single 模式按 parcel 跳过
set -euo pipefail
subject_id="${SUBJECT_ID:-uts02}"

data_name="${DATA_NAME:-lebel}"
dataset_type="${DATASET_TYPE:-lebel}"

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core"
SCRIPT="${PROJECT_ROOT}/train_simple_saeact.py"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

# cache 路径规则：lebel 数据集不加后缀，其它数据集按 data_name 区分
if [[ -z "${data_name}" || "${data_name}" == "lebel" || "${dataset_type}" == "lebel" ]]; then
  CACHE_DIR="${PROJECT_ROOT}/cache_saeact_model"
else
  CACHE_DIR="${PROJECT_ROOT}/cache_saeact_model${data_name}"
fi

if [[ -z "${data_name}" ]]; then
  ASSEMBLY_PATH="${PROJECT_ROOT}/dataset/assembly_lebel_${subject_id}.pkl"
else
  ASSEMBLY_PATH="${PROJECT_ROOT}/dataset/assembly_${data_name}_${subject_id}.pkl"
fi
PARCEL_MAPPING_PATH="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json"
SAE_LOCAL_BASE_DIR="/path/to/local_models/gemma-scope-2b-pt-res"

# ========== 可配置超参数 ==========
RUN_MODE="${RUN_MODE:-single}"                  # all / single
START_PARCEL_ID="${START_PARCEL_ID:-0}"      # single 模式生效
END_PARCEL_ID="${END_PARCEL_ID:-269}"        # single 模式生效
SKIP_EXISTING="${SKIP_EXISTING:-1}"
USE_TRAIN_TEST_SPLIT="${USE_TRAIN_TEST_SPLIT:-1}"
TEST_STORY_FROM_END_DEFAULT="${TEST_STORY_FROM_END:-25}"
EVAL_STORY_FROM_END="${EVAL_STORY_FROM_END:-0}"        # 默认关闭 evaluation split
ENABLE_SEPARATE_ALPHA_SEARCH="${ENABLE_SEPARATE_ALPHA_SEARCH:-1}"  # 1: best alpha 仅在训练集内层CV搜索
PARCEL_IDS="${PARCEL_IDS:-}"                 # all 模式可选：逗号分隔，留空则自动读取 mapping 全部 parcel

sae_paths="layer_0/width_16k/average_l0_105,layer_1/width_16k/average_l0_102,layer_2/width_16k/average_l0_141,layer_3/width_16k/average_l0_59,layer_4/width_16k/average_l0_124,layer_5/width_16k/average_l0_68,layer_6/width_16k/average_l0_70,layer_7/width_16k/average_l0_69,layer_8/width_16k/average_l0_71,layer_9/width_16k/average_l0_73,layer_10/width_16k/average_l0_77,layer_11/width_16k/average_l0_80,layer_12/width_16k/average_l0_82,layer_13/width_16k/average_l0_84,layer_14/width_16k/average_l0_84,layer_15/width_16k/average_l0_78,layer_16/width_16k/average_l0_78,layer_17/width_16k/average_l0_77,layer_18/width_16k/average_l0_74,layer_19/width_16k/average_l0_73,layer_20/width_16k/average_l0_71,layer_21/width_16k/average_l0_70,layer_22/width_16k/average_l0_72,layer_23/width_16k/average_l0_75,layer_24/width_16k/average_l0_73,layer_25/width_16k/average_l0_116"

if [[ "${RUN_MODE}" != "all" && "${RUN_MODE}" != "single" ]]; then
  echo "RUN_MODE 必须是 all 或 single，当前值: ${RUN_MODE}"
  exit 1
fi

echo "配置: subject_id=${subject_id}, dataset_type=${dataset_type}, RUN_MODE=${RUN_MODE}"
echo "配置: SKIP_EXISTING=${SKIP_EXISTING}, USE_TRAIN_TEST_SPLIT=${USE_TRAIN_TEST_SPLIT}"
echo "配置: TEST_STORY_FROM_END_DEFAULT=${TEST_STORY_FROM_END_DEFAULT}, EVAL_STORY_FROM_END=${EVAL_STORY_FROM_END}"
echo "配置: ENABLE_SEPARATE_ALPHA_SEARCH=${ENABLE_SEPARATE_ALPHA_SEARCH}"
if [[ "${RUN_MODE}" == "single" ]]; then
  echo "配置: single 模式 parcel_id 范围 [${START_PARCEL_ID}-${END_PARCEL_ID}]"
else
  echo "配置: all 模式 PARCEL_IDS='${PARCEL_IDS}' (空=从 mapping 自动读取)"
fi

for TEST_STORY_FROM_END in ${TEST_STORY_FROM_END_DEFAULT}; do
  echo "========================================="
  echo "开始处理 TEST_STORY_FROM_END=${TEST_STORY_FROM_END}"
  echo "========================================="

  MODE_TAG="allparcels"
  if [[ "${RUN_MODE}" == "single" ]]; then
    MODE_TAG="singleparcel"
  fi

  LOG_DIR="${PROJECT_ROOT}/logs/${subject_id}/saeact_${MODE_TAG}_test${TEST_STORY_FROM_END}_eval${EVAL_STORY_FROM_END}_mode_${RUN_MODE}_${RUN_STAMP}"
  result_dir="${PROJECT_ROOT}/results/${subject_id}/results_saeact_${MODE_TAG}_test${TEST_STORY_FROM_END}_eval${EVAL_STORY_FROM_END}_mode_${RUN_MODE}_${RUN_STAMP}"
  mkdir -p "${CACHE_DIR}" "${result_dir}" "${LOG_DIR}"

  if [[ "${RUN_MODE}" == "all" ]]; then
    log_file="${LOG_DIR}/all_parcels.log"
    echo "开始 all 模式训练，日志文件: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"

    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
      if python - "${result_dir}" << 'PY'
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
has_result = any(run_dir.is_dir() and ((run_dir / "metrics.pkl").exists() or (run_dir / "evaluation_metrics.pkl").exists())
                 for run_dir in result_dir.glob("run_*"))
sys.exit(0 if has_result else 1)
PY
      then
        echo "all 模式已检测到结果文件，跳过本次任务 (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
        continue
      fi
    fi

    cmd=(python "${SCRIPT}" \
      --assembly_path "${ASSEMBLY_PATH}" \
      --parcel_mapping_path "${PARCEL_MAPPING_PATH}" \
      --sae_release "gemma-scope-2b-pt-res" \
      --sae_local_base_dir "${SAE_LOCAL_BASE_DIR}" \
      --sae_paths "${sae_paths}" \
      --cache_dir "${CACHE_DIR}" \
      --results_dir "${result_dir}" \
      --logger_backend "none" \
      --parcel_level \
      --all_parcels \
      --lookback 256 \
      --dataset_type "${dataset_type}")

    if [[ -n "${PARCEL_IDS}" ]]; then
      cmd+=(--parcel_ids "${PARCEL_IDS}")
    fi
    if [[ "${ENABLE_SEPARATE_ALPHA_SEARCH}" == "1" ]]; then
      # NestedCVModel 在 train/test 模式下默认在训练集内部做 alpha 搜索；
      # 不传 eval split 时，best alpha 与 test 完全解耦。
      :
    fi
    if [[ "${USE_TRAIN_TEST_SPLIT}" == "1" ]]; then
      cmd+=(--use_train_test_split --test_story_from_end "${TEST_STORY_FROM_END}")
      if [[ "${EVAL_STORY_FROM_END}" != "0" ]]; then
        cmd+=(--eval_story_from_end "${EVAL_STORY_FROM_END}")
      fi
    fi

    if "${cmd[@]}" > "${log_file}" 2>&1; then
      echo "all 模式训练完成 (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
    else
      echo "all 模式训练失败，请查看日志: ${log_file} (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
    fi
    continue
  fi

  for parcel_id in $(seq "${START_PARCEL_ID}" "${END_PARCEL_ID}"); do
    log_file="${LOG_DIR}/${parcel_id}.log"
    echo "开始处理 parcel_id=${parcel_id}, TEST_STORY_FROM_END=${TEST_STORY_FROM_END}, 日志文件: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"

    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
      if python - "${result_dir}" "${parcel_id}" << 'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
parcel_id = int(sys.argv[2])
found = False
for hp in result_dir.glob("run_*/hyperparams.json"):
    try:
        with open(hp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("layer_idx")) == parcel_id:
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
        echo "parcel_id=${parcel_id} 已有结果，跳过 (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
        continue
      fi
    fi

    cmd=(python "${SCRIPT}" \
      --assembly_path "${ASSEMBLY_PATH}" \
      --parcel_id "${parcel_id}" \
      --single_parcel \
      --parcel_mapping_path "${PARCEL_MAPPING_PATH}" \
      --sae_release "gemma-scope-2b-pt-res" \
      --sae_local_base_dir "${SAE_LOCAL_BASE_DIR}" \
      --sae_paths "${sae_paths}" \
      --cache_dir "${CACHE_DIR}" \
      --results_dir "${result_dir}" \
      --logger_backend "none" \
      --parcel_level \
      --lookback 256 \
      --dataset_type "${dataset_type}")

    if [[ "${ENABLE_SEPARATE_ALPHA_SEARCH}" == "1" ]]; then
      # NestedCVModel 在 train/test 模式下默认在训练集内部做 alpha 搜索；
      # 不传 eval split 时，best alpha 与 test 完全解耦。
      :
    fi
    if [[ "${USE_TRAIN_TEST_SPLIT}" == "1" ]]; then
      cmd+=(--use_train_test_split --test_story_from_end "${TEST_STORY_FROM_END}")
      if [[ "${EVAL_STORY_FROM_END}" != "0" ]]; then
        cmd+=(--eval_story_from_end "${EVAL_STORY_FROM_END}")
      fi
    fi

    if "${cmd[@]}" > "${log_file}" 2>&1; then
      echo "parcel_id=${parcel_id} 处理完成 (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
    else
      echo "parcel_id=${parcel_id} 处理失败，请查看日志: ${log_file} (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
    fi
  done

  echo "single 模式所有 parcel_id (${START_PARCEL_ID}-${END_PARCEL_ID}) 处理完成 (TEST_STORY_FROM_END=${TEST_STORY_FROM_END})" | tee -a "${LOG_DIR}/batch_run.log"
done

echo "========================================="
echo "所有 TEST_STORY_FROM_END 任务处理完成"
echo "========================================="

