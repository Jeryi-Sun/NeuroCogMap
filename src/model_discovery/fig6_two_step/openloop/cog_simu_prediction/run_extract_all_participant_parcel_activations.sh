#!/usr/bin/env bash
# 提取所有参与者的 parcel 激活特征
# 从JSONL文件中提取所有参与者数据，然后提取每个participant的parcel激活
# 将所有trial的激活合并为一个列表，保存格式为 {participant_id: [activation]}
# 支持处理两个输入文件并分别保存结果
#
# 使用方法:
#   1. 不传参数：自动扫描所有 *_reformatted.jsonl 文件并处理
#      ./run_extract_all_participant_parcel_activations.sh
#
#   2. 传入指定数据集：只处理指定的数据集
#      ./run_extract_all_participant_parcel_activations.sh kool2016when_exp2 kool2017cost_exp2
set -euo pipefail

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction"
SCRIPT="${PROJECT_ROOT}/extract_all_participant_parcel_activations.py"
LOG_DIR="${PROJECT_ROOT}/logs/extract_all_participant_parcel_activations_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PROJECT_ROOT}/results/activations"
DATASET_DIR="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/dataset"

# 控制是否跳过已存在的结果文件，设置为 1 表示跳过，0 表示不跳过
SKIP_EXISTING=${SKIP_EXISTING:-0}

# 模型配置
MODEL_NAME="${MODEL_NAME:-google/gemma-2-2b}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# SAE 相关路径
PARCEL_MAPPING_PATH="${PARCEL_MAPPING_PATH:-/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json}"
SAE_LOCAL_BASE_DIR="${SAE_LOCAL_BASE_DIR:-/path/to/local_models/gemma-scope-2b-pt-res}"
SAE_RELEASE="${SAE_RELEASE:-gemma-scope-2b-pt-res}"

# SAE 路径列表（可选，如果为空则使用默认值）
SAE_PATHS="${SAE_PATHS:-layer_0/width_16k/average_l0_105,layer_1/width_16k/average_l0_102,layer_2/width_16k/average_l0_141,layer_3/width_16k/average_l0_59,layer_4/width_16k/average_l0_124,layer_5/width_16k/average_l0_68,layer_6/width_16k/average_l0_70,layer_7/width_16k/average_l0_69,layer_8/width_16k/average_l0_71,layer_9/width_16k/average_l0_73,layer_10/width_16k/average_l0_77,layer_11/width_16k/average_l0_80,layer_12/width_16k/average_l0_82,layer_13/width_16k/average_l0_84,layer_14/width_16k/average_l0_84,layer_15/width_16k/average_l0_78,layer_16/width_16k/average_l0_78,layer_17/width_16k/average_l0_77,layer_18/width_16k/average_l0_74,layer_19/width_16k/average_l0_73,layer_20/width_16k/average_l0_71,layer_21/width_16k/average_l0_70,layer_22/width_16k/average_l0_72,layer_23/width_16k/average_l0_75,layer_24/width_16k/average_l0_73,layer_25/width_16k/average_l0_116}"

# 创建日志目录（在扫描数据集之前）
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 数据集列表
# 如果通过命令行参数传入，使用传入的数据集；否则自动扫描 train 和 test 目录
if [ $# -gt 0 ]; then
    # 从命令行参数读取数据集列表（同时检查 train 和 test 目录）
    declare -a TRAIN_DATASETS
    declare -a TEST_DATASETS
    
    for DATASET in "$@"; do
        if [ -f "$DATASET_DIR/train/${DATASET}_reformatted.jsonl" ]; then
            TRAIN_DATASETS+=("$DATASET")
        fi
        if [ -f "$DATASET_DIR/test/${DATASET}_reformatted.jsonl" ]; then
            TEST_DATASETS+=("$DATASET")
        fi
    done
    
    echo "使用命令行参数指定的数据集: $@" | tee -a "${LOG_DIR}/batch_run.log"
    echo "train 数据集: ${TRAIN_DATASETS[@]}" | tee -a "${LOG_DIR}/batch_run.log"
    echo "test 数据集: ${TEST_DATASETS[@]}" | tee -a "${LOG_DIR}/batch_run.log"
else
    # 自动扫描 train 和 test 目录，找到所有 *_reformatted.jsonl 文件
    echo "未指定数据集，自动扫描 train 和 test 目录..." | tee -a "${LOG_DIR}/batch_run.log"
    
    # 扫描 train 目录
    declare -A TRAIN_DATASET_SET
    if [ -d "$DATASET_DIR/train" ]; then
        while IFS= read -r file; do
            dataset_name=$(basename "$file" | sed 's/_reformatted\.jsonl$//')
            if [ -n "$dataset_name" ]; then
                TRAIN_DATASET_SET["$dataset_name"]=1
            fi
        done < <(find "$DATASET_DIR/train" -name "*_reformatted.jsonl" -type f | sort)
    fi
    
    # 扫描 test 目录
    declare -A TEST_DATASET_SET
    if [ -d "$DATASET_DIR/test" ]; then
        while IFS= read -r file; do
            dataset_name=$(basename "$file" | sed 's/_reformatted\.jsonl$//')
            if [ -n "$dataset_name" ]; then
                TEST_DATASET_SET["$dataset_name"]=1
            fi
        done < <(find "$DATASET_DIR/test" -name "*_reformatted.jsonl" -type f | sort)
    fi
    
    # 将关联数组的键转换为数组（已排序）
    TRAIN_DATASETS=($(printf '%s\n' "${!TRAIN_DATASET_SET[@]}" | sort))
    TEST_DATASETS=($(printf '%s\n' "${!TEST_DATASET_SET[@]}" | sort))
    
    echo "找到 ${#TRAIN_DATASETS[@]} 个 train 数据集: ${TRAIN_DATASETS[@]}" | tee -a "${LOG_DIR}/batch_run.log"
    echo "找到 ${#TEST_DATASETS[@]} 个 test 数据集: ${TEST_DATASETS[@]}" | tee -a "${LOG_DIR}/batch_run.log"
fi

echo "开始提取所有参与者的 parcel 激活特征" | tee -a "${LOG_DIR}/batch_run.log"
echo "模型: ${MODEL_NAME}" | tee -a "${LOG_DIR}/batch_run.log"
echo "日志目录: ${LOG_DIR}" | tee -a "${LOG_DIR}/batch_run.log"

# 先处理 train 目录的数据集
for DATASET in "${TRAIN_DATASETS[@]}"; do
  INPUT_FILE="$DATASET_DIR/train/${DATASET}_reformatted.jsonl"
  
  # 从输入文件名提取输出文件名，包含 train/test 文件夹信息
  PARENT_DIR="train"
  FILENAME="${DATASET}_reformatted"
  INPUT_BASENAME="${PARENT_DIR}/${DATASET}_reformatted"
  OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_BASENAME}_parcel_activations.json"
  log_file="${LOG_DIR}/${INPUT_BASENAME}.log"
  
  # 确保输出目录和日志目录存在
  mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${log_file}")"
  
  echo "" | tee -a "${LOG_DIR}/batch_run.log"
  echo "处理文件: ${INPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
  echo "输出文件: ${OUTPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
  
  # 检查输出文件是否已存在
  if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${OUTPUT_FILE}" ]; then
    echo "跳过处理，输出文件已存在: ${OUTPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
    continue
  fi
  
  # 检查输入文件是否存在
  if [ ! -f "${INPUT_FILE}" ]; then
    echo "错误: 输入文件不存在: ${INPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
    continue
  fi
  
  # 构建 Python 命令
  python_cmd=(
    python "${SCRIPT}"
    --input "${INPUT_FILE}"
    --output "${OUTPUT_FILE}"
    --model-name "${MODEL_NAME}"
    --parcel-mapping-path "${PARCEL_MAPPING_PATH}"
    --sae-release "${SAE_RELEASE}"
    --sae-local-base-dir "${SAE_LOCAL_BASE_DIR}"
    --max-tokens "${MAX_TOKENS}"
  )
  
  # 如果设置了 SAE_PATHS，则添加该参数
  if [ -n "${SAE_PATHS}" ]; then
    python_cmd+=(--sae-paths "${SAE_PATHS}")
  fi
  
  # 如果设置了 SKIP_EXISTING，则添加该参数
  if [ "${SKIP_EXISTING}" -eq 1 ]; then
    python_cmd+=(--skip-existing)
  fi
  
  # 执行 Python 脚本
  echo "执行命令: ${python_cmd[*]}" | tee -a "${LOG_DIR}/batch_run.log"
  "${python_cmd[@]}" > "${log_file}" 2>&1
  
  if [ $? -eq 0 ]; then
    echo "处理完成，结果已保存到: ${OUTPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
  else
    echo "处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
    exit 1
  fi
done

# 再处理 test 目录的数据集
for DATASET in "${TEST_DATASETS[@]}"; do
  INPUT_FILE="$DATASET_DIR/test/${DATASET}_reformatted.jsonl"
  
  # 从输入文件名提取输出文件名，包含 train/test 文件夹信息
  PARENT_DIR="test"
  FILENAME="${DATASET}_reformatted"
  INPUT_BASENAME="${PARENT_DIR}/${DATASET}_reformatted"
  OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_BASENAME}_parcel_activations.json"
  log_file="${LOG_DIR}/${INPUT_BASENAME}.log"
  
  # 确保输出目录和日志目录存在
  mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${log_file}")"
  
  echo "" | tee -a "${LOG_DIR}/batch_run.log"
  echo "处理文件: ${INPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
  echo "输出文件: ${OUTPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
  
  # 检查输出文件是否已存在
  if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${OUTPUT_FILE}" ]; then
    echo "跳过处理，输出文件已存在: ${OUTPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
    continue
  fi
  
  # 检查输入文件是否存在
  if [ ! -f "${INPUT_FILE}" ]; then
    echo "错误: 输入文件不存在: ${INPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
    continue
  fi
  
  # 构建 Python 命令
  python_cmd=(
    python "${SCRIPT}"
    --input "${INPUT_FILE}"
    --output "${OUTPUT_FILE}"
    --model-name "${MODEL_NAME}"
    --parcel-mapping-path "${PARCEL_MAPPING_PATH}"
    --sae-release "${SAE_RELEASE}"
    --sae-local-base-dir "${SAE_LOCAL_BASE_DIR}"
    --max-tokens "${MAX_TOKENS}"
  )
  
  # 如果设置了 SAE_PATHS，则添加该参数
  if [ -n "${SAE_PATHS}" ]; then
    python_cmd+=(--sae-paths "${SAE_PATHS}")
  fi
  
  # 如果设置了 SKIP_EXISTING，则添加该参数
  if [ "${SKIP_EXISTING}" -eq 1 ]; then
    python_cmd+=(--skip-existing)
  fi
  
  # 执行 Python 脚本
  echo "执行命令: ${python_cmd[*]}" | tee -a "${LOG_DIR}/batch_run.log"
  "${python_cmd[@]}" > "${log_file}" 2>&1
  
  if [ $? -eq 0 ]; then
    echo "处理完成，结果已保存到: ${OUTPUT_FILE}" | tee -a "${LOG_DIR}/batch_run.log"
  else
    echo "处理失败，请查看日志: ${log_file}" | tee -a "${LOG_DIR}/batch_run.log"
    exit 1
  fi
done

echo "" | tee -a "${LOG_DIR}/batch_run.log"
echo "所有处理完成" | tee -a "${LOG_DIR}/batch_run.log"

