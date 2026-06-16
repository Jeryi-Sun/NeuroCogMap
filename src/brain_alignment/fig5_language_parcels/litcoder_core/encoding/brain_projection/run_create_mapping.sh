#!/usr/bin/env bash

# 功能简介：
#   1. 检查并加载 LanA 左、右半球概率体积或球面 per-vertex 数据
#   2. 调用 create_roi_masks.py，将概率图映射到 fsaverage5，并生成语言网络 ROI（默认前 10% 顶点）
#   3. 可通过环境变量启用 neuromaps 球面重采样或跳过已存在结果
# 使用说明：
#   - 先将官方 LanA 数据（LH_LanA_n804.nii.gz、RH_LanA_n804.nii.gz）放在 DATA_DIR 指定目录
#   - 如需修改输出路径或阈值，可调整下方变量

set -euo pipefail

PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core"
DATA_DIR="${PROJECT_ROOT}/dataset/FS"
SCRIPT="${PROJECT_ROOT}/encoding/brain_projection/create_roi_masks.py"
OUTPUT_DIR="${PROJECT_ROOT}/results/roi_masks/language"
TOP_PERCENT=10

LEFT_VOL="${DATA_DIR}/LH_LanA_n804.nii.gz"
RIGHT_VOL="${DATA_DIR}/RH_LanA_n804.nii.gz"
USE_NEUROMAPS=${USE_NEUROMAPS:-1}
NEUROMAPS_DENSITY=${NEUROMAPS_DENSITY:-"10k"}
NEUROMAPS_METHOD=${NEUROMAPS_METHOD:-"linear"}
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-0}

if [[ ! -f "${LEFT_VOL}" || ! -f "${RIGHT_VOL}" ]]; then
  echo "[ERROR] 未找到 LanA 体积文件，请确认路径："
  echo "  ${LEFT_VOL}"
  echo "  ${RIGHT_VOL}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

LEFT_MASK="${OUTPUT_DIR}/language_mask_left.npy"
RIGHT_MASK="${OUTPUT_DIR}/language_mask_right.npy"
if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${LEFT_MASK}" && -f "${RIGHT_MASK}" ]]; then
  echo "[INFO] 检测到语言 ROI 掩码已存在，跳过重新计算。"
  echo "  如需重新运行，请设置 SKIP_IF_EXISTS=0 或删除输出文件。"
  exit 0
fi

echo "[INFO] USE_NEUROMAPS=${USE_NEUROMAPS} (density=${NEUROMAPS_DENSITY}, method=${NEUROMAPS_METHOD})"
PY_ARGS=(
  --lana_left_path "${LEFT_VOL}"
  --lana_right_path "${RIGHT_VOL}"
  --output_dir "${OUTPUT_DIR}"
  --top_percent "${TOP_PERCENT}"
  --visualize
)

if [[ "${USE_NEUROMAPS}" == "1" ]]; then
  PY_ARGS+=(
    --use_neuromaps
    --neuromaps_density "${NEUROMAPS_DENSITY}"
    --neuromaps_method "${NEUROMAPS_METHOD}"
  )
fi

echo "[INFO] 运行 create_roi_masks.py 生成语言网络 ROI"
python "${SCRIPT}" "${PY_ARGS[@]}"

echo "[DONE] LanA 语言网络 ROI 已生成，输出目录：${OUTPUT_DIR}"
