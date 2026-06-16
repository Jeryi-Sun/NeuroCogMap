#!/bin/bash
#
# 功能：合并多个 uts 下的 human-LLM 对齐结果（top_human_parcels_per_llm_sim06.csv），
#      筛选出在同一 human_parcel 下出现的相同 llm_parcel，并将不同 uts 的 rank/accuracy/similarity
#      分别写入独立列，方便后续可视化与统计分析。
#
# 使用示例（当前默认示例为 uts02 与 uts03）：
#   bash run_merge_llm_human_alignments_across_uts.sh
#
# 如需修改输入/输出路径或 uts 标签，可手动编辑下面几行配置。

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

# ===== 用户可修改配置区域 =====

INPUT_UTS02="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/draw_result/uts02/adollshouse/saeact/top_human_parcels_per_llm.csv"
INPUT_UTS03="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/draw_result/uts03/adollshouse/saeact/top_human_parcels_per_llm.csv"

# uts 标签，可按需修改（需要与 INPUT 顺序一一对应）
LABEL_UTS02="uts02"
LABEL_UTS03="uts03"

OUTPUT_CSV="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/draw_result/adollshouse/saeact/merged_top_human_parcels_per_llm_uts02_uts03.csv"

# 是否允许覆盖已存在的输出（true/false），默认为 false 更安全
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-false}"

# ===== 配置结束 =====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="${SCRIPT_DIR}/merge_llm_human_alignments_across_uts.py"

if [[ ! -f "${MERGE_SCRIPT}" ]]; then
  echo "[Error] 找不到合并脚本：${MERGE_SCRIPT}" >&2
  exit 1
fi

ARGS=(
  --inputs
  "${INPUT_UTS02}"
  "${INPUT_UTS03}"
  --labels
  "${LABEL_UTS02}"
  "${LABEL_UTS03}"
  --output
  "${OUTPUT_CSV}"
)

if [[ "${ALLOW_OVERWRITE}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

echo "[Info] 使用 Python 解释器：${PYTHON_BIN}"
echo "[Info] 调用脚本：${MERGE_SCRIPT}"
echo "[Info] 输出文件：${OUTPUT_CSV}"

set -x
"${PYTHON_BIN}" "${MERGE_SCRIPT}" "${ARGS[@]}"
set +x

echo "[Info] 合并完成。"

