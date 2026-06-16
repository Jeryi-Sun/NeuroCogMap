#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 jailbreak 结果绘制"由异常 edge 选 node"的 Parcel/Capability 级
#   circle graph，复用 hallucination 下 circle_graph_edge2node.py，
#   使用 Correct / Incorrect 标签。
#
# 用法：
#   bash safety_explanation/jailbreak/analysis_graphs/run_circle_graph_edge2node.sh
#
# 说明：
#   - 可通过修改脚本内的 LEVEL 变量切换 parcel 或 capability 级别
#   - 可通过修改 EDGE_SELECTION 变量切换 edge 选择策略
#   - FORCE_RECOMPUTE: 控制是否强制重算中间数据（OUTPUT_DATA）
#   - FORCE_REDRAW: 控制是否强制重绘最终图像（OUTPUT_FIG）
# ============================================================

set -e

LEVEL="capability"
EDGE_SELECTION="half_signed"

# 是否强制重算中间数据（覆盖已有 OUTPUT_DATA）：true / false
#   - false：如 OUTPUT_DATA 已存在，则使用已有数据（不重新计算）
#   - true ：强制重新计算中间数据（即使 OUTPUT_DATA 已存在）
FORCE_RECOMPUTE=true

# 是否强制重绘最终图像（覆盖已有 OUTPUT_FIG）：true / false
#   - false：如 OUTPUT_FIG 已存在，则跳过绘图（即使中间数据更新了也不重绘）
#   - true ：强制重新绘制图像（即使 OUTPUT_FIG 已存在）
FORCE_REDRAW=true

# 是否为同色连通子图绘制凸包强调：1=开启，0=关闭
SHOW_HULL=1

if [ "${LEVEL}" = "capability" ]; then
    LEVEL_NAME="capability_level"
    ANOMALOUS_JSON_NAME="anomalous_capability_connections.json"
else
    LEVEL_NAME="parcel_level"
    ANOMALOUS_JSON_NAME="anomalous_connections.json"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ANALYSIS_OUT="${ROOT_DIR}/safety_explanation/jailbreak/results/analysis_output"
SCRIPT="${ROOT_DIR}/safety_explanation/hallucination/analysis_graphs/circle_graph_edge2node.py"
DATA_BASE="${ROOT_DIR}/safety_explanation/jailbreak/analysis_graphs/data"
FIG_BASE="${ROOT_DIR}/safety_explanation/jailbreak/analysis_graphs/figures"

GROUP_HIGH_LABEL="Refuse-Failed"
GROUP_LOW_LABEL="Refuse-Success"

DATASETS=(
    "JBB-Behaviors_gemma-2-9b-it"
    "AdvBench_gemma-2-9b-it"
    "JBB-Behaviors_gemma-2-2b"
    "AdvBench_gemma-2-2b"
)

echo "[INFO] 级别类型: ${LEVEL} (${LEVEL_NAME})"
echo "[INFO] Edge 选择策略: ${EDGE_SELECTION}"
echo "[INFO] 将依次处理 ${#DATASETS[@]} 个数据集"
echo ""

for DATASET in "${DATASETS[@]}"; do
  ANOMALOUS_EDGES_JSON="${ANALYSIS_OUT}/${DATASET}/${LEVEL_NAME}/${ANOMALOUS_JSON_NAME}"
  if [ "${LEVEL}" = "capability" ]; then
    ANALYSIS_COMPLETE_JSON="${ANALYSIS_OUT}/${DATASET}/${LEVEL_NAME}/capability_level_analysis_complete.json"
  else
    ANALYSIS_COMPLETE_JSON="${ANALYSIS_OUT}/${DATASET}/${LEVEL_NAME}/parcel_level_analysis_complete.json"
  fi

  DATA_DIR="${DATA_BASE}/${DATASET}/${LEVEL_NAME}"
  FIG_DIR="${FIG_BASE}/${DATASET}/${LEVEL_NAME}"
  mkdir -p "${DATA_DIR}" "${FIG_DIR}"

  # 在文件名中编码 edge_selection，方便同时保存多种 edge 选择策略的结果
  OUTPUT_DATA="${DATA_DIR}/circle_graph_edge2node_${EDGE_SELECTION}.json"
  OUTPUT_FIG="${FIG_DIR}/circle_graph_edge2node_${EDGE_SELECTION}.pdf"

  # 检查最终图像是否已存在且不需要重绘
  SKIP_DRAW=false
  if [ -f "${OUTPUT_FIG}" ] && [ "${FORCE_REDRAW}" != "true" ]; then
    SKIP_DRAW=true
    echo "[INFO] 检测到已有最终图像 ${OUTPUT_FIG}，FORCE_REDRAW=false，将跳过绘图"
  fi

  # 如果中间数据已存在且不需要重算，且图像也已存在且不需要重绘，则完全跳过
  if [ -f "${OUTPUT_DATA}" ] && [ "${FORCE_RECOMPUTE}" != "true" ] && [ "${SKIP_DRAW}" = "true" ]; then
    echo "[INFO] 跳过 ${DATASET} (${LEVEL_NAME}, ${EDGE_SELECTION})：中间数据和最终图像均已存在，且未开启强制重算/重绘"
    echo ""
    continue
  fi

  # 如果强制重算中间数据，删除旧数据文件
  if [ "${FORCE_RECOMPUTE}" = "true" ] && [ -f "${OUTPUT_DATA}" ]; then
    echo "[INFO] FORCE_RECOMPUTE=true，删除已有中间数据: ${OUTPUT_DATA}"
    rm -f "${OUTPUT_DATA}"
  fi

  # 如果强制重绘，删除旧图像文件
  if [ "${FORCE_REDRAW}" = "true" ] && [ -f "${OUTPUT_FIG}" ]; then
    echo "[INFO] FORCE_REDRAW=true，删除已有最终图像: ${OUTPUT_FIG}"
    rm -f "${OUTPUT_FIG}"
  fi

  echo "========================================"
  echo "[INFO] DATASET               = ${DATASET}"
  echo "[INFO] LEVEL                 = ${LEVEL}"
  echo "[INFO] EDGE_SELECTION        = ${EDGE_SELECTION}"
  echo "[INFO] FORCE_RECOMPUTE       = ${FORCE_RECOMPUTE}"
  echo "[INFO] FORCE_REDRAW          = ${FORCE_REDRAW}"
  echo "[INFO] ANOMALOUS_EDGES_JSON  = ${ANOMALOUS_EDGES_JSON}"
  echo "[INFO] ANALYSIS_COMPLETE_JSON= ${ANALYSIS_COMPLETE_JSON}"
  echo "[INFO] OUTPUT_DATA           = ${OUTPUT_DATA}"
  echo "[INFO] OUTPUT_FIG            = ${OUTPUT_FIG}"
  echo "========================================"

  # 检查输入文件是否存在
  if [ ! -f "${ANOMALOUS_EDGES_JSON}" ]; then
    echo "[WARN] 跳过 ${DATASET}：找不到 ${ANOMALOUS_EDGES_JSON}"
    echo ""
    continue
  fi

  if [ ! -f "${ANALYSIS_COMPLETE_JSON}" ]; then
    echo "[WARN] 跳过 ${DATASET}：找不到 ${ANALYSIS_COMPLETE_JSON}"
    echo ""
    continue
  fi

  # 构建 Python 脚本参数
  PYTHON_ARGS=(
    --anomalous_edges_json "${ANOMALOUS_EDGES_JSON}"
    --analysis_complete_json "${ANALYSIS_COMPLETE_JSON}"
    --output_data "${OUTPUT_DATA}"
    --output_fig "${OUTPUT_FIG}"
    --p_threshold 0.05
    --max_edges 10
    --edge_selection "${EDGE_SELECTION}"
    --level "${LEVEL}"
    --group_high_label "${GROUP_HIGH_LABEL}"
    --group_low_label "${GROUP_LOW_LABEL}"
    --show_hull "${SHOW_HULL}"
  )

  # 根据 FORCE_RECOMPUTE 决定是否传递 --refresh_data
  if [ "${FORCE_RECOMPUTE}" = "true" ]; then
    PYTHON_ARGS+=(--refresh_data)
  fi

  # 根据 FORCE_REDRAW 决定是否传递 --skip_if_exists
  # 如果 FORCE_REDRAW=false 且图像已存在，则跳过绘图
  if [ "${FORCE_REDRAW}" != "true" ] && [ -f "${OUTPUT_FIG}" ]; then
    PYTHON_ARGS+=(--skip_if_exists)
  fi

  python "${SCRIPT}" "${PYTHON_ARGS[@]}"

  echo "[INFO] 完成: ${DATASET}"
  echo ""
done

echo "[INFO] 所有指定数据集的 circle graph (edge2node) 已完成。"
