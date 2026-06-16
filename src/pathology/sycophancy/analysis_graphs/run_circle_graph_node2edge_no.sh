#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 sycophancy 结果绘制 Parcel 级 circle graph（由 top parcels 选节点，
#   再画连接边），复用 hallucination 下 circle_graph_node2edge_no.py，
#   使用 Correct / Incorrect 标签。
#
# 用法：
#   bash safety_explanation/sycophancy/analysis_graphs/run_circle_graph_node2edge_no.sh
# 或指定数据集与方法：
#   bash .../run_circle_graph_node2edge_no.sh answer_gemma-2-9b-it traditional
#
# 说明：
#   - 当前脚本仅支持 parcel 级别（node2edge 使用 top_anomalous_parcels.json）
#   - 可在 Python 命令行加 --skip_if_exists 实现存在则跳过
# ============================================================

set -e

METHOD="${2:-traditional}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ANALYSIS_OUT="${ROOT_DIR}/safety_explanation/sycophancy/results/analysis_output"
SCRIPT="${ROOT_DIR}/safety_explanation/hallucination/analysis_graphs/circle_graph_node2edge_no.py"
LEVEL_NAME="parcel_level"
GROUP_HIGH_LABEL="Incorrect"
GROUP_LOW_LABEL="Correct"

DATASETS=(
    "answer_gemma-2-9b-it"
    "feedback_gemma-2-2b"
)

run_one() {
    local DATASET="$1"
    TOP_PARCELS_JSON="${ANALYSIS_OUT}/${DATASET}/${LEVEL_NAME}/top_anomalous_parcels.json"
    ANALYSIS_COMPLETE_JSON="${ANALYSIS_OUT}/${DATASET}/${LEVEL_NAME}/parcel_level_analysis_complete.json"
    DATA_DIR="${ROOT_DIR}/safety_explanation/sycophancy/analysis_graphs/data/${DATASET}/${LEVEL_NAME}"
    FIG_DIR="${ROOT_DIR}/safety_explanation/sycophancy/analysis_graphs/figures/${DATASET}/${LEVEL_NAME}"
    mkdir -p "${DATA_DIR}" "${FIG_DIR}"
    OUTPUT_DATA="${DATA_DIR}/circle_graph_node2edge_${METHOD}.json"
    OUTPUT_FIG="${FIG_DIR}/circle_graph_node2edge_${METHOD}.pdf"

    if [ ! -f "${TOP_PARCELS_JSON}" ]; then
        echo "[WARN] 跳过 ${DATASET}：找不到 ${TOP_PARCELS_JSON}"
        return 0
    fi
    if [ ! -f "${ANALYSIS_COMPLETE_JSON}" ]; then
        echo "[WARN] 跳过 ${DATASET}：找不到 ${ANALYSIS_COMPLETE_JSON}"
        return 0
    fi

    echo "[INFO] DATASET = ${DATASET}, METHOD = ${METHOD}"
    python "${SCRIPT}" \
      --top_parcels_json "${TOP_PARCELS_JSON}" \
      --analysis_complete_json "${ANALYSIS_COMPLETE_JSON}" \
      --output_data "${OUTPUT_DATA}" \
      --output_fig "${OUTPUT_FIG}" \
      --method "${METHOD}" \
      --p_threshold 0.05 \
      --max_edges 300 \
      --group_high_label "${GROUP_HIGH_LABEL}" \
      --group_low_label "${GROUP_LOW_LABEL}"
    echo "[INFO] 完成: ${DATASET}"
    echo ""
}

if [ -n "${1:-}" ]; then
    run_one "$1"
else
    echo "[INFO] 将依次处理 ${#DATASETS[@]} 个数据集 (METHOD = ${METHOD})"
    for ds in "${DATASETS[@]}"; do
        run_one "${ds}"
    done
    echo "[INFO] 所有 node2edge circle graph 已完成。"
fi
