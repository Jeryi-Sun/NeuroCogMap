#!/usr/bin/env bash
# 简易脚本：调用 circle_graph.py 绘制 Parcel 级 circle graph 网络图
# 用法示例：
#   bash safety_explanation/hallucination/graphs/run_circle_graph.sh truthfulqa_gemma-2-2b traditional
# 第一个参数：数据集名称（例如 truthfulqa_gemma-2-2b）
# 第二个参数：方法 traditional 或 pca_concate（可选，默认 traditional）

set -e

DATASET="${1:-truthfulqa_gemma-2-2b}"
METHOD="${2:-traditional}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

TOP_PARCELS_JSON="${ROOT_DIR}/safety_explanation/hallucination/results/analysis_output/${DATASET}/parcel_level/top_anomalous_parcels.json"
ANALYSIS_COMPLETE_JSON="${ROOT_DIR}/safety_explanation/hallucination/results/analysis_output/${DATASET}/parcel_level/parcel_level_analysis_complete.json"

DATA_DIR="${ROOT_DIR}/safety_explanation/hallucination/graphs/data/${DATASET}/parcel_level"
FIG_DIR="${ROOT_DIR}/safety_explanation/hallucination/graphs/figures/${DATASET}/parcel_level"

mkdir -p "${DATA_DIR}" "${FIG_DIR}"

OUTPUT_DATA="${DATA_DIR}/circle_graph_node2edge_${METHOD}.json"
OUTPUT_FIG="${FIG_DIR}/circle_graph_node2edge_${METHOD}.pdf"

echo "[INFO] DATASET = ${DATASET}"
echo "[INFO] METHOD  = ${METHOD}"
echo "[INFO] TOP_PARCELS_JSON       = ${TOP_PARCELS_JSON}"
echo "[INFO] ANALYSIS_COMPLETE_JSON = ${ANALYSIS_COMPLETE_JSON}"
echo "[INFO] OUTPUT_DATA            = ${OUTPUT_DATA}"
echo "[INFO] OUTPUT_FIG             = ${OUTPUT_FIG}"

python "${ROOT_DIR}/safety_explanation/hallucination/graphs/circle_graph_node2edge_no.py" \
  --top_parcels_json "${TOP_PARCELS_JSON}" \
  --analysis_complete_json "${ANALYSIS_COMPLETE_JSON}" \
  --output_data "${OUTPUT_DATA}" \
  --output_fig "${OUTPUT_FIG}" \
  --method "${METHOD}" \
  --p_threshold 0.05 \
  --max_edges 300 \

