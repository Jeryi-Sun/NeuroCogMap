#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 jailbreak 结果中的 parcel-level anomalous_connections.json
#   连边添加结构连接的正负标记（基于 parcel_connection_matrix.csv 中 M[i,j] 的符号）：
#     - struct_connectivity_sign  (positive / negative / zero)
#     - struct_connectivity_value (结构连接数值)
#   并在同目录下生成新的 JSON：
#     anomalous_connections_with_struct_sign.json
#
# 用法：
#   bash safety_explanation/jailbreak/analysis_graphs/run_annotate_struct_sign_in_anomalous_connections.sh
#
# 说明：
#   - DATASETS：可配置要处理的模型数据集（与 run_circle_graph_edge2node.sh 风格一致）
#   - ANALYSIS_OUT：jailbreak 结果根目录（每个 DATASET 一个子目录）
#   - PARCEL_MATRIX_CSV：parcel-level 结构连接矩阵 CSV
#   - SKIP_IF_EXISTS：若目标输出已存在时是否跳过（true/false）
#
#   该脚本内部通过 for 循环按 DATASET 逐个调用 Python：
#     annotate_struct_sign_in_anomalous_connections.py
# ============================================================

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

ANALYSIS_OUT="${ROOT_DIR}/safety_explanation/jailbreak/results/analysis_output"

# 结构连接矩阵 CSV 路径（与 hallucination / fairness_bias 中保持一致）
PARCEL_MATRIX_CSV="${ROOT_DIR}/neural_area/global_weight/outputs/parcel_connection_matrix.csv"

# 若为 true，则当 anomalous_connections_with_struct_sign.json 已存在时跳过该文件
SKIP_IF_EXISTS=true

# ===== 可配置：要处理的模型数据集（与 run_circle_graph_edge2node.sh 对齐）=====
DATASETS=(
  "JBB-Behaviors_gemma-2-9b-it"
  "AdvBench_gemma-2-9b-it"
  "JBB-Behaviors_gemma-2-2b"
  "AdvBench_gemma-2-2b"
)

echo "[INFO] ANALYSIS_OUT       = ${ANALYSIS_OUT}"
echo "[INFO] PARCEL_MATRIX_CSV  = ${PARCEL_MATRIX_CSV}"
echo "[INFO] SKIP_IF_EXISTS     = ${SKIP_IF_EXISTS}"
echo "[INFO] 将依次处理 ${#DATASETS[@]} 个数据集"
echo ""

for DATASET in "${DATASETS[@]}"; do
  ROOT_RESULTS_DIR="${ANALYSIS_OUT}/${DATASET}/parcel_level"

  if [ ! -d "${ROOT_RESULTS_DIR}" ]; then
    echo "[WARN] 跳过 ${DATASET}：目录不存在 ${ROOT_RESULTS_DIR}"
    echo ""
    continue
  fi

  echo "========================================"
  echo "[INFO] DATASET           = ${DATASET}"
  echo "[INFO] ROOT_RESULTS_DIR  = ${ROOT_RESULTS_DIR}"
  echo "========================================"

  PYTHON_ARGS=(
    --root_results_dir "${ROOT_RESULTS_DIR}"
    --parcel_matrix_csv "${PARCEL_MATRIX_CSV}"
  )

  if [ "${SKIP_IF_EXISTS}" = "true" ]; then
    PYTHON_ARGS+=(--skip_if_exists)
  fi

  python "${ROOT_DIR}/safety_explanation/fairness_bias/analysis_graphs/annotate_struct_sign_in_anomalous_connections.py" "${PYTHON_ARGS[@]}"

  echo "[INFO] 结构连接正负标记处理完成: ${DATASET}"
  echo ""
done

echo "[INFO] 所有指定 jailbreak 数据集的结构连接正负标记批处理完成。"

