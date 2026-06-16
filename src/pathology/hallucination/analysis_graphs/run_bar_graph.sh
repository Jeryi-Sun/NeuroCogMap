#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   调用 bar_graph.py 绘制 activation_diff 条形图
#   支持两种级别（通过脚本内部超参数控制）：parcel / capability
#   当前脚本会在内部 for 循环依次处理若干数据集（无需外部传参）。
#
# 用法示例：
#   直接运行：
#     bash safety_explanation/hallucination/graphs/run_bar_graph.sh
#
# 说明：
#   - 默认仅绘制显著条目（p < 0.05 且 is_significant=true）
#   - 默认若输出 PDF 已存在则跳过（可在脚本内部修改 SKIP_IF_EXISTS 变量）
#   - 默认使用缓存数据（如果存在），如果不存在则从原始文件读取并保存缓存
#   - 如需强制重新计算（忽略缓存），设置 FORCE_RECOMPUTE="--force_recompute"
# ============================================================

set -e

# 是否在结果已存在时跳过（默认开启跳过）
SKIP_IF_EXISTS="" # 如需开启跳过逻辑，改为 "--skip_if_exists"

# 是否强制重新计算（忽略缓存，从原始文件重新读取并处理）
FORCE_RECOMPUTE="--force_recompute" # 如需强制重写，改为 "--force_recompute"

# 级别超参数：parcel 或 capability
LEVEL="capability"  # 可改为 "parcel"

if [ "${LEVEL}" == "capability" ]; then
    LEVEL_NAME="capability_level"
    JSON_NAME="top_anomalous_capabilities.json"
    LEVEL_ARG="--level capability"
    TITLE_PREFIX="Capability"
else
    LEVEL_NAME="parcel_level"
    JSON_NAME="top_anomalous_parcels.json"
    LEVEL_ARG="--level parcel"
    TITLE_PREFIX="Parcel"
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ANALYSIS_DIR="${ROOT_DIR}/safety_explanation/hallucination/results/analysis_output"
SCRIPT="${ROOT_DIR}/safety_explanation/hallucination/analysis_graphs/bar_graph.py"

# 处理单个数据集的函数
process_dataset() {
    local ds="$1"
    local extra="$2"
    TOP_PARCELS_JSON="${ANALYSIS_DIR}/${ds}/${LEVEL_NAME}/${JSON_NAME}"
    FIG_DIR="${ROOT_DIR}/safety_explanation/hallucination/analysis_graphs/figures/${ds}/${LEVEL_NAME}"
    mkdir -p "${FIG_DIR}"

    OUTPUT_FIG="${FIG_DIR}/bar_graph_activation_diff_${LEVEL}.pdf"
    
    # 缓存数据路径
    CACHE_DATA_DIR="${ROOT_DIR}/safety_explanation/hallucination/analysis_graphs/data/${ds}/${LEVEL_NAME}"
    CACHE_DATA_FILE="${CACHE_DATA_DIR}/plot_data.json"

    # 检查输入文件是否存在（如果强制重写或缓存不存在时需要）
    if [ ! -f "${TOP_PARCELS_JSON}" ]; then
        if [ -n "${FORCE_RECOMPUTE}" ]; then
            echo "[WARN] 跳过 ${ds}：强制重写但找不到 ${TOP_PARCELS_JSON}"
            return 0
        elif [ ! -f "${CACHE_DATA_FILE}" ]; then
            echo "[WARN] 跳过 ${ds}：找不到原始文件 ${TOP_PARCELS_JSON} 且缓存文件 ${CACHE_DATA_FILE} 也不存在"
            return 0
        fi
    fi

    echo "========================================"
    echo "[INFO] LEVEL            = ${LEVEL} (${LEVEL_NAME})"
    echo "[INFO] DATASET          = ${ds}"
    if [ -f "${CACHE_DATA_FILE}" ] && [ -z "${FORCE_RECOMPUTE}" ]; then
        echo "[INFO] CACHE_DATA       = ${CACHE_DATA_FILE} (将使用缓存)"
    else
        echo "[INFO] TOP_JSON         = ${TOP_PARCELS_JSON}"
        if [ -n "${FORCE_RECOMPUTE}" ]; then
            echo "[INFO] FORCE_RECOMPUTE = 是（将忽略缓存）"
        fi
    fi
    echo "[INFO] OUTPUT_FIG       = ${OUTPUT_FIG}"
    echo "========================================"

    # 默认仅绘制显著条目，除非传入 --all_parcels
    ONLY_SIG="--only_significant"
    if [[ "${extra}" == *"--all_parcels"* ]]; then
        ONLY_SIG=""
        extra=$(echo "${extra}" | sed 's/--all_parcels//g' | xargs)
    fi

    python "${SCRIPT}" \
        --top_parcels_json "${TOP_PARCELS_JSON}" \
        --output_fig "${OUTPUT_FIG}" \
        --level "${LEVEL}" \
        --title "${TITLE_PREFIX} Activation Difference - ${ds}" \
        --p_threshold 0.05 \
        --cached_data_path "${CACHE_DATA_FILE}" \
        --top_k 5 \
        ${SKIP_IF_EXISTS} \
        ${ONLY_SIG} \
        ${FORCE_RECOMPUTE} \
        ${extra}
        #--extra_non_top20_k 5
    echo "[INFO] 完成: ${ds}"
    echo ""
}

# 主逻辑：固定顺序依次处理若干数据集
DATASETS=(
    # "MedHallu_gemma-2-2b"
    # "HaluEval_gemma-2-2b"
    # "dolly_close_gemma-2-2b"
    # "nq_open_gemma-2-2b"
    # "sciq_gemma-2-2b"
    # "triviaqa_gemma-2-2b"
    "truthfulqa_gemma-2-2b"
)

echo "[INFO] 将依次为以下数据集绘制条形图（仅显著条目，LEVEL = ${LEVEL}）："
for ds in "${DATASETS[@]}"; do
  echo "  - ${ds}"
done
echo ""

for ds in "${DATASETS[@]}"; do
    process_dataset "${ds}" ""
done

echo "[INFO] 所有指定数据集处理完成。"
