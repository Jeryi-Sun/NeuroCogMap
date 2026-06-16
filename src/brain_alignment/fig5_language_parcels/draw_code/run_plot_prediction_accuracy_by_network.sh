#!/bin/bash
# 运行预测准确度按7Networks分组的分析脚本（支持对多个 UST_ID 和 METHOD 批量运行）
# 功能：绘制每个network的柱状图（network均值）+ parcel散点（parcel的Top-K均值）+ 误差线（SEM）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================
# 基本配置（直接在这里改）
# =========================
# UST 被试 ID 列表
UST_IDS=("uts02" "uts03")
# 数据名称 / 故事名称
STORY_NAME="whereisthesmoke"
# 方法名称列表（目前支持：sae / saeact）
METHODS=("sae" "saeact")

# 根目录
PROJECT_ROOT="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs"

# 固定的 Parcel 描述文件
PARCEL_DESCRIPTIONS="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"

# 解析命令行参数（这里只保留与分析逻辑相关的少量参数）
TOP_K=1
OVERWRITE="--overwrite"

while [[ $# -gt 0 ]]; do
    case $1 in
        --top-k)
            TOP_K="$2"
            shift 2
            ;;
        --overwrite)
            OVERWRITE="--overwrite"
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--top-k K] [--overwrite]"
            exit 1
            ;;
    esac
done

OVERALL_EXIT_CODE=0

for UST_ID in "${UST_IDS[@]}"; do
  for METHOD in "${METHODS[@]}"; do
    CURRENT_DATA_DIR="${PROJECT_ROOT}/data4draw/${UST_ID}/${STORY_NAME}/${METHOD}"
    CURRENT_OUTPUT_DIR="${PROJECT_ROOT}/draw_result/${UST_ID}/${STORY_NAME}/${METHOD}/prediction_accuracy_by_network"

    echo "=========================================="
    echo "预测准确度按7Networks分组分析"
    echo "UST_ID: ${UST_ID}"
    echo "STORY_NAME: ${STORY_NAME}"
    echo "METHOD: ${METHOD}"
    echo "数据目录: $CURRENT_DATA_DIR"
    echo "输出目录: $CURRENT_OUTPUT_DIR"
    echo "Top-K: $TOP_K"
    echo "=========================================="

    if [ ! -d "$CURRENT_DATA_DIR" ]; then
        echo "错误: 数据目录不存在: $CURRENT_DATA_DIR"
        OVERALL_EXIT_CODE=1
        continue
    fi

    # 在该 data4draw 目录下寻找最新的 prediction_matrix_*.csv
    LATEST_PRED=$(ls -1 "${CURRENT_DATA_DIR}"/prediction_matrix_*.csv 2>/dev/null | sort | tail -n 1)
    if [ -z "$LATEST_PRED" ]; then
        echo "错误: 在 ${CURRENT_DATA_DIR} 中未找到 prediction_matrix_*.csv"
        echo "请先运行构建 prediction_matrix 的脚本。"
        OVERALL_EXIT_CODE=1
        continue
    fi

    PREDICTION_MATRIX="$LATEST_PRED"
    mkdir -p "$CURRENT_OUTPUT_DIR"

    echo "使用预测矩阵: $PREDICTION_MATRIX"

    # 清理旧版脚本生成的历史图片，确保只保留当前目标图
    rm -f "${CURRENT_OUTPUT_DIR}/prediction_accuracy_bubble_plot.png"
    rm -f "${CURRENT_OUTPUT_DIR}/mean_accuracy_by_network_lineplot.png"

    # 运行Python脚本
    python plot_prediction_accuracy_by_network.py \
        --prediction-matrix "$PREDICTION_MATRIX" \
        --parcel-descriptions "$PARCEL_DESCRIPTIONS" \
        --output-dir "$CURRENT_OUTPUT_DIR" \
        --top-k "$TOP_K" \
        $OVERWRITE

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ 组合 UST_ID=${UST_ID}, METHOD=${METHOD} 分析完成！"
        echo "结果保存在: $CURRENT_OUTPUT_DIR"
        echo "  - prediction_accuracy_bubble_plot_top_${TOP_K}.png"
        echo "  - prediction_accuracy_violin_plot_top_${TOP_K}.png"
        echo "=========================================="
    else
        echo ""
        echo "✗ 组合 UST_ID=${UST_ID}, METHOD=${METHOD} 分析失败，退出码: $exit_code"
        OVERALL_EXIT_CODE=$exit_code
    fi

  done
done

exit $OVERALL_EXIT_CODE