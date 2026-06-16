#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 jailbreak 结果绘制层级激活差异柱状图和小提琴图，复用
#   hallucination 下的 plot_hierarchy_level_bars.py，通过
#   --group-high-label / --group-low-label 传入标签。
#
# 前置：先运行 run_aggregate_hierarchy_level.sh 生成汇总 JSON。
#
# 用法示例：
#   bash safety_explanation/jailbreak/hierarchy_graphs/run_plot_hierarchy_level_bars.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/hallucination/hierarchy_graphs/plot_hierarchy_level_bars.py"
JSON_PATH="$BASE_DIR/safety_explanation/jailbreak/hierarchy_graphs/data/hierarchy_level_all_models.json"
OUTPUT_DIR="$BASE_DIR/safety_explanation/jailbreak/hierarchy_graphs/figures_hierarchy_level_bars"

# 标签：Refuse-Success vs Refuse-Failed（可根据实际数据含义调整）
GROUP_HIGH_LABEL="Refuse-Success"
GROUP_LOW_LABEL="Refuse-Failed"

python3 "$SCRIPT_PATH" \
    --json-path "$JSON_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --group-high-label "$GROUP_HIGH_LABEL" \
    --group-low-label "$GROUP_LOW_LABEL" \
    2>&1 | tee "$BASE_DIR/safety_explanation/jailbreak/hierarchy_graphs/run_plot_hierarchy_level_bars.log"
