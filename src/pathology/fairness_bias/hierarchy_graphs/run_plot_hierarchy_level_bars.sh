#!/usr/bin/env bash
# ============================================================
# 功能介绍：
#   为 fairness_bias 结果绘制 hierarchy level 柱状图和小提琴图，
#   复用 hallucination 下的 plot_hierarchy_level_bars.py。
#
# 用法示例：
#   bash run_plot_hierarchy_level_bars.sh
#   JSON_PATH=/path/to/data.json OUT_DIR=/path/to/output bash run_plot_hierarchy_level_bars.sh
# ============================================================

set -euo pipefail

BASE_DIR="${BASE_DIR:-/path/to/project_root}"
JSON_PATH="${JSON_PATH:-$BASE_DIR/safety_explanation/fairness_bias/hierarchy_graphs/data/hierarchy_level_all_models.json}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/safety_explanation/fairness_bias/hierarchy_graphs/figures_hierarchy_level_bars}"

SCRIPT_PATH="$BASE_DIR/safety_explanation/hallucination/hierarchy_graphs/plot_hierarchy_level_bars.py"

# 修改 Python 脚本中的路径（通过环境变量或直接修改脚本）
# 这里我们直接调用，Python 脚本需要支持从外部传入路径
python3 "$SCRIPT_PATH" \
    --json-path "$JSON_PATH" \
    --output-dir "$OUT_DIR" \
    --project-type fairness_bias \
    --group-high-label "Incorrect" \
    --group-low-label "Correct" \
    2>&1 | tee "$BASE_DIR/safety_explanation/fairness_bias/hierarchy_graphs/run_plot_hierarchy_level_bars.log"
