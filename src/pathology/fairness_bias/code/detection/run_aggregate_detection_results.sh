#!/usr/bin/env bash
# 功能：汇总公平偏见检测的 Baseline 与我们的模型 (our_method) 的 performance 统计结果。
# 从 results/detection、results/detection/baselines、results/llm_detection 读取
# cv_metrics，汇总为表格并保存到 results/detection/all_results。
# 用法：在 code/detection 下执行 ./run_aggregate_detection_results.sh [可选参数]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/path/to/project_root"
FAIRNESS_BIAS_DIR="$BASE_DIR/safety_explanation/fairness_bias"
PYTHON_SCRIPT="$BASE_DIR/safety_explanation/hallucination/code/detection/aggregate_detection_results.py"

cd "$SCRIPT_DIR"

python3 "$PYTHON_SCRIPT" --base_dir "$FAIRNESS_BIAS_DIR" --output_dir "$FAIRNESS_BIAS_DIR/results/detection/all_results" "$@"
