#!/usr/bin/env bash
# 功能：汇总干预效果统计：扫描 results/intervention 下各 strength_* 目录中的评测结果文件
# 从 *_baseline_eval.jsonl 和 *_intervention_eval.jsonl 中的 eval_type 统计：
# - baseline 正确/错误（bias 为错误，no_bias/antibias 为正确）
# - 被干预改正/破坏数
# 并输出汇总表（JSON/CSV）。
# 用法：在 code/intervention 下执行 ./run_aggregate_intervention_effect.sh [可选参数]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/path/to/project_root"
FAIRNESS_BIAS_DIR="$BASE_DIR/safety_explanation/fairness_bias"
INTERVENTION_ROOT="$FAIRNESS_BIAS_DIR/results/intervention"
PYTHON_SCRIPT="$SCRIPT_DIR/aggregate_intervention_effect.py"

cd "$SCRIPT_DIR"

python3 "$PYTHON_SCRIPT" --intervention_root "$INTERVENTION_ROOT" --output_dir "$FAIRNESS_BIAS_DIR/results/intervention/aggregate" "$@"
