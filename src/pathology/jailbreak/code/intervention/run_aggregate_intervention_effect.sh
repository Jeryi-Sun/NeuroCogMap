#!/usr/bin/env bash
# 功能：汇总干预效果统计：扫描 results/intervention 下各干预强度目录中的 *_eval.jsonl
# 从评测结果文件中的 refusal_detected 字段统计 baseline 正确/错误、
# 被干预改正/破坏数，并输出汇总表（JSON/CSV）。
# 用法：在 code/intervention 下执行 ./run_aggregate_intervention_effect.sh [可选参数]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="/path/to/project_root"
JAILBREAK_DIR="$BASE_DIR/safety_explanation/jailbreak"
INTERVENTION_ROOT="$JAILBREAK_DIR/results/intervention"
PYTHON_SCRIPT="$SCRIPT_DIR/aggregate_intervention_effect.py"

cd "$SCRIPT_DIR"

python3 "$PYTHON_SCRIPT" --intervention_root "$INTERVENTION_ROOT" --output_dir "$JAILBREAK_DIR/results/intervention/aggregate" "$@"
