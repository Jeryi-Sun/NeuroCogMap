#!/usr/bin/env bash
# 功能：汇总 results/intervention 下各干预强度目录中的 *_analysis.json，
#       统计 baseline 正确/错误数、被干预改正数等，并输出到 results/intervention/aggregate/。
# 用法：在 code/intervention 下执行 ./run_aggregate_intervention_effect.sh [可选参数]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python aggregate_intervention_effect.py "$@"
