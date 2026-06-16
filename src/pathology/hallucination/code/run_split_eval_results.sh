#!/usr/bin/env bash
#
# 功能: 运行 split_eval_results.py，扫描 results 目录中含 "eval" 的 jsonl，
#       按 is_correct（或 refusal_detected）拆分为 <模型名>/correct.jsonl 与 incorrect.jsonl。
#
# 用法: 按需修改下方变量后执行: bash run_split_eval_results.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 含 eval 的 jsonl 所在目录（与 Python 默认一致，可按需改）
RESULTS_DIR="/path/to/project_root/safety_explanation/hallucination/results/"

# 输出根目录；留空表示与 RESULTS_DIR 相同（不写 --output-root）
OUTPUT_ROOT=""

# 超参数: 1 = 若 correct.jsonl 与 incorrect.jsonl 均已存在则跳过该模型；0 = 始终处理
SKIP_EXISTING=1

# 1 = 仅打印计划不写文件
DRY_RUN=1

PY="$SCRIPT_DIR/split_eval_results.py"
CMD=(python3 "$PY" --results-dir "$RESULTS_DIR")

if [[ -n "$OUTPUT_ROOT" ]]; then
  CMD+=(--output-root "$OUTPUT_ROOT")
fi
if [[ "$SKIP_EXISTING" -eq 1 ]]; then
  CMD+=(--skip-existing)
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
