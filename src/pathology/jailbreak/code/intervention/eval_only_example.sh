#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_intervention.py"
CONFIG_FILE="$SCRIPT_DIR/config.json"

# 修改为需要补评估的结果文件
RESULTS_FILE="/path/to/your/intervention_results.json"
VLLM_URL="http://127.0.0.1:8001/v1"

if [[ ! -f "$RESULTS_FILE" ]]; then
  echo "错误: 未找到结果文件 $RESULTS_FILE"
  exit 1
fi

CMD=(python "$PY_SCRIPT" \
  --mode eval_only \
  --config "$CONFIG_FILE" \
  --eval_only_from_file "$RESULTS_FILE" \
  --vllm_url "$VLLM_URL")

echo "运行命令: ${CMD[*]}"
"${CMD[@]}"

echo "评估完成"

