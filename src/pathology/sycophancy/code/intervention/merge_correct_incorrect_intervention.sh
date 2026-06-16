#!/bin/bash

# 将 correct 和 incorrect 的干预结果合并为符合预期路径的单一文件
# 输出格式：${MODEL_NAME}_${data_type}_intervention.json（与 fairness_bias 一致）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/merge_correct_incorrect_intervention.py"
PYTHON_CMD="python"

# 要合并的强度列表（默认 0.1 0.3 0.5）
STRENGTHS="${STRENGTHS:-0.1 0.3 0.5}"
# 是否跳过已存在的输出文件（yes/no）
SKIP_EXISTING="${SKIP_EXISTING:-no}"

SKIP_FLAG=""
if [[ "$SKIP_EXISTING" == "yes" ]]; then
    SKIP_FLAG="--skip-existing"
fi

echo "🧩 合并 correct/incorrect 干预结果"
echo "强度列表: $STRENGTHS"
echo "跳过已存在: $SKIP_EXISTING"
echo ""

$PYTHON_CMD "$PY_SCRIPT" --strengths $STRENGTHS $SKIP_FLAG

echo ""
echo "🎉 合并完成"
