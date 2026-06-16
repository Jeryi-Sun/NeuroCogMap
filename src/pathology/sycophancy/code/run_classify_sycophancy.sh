#!/bin/bash

set -euo pipefail

# ===== 配置区域 =====

# 评估结果文件路径列表（在这些文件之间循环）
# 路径相对于 dataset/sycophancy-eval 目录
EVAL_FILES=(
  #"../dataset/sycophancy-eval/results/answer_with_groups_gemma-2-2b_gen_eval.jsonl"
  #"../dataset/sycophancy-eval/results/answer_with_groups_gemma-2-9b-it_gen_eval.jsonl"
  "../dataset/sycophancy-eval/results/answer_with_groups_Llama-3.1-8B_gen_eval.jsonl"
  # "../dataset/sycophancy-eval/results/feedback_with_groups_gemma-2-2b_gen_eval.jsonl"
  # "../dataset/sycophancy-eval/results/feedback_with_groups_gemma-2-9b-it_gen_eval.jsonl"
  "../dataset/sycophancy-eval/results/feedback_with_groups_Llama-3.1-8B_gen_eval.jsonl"
  
)

# 输出基础目录
OUTPUT_BASE_DIR="/path/to/project_root/safety_explanation/sycophancy/results"

# 是否跳过已存在的结果: 1=跳过, 0=不跳过
SKIP_EXISTING=1

# ===== 脚本执行区域 =====

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/classify_sycophancy.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] 找不到脚本: $PY_SCRIPT" >&2
  exit 1
fi

# 处理单个评估文件的函数
process_eval_file() {
  local eval_file="$1"
  local eval_basename=$(basename "$eval_file" .jsonl)
  
  echo ""
  echo "=========================================="
  echo "处理评估结果: $eval_basename"
  echo "文件路径: $eval_file"
  echo "=========================================="
  
  ARGS=(
    --eval_file "$eval_file"
    --output_base_dir "$OUTPUT_BASE_DIR"
  )
  
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip_existing)
  fi
  
  python3 "$PY_SCRIPT" "${ARGS[@]}"
  
  if [[ $? -eq 0 ]]; then
    echo "[SUCCESS] $eval_basename 处理完成"
  else
    echo "[ERROR] $eval_basename 处理失败" >&2
  fi
}

# 依次处理 EVAL_FILES 中列出的评估结果文件
echo "[INFO] 将依次分类以下评估结果文件:"
for f in "${EVAL_FILES[@]}"; do
  echo " - $f"
done

for EVAL_FILE in "${EVAL_FILES[@]}"; do
  # 构建完整路径（相对于脚本目录）
  full_path="$SCRIPT_DIR/$EVAL_FILE"
  
  if [[ -f "$full_path" ]]; then
    process_eval_file "$full_path"
  else
    echo "[WARN] 评估结果文件不存在，跳过: $full_path" >&2
  fi
done

echo ""
echo "=========================================="
echo "所有评估结果文件分类完成！"
echo "=========================================="
