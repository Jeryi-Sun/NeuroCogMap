#!/usr/bin/env bash

# 功能: 针对 Llama-3.1-8B 在 Jailbreak 数据集上执行 generate+eval。
# 特点: 仅通过变量配置；支持处理单个 CSV 或目录下全部 CSV；支持跳过已存在结果与 in-context。

set -euo pipefail

# ===== 配置区域 =====

# 运行模式: both | generate | eval
MODE="both"

# 模型配置
MODEL_ID="meta-llama/Llama-3.1-8B"

# 数据集路径（支持单个文件或目录）
# 如果是目录，将处理目录下所有.csv文件
CSV_PATH="/path/to/project_root/safety_explanation/jailbreak/dataset"

# 输出目录
OUTPUT_DIR="/path/to/project_root/safety_explanation/jailbreak/results"

# 是否跳过已存在的结果: 1=跳过, 0=不跳过
SKIP_EXISTING=1

# 是否使用in-context learning: 1=使用, 0=不使用
# 使用时会用前2条数据作为示例，从第3条开始生成
USE_INCONTEXT=1

# 处理样本数限制（0表示全部）
MAX_SAMPLES=0

# 生成参数
MAX_NEW_TOKENS=512
TEMPERATURE=0.0

# 评测服务配置
VLLM_URL="http://0.0.0.0:8001/v1"
API_KEY="abcabc"

# ===== 脚本执行区域 =====

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/jailbreak_generate_and_eval.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] 找不到脚本: $PY_SCRIPT" >&2
  exit 1
fi

process_csv() {
  local csv_file="$1"
  local csv_basename
  csv_basename=$(basename "$csv_file" .csv)

  echo ""
  echo "=========================================="
  echo "处理数据集: $csv_basename"
  echo "文件路径: $csv_file"
  echo "=========================================="

  ARGS=(
    --mode "$MODE"
    --model_id "$MODEL_ID"
    --csv_path "$csv_file"
    --output_dir "$OUTPUT_DIR"
    --max_samples "$MAX_SAMPLES"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --vllm_url "$VLLM_URL"
    --api_key "$API_KEY"
  )

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    ARGS+=(--skip_existing)
  fi

  if [[ "$USE_INCONTEXT" == "1" ]]; then
    ARGS+=(--use_incontext)
  fi

  python3 "$PY_SCRIPT" "${ARGS[@]}"
}

if [[ -f "$CSV_PATH" ]]; then
  process_csv "$CSV_PATH"
elif [[ -d "$CSV_PATH" ]]; then
  echo "[INFO] 检测到目录，将处理目录下所有.csv文件: $CSV_PATH"

  csv_files=()
  while IFS= read -r -d '' file; do
    csv_files+=("$file")
  done < <(find "$CSV_PATH" -maxdepth 1 -type f -name "*.csv" -print0 | sort -z)

  if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "[ERROR] 目录下没有找到.csv文件: $CSV_PATH" >&2
    exit 1
  fi

  echo "[INFO] 找到 ${#csv_files[@]} 个CSV文件"
  for csv_file in "${csv_files[@]}"; do
    process_csv "$csv_file"
  done

  echo ""
  echo "=========================================="
  echo "所有数据集处理完成！"
  echo "=========================================="
else
  echo "[ERROR] 路径不存在: $CSV_PATH" >&2
  exit 1
fi

