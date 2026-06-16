#!/usr/bin/env bash

# 连接组分析增强版 一键运行脚本（简化版）
# ------------------------------------------------------------
# 用法：
#   1) 直接编辑本文件顶部的参数区（不需要命令行传参）
#   2) 保存后运行：
#        bash run_connection_analysis_plus.sh
#   3) 结果会输出到 OUTPUT_DIR 指定目录，并生成：
#        - connection_analysis_results.json      总结JSON
#        - delta_connectivity_matrix.npy         ΔFC矩阵
#        - analysis_report.txt                   文本报告
#        - visualizations/                       多张可视化图片
#
# 依赖：
#   - Python 3（建议3.9+）
#   - Python包：numpy, networkx, matplotlib, seaborn, scipy
#   - 脚本：connection_analysis_plus.py（已在同目录）
# ------------------------------------------------------------

#set -euo pipefail  # 如需严格模式可取消注释

############################ 参数区（按需修改） ############################

# 分析数据集列表（目录名）
DATASETS=("MedHallu_gemma-2-2b" "truthfulqa_gemma-2-2b" "HaluEval_gemma-2-2b" "nq_open_gemma-2-2b" "sciq_gemma-2-2b" "triviaqa_gemma-2-2b" "dolly_close_gemma-2-2b")

# 结果根目录（每个数据集都会在该目录下寻找连接矩阵并输出结果）
BASE_OUTPUT_ROOT="/path/to/project_root/safety_explanation/hallucination/results/analysis_output"

# 检测阈值
Z_THRESHOLD=2.0          # Z 分数阈值（例如 2.0）
TOP_K_PERCENT=5.0        # 取 ΔFC 绝对值 Top-K 百分比（例如 5.0）

# Python 可执行文件
PYTHON_BIN="python3"
########################## 参数区结束（以上修改） ##########################

# ------------- 无需修改的运行逻辑 -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PY="${SCRIPT_DIR}/connection_analysis_plus.py"

echo "[INFO] 使用脚本: ${SCRIPT_PY}"

if [[ ! -f "$SCRIPT_PY" ]]; then
  echo "[ERROR] 未找到 ${SCRIPT_PY}，请确认脚本已生成在本目录。" >&2
  exit 1
fi

# 循环运行多个数据集
for DS in "${DATASETS[@]}"; do
  echo "\n===================== [DATASET] $DS ====================="
  MATRIX_DIR="${BASE_OUTPUT_ROOT}/${DS}/parcel_level/connectivity_matrices"
  CORRECT_MATRIX="${MATRIX_DIR}/correct_connectivity_matrix.npy"
  INCORRECT_MATRIX="${MATRIX_DIR}/incorrect_connectivity_matrix.npy"
  NODE_NAMES="${MATRIX_DIR}/parcel_node_names.json"
  OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${DS}/parcel_level/connection_plus"

  # 基础检查
  if [[ ! -f "$CORRECT_MATRIX" ]]; then
    echo "[ERROR] 正确样本矩阵不存在: $CORRECT_MATRIX" >&2
    continue
  fi
  if [[ ! -f "$INCORRECT_MATRIX" ]]; then
    echo "[ERROR] 幻觉样本矩阵不存在: $INCORRECT_MATRIX" >&2
    continue
  fi
  if [[ ! -f "$NODE_NAMES" ]]; then
    echo "[WARN] 节点名称文件不存在，将在脚本中使用默认名称: $NODE_NAMES"
  fi

  # 创建输出目录
  mkdir -p "$OUTPUT_DIR"

  # 展示关键参数
  cat <<EOF
[INFO] 连接组分析运行参数：
  - DATASET         = $DS
  - CORRECT_MATRIX   = $CORRECT_MATRIX
  - INCORRECT_MATRIX = $INCORRECT_MATRIX
  - NODE_NAMES       = $NODE_NAMES
  - OUTPUT_DIR       = $OUTPUT_DIR
  - Z_THRESHOLD      = $Z_THRESHOLD
  - TOP_K_PERCENT    = $TOP_K_PERCENT
EOF

  # 执行分析
  "$PYTHON_BIN" "$SCRIPT_PY" \
    --correct_matrix "$CORRECT_MATRIX" \
    --incorrect_matrix "$INCORRECT_MATRIX" \
    --node_names "$NODE_NAMES" \
    --output_dir "$OUTPUT_DIR" \
    --z_threshold "$Z_THRESHOLD" \
    --top_k_percent "$TOP_K_PERCENT"

  EXIT_CODE=$?
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[ERROR] 分析失败（$DS），退出码：$EXIT_CODE" >&2
  else
    echo "[DONE] 分析完成（$DS）。结果目录：$OUTPUT_DIR"
  fi

done
