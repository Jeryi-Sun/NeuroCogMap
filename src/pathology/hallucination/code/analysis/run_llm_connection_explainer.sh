#!/usr/bin/env bash

# LLM 连接组结果解释 一键运行脚本（基于 gpt-oss，简化版）
# ------------------------------------------------------------
# 用法：
#   1) 直接编辑本文件顶部参数区（不需要命令行传参）
#   2) 运行：
#        bash run_llm_connection_explainer.sh
#   3) 将在每个数据集的 connection_plus 目录生成：
#        - llm_connection_report_<dataset>.md
#
# 依赖：
#   - Python 3（建议3.9+）
#   - Python包：requests, numpy
#   - 脚本：llm_connection_explainer.py（已在同目录）
# ------------------------------------------------------------

#set -euo pipefail  # 如需严格模式可取消注释

############################ 参数区（按需修改） ############################
# 分析数据集列表（目录名，需与 analysis_output 下的目录名一致）
DATASETS=(
  "MedHallu_gemma-2-2b"
  "truthfulqa_gemma-2-2b"
  "HaluEval_gemma-2-2b"
  "nq_open_gemma-2-2b"
  "sciq_gemma-2-2b"
  "triviaqa_gemma-2-2b"
  "dolly_close_gemma-2-2b"
)

# analysis_output 根目录（里面包含各数据集子目录）
BASE_OUTPUT_ROOT="/path/to/project_root/safety_explanation/hallucination/results/analysis_output"

# vLLM 配置（参考 analysis_llm_summary.py）
VLLM_URL="http://0.0.0.0:8001/v1"
API_KEY="abcabc"
MODEL_PATH="/path/to/local_models/gpt-oss-20b"

# Python 可执行
PYTHON_BIN="python3"
########################## 参数区结束（以上修改） ##########################

# ------------- 无需修改的运行逻辑 -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PY="${SCRIPT_DIR}/llm_connection_explainer.py"

echo "[INFO] 使用脚本: ${SCRIPT_PY}"

if [[ ! -f "$SCRIPT_PY" ]]; then
  echo "[ERROR] 未找到 ${SCRIPT_PY}，请确认脚本在本目录。" >&2
  exit 1
fi

if [[ ! -d "$BASE_OUTPUT_ROOT" ]]; then
  echo "[ERROR] 根目录不存在: $BASE_OUTPUT_ROOT" >&2
  exit 1
fi

# 打印参数
cat <<EOF
[INFO] LLM 解释运行参数：
  - BASE_OUTPUT_ROOT = $BASE_OUTPUT_ROOT
  - DATASETS         = ${DATASETS[*]}
  - VLLM_URL         = $VLLM_URL
  - MODEL_PATH       = $MODEL_PATH
EOF

# 运行（一次性传入多个数据集）
"$PYTHON_BIN" "$SCRIPT_PY" \
  --base_output_root "$BASE_OUTPUT_ROOT" \
  --datasets ${DATASETS[*]} \
  --vllm_url "$VLLM_URL" \
  --api_key "$API_KEY" \
  --model "$MODEL_PATH"

EXIT_CODE=$?
if [[ $EXIT_CODE -ne 0 ]]; then
  echo "[ERROR] 生成 LLM 报告失败，退出码：$EXIT_CODE" >&2
  exit $EXIT_CODE
fi

echo "[DONE] 已为上述数据集生成 LLM 报告（如有可用结果）。"
