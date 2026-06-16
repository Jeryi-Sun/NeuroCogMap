#!/usr/bin/env bash
# 功能：生成 Step1 直接相关性分析报告（基于训练集 train set 的 parcel 激活 vs AIC 的 PCC）
#
# 流程：
# 1) 运行 analyze_parcel_features_against_predictions.py，补齐/计算 step1_train_feature_pcc_ranked_by_{aic,nll}.csv
# 2) 运行 generate_step1_report.py，生成 Nature_Report_step1_direct_correlation.md
#
# 跳过逻辑：
# - analyze 步骤：通过 --skip-existing 控制（脚本内部会在缺少 step1_train_* 时自动补算）
# - report 步骤：若目标 md 已存在且开启 --skip-md-existing=1，则跳过 report 生成
#
# 示例：
#   bash run_step1_train_report.sh --top-k 10 --skip-existing-analysis 1 --skip-md-existing 1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

SCRIPT_ANALYZE="${PROJECT_DIR}/analyze_parcel_features_against_predictions.py"
SCRIPT_REPORT="${PROJECT_DIR}/generate_step1_report.py"

FEATURE_ANALYSIS_DIR="${PROJECT_DIR}/results/feature_analysis"
ACTIVATIONS_ROOT="${PROJECT_DIR}/results/activations"
PREDICTIONS_ROOT="${PROJECT_DIR}/results/predictions"

OUTPUT_MD="${FEATURE_ANALYSIS_DIR}/Nature_Report_step1_direct_correlation.md"

TOP_K=10
SKIP_EXISTING_ANALYSIS=0   # 1 表示 analyze 时传 --skip-existing
SKIP_MD_EXISTING=0         # 1 表示若 OUTPUT_MD 已存在则跳过生成

while [[ $# -gt 0 ]]; do
  case "$1" in
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --skip-existing-analysis)
      SKIP_EXISTING_ANALYSIS="$2"
      shift 2
      ;;
    --skip-md-existing)
      SKIP_MD_EXISTING="$2"
      shift 2
      ;;
    *)
      echo "[ERROR] 未识别参数: $1"
      exit 1
      ;;
  esac
done

if [[ ! -f "${SCRIPT_ANALYZE}" ]]; then
  echo "[ERROR] 找不到 analyze 脚本: ${SCRIPT_ANALYZE}"
  exit 1
fi
if [[ ! -f "${SCRIPT_REPORT}" ]]; then
  echo "[ERROR] 找不到 report 脚本: ${SCRIPT_REPORT}"
  exit 1
fi

mkdir -p "${FEATURE_ANALYSIS_DIR}"

ANALYZE_ARGS=(
  --activations-root "${ACTIVATIONS_ROOT}"
  --predictions-root "${PREDICTIONS_ROOT}"
  --output-root "${FEATURE_ANALYSIS_DIR}"
)
if [[ "${SKIP_EXISTING_ANALYSIS}" == "1" ]]; then
  ANALYZE_ARGS+=(--skip-existing)
fi

echo "[INFO] 开始 Step1-train PCC 计算：$(date)"
python "${SCRIPT_ANALYZE}" "${ANALYZE_ARGS[@]}"
echo "[INFO] Step1-train PCC 计算完成：$(date)"

if [[ "${SKIP_MD_EXISTING}" == "1" && -f "${OUTPUT_MD}" ]]; then
  echo "[SKIP] 已存在输出报告：${OUTPUT_MD}"
  exit 0
fi

echo "[INFO] 开始生成 Step1 报告（top-k=${TOP_K}）：$(date)"
python "${SCRIPT_REPORT}" --top-k "${TOP_K}"
echo "[INFO] Step1 报告生成完成：$(date)"

