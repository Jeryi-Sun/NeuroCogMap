#!/usr/bin/env bash
#
# 功能：比较三份 CSV 在“对应行”上的 AIC（full vs baseline / full vs simple），并输出 win/tie/loss 汇总与逐行明细。
#
# 用法示例：
#   bash run_compare_aic.sh \
#     --full    /abs/path/to/full.csv \
#     --baseline /abs/path/to/baseline.csv \
#     --simple  /abs/path/to/simple.csv \
#     --outdir  /abs/path/to/output_dir \
#     --eps 0 \
#     --skip_existing
#
# 说明：
# - 脚本会调用同目录下的 compare_aic_kool2017cost_exp2.py
# - 输出目录会生成两个文件：*.detail.csv 和 *.summary.csv（文件名由 python 脚本决定）
set -euo pipefail

# =========================
# 在这里直接配置输入/输出（默认使用）
# =========================
# 只需要指定 “文件类型”，脚本会自动选择对应的 results 子目录与文件名模式
# 可选类型（如需更多类型我可以继续加）：
# - human_kool2017cost_exp2
# - testhuman_kool2016when_exp2_v2
FILE_TYPE="testhuman_kool2016when_exp2_v2"

# openloop/results 根目录（一般不用改）
RESULTS_DIR="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/results"

# 输出目录：默认跟随类型（会自动设到 RESULTS_DIR 或其子目录）
OUTDIR=""
# AIC 判定 tie 的容差（默认 0.0；如处理浮点误差可设 1e-9）
EPS="0.0"
# 若输出文件已存在则跳过（不覆盖）：1 开启 / 0 关闭
SKIP_EXISTING="0"

usage() {
  cat <<'EOF'
Usage:
  直接编辑脚本顶部的 FILE_TYPE（以及可选 OUTDIR），然后运行：
    bash run_compare_aic.sh

  （可选）也支持命令行参数覆盖脚本内配置：
    bash run_compare_aic.sh --full FULL.csv --baseline BASELINE.csv --simple SIMPLE.csv --outdir OUTDIR [--eps EPS] [--skip_existing]

FILE_TYPE:
  - human_kool2017cost_exp2
  - testhuman_kool2016when_exp2_v2

Args:
  --full           full 方法的 CSV（例如：GPT5.2+NeuroCogMap）
  --baseline       baseline 方法的 CSV（例如：GPT5.2）
  --simple         simple 方法的 CSV（例如：Cognitive Model）
  --outdir         输出目录（建议 openloop/results/）
  --eps            AIC 判定 tie 的容差（默认 0.0；如处理浮点误差可设 1e-9）
  --skip_existing  若输出文件已存在则跳过（不覆盖）
EOF
}

FULL=""
BASELINE=""
SIMPLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full) FULL="${2:-}"; shift 2;;
    --baseline) BASELINE="${2:-}"; shift 2;;
    --simple) SIMPLE="${2:-}"; shift 2;;
    --outdir) OUTDIR="${2:-}"; shift 2;;
    --eps) EPS="${2:-}"; shift 2;;
    --skip_existing) SKIP_EXISTING="1"; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$FULL" || -z "$BASELINE" || -z "$SIMPLE" ]]; then
  # 若未从命令行显式传入，则根据 FILE_TYPE 自动组装路径
  case "$FILE_TYPE" in
    human_kool2017cost_exp2)
      STEM="!human_kool2017cost_exp2"
      SUBDIR=""
      SUFFIX="fix_bug"
      ;;
    testhuman_kool2016when_exp2_v2)
      STEM="!testhuman_kool2016when_exp2"
      SUBDIR="test_part"
      SUFFIX="fix_bug_v2"
      ;;
    *)
      echo "[ERROR] 未知 FILE_TYPE: $FILE_TYPE" >&2
      usage
      exit 2
      ;;
  esac

  DIR="$RESULTS_DIR"
  if [[ -n "$SUBDIR" ]]; then
    DIR="$RESULTS_DIR/$SUBDIR"
  fi

  FULL="$DIR/${STEM}.csv_full_${SUFFIX}.csv"
  BASELINE="$DIR/${STEM}.csv_baseline_${SUFFIX}.csv"
  SIMPLE="$DIR/${STEM}.csv_simple_${SUFFIX}.csv"

  if [[ -z "$OUTDIR" ]]; then
    OUTDIR="$DIR"
  fi
fi

if [[ -z "$FULL" || -z "$BASELINE" || -z "$SIMPLE" || -z "$OUTDIR" ]]; then
  echo "[ERROR] 必填参数缺失（FULL/BASELINE/SIMPLE/OUTDIR）" >&2
  usage
  exit 2
fi

for f in "$FULL" "$BASELINE" "$SIMPLE"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] 找不到输入文件: $f" >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/compare_aic_kool2017cost_exp2.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "[ERROR] 找不到 Python 脚本: $PY_SCRIPT" >&2
  exit 1
fi

cmd=(python "$PY_SCRIPT" --full "$FULL" --baseline "$BASELINE" --simple "$SIMPLE" --outdir "$OUTDIR" --eps "$EPS")
if [[ "$SKIP_EXISTING" == "1" ]]; then
  cmd+=(--skip_existing)
fi

echo "[INFO] Running:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

