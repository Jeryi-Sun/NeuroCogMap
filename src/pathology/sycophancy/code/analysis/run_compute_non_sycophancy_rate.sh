#!/bin/bash
#
# 统计“非谄媚率(non-sycophancy rate)”的批量运行脚本
#
# 功能：
# - 扫描 intervention 评测输出目录（默认：.../results/intervention/strength_*）
# - 对每个 strength 目录下的 *_eval.jsonl 计算非谄媚率
# - 可选写出 *_non_sycophancy_summary.json（默认写出）
# - 支持跳过已存在 summary（SKIP_EXISTING=true）
#
# 用法示例：
#   bash run_compute_non_sycophancy_rate.sh
#   SKIP_EXISTING=false bash run_compute_non_sycophancy_rate.sh
#   STRENGTH_LIST="0.1 0.3 0.5" bash run_compute_non_sycophancy_rate.sh
#   INTERVENTION_BASE_DIR="/path/to/results/intervention" bash run_compute_non_sycophancy_rate.sh
#

set -e

# ==================== 配置区域（可用环境变量覆盖） ====================
BASE_DIR="${BASE_DIR:-/path/to/project_root}"
INTERVENTION_BASE_DIR="${INTERVENTION_BASE_DIR:-$BASE_DIR/safety_explanation/sycophancy/results/intervention}"

# strength 列表；可用环境变量覆盖（空则自动探测 strength_* 目录）
STRENGTH_LIST="${STRENGTH_LIST:-}"

# 是否写 summary（建议保持 true）
WRITE_SUMMARY="${WRITE_SUMMARY:-true}"

# 是否跳过已存在 summary
SKIP_EXISTING="${SKIP_EXISTING:-true}"

# 扫描模式
PATTERN="${PATTERN:-*_eval.jsonl}"
RECURSIVE="${RECURSIVE:-false}"

# Python
PYTHON_BIN="${PYTHON_BIN:-python3}"
PY_SCRIPT="$BASE_DIR/safety_explanation/sycophancy/code/analysis/compute_non_sycophancy_rate.py"

# ==================== 工具函数 ====================
print_info() { echo -e "\033[0;32m[INFO]\033[0m $1"; }
print_warn() { echo -e "\033[0;33m[WARN]\033[0m $1"; }
print_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

check_environment() {
  if [ ! -f "$PY_SCRIPT" ]; then
    print_error "找不到 Python 脚本: $PY_SCRIPT"
    exit 1
  fi
  if ! command -v "$PYTHON_BIN" &>/dev/null; then
    print_error "找不到 PYTHON_BIN: $PYTHON_BIN"
    exit 1
  fi
  if [ ! -d "$INTERVENTION_BASE_DIR" ]; then
    print_error "INTERVENTION_BASE_DIR 不存在: $INTERVENTION_BASE_DIR"
    exit 1
  fi
}

bool_to_flag() {
  # 将 true/false 转成是否附加 flag
  local v="$1"
  local flag="$2"
  if [ "$v" = "true" ] || [ "$v" = "TRUE" ] || [ "$v" = "1" ]; then
    echo "$flag"
  else
    echo ""
  fi
}

discover_strength_list() {
  # 自动从 INTERVENTION_BASE_DIR 下找 strength_* 目录
  local dirs
  dirs=$(find "$INTERVENTION_BASE_DIR" -maxdepth 1 -type d -name "strength_*" 2>/dev/null | sort || true)
  if [ -z "$dirs" ]; then
    echo ""
    return
  fi
  # strength_0.1 -> 0.1
  echo "$dirs" | sed 's#.*/strength_##g' | tr '\n' ' '
}

run_one_strength() {
  local strength="$1"
  local strength_dir="$INTERVENTION_BASE_DIR/strength_$strength"
  if [ ! -d "$strength_dir" ]; then
    print_warn "强度目录不存在，跳过: $strength_dir"
    return
  fi

print_info "处理强度 strength_$strength: $strength_dir"

local cmd="$PYTHON_BIN $PY_SCRIPT --input_dir \"$strength_dir\" --pattern \"$PATTERN\""
cmd="$cmd $(bool_to_flag "$RECURSIVE" "--recursive")"
cmd="$cmd $(bool_to_flag "$WRITE_SUMMARY" "--write_summary")"
cmd="$cmd $(bool_to_flag "$SKIP_EXISTING" "--skip_existing")"

print_info "执行命令: $cmd"
set +e
eval "$cmd"
local code=$?
set -e
if [ $code -ne 0 ]; then
  print_warn "该强度统计失败(code=$code)，继续下一个。"
fi
}

# ==================== 主程序 ====================
main() {
  print_info "非谄媚率批量统计脚本"
  print_info "INTERVENTION_BASE_DIR: $INTERVENTION_BASE_DIR"
  print_info "PATTERN: $PATTERN | RECURSIVE: $RECURSIVE | WRITE_SUMMARY: $WRITE_SUMMARY | SKIP_EXISTING: $SKIP_EXISTING"

  check_environment

  local strengths="$STRENGTH_LIST"
  if [ -z "$strengths" ]; then
    strengths=$(discover_strength_list)
  fi
  if [ -z "$strengths" ]; then
    print_error "未发现任何 strength_* 目录，也未指定 STRENGTH_LIST"
    exit 1
  fi

  print_info "强度列表: $strengths"

  for s in $strengths; do
    run_one_strength "$s"
  done

  print_info "全部完成。"
}

main "$@"

