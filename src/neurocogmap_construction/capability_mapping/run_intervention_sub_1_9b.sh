#!/bin/bash

# 等待上游进程结束后，启动 9b-it 的 Parcel Intervention 运行脚本

# 要等待的进程 PID（可通过环境变量覆盖）
TARGET_PID=${TARGET_PID:-0}

check_process_exists() {
  local pid=$1
  if kill -0 $pid 2>/dev/null; then return 0; fi
  if ps -p $pid >/dev/null 2>&1; then return 0; fi
  if [[ -d "/proc/$pid" ]]; then return 0; fi
  return 1
}

get_process_info() {
  local pid=$1
  if ps -p $pid >/dev/null 2>&1; then
    local process_info=$(ps -p $pid -o pid,ppid,user,comm,etime --no-headers 2>/dev/null)
    if [[ -n "$process_info" ]]; then
      echo "进程信息: $process_info"
      return
    fi
  fi
  if [[ -d "/proc/$pid" ]]; then
    local cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ')
    local status=$(cat "/proc/$pid/status" 2>/dev/null | grep "State:" | cut -f2)
    echo "命令行: $cmdline"
    echo "状态: $status"
    return
  fi
  echo "无法获取进程详细信息"
}

if [ "$TARGET_PID" -ne 0 ]; then
  echo "开始检测进程 $TARGET_PID..."
  if ! check_process_exists $TARGET_PID; then
    echo "进程 $TARGET_PID 不存在，直接开始执行"
  else
    echo "进程 $TARGET_PID 正在运行，等待其结束..."
    get_process_info $TARGET_PID
    while check_process_exists $TARGET_PID; do
      echo "Process $TARGET_PID is still running..."
      sleep 5
    done
    echo "Process $TARGET_PID is no longer running"
  fi
fi

echo "Process $TARGET_PID finished. Starting 9b intervention..."

# 进入脚本目录并调用 9b 运行脚本
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 示例：处理所有数据集前 50 个 parcel 的单数据集运行
# 使用: bash run_intervention_sub_1_9b.sh DATASET_NAME

bash "$SCRIPT_DIR/run_intervention_9b.sh" --all --top-k 50


