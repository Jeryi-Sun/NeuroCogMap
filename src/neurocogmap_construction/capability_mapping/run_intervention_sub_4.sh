
# 要等待的进程 PID
TARGET_PID=2096060

# 改进的进程检测函数
check_process_exists() {
    local pid=$1
    
    # 方法1: 使用 kill -0 (需要权限)
    if kill -0 $pid 2>/dev/null; then
        return 0
    fi
    
    # 方法2: 使用 ps 命令检查
    if ps -p $pid >/dev/null 2>&1; then
        return 0
    fi
    
    # 方法3: 使用 pgrep 命令检查
    if pgrep -f "^$pid$" >/dev/null 2>&1; then
        return 0
    fi
    
    # 方法4: 检查 /proc 文件系统
    if [[ -d "/proc/$pid" ]]; then
        return 0
    fi
    
    return 1
}

# 获取进程信息
get_process_info() {
    local pid=$1
    
    # 尝试获取进程详细信息
    if ps -p $pid >/dev/null 2>&1; then
        local process_info=$(ps -p $pid -o pid,ppid,user,comm,etime --no-headers 2>/dev/null)
        if [[ -n "$process_info" ]]; then
            echo "进程信息: $process_info"
            return
        fi
    fi
    
    # 尝试从 /proc 获取信息
    if [[ -d "/proc/$pid" ]]; then
        local cmdline=$(cat "/proc/$pid/cmdline" 2>/dev/null | tr '\0' ' ')
        local status=$(cat "/proc/$pid/status" 2>/dev/null | grep "State:" | cut -f2)
        echo "命令行: $cmdline"
        echo "状态: $status"
        return
    fi
    
    echo "无法获取进程详细信息"
}

# 检查进程是否还存在，存在就等待
echo "开始检测进程 $TARGET_PID..."

# 首先检查进程是否存在
if ! check_process_exists $TARGET_PID; then
    echo "进程 $TARGET_PID 不存在，直接开始执行"
else
    echo "进程 $TARGET_PID 正在运行，等待其结束..."
    get_process_info $TARGET_PID
    
    # 等待进程结束
    while check_process_exists $TARGET_PID; do
        echo "Process $TARGET_PID is still running..."
        sleep 5
    done
    
    echo "Process $TARGET_PID is no longer running"
fi


echo "Process $TARGET_PID finished. Starting vLLM server..."

########################################################正式代码
export CUDA_VISIBLE_DEVICES=7
bash run_intervention.sh --all --top-k 50