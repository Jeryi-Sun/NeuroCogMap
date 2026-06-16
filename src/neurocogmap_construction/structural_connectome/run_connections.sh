#!/usr/bin/env bash
set -euo pipefail

# Parcel-Parcel结构连接构建脚本
# 使用方法: ./run_connections.sh [test|full|visualize]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置Python路径
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# 配置参数
CONFIG_FILE="configs/paths_8b.yaml"
PYTHON_BIN="python"

# 检查配置文件是否存在
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查Python依赖
echo "🔍 检查Python依赖..."
$PYTHON_BIN -c "import numpy, pandas, matplotlib, seaborn, yaml, tqdm" 2>/dev/null || {
    echo "❌ 缺少必要的Python依赖，请安装："
    echo "   pip install numpy pandas matplotlib seaborn pyyaml tqdm"
    exit 1
}

# 创建输出目录
mkdir -p outputs

# 根据参数选择运行模式
case "${1:-test}" in
    "test")
        echo "🧪 运行测试模式（计算少量连接）..."
        $PYTHON_BIN build_parcel_connections.py \
            --config "$CONFIG_FILE" \
            --test \
            --max_connections 50 \
            --visualize
        ;;
    "full")
        echo "🚀 运行完整模式（计算所有连接）..."
        $PYTHON_BIN build_parcel_connections.py \
            --config "$CONFIG_FILE" \
            --visualize
        ;;
    "visualize")
        echo "🎨 仅生成可视化图表..."
        $PYTHON_BIN build_parcel_connections.py \
            --config "$CONFIG_FILE" \
            --test \
            --max_connections 10 \
            --visualize
        ;;
    "help"|"-h"|"--help")
        echo "使用方法: $0 [test|full|visualize|help]"
        echo ""
        echo "模式说明:"
        echo "  test      - 测试模式，计算少量连接（默认）"
        echo "  full      - 完整模式，计算所有连接"
        echo "  visualize - 仅生成可视化图表"
        echo "  help      - 显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  $0 test      # 运行测试"
        echo "  $0 full      # 运行完整计算"
        echo "  $0 visualize # 仅生成图表"
        exit 0
        ;;
    *)
        echo "❌ 未知模式: $1"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac

echo "✅ 脚本执行完成！"
echo "📁 结果文件保存在: outputs/"
