#!/bin/bash
# 运行脚本：为 Schaefer2018_100Parcels_7Networks 生成功能描述
# 功能：读取 CSV 文件中的 cognitive term 关联强度，调用 LLM 为每个 Parcel 生成功能描述

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 默认参数
CSV_PATH="$SCRIPT_DIR/ns_scale100.csv"
OUTPUT_FILE="$SCRIPT_DIR/parcel_descriptions.json"
VLLM_URL="http://0.0.0.0:8000/v1"
API_KEY="abcabc"
SKIP_EXISTING="--skip_existing"
CHECK_API="--check_api"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --csv_path)
            CSV_PATH="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --vllm_url)
            VLLM_URL="$2"
            shift 2
            ;;
        --api_key)
            API_KEY="$2"
            shift 2
            ;;
        --no_skip_existing)
            SKIP_EXISTING="--no_skip_existing"
            shift
            ;;
        --no_check_api)
            CHECK_API=""
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --csv_path PATH        输入 CSV 文件路径 (默认: $CSV_PATH)"
            echo "  --output_file FILE     输出 JSON 文件路径 (默认: $OUTPUT_FILE)"
            echo "  --vllm_url URL         vLLM API URL (默认: $VLLM_URL)"
            echo "  --api_key KEY          API Key (默认: $API_KEY)"
            echo "  --no_skip_existing     不跳过已存在的 parcel，重新生成"
            echo "  --no_check_api         不检查 API 是否可用"
            echo "  -h, --help            显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查 Python 脚本是否存在
PYTHON_SCRIPT="$SCRIPT_DIR/generate_parcel_descriptions.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 检查 CSV 文件是否存在
if [ ! -f "$CSV_PATH" ]; then
    echo "错误: 找不到 CSV 文件: $CSV_PATH"
    exit 1
fi

# 运行 Python 脚本
echo "开始生成 Parcel 功能描述..."
echo "CSV 文件: $CSV_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "vLLM URL: $VLLM_URL"
echo ""

python3 "$PYTHON_SCRIPT" \
    --csv_path "$CSV_PATH" \
    --output_file "$OUTPUT_FILE" \
    --vllm_url "$VLLM_URL" \
    --api_key "$API_KEY" \
    $SKIP_EXISTING \
    $CHECK_API

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ 所有任务完成！"
else
    echo ""
    echo "✗ 任务执行失败，退出码: $exit_code"
    exit $exit_code
fi

