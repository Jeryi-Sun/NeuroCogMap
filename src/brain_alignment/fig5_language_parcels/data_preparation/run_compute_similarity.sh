#!/bin/bash
# 运行脚本：计算 Human Parcel 和 LLM Parcel 之间的语义相似度矩阵
# 使用 Qwen3-8b-embedding 模型

# 设置脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# 默认参数
HUMAN_PARCEL_FILE="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
LLM_PARCEL_FILE="/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
OUTPUT_FILE="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/semantic_matrix_gemma2_2b.csv"
MODEL_NAME="/path/to/local_models/Qwen3-Embedding-8B"
BATCH_SIZE=32
LOG_FILE="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/compute_similarity_gemma2_2b.log"
DEVICE="cuda"  # 使用 GPU，如果 GPU 不可用会自动回退到 CPU

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --human_parcel_file)
            HUMAN_PARCEL_FILE="$2"
            shift 2
            ;;
        --llm_parcel_file)
            LLM_PARCEL_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --log_file)
            LOG_FILE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --human_parcel_file FILE   Human Parcel JSON 文件路径 (默认: $HUMAN_PARCEL_FILE)"
            echo "  --llm_parcel_file FILE     LLM Parcel JSON 文件路径 (默认: $LLM_PARCEL_FILE)"
            echo "  --output_file FILE         输出 CSV 文件路径 (默认: $OUTPUT_FILE)"
            echo "  --model_name NAME          Qwen3-8b-embedding 模型名称 (默认: $MODEL_NAME)"
            echo "  --batch_size SIZE          批处理大小 (默认: $BATCH_SIZE)"
            echo "  --log_file FILE            日志文件路径 (默认: $LOG_FILE)"
            echo "  --device DEVICE            设备 (cuda/cpu, 默认: $DEVICE, 自动检测)"
            echo "  -h, --help                 显示此帮助信息"
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
PYTHON_SCRIPT="$SCRIPT_DIR/compute_parcel_similarity.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到 Python 脚本: $PYTHON_SCRIPT"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$HUMAN_PARCEL_FILE" ]; then
    echo "错误: 找不到 Human Parcel 文件: $HUMAN_PARCEL_FILE"
    exit 1
fi

if [ ! -f "$LLM_PARCEL_FILE" ]; then
    echo "错误: 找不到 LLM Parcel 文件: $LLM_PARCEL_FILE"
    exit 1
fi

# 运行 Python 脚本（使用 conda 的 lit 环境）
echo "开始计算 Human-LLM Parcel 语义相似度矩阵..."
echo "Human Parcel 文件: $HUMAN_PARCEL_FILE"
echo "LLM Parcel 文件: $LLM_PARCEL_FILE"
echo "输出文件: $OUTPUT_FILE"
echo "模型: $MODEL_NAME"
echo "批处理大小: $BATCH_SIZE"
echo "设备: $DEVICE (自动检测，优先使用 GPU)"
echo ""

conda run -n lit python "$PYTHON_SCRIPT" \
    --human_parcel_file "$HUMAN_PARCEL_FILE" \
    --llm_parcel_file "$LLM_PARCEL_FILE" \
    --output_file "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --log_file "$LOG_FILE" \
    --device "$DEVICE"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ 所有任务完成！"
    echo "结果已保存到: $OUTPUT_FILE"
    echo "日志已保存到: $LOG_FILE"
else
    echo ""
    echo "✗ 任务执行失败，退出码: $exit_code"
    echo "请查看日志文件: $LOG_FILE"
    exit $exit_code
fi

