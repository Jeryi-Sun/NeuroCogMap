#!/usr/bin/env bash
# Parcel功能分析运行脚本
# 要等待的进程 PID
TARGET_PID=1943271

# 检查进程是否还存在，存在就等待
while kill -0 $TARGET_PID 2>/dev/null; do
    echo "Process $TARGET_PID is still running..."
    sleep 5
done

echo "Process $TARGET_PID finished. Starting vLLM server..."

set -euo pipefail

# 项目路径
PROJ="/path/to/project_root/neural_area/divide_area_by_sae_act"
CLUSTER_OUTPUT_DIR="$PROJ/cluster_output_llama_8b_pt_neural"
OUTPUT_DIR="$PROJ/parcel_functionality_analysis" # 这个没啥用, 直接保存到输入数据的文件夹里

cd "$PROJ"

echo "🚀 开始Parcel功能分析"
echo "=" * 50

# 检查vLLM服务是否运行
echo "检查vLLM服务状态..."
if curl -s http://0.0.0.0:8001/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM服务正在运行"
else
    echo "❌ vLLM服务未运行，请先启动服务："
    echo "   vllm serve /path/to/local_models/gpt-oss-20b --host 0.0.0.0 --port 8001 --api-key abcabc"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo ""
echo "选择分析模式："
echo "1. 分析单个文件（测试用）"
echo "2. 批量分析所有文件"
echo "3. 分析特定模式的文件"
echo "4. 只生成汇总报告（跳过API调用）"
read -p "请选择 (1-4): " choice
case $choice in
    1)
        echo "分析单个文件..."
        # 查找第一个topsamples文件进行测试
        TOPSAMPLES_FILE=$(find "$CLUSTER_OUTPUT_DIR" -name "latent_parcel_topsamples.json" | head -1)
        if [ -z "$TOPSAMPLES_FILE" ]; then
            echo "❌ 未找到topsamples文件"
            exit 1
        fi
        echo "使用文件: $TOPSAMPLES_FILE"
        
        python analyze_parcel_functionality.py \
            --topsamples_file "$TOPSAMPLES_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --n_top_samples 1000 \
            --delay 2.0
        ;;
    2)
        echo "批量分析所有文件..."
        python batch_analyze_parcels.py \
            --cluster_output_dir "$CLUSTER_OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --n_top_samples 1000     \
            --delay 1.5 \
            # --skip_existing
        ;;
    3)
        echo "分析特定模式的文件..."
        pattern="clustering_results_sentence"
        python batch_analyze_parcels.py \
            --cluster_output_dir "$CLUSTER_OUTPUT_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --n_top_samples 1000 \
            --delay 1.5 \
            --filter_pattern "$pattern" \
            --vllm_url "http://0.0.0.0:8001/v1" \
            --skip_existing
        ;;
    4)
        echo "只生成汇总报告..."
        # 查找已有的分析文件
        ANALYSIS_FILES=$(find "$OUTPUT_DIR" -name "*_functionality_analysis.json")
        if [ -z "$ANALYSIS_FILES" ]; then
            echo "❌ 未找到已有的分析文件"
            exit 1
        fi
        
        for analysis_file in $ANALYSIS_FILES; do
            base_name=$(basename "$analysis_file" _functionality_analysis.json)
            summary_file="$OUTPUT_DIR/${base_name}_functionality_summary.json"
            
            echo "处理: $base_name"
            python -c "
import json
from analyze_parcel_functionality import ParcelFunctionalityAnalyzer

analyzer = ParcelFunctionalityAnalyzer()
with open('$analysis_file', 'r', encoding='utf-8') as f:
    analyses = json.load(f)
analyzer.generate_summary_report(analyses, '$summary_file')
print('完成: $base_name')
"
        done
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "✅ 分析完成！"
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "查看结果："
echo "  ls -la $OUTPUT_DIR"
echo ""
echo "查看汇总报告："
echo "  find $OUTPUT_DIR -name '*_functionality_summary.json' | head -1 | xargs cat" 