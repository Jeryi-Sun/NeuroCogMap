#!/usr/bin/env bash

# 分析示例脚本：提取指定 Parcel ID 上激活最大的 K 个案例
# 该脚本演示如何使用 extract_top_k_cases.py 进行案例分析

# 设置基本路径
MODEL_NAME="nq_open_gemma-2-2b"
Correct_or_Incorrect="incorrect"
SCRIPT_DIR="/path/to/project_root/safety_explanation/hallucination/code/analysis"
TOKEN_PARCELS_FILE="/path/to/project_root/safety_explanation/hallucination/results/${MODEL_NAME}/parcels_token_acts/${Correct_or_Incorrect}/token_parcels.jsonl"
ORIGINAL_FILE="/path/to/project_root/safety_explanation/hallucination/results/${MODEL_NAME}/${Correct_or_Incorrect}.jsonl"
echo "开始 Parcel 案例分析... ${MODEL_NAME}"
echo "=========================================="

# 示例1: 分析 Parcel 0 的 Top-5 案例
echo "示例1: 分析 Parcel 0 的 Top-5 案例"
python3 "$SCRIPT_DIR/extract_top_k_cases.py" \
    --parcel-id 167 \
    --k 10 \
    --token-parcels-file "$TOKEN_PARCELS_FILE" \
    --original-file "$ORIGINAL_FILE" \
    --output-file "/path/to/project_root/safety_explanation/hallucination/results/analysis_output/${MODEL_NAME}/extract_top_k_cases/parcel_167_top_5_cases.jsonl" \
    --aggregation-method sum \
    --verbose

echo ""
echo "=========================================="

# # 示例2: 分析 Parcel 10 的 Top-10 案例，使用最大值聚合
# echo "示例2: 分析 Parcel 10 的 Top-10 案例 (最大值聚合)"
# python3 "$SCRIPT_DIR/extract_top_k_cases.py" \
#     --parcel-id 10 \
#     --k 10 \
#     --token-parcels-file "$TOKEN_PARCELS_FILE" \
#     --original-file "$ORIGINAL_FILE" \
#     --output-file "parcel_10_top_10_cases_max.jsonl" \
#     --aggregation-method max \
#     --verbose

# echo ""
# echo "=========================================="

# # 示例3: 分析 Parcel 50 的 Top-20 案例，使用平均值聚合
# echo "示例3: 分析 Parcel 50 的 Top-20 案例 (平均值聚合)"
# python3 "$SCRIPT_DIR/extract_top_k_cases.py" \
#     --parcel-id 50 \
#     --k 20 \
#     --token-parcels-file "$TOKEN_PARCELS_FILE" \
#     --original-file "$ORIGINAL_FILE" \
#     --output-file "parcel_50_top_20_cases_mean.jsonl" \
#     --aggregation-method mean \
#     --verbose

# echo ""
# echo "=========================================="

# # 示例4: 批量分析多个 Parcel
# echo "示例4: 批量分析多个 Parcel (Parcels 0, 10, 20, 30, 40)"
# for parcel_id in 0 10 20 30 40; do
#     echo "分析 Parcel $parcel_id..."
#     python3 "$SCRIPT_DIR/extract_top_k_cases.py" \
#         --parcel-id "$parcel_id" \
#         --k 10 \
#         --token-parcels-file "$TOKEN_PARCELS_FILE" \
#         --original-file "$ORIGINAL_FILE" \
#         --output-file "parcel_${parcel_id}_top_10_cases.jsonl" \
#         --aggregation-method sum \
#         --skip-existing
# done

echo ""
echo "=========================================="
echo "所有分析完成！"
echo "生成的文件："
ls -la parcel_*_top_*_cases.jsonl 2>/dev/null || echo "没有找到输出文件"
