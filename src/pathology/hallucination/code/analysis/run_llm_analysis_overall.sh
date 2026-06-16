#!/bin/bash
# LLM分析报告生成示例脚本

# 设置路径
BASE_DIR="/path/to/project_root"
ANALYSIS_DIR="${BASE_DIR}/safety_explanation/hallucination/code/analysis"
RESULTS_DIR="${BASE_DIR}/safety_explanation/hallucination/results/analysis_output"

# 输入文件路径
PARCEL_DESC="${BASE_DIR}/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
CAP_DESC="${BASE_DIR}/capability_analysis/data/capability_descriptions/capability_descriptions_run2.json"
COG_MAPPING="${BASE_DIR}/capability_analysis/data/capability_cog_mapping.json"

MODEL_DATA_LIST=(
    "MedHallu_gemma-2-2b"
    "HaluEval_gemma-2-2b"
    "dolly_close_gemma-2-2b"
    "nq_open_gemma-2-2b"
    "sciq_gemma-2-2b"
    "triviaqa_gemma-2-2b"
)

echo "开始生成LLM分析报告..."
echo "Parcel描述文件: ${PARCEL_DESC}"
echo "Capability描述文件: ${CAP_DESC}"
echo "认知层级映射文件: ${COG_MAPPING}"

# 检查基础输入文件是否存在
for file in "${PARCEL_DESC}" "${CAP_DESC}" "${COG_MAPPING}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 文件不存在: $file"
        exit 1
    fi
done

# 遍历每个模型数据
for MODEL_DATA in "${MODEL_DATA_LIST[@]}"; do
    echo "=========================================="
    echo "处理模型: ${MODEL_DATA}"
    echo "=========================================="
    
    # 分析结果文件路径
    PARCEL_DIFF="${RESULTS_DIR}/${MODEL_DATA}/parcel_level/parcel_activation_diff.json"
    CAP_DIFF="${RESULTS_DIR}/${MODEL_DATA}/capability_level/capability_activation_diff.json"
    CORRECT_JSONL="$BASE_DIR/safety_explanation/hallucination/results/${MODEL_DATA}/correct.jsonl"
    INCORRECT_JSONL="$BASE_DIR/safety_explanation/hallucination/results/${MODEL_DATA}/incorrect.jsonl"
    # 输出文件路径
    OUTPUT_REPORT="${RESULTS_DIR}/${MODEL_DATA}/llm_analysis/llm_analysis_report.md"
    
    echo "Parcel分析结果: ${PARCEL_DIFF}"
    echo "Capability分析结果: ${CAP_DIFF}"
    echo "输出报告: ${OUTPUT_REPORT}"
    
    # 检查模型相关的输入文件是否存在
    for file in "${PARCEL_DIFF}" "${CAP_DIFF}" "${CORRECT_JSONL}" "${INCORRECT_JSONL}"; do
        if [ ! -f "$file" ]; then
            echo "错误: 文件不存在: $file"
            echo "跳过模型: ${MODEL_DATA}"
            continue 2
        fi
    done
    
    # 创建输出目录（如果不存在）
    mkdir -p "$(dirname "${OUTPUT_REPORT}")"
    
    # 运行LLM分析脚本
    python3 "${ANALYSIS_DIR}/analysis_llm_summary.py" \
        --parcel_desc "${PARCEL_DESC}" \
        --cap_desc "${CAP_DESC}" \
        --parcel_diff "${PARCEL_DIFF}" \
        --cap_diff "${CAP_DIFF}" \
        --cog_mapping "${COG_MAPPING}" \
        --correct_data "${CORRECT_JSONL}" \
        --incorrect_data "${INCORRECT_JSONL}" \
        --out "${OUTPUT_REPORT}" \
        --vllm_url "http://0.0.0.0:8001/v1" \
        --api_key "abcabc" \
        --data_type "factqa"\
        --model_data "${MODEL_DATA}"
    
    if [ $? -eq 0 ]; then
        echo "模型 ${MODEL_DATA} 的LLM分析报告生成成功！"
        echo "报告保存位置: ${OUTPUT_REPORT}"
    else
        echo "模型 ${MODEL_DATA} 的LLM分析报告生成失败！"
        echo "继续处理下一个模型..."
    fi
    
    echo ""
done

echo "所有模型的LLM分析报告生成完成！"
