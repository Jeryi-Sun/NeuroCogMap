#!/bin/bash

# 运行capability-parcel语义相似性分析脚本
# 使用BGE大模型进行语义相似性计算

echo "开始运行capability-parcel语义相似性分析..."
echo "使用模型: Qwen3-Embedding-8B"

# 设置工作目录
cd /path/to/project_root/neural_area/connect_cap_parcel/code

# 运行Python脚本
python3 compute_semantic_similarity.py \
    --capability_file /path/to/project_root/capability_analysis/data/capability_descriptions/capability_descriptions_run2.json \
    --parcel_file /path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_llama_8b_pt/clustering_results_sentence_prep0.01_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels240/latent_parcel_topsamples_functionality_analysis.json \
    --output_file /path/to/project_root/neural_area/connect_cap_parcel/results/cap_parcel_similarity/capability_parcel_similarity_matrix_qwen_8b.csv \
    --model_name /path/to/local_models/Qwen3-Embedding-8B/

echo "分析完成！"
echo "结果已保存到: capability_parcel_similarity_matrix_qwen_9b.csv"
echo "详细结果保存到: capability_parcel_similarity_matrix_qwen_9b_detailed.csv" 