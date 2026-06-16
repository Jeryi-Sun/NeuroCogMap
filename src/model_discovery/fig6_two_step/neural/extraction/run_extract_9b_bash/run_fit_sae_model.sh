#!/bin/bash
# 运行 Ridge 回归拟合脚本
# 新功能：
# - 支持一次预测多个 ROI（用逗号分隔）
# - 不指定 --roi 参数时，默认预测所有可用的 ROI
# - 使用相关性评估和 FDR 校正
# - 结果保存为 JSON 格式

# 使用 language_model 提取器，第 10 层，预测所有 ROI（不指定 --roi 参数）
# layer id 写个 for循环从 0 到 25
# for layer in {0..25}
# do
#   python fit.py --model google/gemma-2-2b --extractor_type language_model --layer $layer --participant_num 10
# done


for layer in {0..269}
do
  python fit.py --model google/gemma-2-9b-it --extractor_type sae_model --parcel_id $layer --participant_num 10
done

# # 使用 language_model 提取器，第 10 层，预测单个 ROI
# python fit.py --model google/gemma-2-2b --extractor_type language_model --layer 10 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)"

# # 使用 language_model 提取器，第 10 层，预测多个 ROI
# python fit.py --model google/gemma-2-2b --extractor_type language_model --layer 10 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex),Left Accumbens,Frontal Medial Cortex"

# # 使用 language_model 提取器，第 10 层，自定义 FDR 显著性水平
# python fit.py --model google/gemma-2-2b --extractor_type language_model --layer 10 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)" --alpha_fdr 0.01

# # 使用 sae_model 提取器，parcel 0，预测单个 ROI
# python fit.py --model google/gemma-2-2b --extractor_type sae_model --parcel_id 0 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)"

# # 使用 sae_model 提取器，parcel 0，预测多个 ROI
# python fit.py --model google/gemma-2-2b --extractor_type sae_model --parcel_id 0 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex),Left Accumbens"

# # 使用 saeact_model 提取器，parcel 5，预测单个 ROI
# python fit.py --model google/gemma-2-2b --extractor_type saeact_model --parcel_id 5 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)"

# # 使用 saeact_model 提取器，parcel 5，预测多个 ROI
# python fit.py --model google/gemma-2-2b --extractor_type saeact_model --parcel_id 5 --roi "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex),Left Accumbens,Frontal Medial Cortex"