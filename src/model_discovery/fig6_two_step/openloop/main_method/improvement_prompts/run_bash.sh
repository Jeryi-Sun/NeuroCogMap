#!/usr/bin/env bash
#
# 功能: 使用 comparison_results 目录中的 parcel_activation_cogneuromap 改进建议 JSON，
# 生成两套用于“改进 Dual-Systems 认知模型实现”的 prompt 文件：
#   1) simple 行为数据版
#   2) full CogNeuroMap 版
#
# 生成的 prompt 在模板中写入不同的 MODEL_SUFFIX / RECORD_SUFFIX，
# 便于后续 LLM 生成的改进模型代码和 markdown record 使用不同后缀区分。

set -e

PYTHON_BIN=python3
SCRIPT="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_prompts/generate_variant_improvement_prompts.py"
TEMPLATE_MD="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_prompts/improvement_prompt.md"
OUTPUT_DIR="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_prompts"

# simple 行为数据版：后缀 _cogneuromap_simple
$PYTHON_BIN "$SCRIPT" \
  --source_schema parcel_activation \
  --suggestions_json /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_suggestion_results/parcel_activation_cogneuromap_improvement_suggestions_simple.json \
  --template_md "$TEMPLATE_MD" \
  --output_dir "$OUTPUT_DIR" \
  --model_suffix "_cogneuromap_simple" \
  --record_suffix "_cogneuromap_simple"

# full CogNeuroMap 版：后缀 _cogneuromap_full
$PYTHON_BIN "$SCRIPT" \
  --source_schema parcel_activation \
  --suggestions_json /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_suggestion_results/parcel_activation_cogneuromap_improvement_suggestions.json \
  --template_md "$TEMPLATE_MD" \
  --output_dir "$OUTPUT_DIR" \
  --model_suffix "_cogneuromap_full" \
  --record_suffix "_cogneuromap_full"