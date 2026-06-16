story_name="adventuresinsayingyes"
python /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data_preparation/map_capability_to_cog_category_and_analyze.py \
  --capability_matrix_file /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/${story_name}/prediction_matrix_gemma2_2b_capability.csv \
  --capability_cog_mapping_file /path/to/project_root/capability_analysis/data/capability_cog_mapping_flat.json \
  --parcel_desc_file /path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json \
  --output_category_matrix_file /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/${story_name}/prediction_matrix_gemma2_2b_cog_category.csv \
  --output_yeo7_summary_file /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/${story_name}/yeo7_cog_category_distribution.csv \
  --ignore_hemisphere \
  # --use_first_letter_only