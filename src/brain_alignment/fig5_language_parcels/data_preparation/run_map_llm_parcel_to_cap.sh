story_name="smoke_story"
python /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data_preparation/map_llm_parcel_to_capability.py \
  --input_file /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/${story_name}/prediction_matrix_gemma2_2b.csv \
  --mapping_json /path/to/project_root/neural_area/connect_cap_parcel/results/aggrate_final_9b/final_capability_parcel_all.json \
  --output_file /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/${story_name}/prediction_matrix_gemma2_2b_capability.csv \
  --parcel_desc_file /path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json \
  --top_k 5 \
  --topk_output_file /path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/draw_result/${story_name}/fig2_top_llm_capabilities_per_human.csv