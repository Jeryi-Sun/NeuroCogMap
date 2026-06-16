#!/usr/bin/env bash
# 功能：运行 panel_a 指定文件分析脚本，生成每个文件的独立 summary 与整合 CSV
# 输出位置：analysis_results/result/panel_a
#
# 关键说明：
# - Pearson 相关系数聚合使用 Fisher Z 变换；输出含 std 与 95% bootstrap CI
# - 可选参与者筛选（前 N 个或按 participant_id 列表/文件）
# - 默认覆盖已有结果（--no-skip-existing）；如需增量跳过已存在 summary，请使用 --skip-existing
#
# 常用示例：
# 1) 使用全部参与者（覆盖写入，bootstrap=5000）：
#    bash run_panel_a.sh
#
# 2) 仅使用前 10 个参与者：
#    python panel_a_analyze_selected_fits.py \
#      --participant-first-n 10 \
#      --n-bootstrap 5000 \
#      --seed 42
#
# 3) 指定 participant_id 列表：
#    python panel_a_analyze_selected_fits.py \
#      --participant-ids '0,3,7' \
#      --n-bootstrap 5000 \
#      --seed 42
#
# 4) 从文件读取 participant_id（每行一个；允许空行和 # 注释）：
#    python panel_a_analyze_selected_fits.py \
#      --participant-ids-file /abs/path/participant_ids.txt \
#      --n-bootstrap 5000 \
#      --seed 42
#
# 5) 增量执行（已存在 summary 则跳过）：
#    python panel_a_analyze_selected_fits.py \
#      --skip-existing \
#      --n-bootstrap 5000 \
#      --seed 42

python /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/analysis_results/code/panel_a_analyze_selected_fits.py \
  --n-bootstrap 5000 \
  --participant-ids "1, 8, 11, 17, 27, 50, 55, 60, 70, 87" \
  --seed 42

