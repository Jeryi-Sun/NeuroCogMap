#!/usr/bin/env bash
# 功能：批量汇总 neural/fits 下每个 JSON 的 participant mean_score
# - Pearson 相关性会先做 Fisher Z 变换再聚合
# - 输出 std、95% bootstrap CI
# - 保存每个 participant 的 mean_score/median/std/min/max，便于后续显著性检验
# - 结果写回源文件同目录，文件名后缀为 *_summary.json
#
# 使用说明（示例）：
# 1) 默认：对全部 participant 统计（覆盖已有 summary）
#    bash run_summary_participant.sh
#
# 2) 只用前 10 个 participant（按 participant_0, participant_1... 顺序）
#    python summarize_participant_mean_score.py \
#      --fits-root /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits \
#      --no-skip-existing \
#      --participant-first-n 10 \
#      --n-bootstrap 5000 \
#      --seed 42
#
# 3) 只用指定 participant_id（逗号分隔）
#    python summarize_participant_mean_score.py \
#      --fits-root /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits \
#      --no-skip-existing \
#      --participant-ids '0,3,7' \
#      --n-bootstrap 5000 \
#      --seed 42
#
# 4) 从文件读取 participant_id（每行一个；允许空行和 # 注释）
#    # 例：/abs/path/participant_ids.txt 内容：
#    #   0
#    #   3
#    #   7
#    python summarize_participant_mean_score.py \
#      --fits-root /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits \
#      --no-skip-existing \
#      --participant-ids-file /abs/path/participant_ids.txt \
#      --n-bootstrap 5000 \
#      --seed 42
#
# 5) 只想增量生成（summary 已存在则跳过，不覆盖）
#    python summarize_participant_mean_score.py \
#      --fits-root /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits \
#      --skip-existing \
#      --n-bootstrap 5000 \
#      --seed 42

# python summarize_participant_mean_score.py \
#   --fits-root /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits \
#   --no-skip-existing \
#   --n-bootstrap 2000 \
#   --seed 42

python summarize_participant_mean_score.py \
    --fits-root /path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits \
    --no-skip-existing \
    --participant-first-n 10 \
    --n-bootstrap 5000 \
    --seed 42