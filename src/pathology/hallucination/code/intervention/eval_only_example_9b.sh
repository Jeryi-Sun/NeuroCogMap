#!/bin/bash
# 仅评估模式：从已有干预结果文件中读取数据，只进行评估（不重新生成干预结果）。
# 按不同干预强度分别处理 9b 模型结果：遍历 strength_*/ 下所有 gemma-2-9b-it_*_intervention.json。
# 用法：
#   ./eval_only_example_9b.sh                  # 处理所有强度、所有数据集
#   ./eval_only_example_9b.sh 0.1 0.3         # 仅处理强度 0.1 和 0.3
#   SKIP_EXISTING=1 ./eval_only_example_9b.sh # 若已有 _intervention_analysis.json 则跳过

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/run_intervention.py"
CONFIG_FILE="$SCRIPT_DIR/config_9b.json"
INTERVENTION_ROOT="/path/to/project_root/safety_explanation/hallucination/results/intervention"

# 9b 模型：文件名前缀与数据集
MODEL_PREFIX="gemma-2-9b-it"
DATASETS="MedHallu nq_open truthfulqa"

VLLM_URL="${VLLM_URL:-http://127.0.0.1:8001/v1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
# 若已存在 _intervention_analysis.json 是否跳过（0=不跳过 1=跳过）
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# 若传入参数则视为要处理的强度列表，否则扫描所有 strength_*
if [[ $# -ge 1 ]]; then
    STRENGTHS=("$@")
else
    STRENGTHS=()
    for d in "$INTERVENTION_ROOT"/strength_*; do
        [[ -d "$d" ]] || continue
        s="${d##*strength_}"
        STRENGTHS+=("$s")
    done
    # 按数值排序
    IFS=$'\n' sorted=($(sort -V <<<"${STRENGTHS[*]}")); unset IFS
    STRENGTHS=("${sorted[@]}")
fi

python_cmd="python -u"

echo "📊 仅评估模式 (9b) - 按干预强度分别处理"
echo "============================================================"
echo "干预根目录: $INTERVENTION_ROOT"
echo "强度列表: ${STRENGTHS[*]}"
echo "数据集: $DATASETS"
echo "vLLM地址: $VLLM_URL"
echo "SKIP_EXISTING: $SKIP_EXISTING"
echo "============================================================"

total=0
skipped=0
failed=0
for strength in "${STRENGTHS[@]}"; do
    for dataset in $DATASETS; do
        results_file="$INTERVENTION_ROOT/strength_${strength}/${MODEL_PREFIX}_${dataset}_intervention.json"
        analysis_file="$INTERVENTION_ROOT/strength_${strength}/${MODEL_PREFIX}_${dataset}_intervention_analysis.json"
        if [[ ! -f "$results_file" ]]; then
            echo "[SKIP] 不存在: $results_file"
            ((skipped++)) || true
            continue
        fi
        if [[ "$SKIP_EXISTING" == "1" ]] && [[ -f "$analysis_file" ]]; then
            echo "[SKIP] 已存在分析: $analysis_file"
            ((skipped++)) || true
            continue
        fi
        ((total++)) || true
        echo ""
        echo "[INFO] 处理 strength=${strength} dataset=${dataset} -> $results_file"
        cmd="$python_cmd $PY_SCRIPT --mode eval_only --config $CONFIG_FILE"
        cmd="$cmd --eval_only_from_file $results_file --vllm_url $VLLM_URL --log_level $LOG_LEVEL"
        if ! $cmd; then
            echo "[ERROR] 评估失败: $results_file"
            ((failed++)) || true
        fi
    done
done

echo ""
echo "============================================================"
echo "处理数: $total  跳过: $skipped  失败: $failed"
if [[ $failed -gt 0 ]]; then
    echo "❌ 存在失败任务"
    exit 1
fi
echo "✅ 评估完成"
