#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
谄媚干预结果评测脚本

参考 fairness_bias 的 eval_intervention.py，对谄媚干预结果进行评测。
流程：
1. 从干预 JSON 提取 baseline_results 和 intervention_results
2. 从原始 gen_eval 文件按 index 对齐获取 original_record
3. 构造 baseline_gen.jsonl、intervention_gen.jsonl（sycophancy_eval 输入格式）
4. 调用 sycophancy_eval 逻辑进行组内评估（answer: is_correct；feedback: 成对比较）
5. 调用 classify_sycophancy 逻辑按组分类谄媚/非谄媚
6. 输出 baseline_eval.jsonl、intervention_eval.jsonl、summary.json

使用示例:
python eval_sycophancy_intervention.py \
  --intervention_json results/intervention/strength_0.1/gemma-2-2b_answer_intervention.json \
  --output_dir results/intervention/strength_0.1 \
  --vllm_url http://0.0.0.0:8001/v1 \
  --skip_existing
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加 sycophancy-eval 和 code 到路径
BASE = Path(__file__).resolve().parent.parent.parent
SYCOPHANCY_EVAL_DIR = BASE / "dataset" / "sycophancy-eval"
CODE_DIR = BASE / "code"
for d in [str(SYCOPHANCY_EVAL_DIR), str(CODE_DIR)]:
    if d not in sys.path:
        sys.path.insert(0, d)

from sycophancy_eval import (  # noqa: E402
    eval_answer_records,
    eval_feedback_records,
    load_jsonl,
    load_existing_indices,
)
from classify_sycophancy import (  # noqa: E402
    classify_answer_records,
    classify_feedback_records,
    load_jsonl as load_jsonl_classify,
)


def load_intervention_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def build_gen_records_from_intervention(
    baseline_results: List[Dict],
    intervention_results: List[Dict],
    original_records: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    从 baseline_results 和 intervention_results 构造 sycophancy_eval 期望的 gen_records。
    - 优先使用 baseline_results / intervention_results 中嵌入的 original_record（来自 run_intervention 的 meta）
    - 若无，则按位置从 original_records（gen_eval 文件）兜底对齐（强制启用）
    """
    baseline_gen = []
    interv_gen = []
    skipped_no_original = 0
    empty_baseline_answer = 0
    empty_interv_answer = 0

    for i, (br, ir) in enumerate(zip(baseline_results, intervention_results)):
        # 1) 优先使用已有 original_record
        orec = br.get("original_record") or ir.get("original_record")
        # 2) 兜底使用 gen_eval 原始记录（强制对齐）
        if orec is None and i < len(original_records):
            src = original_records[i]
            orec = src.get("original_record", src)
        # 3) 仍然缺失则跳过
        if orec is None:
            skipped_no_original += 1
            continue

        idx = br.get("index", ir.get("index", i + 1))
        b_ans = br.get("baseline_text", "")
        i_ans = ir.get("intervention_text", "")
        if not isinstance(b_ans, str) or not b_ans.strip():
            empty_baseline_answer += 1
        if not isinstance(i_ans, str) or not i_ans.strip():
            empty_interv_answer += 1

        baseline_gen.append({
            "index": idx,
            "original_record": orec,
            "model_answer": b_ans,
        })
        interv_gen.append({
            "index": idx,
            "original_record": orec,
            "model_answer": i_ans,
        })
    # 诊断日志（构造阶段）
    print("\n[DIAG] 构造 gen_records 诊断信息")
    print(f"[DIAG] baseline_results 条数: {len(baseline_results)}")
    print(f"[DIAG] intervention_results 条数: {len(intervention_results)}")
    print(f"[DIAG] original_records 条数: {len(original_records)}")
    print(f"[DIAG] 生成 baseline_gen 条数: {len(baseline_gen)}")
    print(f"[DIAG] 生成 intervention_gen 条数: {len(interv_gen)}")
    print(f"[DIAG] 因缺失 original_record 被跳过条数: {skipped_no_original}")
    print(f"[DIAG] baseline 空回答条数: {empty_baseline_answer}")
    print(f"[DIAG] intervention 空回答条数: {empty_interv_answer}\n")

    return baseline_gen, interv_gen


def load_original_records(dataset_path: str) -> List[Dict]:
    """从 gen_eval 文件加载记录，按文件顺序返回。"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"原始数据文件不存在: {dataset_path}")
    return load_jsonl(dataset_path)


def run_eval_and_classify(
    gen_records: List[Dict],
    dataset_type: str,
    eval_file: str,
    args: Any,
    evaluated_indices: set,
) -> List[Dict]:
    """运行 sycophancy_eval 和 classify，返回 eval_records 及谄媚率"""
    if dataset_type == "feedback":
        eval_feedback_records(gen_records, args, eval_file, evaluated_indices)
    else:
        eval_answer_records(gen_records, args, eval_file, evaluated_indices)

    eval_records = load_jsonl_classify(eval_file)
    if dataset_type == "answer":
        incorrect, correct = classify_answer_records(eval_records)
    else:
        incorrect, correct = classify_feedback_records(eval_records)

    total = len(incorrect) + len(correct)
    sycophancy_rate = (len(incorrect) / total * 100.0) if total > 0 else 0.0
    return eval_records, len(incorrect), len(correct), sycophancy_rate


def main():
    import argparse

    parser = argparse.ArgumentParser(description="谄媚干预结果评测")
    parser.add_argument("--intervention_json", type=str, required=True, help="干预结果 JSON 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录（与 fairness_bias 一致，保存到干预文件同目录）")
    parser.add_argument("--vllm_url", type=str, default="http://0.0.0.0:8001/v1", help="vLLM 地址")
    parser.add_argument("--api_key", type=str, default="abcabc", help="API 密钥")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已评测记录")
    parser.add_argument("--intervention_strength", type=str, default=None, help="指定干预强度，默认取第一个")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    # 加载干预结果
    intervention_data = load_intervention_json(args.intervention_json)
    dataset_info = intervention_data.get("dataset_info", {})
    dataset_path = dataset_info.get("dataset_path", "")
    dataset_name = dataset_info.get("dataset_name", "answer")
    model_name = dataset_info.get("model_name", "gemma-2-2b")
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    intervention_results = intervention_data.get("intervention_results", {})
    if not intervention_results:
        print("[ERROR] 干预结果中无 intervention_results 字段")
        sys.exit(1)

    strength_key = args.intervention_strength or list(intervention_results.keys())[0]
    if strength_key not in intervention_results:
        print(f"[ERROR] 强度 {strength_key} 不存在，可用: {list(intervention_results.keys())}")
        sys.exit(1)

    strength_data = intervention_results[strength_key]
    baseline_results = strength_data.get("baseline_results", [])
    interv_results = strength_data.get("intervention_results", [])

    if not baseline_results or not interv_results:
        print("[ERROR] 缺少 baseline_results 或 intervention_results")
        sys.exit(1)

    # 若未配置 dataset_path，从文件名推断（保持原逻辑）
    if not dataset_path:
        base_dir = BASE / "dataset" / "sycophancy-eval" / "results"
        dataset_path = str(base_dir / f"{dataset_name}_with_groups_{model_name}_gen_eval.jsonl")

    # 强制加载原始记录用于对齐兜底（不再以是否带 original_record 为条件）
    try:
        records_list = load_original_records(dataset_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    baseline_gen, interv_gen = build_gen_records_from_intervention(
        baseline_results, interv_results, records_list
    )

    if not baseline_gen or not interv_gen:
        print("[ERROR] 无法构造 gen_records，请检查 original_record 对齐")
        sys.exit(1)

    # 检测数据集类型
    dataset_type = "feedback" if "feedback" in dataset_name else "answer"

    # 输出文件命名（与 fairness_bias 一致）
    # gemma-2-2b_answer_intervention.json -> gemma-2-2b_answer_baseline_eval.jsonl
    stem = Path(args.intervention_json).stem.replace("_intervention", "")
    baseline_eval_file = os.path.join(args.output_dir, f"{stem}_baseline_eval.jsonl")
    interv_eval_file = os.path.join(args.output_dir, f"{stem}_intervention_eval.jsonl")
    summary_file = os.path.join(args.output_dir, f"{stem}_summary.json")

    baseline_evaluated = load_existing_indices(baseline_eval_file) if args.skip_existing else set()
    interv_evaluated = load_existing_indices(interv_eval_file) if args.skip_existing else set()

    if not args.skip_existing:
        for f in [baseline_eval_file, interv_eval_file, summary_file]:
            if os.path.exists(f):
                os.remove(f)

    # 评测 baseline
    print("\n" + "=" * 60 + "\n开始评测 baseline\n" + "=" * 60)
    _, base_inc, base_corr, base_rate = run_eval_and_classify(
        baseline_gen, dataset_type, baseline_eval_file, args, baseline_evaluated
    )

    # 评测 intervention
    print("\n" + "=" * 60 + "\n开始评测 intervention\n" + "=" * 60)
    _, int_inc, int_corr, int_rate = run_eval_and_classify(
        interv_gen, dataset_type, interv_eval_file, args, interv_evaluated
    )

    # 汇总
    summary = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "intervention_strength": strength_key,
        "baseline_stats": {
            "sycophantic_count": base_inc,
            "non_sycophantic_count": base_corr,
            "sycophantic_rate": base_rate,
        },
        "intervention_stats": {
            "sycophantic_count": int_inc,
            "non_sycophantic_count": int_corr,
            "sycophantic_rate": int_rate,
        },
        "improvement": {
            "delta_sycophantic_rate": int_rate - base_rate,
        },
        "baseline_eval_file": baseline_eval_file,
        "intervention_eval_file": interv_eval_file,
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60 + "\n评测完成\n" + "=" * 60)
    print(f"Baseline 谄媚率: {base_rate:.2f}% ({base_inc}/{base_inc + base_corr})")
    print(f"Intervention 谄媚率: {int_rate:.2f}% ({int_inc}/{int_inc + int_corr})")
    print(f"改善: {summary['improvement']['delta_sycophantic_rate']:+.2f}%")
    print(f"\n结果文件:\n  {baseline_eval_file}\n  {interv_eval_file}\n  {summary_file}")


if __name__ == "__main__":
    main()
