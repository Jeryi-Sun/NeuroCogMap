#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 correct 和 incorrect 的干预结果合并为符合预期路径的单一文件。

输入：gemma-2-2b_answer_correct_intervention.json + gemma-2-2b_answer_incorrect_intervention.json
输出：gemma-2-2b_answer_intervention.json（与 fairness_bias 格式一致）

用法：
  python merge_correct_incorrect_intervention.py [--strengths 0.1 0.3 0.5] [--skip-existing]
"""

import argparse
import json
import os
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reindex_results(results: list, start_index: int = 1) -> list:
    """重新编号 index 字段"""
    out = []
    for i, r in enumerate(results, start=start_index):
        r = dict(r)
        r["index"] = i
        out.append(r)
    return out


def compute_comparison(baseline_results: list) -> dict:
    """根据 baseline_results 计算 comparison 统计"""
    if not baseline_results:
        return {"avg_baseline_length": 0, "avg_intervention_length": 0, "length_change": 0}
    total_baseline = sum(len(r.get("baseline_text", "") or "") for r in baseline_results)
    total_intervention = sum(len(r.get("intervention_text", "") or "") for r in baseline_results)
    n = len(baseline_results)
    avg_baseline = total_baseline / n
    avg_intervention = total_intervention / n
    return {
        "avg_baseline_length": avg_baseline,
        "avg_intervention_length": avg_intervention,
        "length_change": avg_intervention - avg_baseline,
    }


def merge_strength_data(correct_data: dict, incorrect_data: dict, strength_key: str) -> dict:
    """合并单个强度下的 correct 和 incorrect 数据"""
    correct_st = correct_data.get("intervention_results", {}).get(strength_key, {})
    incorrect_st = incorrect_data.get("intervention_results", {}).get(strength_key, {})

    correct_baseline = correct_st.get("baseline_results", [])
    incorrect_baseline = incorrect_st.get("baseline_results", [])

    correct_intervention = correct_st.get("intervention_results", [])
    incorrect_intervention = incorrect_st.get("intervention_results", [])

    merged_baseline = reindex_results(correct_baseline + incorrect_baseline)
    merged_intervention = reindex_results(correct_intervention + incorrect_intervention)

    merged = {
        "intervention_strength": correct_st.get("intervention_strength", float(strength_key)),
        "parcel_ids": correct_st.get("parcel_ids", []),
        "num_samples": len(merged_baseline),
        "baseline_results": merged_baseline,
        "comparison": compute_comparison(merged_baseline),
    }
    if merged_intervention:
        merged["intervention_results"] = merged_intervention

    return merged


def merge_intervention_files(
    correct_path: str,
    incorrect_path: str,
    output_path: str,
    strength_keys: list[str],
    data_type: str,
    model_name: str,
) -> None:
    """合并 correct 和 incorrect 干预结果"""
    correct_data = load_json(correct_path)
    incorrect_data = load_json(incorrect_path)

    merged_intervention_results = {}
    for sk in strength_keys:
        correct_has = sk in correct_data.get("intervention_results", {})
        incorrect_has = sk in incorrect_data.get("intervention_results", {})
        if not correct_has and not incorrect_has:
            continue
        if not correct_has:
            merged_intervention_results[sk] = incorrect_data["intervention_results"][sk]
        elif not incorrect_has:
            merged_intervention_results[sk] = correct_data["intervention_results"][sk]
        else:
            merged_intervention_results[sk] = merge_strength_data(correct_data, incorrect_data, sk)

    total_samples = 0
    if merged_intervention_results:
        total_samples = next(iter(merged_intervention_results.values())).get("num_samples", 0)

    merged = {
        "experiment_info": {
            **correct_data["experiment_info"],
            "num_samples": total_samples,
        },
        "parcel_scaler_info": correct_data["parcel_scaler_info"],
        "intervention_results": merged_intervention_results,
        "dataset_info": {
            "dataset_name": data_type,
            "dataset_path": f"/path/to/project_root/safety_explanation/sycophancy/dataset/sycophancy-eval/results/{data_type}_with_groups_{model_name}_gen_eval.jsonl",
            "model_name": correct_data["experiment_info"]["model_name"],
            "parcel_ids": correct_data["experiment_info"]["parcel_ids"],
            "intervention_strength": float(strength_keys[0]) if strength_keys else 0.1,
            "max_samples": 1500,
            "enable_evaluation": correct_data.get("dataset_info", {}).get("enable_evaluation", False),
        },
    }

    save_json(merged, output_path)


def main():
    parser = argparse.ArgumentParser(description="合并 correct/incorrect 干预结果为预期路径格式")
    parser.add_argument(
        "--strengths",
        nargs="+",
        default=["0.1", "0.3", "0.5"],
        help="要处理的强度列表",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若输出文件已存在则跳过",
    )
    parser.add_argument(
        "--results-root",
        default="/path/to/project_root/safety_explanation/sycophancy/results/intervention",
        help="干预结果根目录",
    )
    args = parser.parse_args()

    models = ["gemma-2-2b", "gemma-2-9b-it"]
    data_types = ["answer", "feedback"]

    for strength in args.strengths:
        strength_dir = Path(args.results_root) / f"strength_{strength}"
        if not strength_dir.exists():
            print(f"[WARN] 目录不存在，跳过: {strength_dir}")
            continue

        for model in models:
            for data_type in data_types:
                correct_path = strength_dir / f"{model}_{data_type}_correct_intervention.json"
                incorrect_path = strength_dir / f"{model}_{data_type}_incorrect_intervention.json"
                output_path = strength_dir / f"{model}_{data_type}_intervention.json"

                if not correct_path.exists():
                    print(f"[WARN] 未找到 correct 文件，跳过: {correct_path}")
                    continue
                if not incorrect_path.exists():
                    print(f"[WARN] 未找到 incorrect 文件，跳过: {incorrect_path}")
                    continue

                if args.skip_existing and output_path.exists():
                    print(f"[SKIP] 已存在，跳过: {output_path}")
                    continue

                merged_strengths = list(
                    set(
                        list(load_json(str(correct_path)).get("intervention_results", {}).keys())
                        + list(load_json(str(incorrect_path)).get("intervention_results", {}).keys())
                    )
                )
                merged_strengths = sorted(merged_strengths, key=float)

                merge_intervention_files(
                    str(correct_path),
                    str(incorrect_path),
                    str(output_path),
                    merged_strengths,
                    data_type,
                    model,
                )
                print(f"[OK] 已合并: {output_path}")


if __name__ == "__main__":
    main()
