#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
聚合 sycophancy 干预评测结果，生成类似 fairness_bias 的总结表

输入目录结构（已存在的评测输出）:
  /path/to/project_root/safety_explanation/sycophancy/results/intervention/
    strength_0.1/
      gemma-2-2b_answer_summary.json
      gemma-2-2b_feedback_summary.json
      ...
    strength_0.3/
      ...
    ...

summary.json 内容示例（eval_sycophancy_intervention.py 生成）:
{
  "dataset_name": "answer_with_groups_gemma-2-2b_gen_eval",
  "model_name": "gemma-2-2b",
  "intervention_strength": "0.5",
  "baseline_stats": {
    "sycophantic_count": 128,
    "non_sycophantic_count": 339,
    "sycophantic_rate": 27.40
  },
  "intervention_stats": {
    "sycophantic_count": 26,
    "non_sycophantic_count": 18,
    "sycophantic_rate": 59.09
  },
  "improvement": {
    "delta_sycophantic_rate": 31.68
  },
  ...
}

本脚本会遍历各个 strength_* 目录下的 *_summary.json，汇总生成:
- CSV: intervention_sycophancy_table_strengths.csv
- JSON: intervention_sycophancy_table_strengths.json

CSV 字段（列，严格对齐 fairness_bias）:
- strength
- model
- dataset
- baseline_correct
- intervention_correct
- total_baseline
- baseline_accuracy
- intervention_accuracy

使用:
  python aggregate_intervention_sycophancy_table.py \
    --intervention_base_dir /path/to/project_root/safety_explanation/sycophancy/results/intervention \
    --output_dir /path/to/project_root/safety_explanation/sycophancy/results/intervention/aggregate
"""

import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List


def safe_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """安全地按层级取值，任意一层缺失则返回 default，并打印警告。"""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            print(f"[WARN] 字段缺失: {'/'.join(keys)}，使用默认值 {default}")
            return default
        cur = cur[k]
    return cur


def collect_summary_rows(intervention_base_dir: str) -> List[Dict[str, Any]]:
    base = Path(intervention_base_dir)
    if not base.exists():
        raise FileNotFoundError(f"干预结果目录不存在: {intervention_base_dir}")

    rows: List[Dict[str, Any]] = []
    # strength 目录命名: strength_0.1, strength_0.3, ...
    for strength_dir in sorted(base.glob("strength_*")):
        if not strength_dir.is_dir():
            continue
        strength = strength_dir.name.replace("strength_", "")
        # 读取 *_answer_summary.json 与 *_feedback_summary.json
        # 过滤掉 *_baseline_non_sycophancy_summary.json / *_intervention_non_sycophancy_summary.json 等
        for summary_path in sorted(strength_dir.glob("*_summary.json")):
            stem = summary_path.stem
            if not (stem.endswith("_answer_summary") or stem.endswith("_feedback_summary")):
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as f:
                    summ = json.load(f)
            except Exception as e:
                print(f"[ERROR] 读取失败: {summary_path} -> {e}")
                continue

            dataset_name = safe_get(summ, "dataset_name", default="unknown_dataset")
            model_name = safe_get(summ, "model_name", default="unknown_model")
            # 优先使用文件内 strength（字符串），否则用目录名
            str_from_file = safe_get(summ, "intervention_strength", default=None)
            str_value = str_from_file if str_from_file is not None else strength

            b_syc = safe_get(summ, "baseline_stats", "sycophantic_count", default=0)
            b_non = safe_get(summ, "baseline_stats", "non_sycophantic_count", default=0)
            b_rate = safe_get(summ, "baseline_stats", "sycophantic_rate", default=0.0)

            i_syc = safe_get(summ, "intervention_stats", "sycophantic_count", default=0)
            i_non = safe_get(summ, "intervention_stats", "non_sycophantic_count", default=0)
            i_rate = safe_get(summ, "intervention_stats", "sycophantic_rate", default=0.0)

            delta_rate = safe_get(summ, "improvement", "delta_sycophantic_rate", default=(i_rate - b_rate))

            # 计算总数
            try:
                b_total = int(b_syc) + int(b_non)
            except Exception as e:
                print(f"[WARN] baseline 计数解析失败({summary_path}): {e}")
                b_total = 0
            try:
                i_total = int(i_syc) + int(i_non)
            except Exception as e:
                print(f"[WARN] intervention 计数解析失败({summary_path}): {e}")
                i_total = 0

            row = {
                "strength": str(str_value),
                "model": str(model_name),
                "dataset": str(dataset_name),
                "baseline_sycophantic_count": int(b_syc) if isinstance(b_syc, (int, float)) else 0,
                "baseline_non_sycophantic_count": int(b_non) if isinstance(b_non, (int, float)) else 0,
                "baseline_total": int(b_total),
                "baseline_sycophantic_rate": float(b_rate) if isinstance(b_rate, (int, float)) else 0.0,
                "intervention_sycophantic_count": int(i_syc) if isinstance(i_syc, (int, float)) else 0,
                "intervention_non_sycophantic_count": int(i_non) if isinstance(i_non, (int, float)) else 0,
                "intervention_total": int(i_total),
                "intervention_sycophantic_rate": float(i_rate) if isinstance(i_rate, (int, float)) else 0.0,
                "delta_sycophantic_rate": float(delta_rate) if isinstance(delta_rate, (int, float)) else (float(i_rate) - float(b_rate)),
                "source_file": str(summary_path),
            }
            rows.append(row)
    return rows


def save_csv(rows: List[Dict[str, Any]], output_csv: str) -> None:
    import csv
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 列顺序：严格对齐 fairness_bias/intervention_accuracy_table*.csv
    fieldnames = [
        "strength",
        "model",
        "dataset",
        "baseline_correct",
        "intervention_correct",
        "total_baseline",
        "baseline_accuracy",
        "intervention_accuracy",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            baseline_total = row.get("baseline_total", 0)
            intervention_total = row.get("intervention_total", 0)

            baseline_non = row.get("baseline_non_sycophantic_count", 0)
            intervention_non = row.get("intervention_non_sycophantic_count", 0)

            # 与 fairness 语义对齐：
            # correct <- non_sycophantic_count
            # accuracy <- non_sycophantic_rate / 100
            out = {
                "strength": row.get("strength"),
                "model": row.get("model"),
                "dataset": row.get("dataset"),
                "baseline_correct": baseline_non,
                "intervention_correct": intervention_non,
                "total_baseline": baseline_total,
                "baseline_accuracy": (baseline_non / baseline_total) if baseline_total else 0.0,
                "intervention_accuracy": (intervention_non / intervention_total) if intervention_total else 0.0,
            }
            writer.writerow(out)
    print(f"[INFO] CSV 已保存: {out_path}")


def save_json(rows: List[Dict[str, Any]], output_json: str) -> None:
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[INFO] JSON 已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 sycophancy 干预 summary 为 CSV/JSON")
    parser.add_argument(
        "--intervention_base_dir",
        type=str,
        default="/path/to/project_root/safety_explanation/sycophancy/results/intervention",
        help="干预结果根目录（包含 strength_* 子目录）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/project_root/safety_explanation/sycophancy/results/intervention/aggregate",
        help="输出目录（保存 CSV 和 JSON）",
    )

    args = parser.parse_args()
    rows = collect_summary_rows(args.intervention_base_dir)
    if not rows:
        print("[WARN] 未收集到任何 summary 行，请确认评测已生成 *_summary.json")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "intervention_sycophancy_table_strengths.csv"
    json_path = output_dir / "intervention_sycophancy_table_strengths.json"

    save_csv(rows, str(csv_path))
    save_json(rows, str(json_path))


if __name__ == "__main__":
    main()

