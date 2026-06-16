#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计非谄媚率(non-sycophancy rate)的小工具。

背景：
- intervention 评测会生成 *_baseline_eval.jsonl / *_intervention_eval.jsonl
- 其中 answer 数据集的逐条记录含 is_correct；feedback 数据集含 positive(=sycophancy 标签)
- classify_sycophancy.py 已实现把 eval_records 划分为：
    - incorrect: 谄媚样本
    - correct:   非谄媚样本

本脚本输出：
  non_sycophantic_rate = non_sycophantic_count / (sycophantic_count + non_sycophantic_count) * 100

用法示例：
1) 单文件：
  python compute_non_sycophancy_rate.py \
    --eval_jsonl /path/to/gemma-2-9b-it_answer_baseline_eval.jsonl

2) 扫描目录（递归）：
  python compute_non_sycophancy_rate.py \
    --input_dir /path/to/project_root/safety_explanation/sycophancy/results/intervention \
    --pattern "*_eval.jsonl" \
    --write_summary

注意：
- 默认只打印到 stdout；加 --write_summary 才会写 summary.json 文件
- 若启用 --skip_existing，且 summary 已存在，则跳过对应文件
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# 复用现有分类逻辑（与 intervention 评测保持一致）
# 这里不假设 safety_explanation/ 是 python package，改用 sys.path 注入保证可运行
THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent  # .../sycophancy/code
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from classify_sycophancy import (  # noqa: E402
    classify_answer_records,
    classify_feedback_records,
    load_jsonl,
)


def detect_dataset_type_from_path(eval_path: Path, eval_records: List[Dict[str, Any]]) -> str:
    stem = eval_path.stem.lower()
    if "feedback" in stem:
        return "feedback"
    if "answer" in stem:
        return "answer"
    if eval_records:
        r0 = eval_records[0]
        if "positive" in r0:
            return "feedback"
        if "is_correct" in r0:
            return "answer"
    raise ValueError(f"无法判断数据集类型（文件名与内容均无特征）: {str(eval_path)}")


def compute_counts(eval_path: Path) -> Dict[str, Any]:
    if not eval_path.exists():
        raise FileNotFoundError(f"eval 文件不存在: {str(eval_path)}")

    eval_records = load_jsonl(str(eval_path))
    dataset_type = detect_dataset_type_from_path(eval_path, eval_records)

    if dataset_type == "answer":
        syco, non_syco = classify_answer_records(eval_records)
    else:
        syco, non_syco = classify_feedback_records(eval_records)

    sycophantic_count = len(syco)
    non_sycophantic_count = len(non_syco)
    total = sycophantic_count + non_sycophantic_count
    non_sycophantic_rate = (non_sycophantic_count / total * 100.0) if total > 0 else 0.0
    sycophantic_rate = (sycophantic_count / total * 100.0) if total > 0 else 0.0

    return {
        "eval_jsonl": str(eval_path),
        "dataset_type": dataset_type,
        "sycophantic_count": sycophantic_count,
        "non_sycophantic_count": non_sycophantic_count,
        "total_count": total,
        "sycophantic_rate": sycophantic_rate,
        "non_sycophantic_rate": non_sycophantic_rate,
    }


def default_summary_path(eval_path: Path) -> Path:
    # xxx_eval.jsonl -> xxx_non_sycophancy_summary.json
    name = eval_path.name
    if name.endswith("_eval.jsonl"):
        name = name[: -len("_eval.jsonl")]
    else:
        name = eval_path.stem
    return eval_path.parent / f"{name}_non_sycophancy_summary.json"


def write_summary(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def iter_eval_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir 不存在: {str(input_dir)}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir 不是目录: {str(input_dir)}")
    if recursive:
        return sorted(input_dir.rglob(pattern))
    return sorted(input_dir.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="统计非谄媚率（non-sycophancy rate）")
    parser.add_argument("--eval_jsonl", type=str, default=None, help="单个 *_eval.jsonl 文件路径")
    parser.add_argument("--input_dir", type=str, default=None, help="目录：扫描并统计多个 *_eval.jsonl")
    parser.add_argument("--pattern", type=str, default="*_eval.jsonl", help="目录扫描的文件名 pattern（glob）")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录")

    parser.add_argument("--write_summary", action="store_true", help="将每个文件的统计写入 summary.json")
    parser.add_argument("--summary_path", type=str, default=None, help="单文件模式下，指定 summary 输出路径")
    parser.add_argument("--skip_existing", action="store_true", help="若 summary 已存在则跳过写入（仅在 --write_summary 时生效）")
    args = parser.parse_args()

    if bool(args.eval_jsonl) == bool(args.input_dir):
        raise ValueError("必须且只能指定 --eval_jsonl 或 --input_dir 其中之一")

    summaries: List[Dict[str, Any]] = []

    if args.eval_jsonl:
        eval_path = Path(args.eval_jsonl)
        summary = compute_counts(eval_path)
        summaries.append(summary)

        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if args.write_summary:
            out_path = Path(args.summary_path) if args.summary_path else default_summary_path(eval_path)
            if args.skip_existing and out_path.exists():
                print(f"[INFO] summary 已存在，跳过: {str(out_path)}")
                return
            write_summary(summary, out_path)
            print(f"[INFO] 已写入: {str(out_path)}")
        return

    # 批量目录模式
    input_dir = Path(args.input_dir)
    eval_files = iter_eval_files(input_dir, args.pattern, args.recursive)
    if not eval_files:
        raise FileNotFoundError(f"未找到匹配文件: dir={str(input_dir)}, pattern={args.pattern}, recursive={args.recursive}")

    for p in eval_files:
        try:
            summary = compute_counts(p)
            summaries.append(summary)
            print(f"[OK] {p.name}: non_sycophancy_rate={summary['non_sycophantic_rate']:.2f}% "
                  f"({summary['non_sycophantic_count']}/{summary['total_count']})")

            if args.write_summary:
                out_path = default_summary_path(p)
                if args.skip_existing and out_path.exists():
                    print(f"[INFO] summary 已存在，跳过: {str(out_path)}")
                    continue
                write_summary(summary, out_path)
        except Exception as e:
            # 这里不吞异常：打印出来，继续处理其他文件
            print(f"[ERROR] 处理失败: {str(p)} -> {e}")

    # 写一个聚合结果到 stdout（方便复制）
    agg = {
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "recursive": bool(args.recursive),
        "num_files": len(summaries),
        "summaries": summaries,
    }
    print("\n[AGGREGATE]\n" + json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

