#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将公平偏见评估结果按 unbiased/antibias/bias 分组。

输入：包含 `eval_type` 字段的 JSONL 评估文件（文件名需包含 `_eval`）。
输出：按模型名划分目录，在其中生成 `correct.jsonl`、`anticorrect.jsonl` 与 `incorrect.jsonl`。

分类规则：
- eval_type == "no_bias"        -> 归入 correct.jsonl
- eval_type == "antibias"       -> 归入 anticorrect.jsonl（回答未与刻板印象一致）
- eval_type == "bias"           -> 归入 incorrect.jsonl
- 其他情况（缺失或值异常）     -> 跳过，并输出警告

使用示例：
python split_bias_results.py \
  --results-dir /path/to/project_root/safety_explanation/fairness_bias/results \
  --output-root /path/to/project_root/safety_explanation/fairness_bias/results \
  --skip-existing
"""

import argparse
import json
import os
import sys
from typing import Tuple


DEFAULT_RESULTS_DIR = "/path/to/project_root/safety_explanation/fairness_bias/results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 unbiased/antibias/bias 将公平偏见评估 JSONL 文件拆分为三组。"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="包含评估 JSONL 文件的目录（文件名需包含 'eval'）。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="输出根目录；若不指定则与 --results-dir 相同。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="当目标输出文件均已存在时跳过该输入文件。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划行动，不实际写入文件。",
    )
    return parser.parse_args()


def derive_model_name(filename: str) -> str:
    base = os.path.basename(filename)
    stem = base[:-6] if base.endswith(".jsonl") else base
    if stem.endswith("_eval"):
        stem = stem[: -len("_eval")]
    return stem


def ensure_dir(path: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] mkdir -p {path}")
        return
    os.makedirs(path, exist_ok=True)


def target_paths(output_root: str, model_name: str) -> Tuple[str, str, str]:
    model_dir = os.path.join(output_root, model_name)
    return (
        os.path.join(model_dir, "correct.jsonl"),
        os.path.join(model_dir, "anticorrect.jsonl"),
        os.path.join(model_dir, "incorrect.jsonl"),
    )


def outputs_exist(correct_path: str, anticorrect_path: str, incorrect_path: str) -> bool:
    return os.path.exists(correct_path) and os.path.exists(anticorrect_path) and os.path.exists(incorrect_path)


def split_file(
    src_path: str,
    correct_out_path: str,
    anticorrect_out_path: str,
    incorrect_out_path: str,
    dry_run: bool = False,
) -> Tuple[int, int, int, int]:
    total = 0
    num_correct = 0
    num_anticorrect = 0
    num_incorrect = 0

    if dry_run:
        print(
            f"[DRY-RUN] 会拆分 {src_path} -> {correct_out_path} / {anticorrect_out_path} / {incorrect_out_path}"
        )
        return total, num_correct, num_anticorrect, num_incorrect

    ensure_dir(os.path.dirname(correct_out_path), dry_run=False)
    ensure_dir(os.path.dirname(anticorrect_out_path), dry_run=False)
    ensure_dir(os.path.dirname(incorrect_out_path), dry_run=False)

    try:
        with open(src_path, "r", encoding="utf-8") as fin, \
             open(correct_out_path, "w", encoding="utf-8") as f_correct, \
             open(anticorrect_out_path, "w", encoding="utf-8") as f_anticorrect, \
             open(incorrect_out_path, "w", encoding="utf-8") as f_incorrect:
            for line_idx, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except Exception as ex:
                    print(
                        f"[ERROR] JSON解析失败 {src_path} 第{line_idx}行: {ex}",
                        file=sys.stderr,
                    )
                    continue

                eval_type = record.get("eval_type")
                if eval_type == "no_bias":
                    num_correct += 1
                    f_correct.write(json.dumps(record, ensure_ascii=False) + "\n")
                elif eval_type == "antibias":
                    num_anticorrect += 1
                    f_anticorrect.write(json.dumps(record, ensure_ascii=False) + "\n")
                elif eval_type == "bias":
                    num_incorrect += 1
                    f_incorrect.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    print(
                        f"[WARN] eval_type 缺失或异常 {src_path} 第{line_idx}行: {eval_type}",
                        file=sys.stderr,
                    )
                    continue
    except Exception as ex:
        print(f"[ERROR] 处理文件失败 {src_path}: {ex}", file=sys.stderr)

    return total, num_correct, num_anticorrect, num_incorrect


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_root = os.path.abspath(args.output_root) if args.output_root else results_dir

    if not os.path.isdir(results_dir):
        print(f"[ERROR] results目录不存在: {results_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        entries = sorted(os.listdir(results_dir))
    except Exception as ex:
        print(f"[ERROR] 读取目录失败 {results_dir}: {ex}", file=sys.stderr)
        sys.exit(1)

    candidates = [
        os.path.join(results_dir, name)
        for name in entries
        if name.endswith(".jsonl") and "eval" in name
    ]

    if not candidates:
        print(f"[INFO] 在 {results_dir} 未找到评估JSONL文件")
        return

    print(f"[INFO] 共发现 {len(candidates)} 个评估文件")

    for src_path in candidates:
        model_name = derive_model_name(src_path)
        correct_path, anticorrect_path, incorrect_path = target_paths(output_root, model_name)

        if args.skip_existing and outputs_exist(correct_path, anticorrect_path, incorrect_path):
            print(
                f"[SKIP] 结果已存在，跳过 '{model_name}':\n"
                f"       {correct_path}\n"
                f"       {anticorrect_path}\n"
                f"       {incorrect_path}"
            )
            continue

        ensure_dir(os.path.dirname(correct_path), dry_run=args.dry_run)
        ensure_dir(os.path.dirname(anticorrect_path), dry_run=args.dry_run)
        ensure_dir(os.path.dirname(incorrect_path), dry_run=args.dry_run)

        total, n_correct, n_anticorrect, n_incorrect = split_file(
            src_path=src_path,
            correct_out_path=correct_path,
            anticorrect_out_path=anticorrect_path,
            incorrect_out_path=incorrect_path,
            dry_run=args.dry_run,
        )

        print(
            f"[DONE] {os.path.basename(src_path)} -> model='{model_name}': "
            f"total={total}, correct={n_correct}, anticorrect={n_anticorrect}, incorrect={n_incorrect}"
        )


if __name__ == "__main__":
    main()


