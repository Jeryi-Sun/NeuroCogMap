#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将各个模型在 analysis_output 中的 hierarchy_level_activation_diff.json
整合为一个汇总文件，便于绘图与下游分析。

输入来源：
  <BASE_DIR>/safety_explanation/fairness_bias/results/analysis_output/<MODEL_DATA>/hierarchy_level/hierarchy_level_activation_diff.json

输出：
  <BASE_DIR>/safety_explanation/fairness_bias/hierarchy_graphs/data/hierarchy_level_all_models.json

结构示例：
{
  "bbq_age_gemma-2-2b": { ... 原始层级 JSON ... },
  "bbq_nationality_gemma-2-2b": { ... },
  ...
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} 顶层必须是 dict")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总多个模型的 hierarchy_level_activation_diff.json 为一个总文件。"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/path/to/project_root",
        help="项目 BASE_DIR，默认是当前工程路径。",
    )
    parser.add_argument(
        "--model-data-list",
        type=str,
        nargs="+",
        default=[
            "bbq_age_gemma-2-2b",
            "bbq_nationality_gemma-2-2b",
            "bbq_gender_identity_gemma-2-2b",
            "bbq_disability_status_gemma-2-2b",
            "bbq_age_gemma-2-9b-it",
            "bbq_nationality_gemma-2-9b-it",
            "bbq_gender_identity_gemma-2-9b-it",
            "bbq_disability_status_gemma-2-9b-it",
        ],
        help="需要汇总的模型列表（fairness_bias 数据集）。",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "汇总输出 JSON 路径，"
            "默认: <base-dir>/safety_explanation/fairness_bias/hierarchy_graphs/data/hierarchy_level_all_models.json"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出文件已存在则跳过（不覆盖）。",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if args.output_json is not None:
        output_path = Path(args.output_json)
    else:
        output_path = (
            base_dir
            / "safety_explanation"
            / "fairness_bias"
            / "hierarchy_graphs"
            / "data"
            / "hierarchy_level_all_models.json"
        )

    if args.skip_existing and output_path.exists():
        print(f"[SKIP] 汇总文件已存在，跳过生成: {output_path}")
        return

    results: Dict[str, dict] = {}
    missing: List[str] = []

    for model_data in args.model_data_list:
        src = (
            base_dir
            / "safety_explanation"
            / "fairness_bias"
            / "results"
            / "analysis_output"
            / model_data
            / "hierarchy_level"
            / "hierarchy_level_activation_diff.json"
        )

        if not src.exists():
            missing.append(f"{model_data} -> {src}")
            print(
                f"[WARN] 模型 {model_data} 缺少 hierarchy_level 结果，已跳过。期望文件: {src}",
                file=sys.stderr,
            )
            continue

        try:
            results[model_data] = _load_json(src)
        except Exception as e:
            print(f"[ERROR] 读取 {src} 失败，已跳过。错误: {e}", file=sys.stderr)
            continue

    if not results:
        print("[ERROR] 没有任何模型的数据可以汇总，未生成输出文件。", file=sys.stderr)
        if missing:
            print("[ERROR] 缺失的模型及路径示例：", file=sys.stderr)
            for line in missing[:10]:
                print(f"  - {line}", file=sys.stderr)
            if len(missing) > 10:
                print(f"  ... 以及另外 {len(missing) - 10} 个", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] 已写出汇总文件: {output_path}")
    print(f"[INFO] 包含模型数量: {len(results)}")


if __name__ == "__main__":
    main()
