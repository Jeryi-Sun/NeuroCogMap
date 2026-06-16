#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steer Vector 提取脚本

示例：
python extract_steer_vector.py \
  --correct_jsonl /path/to/correct.jsonl \
  --incorrect_jsonl /path/to/incorrect.jsonl \
  --output_file /path/to/output.pt
"""

import os
import json
import argparse
from typing import Any, Dict, List

from steer_vector_extractor import SteerVectorExtractor


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Steer Vector 提取脚本")
    parser.add_argument("--config", type=str,
                        default="/path/to/project_root/safety_explanation/hallucination/code/intervention/config.json",
                        help="配置文件路径")
    parser.add_argument("--correct_jsonl", type=str, required=True,
                        help="correct 数据 JSONL 文件路径")
    parser.add_argument("--incorrect_jsonl", type=str, required=True,
                        help="incorrect 数据 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="steer vector 输出文件路径 (.pt)")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="用于计算的样本数量（correct/incorrect 各选 num_samples 条，0 表示使用全部）")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="要提取的层列表，不指定则使用配置文件中的全部层")
    parser.add_argument("--sae_paths", type=str, nargs="*",
                        help="覆盖配置文件中的 SAE 路径列表")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    if not os.path.exists(args.correct_jsonl):
        raise FileNotFoundError(f"correct 数据文件不存在: {args.correct_jsonl}")
    if not os.path.exists(args.incorrect_jsonl):
        raise FileNotFoundError(f"incorrect 数据文件不存在: {args.incorrect_jsonl}")
    output_path = os.path.abspath(args.output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config = load_config(args.config)
    sae_paths: List[str] = args.sae_paths if args.sae_paths else config.get("sae_paths", [])

    extractor = SteerVectorExtractor(
        model_name=config.get("model_name", "google/gemma-2-2b"),
        sae_release=config.get("sae_release", "gemma-scope-2b-pt-res"),
        sae_local_base_dir=config.get("sae_local_base_dir", "/path/to/local_models/gemma-scope-2b-pt-res"),
        results_dir=config.get("results_dir", "/path/to/project_root/safety_explanation/hallucination/results/intervention/baseline"),
        is_instruct=config.get("is_instruct", False),
        layers=args.layers if args.layers else config.get("layers"),
        num_samples=args.num_samples,
        seed=args.seed,
    )

    extractor.run(
        correct_jsonl_path=args.correct_jsonl,
        incorrect_jsonl_path=args.incorrect_jsonl,
        sae_paths=sae_paths,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

