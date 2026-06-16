#!/usr/bin/env python3
"""
合并多个 uts 下的 human-LLM 对齐结果（human -> LLM 的 top_k 配对）。

功能概述
--------
给定若干个 CSV 文件（形如 `top_human_parcels_per_llm_sim06.csv`），
在 **相同 human_parcel** 下筛选出 **相同的 llm_parcel** 组合，
并将不同 uts 的 rank / accuracy（以及 similarity）分别存入不同列。

示例：
    python merge_llm_human_alignments_across_uts.py \
        --inputs  uts02_top.csv uts03_top.csv \
        --labels  uts02 uts03 \
        --output  merged_uts02_uts03_top_human_parcels_per_llm_sim06.csv

列设计
------
输出的主键为：
    - human_parcel
    - llm_parcel

其他元信息只保留一份（来自第一个出现该 pair 的输入文件）：
    - human_function
    - human_parcel_name
    - llm_function

针对每个 uts(label) 追加以下列（label 由 --labels 或自动从路径推断）：
    - rank_by_acc_{label}
    - prediction_accuracy_{label}
    - semantic_similarity_{label}

批量处理规则
------------
- 默认为安全模式：如果输出文件已存在且未指定 --overwrite，则直接报错并退出。
- 如需覆盖，显式传入 --overwrite。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def infer_label_from_path(path: Path) -> str:
    """
    从路径中自动推断 uts label，例如:
    - .../draw_result/uts02/whereis.../top_human_parcels_per_llm_sim06.csv -> 'uts02'

    若无法推断，则返回去掉后缀的文件名。
    """
    parts = list(path.parts)
    for p in parts:
        if p.startswith("uts") and len(p) >= 4 and p[3:].isdigit():
            return p
    stem = path.stem
    return stem.replace(".", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并多个 uts 下的 top_human_parcels_per_llm(_simXX).csv，"
        "保留在同一 human_parcel 下出现的相同 llm_parcel，并为不同 uts 写入独立 rank/accuracy 列。"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        required=True,
        help="需要合并的 CSV 文件列表（例如多个 uts*/top_human_parcels_per_llm_sim06.csv）。",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        type=str,
        default=None,
        help="与 --inputs 一一对应的 uts 标记（如 uts02 uts03）。"
        "若不提供，则会从路径中自动推断（例如目录名 uts02）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="合并后的输出 CSV 路径。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如输出文件已存在，则允许覆盖。",
    )
    return parser.parse_args()


def load_and_prepare(
    csv_path: Path,
    label: str,
) -> Tuple[pd.DataFrame, str]:
    """
    读取单个 CSV，并保留后续合并所需的列。

    要求输入至少包含：
        human_parcel, human_function, human_parcel_name,
        llm_parcel, llm_function,
        rank_by_acc, prediction_accuracy, semantic_similarity

    返回：
        df: 仅包含上述列，便于后续 inner join。
        label: 实际使用的 label（可能做了 sanitize）。
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读取 CSV 失败：{csv_path}，异常：{exc}") from exc

    required_cols = [
        "human_parcel",
        "human_function",
        "human_parcel_name",
        "llm_parcel",
        "llm_function",
        "rank_by_acc",
        "prediction_accuracy",
        "semantic_similarity",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"文件 {csv_path} 缺少必要列：{missing}")

    # 只保留必要列，并显式拷贝一份，避免后续链式赋值警告
    df = df[required_cols].copy()

    # 确保 human_parcel / llm_parcel 是字符串（以避免 merge 时类型不一致）
    df["human_parcel"] = df["human_parcel"].astype(str)
    df["llm_parcel"] = df["llm_parcel"].astype(str)

    # label 中不允许有空格一类的字符，统一转为下划线
    safe_label = label.replace(" ", "_")
    return df, safe_label


def merge_across_uts(inputs: List[Path], labels: List[str]) -> pd.DataFrame:
    """
    在 (human_parcel, llm_parcel) 维度上，对多个 uts 结果做 inner join。

    只保留出现在所有 uts 中的 pair，且：
    - human/llm 的元信息（function / parcel_name）只保留一份
      （以第一个 DataFrame 为基准）
    - 每个 uts 单独拥有 rank/accuracy/similarity 列
    """
    if len(inputs) != len(labels):
        raise ValueError(
            f"--inputs 和 --labels 数量不一致：{len(inputs)} vs {len(labels)}"
        )

    dfs: List[pd.DataFrame] = []
    for csv_path, label in zip(inputs, labels, strict=True):
        df_i, lab = load_and_prepare(csv_path, label)
        # 为当前 uts 添加专属列名
        df_i = df_i.rename(
            columns={
                "rank_by_acc": f"rank_by_acc_{lab}",
                "prediction_accuracy": f"prediction_accuracy_{lab}",
                "semantic_similarity": f"semantic_similarity_{lab}",
            }
        )
        dfs.append(df_i)

    # 第一个 DataFrame 提供公共的元信息列
    base = dfs[0]
    # 确保用于连接的键存在
    key_cols = ["human_parcel", "llm_parcel"]

    # 后续 DataFrame 仅保留键 + 本 uts 的数值列
    merged = base
    for df_i in dfs[1:]:
        value_cols = [c for c in df_i.columns if c not in ("human_function", "human_parcel_name", "llm_function")]
        merged = merged.merge(df_i[value_cols], on=key_cols, how="inner")

    # 按 human_parcel / llm_parcel 做一个稳定排序，便于阅读
    merged = merged.sort_values(by=["human_parcel", "llm_parcel"]).reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()

    # 输出存在性检查（符合“批量处理时可控跳过/覆盖”的规则）
    output_path: Path = args.output
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"输出文件已存在：{output_path}；如需覆盖请添加 --overwrite。"
        )

    # 处理 labels：若未提供则根据路径自动推断
    if args.labels is None:
        labels = [infer_label_from_path(p) for p in args.inputs]
    else:
        labels = args.labels

    try:
        merged_df = merge_across_uts(args.inputs, labels)
    except Exception as exc:  # noqa: BLE001
        # 明确打印异常信息，避免静默吞掉错误
        print(f"[Error] 合并 uts 结果时发生异常：{exc}")
        raise

    merged_df.to_csv(output_path, index=False)
    print(f"[Saved] 合并结果已写入：{output_path}")


if __name__ == "__main__":
    main()

