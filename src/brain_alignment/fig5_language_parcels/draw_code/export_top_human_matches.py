#!/usr/bin/env python3
"""
导出所有 LLM parcel 和 human parcel 的 top-k 配对关系。

功能：
1. 为每个 LLM parcel 找到预测准确率最高的 top-k human parcels
2. 为每个 human parcel 找到预测准确率最高的 top-k LLM parcels

输出两个 CSV 文件：
- {output}: LLM -> Human 的配对
- {output}_human: Human -> LLM 的配对
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
from typing import Iterable

import pandas as pd

from common import (
    RESULT_DIR,
    add_common_arguments,
    ensure_output_dir,
    load_label_mapping,
    load_matrix,
)


def extract_parcel_id(label: str) -> str:
    """从标签中提取数字 ID（如 'LLM_Parcel_3' -> '3'）。"""
    parts = label.split("_")
    if len(parts) >= 2:
        try:
            return str(int(parts[-1]))
        except ValueError:
            return label
    return label


def load_parcel_descriptions(json_path: Path) -> dict[str, str]:
    """加载 parcel_descriptions.json，返回 parcel_id -> parcel_name 的映射（字符串 key）。"""
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 parcel 描述文件：{json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"parcel_descriptions.json 应该是列表格式，但得到 {type(data)}")
    mapping: dict[str, str] = {}
    for item in data:
        parcel_id = item.get("parcel_id")
        parcel_name = item.get("parcel_name", "")
        if parcel_id is None:
            continue
        try:
            key = str(int(parcel_id))
        except (TypeError, ValueError) as exc:
            print(f"[Warning] parcel_id={parcel_id} 无法转换为整数，跳过: {exc}")
            continue
        mapping[key] = parcel_name or ""
    return mapping


def collect_top_matches_for_llm(
    acc: pd.DataFrame,
    sim: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    random_k: int,
    human_mapping: dict,
    llm_mapping: dict,
    llm_subset: Iterable[str] | None = None,
    parcel_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """对每个 LLM parcel 找到预测准确率最高的 top-k human parcels。"""
    if acc.shape != sim.shape:
        raise ValueError(f"A 与 S 形状不一致：{acc.shape} vs {sim.shape}")
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    if bottom_k < 0:
        raise ValueError("bottom_k 不能为负数")
    if random_k < 0:
        raise ValueError("random_k 不能为负数")

    n_rows = acc.shape[0]
    max_top_k = min(top_k, n_rows)
    max_bottom_k = min(bottom_k, n_rows)
    records: list[dict] = []
    allowed_llm = set(llm_subset) if llm_subset is not None else None
    for col in acc.columns:
        if allowed_llm is not None and col not in allowed_llm:
            continue
        acc_col = acc[col]
        # top-k
        top_humans = list(acc_col.nlargest(max_top_k).index)
        # bottom-k
        bottom_humans: list[str] = []
        if max_bottom_k > 0:
            bottom_humans = list(acc_col.nsmallest(max_bottom_k).index)
        # random from remaining (非 topk、非 bottomk)
        random_humans: list[str] = []
        if random_k > 0:
            remaining_index = acc_col.index.difference(top_humans + bottom_humans)
            remaining_n = len(remaining_index)
            if remaining_n <= 0:
                print(
                    f"[Warning] LLM 列 {col} 没有可用于 random 采样的 human parcels "
                    f"(top_k={top_k}, bottom_k={bottom_k})"
                )
            else:
                sample_n = min(random_k, remaining_n)
                if random_k > remaining_n:
                    print(
                        f"[Warning] LLM 列 {col} 中可用于 random 采样的数量只有 {remaining_n}，"
                        f"小于请求的 random_k={random_k}，将全部使用剩余项。"
                    )
                random_humans = list(
                    acc_col.loc[remaining_index].sample(n=sample_n, replace=False, random_state=None).index
                )
        llm_id = extract_parcel_id(col)
        llm_func = llm_mapping.get(llm_id, col)
        # 依次记录 top / bottom / random 三类
        for rank, human_label in enumerate(top_humans, start=1):
            acc_val = float(acc.at[human_label, col])
            sim_val = float(sim.at[human_label, col])
            human_id = extract_parcel_id(human_label)
            human_func = human_mapping.get(human_id, human_label)
            human_name = None
            if parcel_names is not None:
                human_name = parcel_names.get(human_id)
            records.append(
                {
                    "llm_parcel": col,
                    "llm_function": llm_func,
                    "human_parcel": human_label,
                    "human_function": human_func,
                    "human_parcel_name": human_name or human_func,
                    "rank_by_acc": rank,
                    "selection_type": "top",
                    "prediction_accuracy": acc_val,
                    "semantic_similarity": sim_val,
                }
            )
        for rank, human_label in enumerate(bottom_humans, start=1):
            acc_val = float(acc.at[human_label, col])
            sim_val = float(sim.at[human_label, col])
            human_id = extract_parcel_id(human_label)
            human_func = human_mapping.get(human_id, human_label)
            human_name = None
            if parcel_names is not None:
                human_name = parcel_names.get(human_id)
            records.append(
                {
                    "llm_parcel": col,
                    "llm_function": llm_func,
                    "human_parcel": human_label,
                    "human_function": human_func,
                    "human_parcel_name": human_name or human_func,
                    "rank_by_acc": rank,
                    "selection_type": "bottom",
                    "prediction_accuracy": acc_val,
                    "semantic_similarity": sim_val,
                }
            )
        for human_label in random_humans:
            acc_val = float(acc.at[human_label, col])
            sim_val = float(sim.at[human_label, col])
            human_id = extract_parcel_id(human_label)
            human_func = human_mapping.get(human_id, human_label)
            human_name = None
            if parcel_names is not None:
                human_name = parcel_names.get(human_id)
            records.append(
                {
                    "llm_parcel": col,
                    "llm_function": llm_func,
                    "human_parcel": human_label,
                    "human_function": human_func,
                    "human_parcel_name": human_name or human_func,
                    "rank_by_acc": None,
                    "selection_type": "random",
                    "prediction_accuracy": acc_val,
                    "semantic_similarity": sim_val,
                }
            )
    return pd.DataFrame(records)


def collect_top_matches_for_human(
    acc: pd.DataFrame,
    sim: pd.DataFrame,
    top_k: int,
    bottom_k: int,
    random_k: int,
    human_mapping: dict,
    llm_mapping: dict,
    parcel_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """对每个 human parcel 找到预测准确率最高的 top-k LLM parcels。"""
    if acc.shape != sim.shape:
        raise ValueError(f"A 与 S 形状不一致：{acc.shape} vs {sim.shape}")
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    if bottom_k < 0:
        raise ValueError("bottom_k 不能为负数")
    if random_k < 0:
        raise ValueError("random_k 不能为负数")

    n_cols = acc.shape[1]
    max_top_k = min(top_k, n_cols)
    max_bottom_k = min(bottom_k, n_cols)
    records: list[dict] = []
    for human_label in acc.index:
        acc_row = acc.loc[human_label]
        # top-k
        top_llms = list(acc_row.nlargest(max_top_k).index)
        # bottom-k
        bottom_llms: list[str] = []
        if max_bottom_k > 0:
            bottom_llms = list(acc_row.nsmallest(max_bottom_k).index)
        # random from remaining (非 topk、非 bottomk)
        random_llms: list[str] = []
        if random_k > 0:
            remaining_index = acc_row.index.difference(top_llms + bottom_llms)
            remaining_n = len(remaining_index)
            if remaining_n <= 0:
                print(
                    f"[Warning] human 行 {human_label} 没有可用于 random 采样的 LLM parcels "
                    f"(top_k={top_k}, bottom_k={bottom_k})"
                )
            else:
                sample_n = min(random_k, remaining_n)
                if random_k > remaining_n:
                    print(
                        f"[Warning] human 行 {human_label} 中可用于 random 采样的数量只有 {remaining_n}，"
                        f"小于请求的 random_k={random_k}，将全部使用剩余项。"
                    )
                random_llms = list(
                    acc_row.loc[remaining_index].sample(n=sample_n, replace=False, random_state=None).index
                )
        human_id = extract_parcel_id(human_label)
        human_func = human_mapping.get(human_id, human_label)
        # 依次记录 top / bottom / random 三类
        for rank, llm_label in enumerate(top_llms, start=1):
            acc_val = float(acc.at[human_label, llm_label])
            sim_val = float(sim.at[human_label, llm_label])
            llm_id = extract_parcel_id(llm_label)
            llm_func = llm_mapping.get(llm_id, llm_label)
            human_name = None
            if parcel_names is not None:
                human_name = parcel_names.get(human_id)
            records.append(
                {
                    "human_parcel": human_label,
                    "human_function": human_func,
                    "human_parcel_name": human_name or human_func,
                    "llm_parcel": llm_label,
                    "llm_function": llm_func,
                    "rank_by_acc": rank,
                    "selection_type": "top",
                    "prediction_accuracy": acc_val,
                    "semantic_similarity": sim_val,
                }
            )
        for rank, llm_label in enumerate(bottom_llms, start=1):
            acc_val = float(acc.at[human_label, llm_label])
            sim_val = float(sim.at[human_label, llm_label])
            llm_id = extract_parcel_id(llm_label)
            llm_func = llm_mapping.get(llm_id, llm_label)
            human_name = None
            if parcel_names is not None:
                human_name = parcel_names.get(human_id)
            records.append(
                {
                    "human_parcel": human_label,
                    "human_function": human_func,
                    "human_parcel_name": human_name or human_func,
                    "llm_parcel": llm_label,
                    "llm_function": llm_func,
                    "rank_by_acc": rank,
                    "selection_type": "bottom",
                    "prediction_accuracy": acc_val,
                    "semantic_similarity": sim_val,
                }
            )
        for llm_label in random_llms:
            acc_val = float(acc.at[human_label, llm_label])
            sim_val = float(sim.at[human_label, llm_label])
            llm_id = extract_parcel_id(llm_label)
            llm_func = llm_mapping.get(llm_id, llm_label)
            human_name = None
            if parcel_names is not None:
                human_name = parcel_names.get(human_id)
            records.append(
                {
                    "human_parcel": human_label,
                    "human_function": human_func,
                    "human_parcel_name": human_name or human_func,
                    "llm_parcel": llm_label,
                    "llm_function": llm_func,
                    "rank_by_acc": None,
                    "selection_type": "random",
                    "prediction_accuracy": acc_val,
                    "semantic_similarity": sim_val,
                }
            )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出所有 LLM 和 human parcels 的 top-k 配对关系。")
    add_common_arguments(parser)
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="每个 LLM parcel 保留的 human parcels 数目（默认 3）。",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=0,
        help="每个 LLM parcel 额外保留的 bottom-k human parcels 数目（默认 0，表示不启用）。",
    )
    parser.add_argument(
        "--random-k",
        type=int,
        default=0,
        help="从非 top/bottom 的 human parcels 中随机采样的数量（默认 0，表示不启用）。",
    )
    parser.add_argument(
        "--top-k-human",
        type=int,
        default=3,
        help="每个 human parcel 保留的 LLM parcels 数目（默认 3）。",
    )
    parser.add_argument(
        "--bottom-k-human",
        type=int,
        default=3,
        help="每个 human parcel 额外保留的 bottom-k LLM parcels 数目（默认 0，表示不启用）。",
    )
    parser.add_argument(
        "--random-k-human",
        type=int,
        default=3,
        help="从非 top/bottom 的 LLM parcels 中为每个 human parcel 随机采样的数量（默认 0，表示不启用）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULT_DIR / "fig2_top_human_parcels_per_llm.csv",
        help="输出 CSV 路径（默认 draw_result/fig2_top_human_parcels_per_llm.csv）。",
    )
    parser.add_argument(
        "--parcel-descriptions",
        type=Path,
        default=Path(
            "/path/to/project_root/"
            "Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
        ),
        help="parcel_descriptions.json 文件路径，用于获取 parcel_name（默认路径为数据集目录）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dir()
    # 在写入任何文件之前，同时检查 LLM 和 Human 两个输出是否已存在，确保原子性
    llm_output = os.path.join(args.output, "top_llm_parcels_per_human.csv")
    human_output = os.path.join(args.output, "top_human_parcels_per_llm.csv")
    llm_output_sim06 = os.path.join(args.output, "top_llm_parcels_per_human_sim06.csv")
    human_output_sim06 = os.path.join(args.output, "top_human_parcels_per_llm_sim06.csv")

    if not args.overwrite:
        existing = []
        for path in (llm_output, human_output, llm_output_sim06, human_output_sim06):
            if os.path.exists(path):
                existing.append(str(path))
        if existing:
            raise FileExistsError(
                "以下输出文件已存在，如需覆盖请添加 --overwrite：\n"
                + "\n".join(existing)
            )

    acc = load_matrix(args.prediction_matrix)
    sim = load_matrix(args.semantic_matrix)
    mapping = load_label_mapping(args.mapping_file)
    human_mapping = mapping.get("human_parcels", {})
    llm_mapping = mapping.get("llm_parcels", {})
    parcel_names = load_parcel_descriptions(args.parcel_descriptions)
    
    # 为所有 LLM parcels 找到 top-k human parcels
    result_llm_df = collect_top_matches_for_llm(
        acc,
        sim,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        random_k=args.random_k,
        human_mapping=human_mapping,
        llm_mapping=llm_mapping,
        llm_subset=None,  # None 表示处理所有 LLM parcels
        parcel_names=parcel_names,
    )
    result_llm_df.to_csv(llm_output, index=False)
    print(f"[Saved] {llm_output}")

    # 额外保存一份：仅保留 semantic_similarity >= 0.6 的行
    result_llm_df_sim06 = result_llm_df[result_llm_df["semantic_similarity"] >= 0.6].copy()
    result_llm_df_sim06.to_csv(llm_output_sim06, index=False)
    print(f"[Saved] {llm_output_sim06}")

    result_human_df = collect_top_matches_for_human(
        acc,
        sim,
        top_k=args.top_k_human,
        bottom_k=args.bottom_k_human,
        random_k=args.random_k_human,
        human_mapping=human_mapping,
        llm_mapping=llm_mapping,
        parcel_names=parcel_names,
    )
    result_human_df.to_csv(human_output, index=False)
    print(f"[Saved] {human_output}")

    # 额外保存一份：仅保留 semantic_similarity >= 0.6 的行
    result_human_df_sim06 = result_human_df[result_human_df["semantic_similarity"] >= 0.6].copy()
    result_human_df_sim06.to_csv(human_output_sim06, index=False)
    print(f"[Saved] {human_output_sim06}")


if __name__ == "__main__":
    main()

