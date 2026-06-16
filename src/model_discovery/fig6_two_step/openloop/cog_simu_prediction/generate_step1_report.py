#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成各数据集的 Step1 直接相关性分析报告（基于原始 parcel 激活与 AIC 的 Pearson 相关系数）。

与 Step2A 不同，Step1 关注的是：
- 原始特征激活值（未经过线性映射）与真实 AIC 的直接相关性
- 现在在训练集（train set）上计算该直接相关性
- 需要明确区分正相关（激活↑ → AIC↑，可能表示干扰/不匹配）和负相关（激活↑ → AIC↓，表示有益通路）
- 按正负相关性分别提取 Top-K 特征进行分析
"""

import json
import os
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(__file__)
FEATURE_ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "feature_analysis")
PARCEL_JSON = "/path/to/project_root/neurocogmap/2b_model_data/parcel.json"
EXPERIMENT_REPORT_JSON = "/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/dataset/experiment_analysis_report.json"
OUTPUT_MD = os.path.join(FEATURE_ANALYSIS_DIR, "Nature_Report_step1_direct_correlation.md")
OUTPUT_JSON = os.path.join(FEATURE_ANALYSIS_DIR, "Nature_Report_step1_direct_correlation_stats.json")
OUTPUT_MAPPED_MD = os.path.join(
    FEATURE_ANALYSIS_DIR, "Nature_Report_step1_mapped_capability_direct_correlation.md"
)
OUTPUT_MAPPED_JSON = os.path.join(
    FEATURE_ANALYSIS_DIR, "Nature_Report_step1_mapped_capability_direct_correlation_stats.json"
)


def load_parcel_map() -> Dict[int, Dict[str, Any]]:
    """加载 parcel.json，建立 parcel_id -> 信息 的映射。"""
    with open(PARCEL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    parcels = data.get("parcel_summaries", [])
    parcel_map: Dict[int, Dict[str, Any]] = {}
    for p in parcels:
        pid = p.get("parcel_id")
        if isinstance(pid, int):
            parcel_map[pid] = p
    return parcel_map


def load_experiment_report() -> Dict[str, Dict[str, Any]]:
    """加载 experiment_analysis_report.json，返回键到条目的映射。"""
    with open(EXPERIMENT_REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapped: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        if "/" in k and k.endswith(".csv"):
            prefix, fname = k.split("/", 1)
            stem = os.path.splitext(fname)[0]
            dataset_key = f"{prefix}_{stem}_csv"
            mapped[dataset_key] = v
    return mapped


def get_dataset_dirs() -> List[str]:
    """列出 feature_analysis 下所有数据集子目录。"""
    dirs: List[str] = []
    for name in os.listdir(FEATURE_ANALYSIS_DIR):
        path = os.path.join(FEATURE_ANALYSIS_DIR, name)
        if os.path.isdir(path) and name.endswith("_csv"):
            dirs.append(name)
    return sorted(dirs)


def load_step1_results(dataset_key: str) -> pd.DataFrame:
    """
    读取指定数据集的 Step1 结果（直接相关性）。
    
    返回包含以下列的 DataFrame：
    - feature_idx
    - pcc_with_train_aic (带符号的 PCC)
    - abs_pcc_with_train_aic (绝对值)
    """
    csv_path = os.path.join(
        FEATURE_ANALYSIS_DIR, dataset_key, "step1_train_feature_pcc_ranked_by_aic.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 Step1 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.copy()

    expected_cols = {"feature_idx", "pcc_with_train_aic", "abs_pcc_with_train_aic"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_key} Step1 文件缺少列: {sorted(missing)}")

    return df


def load_step1_mapped_capability_results(dataset_key: str) -> pd.DataFrame:
    """
    读取指定数据集的 Step1 mapped capability 结果（直接相关性）。

    返回包含以下关键列的 DataFrame：
    - capability_name
    - pcc_with_train_aic
    - abs_pcc_with_train_aic
    """
    csv_path = os.path.join(
        FEATURE_ANALYSIS_DIR, dataset_key, "step1_train_feature_pcc_mapped_capability_by_aic.csv"
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 mapped capability Step1 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.copy()

    expected_cols = {"capability_name", "pcc_with_train_aic", "abs_pcc_with_train_aic"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_key} mapped capability Step1 文件缺少列: {sorted(missing)}")

    # 某些上游 CSV 的 abs_pcc_with_train_aic 与 |pcc_with_train_aic| 不一致。
    # 为避免报告排序和展示被污染，这里统一按 |pcc_with_train_aic| 重算并覆盖。
    recomputed_abs = df["pcc_with_train_aic"].abs()
    mismatch_mask = (df["abs_pcc_with_train_aic"] - recomputed_abs).abs() > 1e-8
    mismatch_count = int(mismatch_mask.sum())
    if mismatch_count > 0:
        sample_caps = df.loc[mismatch_mask, "capability_name"].head(5).tolist()
        print(
            f"[WARN] {dataset_key} 检测到 {mismatch_count} 行 abs_pcc_with_train_aic 与 |pcc_with_train_aic| 不一致，"
            f"已改为使用重算值。示例 capability: {sample_caps}"
        )
    df["abs_pcc_with_train_aic"] = recomputed_abs

    return df


def split_by_sign(df: pd.DataFrame, top_k: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将 DataFrame 按 PCC 符号分为正相关和负相关两组，每组取 Top-K（按绝对值）。
    
    返回: (positive_df, negative_df)
    """
    # 过滤掉 NaN
    df_clean = df.dropna(subset=["pcc_with_train_aic"])
    
    # 正相关：PCC > 0
    positive = df_clean[df_clean["pcc_with_train_aic"] > 0].copy()
    positive = positive.sort_values("abs_pcc_with_train_aic", ascending=False)
    positive_top = positive.head(top_k).copy()
    
    # 负相关：PCC < 0
    negative = df_clean[df_clean["pcc_with_train_aic"] < 0].copy()
    negative = negative.sort_values("abs_pcc_with_train_aic", ascending=False)
    negative_top = negative.head(top_k).copy()
    
    return positive_top, negative_top


def split_mapped_capability_by_sign(
    df: pd.DataFrame, top_k: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将 mapped capability DataFrame 按 PCC 符号分为正相关和负相关两组，每组取 Top-K（按绝对值）。

    返回: (positive_df, negative_df)
    """
    df_clean = df.dropna(subset=["pcc_with_train_aic"])

    positive = df_clean[df_clean["pcc_with_train_aic"] > 0].copy()
    positive = positive.sort_values("abs_pcc_with_train_aic", ascending=False)
    positive_top = positive.head(top_k).copy()

    negative = df_clean[df_clean["pcc_with_train_aic"] < 0].copy()
    negative = negative.sort_values("abs_pcc_with_train_aic", ascending=False)
    negative_top = negative.head(top_k).copy()

    return positive_top, negative_top


def safe_get(d: Dict[str, Any], key: str, default: str = "") -> str:
    """从字典中安全取字符串字段。"""
    val = d.get(key, default)
    if val is None:
        return default
    return str(val)


def clean_function_name(fname: str) -> str:
    """清洗 parcel function_name（用于 markdown/json 展示）。"""
    fname = fname.strip()
    if fname.startswith("**"):
        return fname.lstrip("* ").strip()
    return fname


def clean_function_description(fdesc: str, max_len: int = 160) -> str:
    """清洗 parcel function_description（用于功能简述）。"""
    short_desc = fdesc.replace("\n", " ").strip()
    if len(short_desc) > max_len:
        short_desc = short_desc[:max_len] + "..."
    return short_desc


def extract_step1_stats_for_dataset(
    dataset_key: str,
    df_step1: pd.DataFrame,
    parcel_map: Dict[int, Dict[str, Any]],
    top_k: int = 10,
    near_zero_abs_threshold: float = 0.1,
) -> Dict[str, List[Dict[str, Any]]]:
    """提取 Step1 的三类（正/0/负）Top-K 纯数据统计，用于 JSON 输出。"""
    # 正/负相关 Top-K（按 abs PCC 排序）
    positive_top, negative_top = split_by_sign(df_step1, top_k=top_k)

    # 接近零相关（|PCC|<阈值）Top-K：abs 从小到大（与 markdown 保持一致）
    df_clean = df_step1.dropna(subset=["pcc_with_train_aic", "abs_pcc_with_train_aic"]).copy()
    near_zero_mask = df_clean["abs_pcc_with_train_aic"] < near_zero_abs_threshold
    df_near_zero = df_clean[near_zero_mask].sort_values("abs_pcc_with_train_aic", ascending=True)
    near_zero_top = df_near_zero.head(top_k)

    def _row_to_item(rank: int, row: Any) -> Dict[str, Any]:
        feature_idx = int(getattr(row, "feature_idx"))
        pcc_val = float(getattr(row, "pcc_with_train_aic"))

        parcel_info = parcel_map.get(feature_idx, {})
        fname = safe_get(parcel_info, "function_name", "")
        fdesc = safe_get(parcel_info, "function_description", "")

        return {
            "rank": rank,
            "feature_idx": feature_idx,
            "pcc": pcc_val,
            "parcel_function_name": clean_function_name(fname),
            "function_summary": clean_function_description(fdesc),
        }

    positive_items: List[Dict[str, Any]] = []
    for rank, row in enumerate(positive_top.itertuples(index=False), start=1):
        positive_items.append(_row_to_item(rank, row))

    negative_items: List[Dict[str, Any]] = []
    for rank, row in enumerate(negative_top.itertuples(index=False), start=1):
        negative_items.append(_row_to_item(rank, row))

    zero_items: List[Dict[str, Any]] = []
    for rank, row in enumerate(near_zero_top.itertuples(index=False), start=1):
        zero_items.append(_row_to_item(rank, row))

    # 按用户要求的三类命名：正相关 / 0相关 / 负相关
    return {"positive": positive_items, "zero": zero_items, "negative": negative_items}


def extract_step1_mapped_capability_stats_for_dataset(
    dataset_key: str,
    df_step1_mapped: pd.DataFrame,
    top_k: int = 10,
    near_zero_abs_threshold: float = 0.1,
) -> Dict[str, List[Dict[str, Any]]]:
    """提取 Step1 mapped capability 的三类（正/0/负）Top-K 纯数据统计，用于 JSON 输出。"""
    positive_top, negative_top = split_mapped_capability_by_sign(df_step1_mapped, top_k=top_k)

    df_clean = df_step1_mapped.dropna(
        subset=["pcc_with_train_aic", "abs_pcc_with_train_aic"]
    ).copy()
    near_zero_mask = df_clean["abs_pcc_with_train_aic"] < near_zero_abs_threshold
    df_near_zero = df_clean[near_zero_mask].sort_values("abs_pcc_with_train_aic", ascending=True)
    near_zero_top = df_near_zero.head(top_k)

    def _mapped_row_to_item(rank: int, row: Any) -> Dict[str, Any]:
        item = {
            "rank": rank,
            "capability_name": str(getattr(row, "capability_name")),
            "pcc": float(getattr(row, "pcc_with_train_aic")),
        }

        for optional_col in [
            "mapped_parcel_count",
            "mapping_total_parcel_count",
            "weight_coverage_ratio",
            "selected_top_k_max",
            "selected_top_k_min",
        ]:
            if optional_col in df_step1_mapped.columns:
                item[optional_col] = getattr(row, optional_col)
        return item

    positive_items: List[Dict[str, Any]] = []
    for rank, row in enumerate(positive_top.itertuples(index=False), start=1):
        positive_items.append(_mapped_row_to_item(rank, row))

    negative_items: List[Dict[str, Any]] = []
    for rank, row in enumerate(negative_top.itertuples(index=False), start=1):
        negative_items.append(_mapped_row_to_item(rank, row))

    zero_items: List[Dict[str, Any]] = []
    for rank, row in enumerate(near_zero_top.itertuples(index=False), start=1):
        zero_items.append(_mapped_row_to_item(rank, row))

    return {"positive": positive_items, "zero": zero_items, "negative": negative_items}


def build_single_dataset_section(
    dataset_key: str,
    parcel_map: Dict[int, Dict[str, Any]],
    exp_report_map: Dict[str, Dict[str, Any]],
    top_k: int = 10,
) -> str:
    """构建某个数据集的 Step1 直接相关性分析 markdown 段落。"""
    lines: List[str] = []
    lines.append(f"### 数据集：`{dataset_key}`")
    lines.append("")
    
    # 实验任务简介
    exp_info = exp_report_map.get(dataset_key)
    if exp_info is not None:
        instr_preview = safe_get(exp_info, "instruction_preview")
        trails_preview = exp_info.get("trails_preview", [])
        example_trail = ""
        if isinstance(trails_preview, list) and trails_preview:
            example_trail = str(trails_preview[0])
        lines.append("**任务简介（基于原始实验描述）**")
        lines.append("")
        lines.append(f"- **instruction_preview**：{instr_preview.strip()[:400]}...")
        if example_trail:
            lines.append(f"- **trials_preview 示例**：{example_trail.strip()[:400]}...")
        lines.append("")
    else:
        lines.append("**任务简介**：无法在 `experiment_analysis_report.json` 中找到对应条目。")
        lines.append("")
    
    # 加载 Step1 结果
    try:
        df_step1 = load_step1_results(dataset_key)
    except FileNotFoundError as e:
        print(f"[ERROR] {dataset_key} 未找到 Step1 文件: {e}")
        lines.append(f"未找到 Step1 文件：{e}")
        lines.append("")
        return "\n".join(lines)
    
    # 按符号分组（正相关 / 负相关）
    positive_top, negative_top = split_by_sign(df_step1, top_k=top_k)
    
    # 负相关特征（激活↑ → AIC↓，有益通路）
    lines.append(f"**负相关特征 Top-{top_k}（parcel 激活越高，AIC 越低，表示有益通路）**")
    lines.append("")
    lines.append("| 排名 | feature_idx | PCC | parcel function_name | 功能简述 |")
    lines.append("|------|-------------|-----|---------------------|----------|")
    
    if len(negative_top) == 0:
        lines.append("| - | - | - | 无显著负相关特征 | - |")
    else:
        for rank, row in enumerate(negative_top.itertuples(index=False), start=1):
            feature_idx = int(getattr(row, "feature_idx"))
            pcc_val = float(getattr(row, "pcc_with_train_aic"))
            abs_pcc = float(getattr(row, "abs_pcc_with_train_aic"))
            
            parcel_info = parcel_map.get(feature_idx, {})
            fname = safe_get(parcel_info, "function_name", "").strip()
            fdesc = safe_get(parcel_info, "function_description", "").strip()
            
            if fname.startswith("**"):
                fname_clean = fname.lstrip("* ").strip()
            else:
                fname_clean = fname
            
            short_desc = fdesc.replace("\n", " ").strip()
            if len(short_desc) > 160:
                short_desc = short_desc[:160] + "..."
            
            lines.append(
                f"| {rank} | {feature_idx} | {pcc_val:.3f} (abs={abs_pcc:.3f}) | {fname_clean} | {short_desc} |"
            )
    
    lines.append("")
    
    # 正相关特征（激活↑ → AIC↑，可能表示干扰/不匹配）
    lines.append(f"**正相关特征 Top-{top_k}（parcel 激活越高，AIC 越高，可能表示干扰或不匹配通路）**")
    lines.append("")
    lines.append("| 排名 | feature_idx | PCC | parcel function_name | 功能简述 |")
    lines.append("|------|-------------|-----|---------------------|----------|")
    
    if len(positive_top) == 0:
        lines.append("| - | - | - | 无显著正相关特征 | - |")
    else:
        for rank, row in enumerate(positive_top.itertuples(index=False), start=1):
            feature_idx = int(getattr(row, "feature_idx"))
            pcc_val = float(getattr(row, "pcc_with_train_aic"))
            abs_pcc = float(getattr(row, "abs_pcc_with_train_aic"))
            
            parcel_info = parcel_map.get(feature_idx, {})
            fname = safe_get(parcel_info, "function_name", "").strip()
            fdesc = safe_get(parcel_info, "function_description", "").strip()
            
            if fname.startswith("**"):
                fname_clean = fname.lstrip("* ").strip()
            else:
                fname_clean = fname
            
            short_desc = fdesc.replace("\n", " ").strip()
            if len(short_desc) > 160:
                short_desc = short_desc[:160] + "..."
            
            lines.append(
                f"| {rank} | {feature_idx} | {pcc_val:.3f} (abs={abs_pcc:.3f}) | {fname_clean} | {short_desc} |"
            )
    
    lines.append("")

    # 接近零相关特征（|PCC| 最小的一批），用于辅助专家理解“背景通路”
    df_clean = df_step1.dropna(subset=["pcc_with_train_aic", "abs_pcc_with_train_aic"]).copy()
    near_zero_mask = df_clean["abs_pcc_with_train_aic"] < 0.1
    df_near_zero = df_clean[near_zero_mask].sort_values("abs_pcc_with_train_aic", ascending=True)
    near_zero_top = df_near_zero.head(top_k)

    lines.append(f"**接近零相关特征 Top-{top_k}（|PCC| 最小，parcel 激活变化与 AIC 几乎无关，可视为背景通路）**")
    lines.append("")
    lines.append("| 排名 | feature_idx | PCC | parcel function_name | 功能简述 |")
    lines.append("|------|-------------|-----|---------------------|----------|")

    if len(near_zero_top) == 0:
        lines.append("| - | - | - | 无明显接近零相关特征 | - |")
    else:
        for rank, row in enumerate(near_zero_top.itertuples(index=False), start=1):
            feature_idx = int(getattr(row, "feature_idx"))
            pcc_val = float(getattr(row, "pcc_with_train_aic"))
            abs_pcc = float(getattr(row, "abs_pcc_with_train_aic"))

            parcel_info = parcel_map.get(feature_idx, {})
            fname = safe_get(parcel_info, "function_name", "").strip()
            fdesc = safe_get(parcel_info, "function_description", "").strip()

            if fname.startswith("**"):
                fname_clean = fname.lstrip("* ").strip()
            else:
                fname_clean = fname

            short_desc = fdesc.replace("\n", " ").strip()
            if len(short_desc) > 160:
                short_desc = short_desc[:160] + "..."

            lines.append(
                f"| {rank} | {feature_idx} | {pcc_val:.3f} (abs={abs_pcc:.3f}) | {fname_clean} | {short_desc} |"
            )

    lines.append("")
    
    # 统计信息
    total_features = len(df_step1)
    valid_features = len(df_step1.dropna(subset=["pcc_with_train_aic"]))
    positive_count = len(df_step1[(df_step1["pcc_with_train_aic"] > 0) & (~df_step1["pcc_with_train_aic"].isna())])
    negative_count = len(df_step1[(df_step1["pcc_with_train_aic"] < 0) & (~df_step1["pcc_with_train_aic"].isna())])
    near_zero_count = len(df_step1[(df_step1["abs_pcc_with_train_aic"] < 0.1) & (~df_step1["abs_pcc_with_train_aic"].isna())])
    
    lines.append("**统计信息**")
    lines.append("")
    lines.append(f"- 总特征数：{total_features}")
    lines.append(f"- 有效特征数（非NaN）：{valid_features}")
    lines.append(f"- 正相关特征数：{positive_count}")
    lines.append(f"- 负相关特征数：{negative_count}")
    lines.append(f"- 接近零相关特征数（|PCC| < 0.1）：{near_zero_count}")
    if valid_features > 0:
        max_abs_pcc = df_step1["abs_pcc_with_train_aic"].max()
        lines.append(f"- 最大绝对相关系数：{max_abs_pcc:.3f}")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def build_single_dataset_mapped_capability_section(
    dataset_key: str,
    exp_report_map: Dict[str, Dict[str, Any]],
    top_k: int = 10,
) -> str:
    """构建某个数据集的 Step1 mapped capability 直接相关性分析 markdown 段落。"""
    lines: List[str] = []
    lines.append(f"### 数据集：`{dataset_key}`")
    lines.append("")

    exp_info = exp_report_map.get(dataset_key)
    if exp_info is not None:
        instr_preview = safe_get(exp_info, "instruction_preview")
        trails_preview = exp_info.get("trails_preview", [])
        example_trail = ""
        if isinstance(trails_preview, list) and trails_preview:
            example_trail = str(trails_preview[0])
        lines.append("**任务简介（基于原始实验描述）**")
        lines.append("")
        lines.append(f"- **instruction_preview**：{instr_preview.strip()[:400]}...")
        if example_trail:
            lines.append(f"- **trials_preview 示例**：{example_trail.strip()[:400]}...")
        lines.append("")
    else:
        lines.append("**任务简介**：无法在 `experiment_analysis_report.json` 中找到对应条目。")
        lines.append("")

    try:
        df_mapped = load_step1_mapped_capability_results(dataset_key)
    except FileNotFoundError as e:
        print(f"[ERROR] {dataset_key} 未找到 mapped capability Step1 文件: {e}")
        lines.append(f"未找到 mapped capability Step1 文件：{e}")
        lines.append("")
        return "\n".join(lines)

    positive_top, negative_top = split_mapped_capability_by_sign(df_mapped, top_k=top_k)

    lines.append(f"**负相关 capability Top-{top_k}（capability 激活越高，AIC 越低，表示有益能力）**")
    lines.append("")
    lines.append("| 排名 | capability_name | PCC | mapped_parcel_count | coverage_ratio | selected_top_k_max/min |")
    lines.append("|------|------------------|-----|---------------------|----------------|------------------------|")

    if len(negative_top) == 0:
        lines.append("| - | - | - | 无显著负相关 capability | - | - |")
    else:
        for rank, row in enumerate(negative_top.itertuples(index=False), start=1):
            capability_name = str(getattr(row, "capability_name"))
            pcc_val = float(getattr(row, "pcc_with_train_aic"))
            abs_pcc = float(getattr(row, "abs_pcc_with_train_aic"))
            mapped_count = getattr(row, "mapped_parcel_count", "-")
            coverage = getattr(row, "weight_coverage_ratio", "-")
            top_k_max = getattr(row, "selected_top_k_max", "-")
            top_k_min = getattr(row, "selected_top_k_min", "-")
            coverage_str = f"{float(coverage):.3f}" if coverage != "-" else "-"

            lines.append(
                f"| {rank} | {capability_name} | {pcc_val:.3f} (abs={abs_pcc:.3f}) | "
                f"{mapped_count} | {coverage_str} | {top_k_max}/{top_k_min} |"
            )

    lines.append("")

    lines.append(f"**正相关 capability Top-{top_k}（capability 激活越高，AIC 越高，可能表示干扰能力）**")
    lines.append("")
    lines.append("| 排名 | capability_name | PCC | mapped_parcel_count | coverage_ratio | selected_top_k_max/min |")
    lines.append("|------|------------------|-----|---------------------|----------------|------------------------|")

    if len(positive_top) == 0:
        lines.append("| - | - | - | 无显著正相关 capability | - | - |")
    else:
        for rank, row in enumerate(positive_top.itertuples(index=False), start=1):
            capability_name = str(getattr(row, "capability_name"))
            pcc_val = float(getattr(row, "pcc_with_train_aic"))
            abs_pcc = float(getattr(row, "abs_pcc_with_train_aic"))
            mapped_count = getattr(row, "mapped_parcel_count", "-")
            coverage = getattr(row, "weight_coverage_ratio", "-")
            top_k_max = getattr(row, "selected_top_k_max", "-")
            top_k_min = getattr(row, "selected_top_k_min", "-")
            coverage_str = f"{float(coverage):.3f}" if coverage != "-" else "-"

            lines.append(
                f"| {rank} | {capability_name} | {pcc_val:.3f} (abs={abs_pcc:.3f}) | "
                f"{mapped_count} | {coverage_str} | {top_k_max}/{top_k_min} |"
            )

    lines.append("")

    df_clean = df_mapped.dropna(subset=["pcc_with_train_aic", "abs_pcc_with_train_aic"]).copy()
    near_zero_mask = df_clean["abs_pcc_with_train_aic"] < 0.1
    df_near_zero = df_clean[near_zero_mask].sort_values("abs_pcc_with_train_aic", ascending=True)
    near_zero_top = df_near_zero.head(top_k)

    lines.append(f"**接近零相关 capability Top-{top_k}（|PCC| 最小，能力与 AIC 几乎无关）**")
    lines.append("")
    lines.append("| 排名 | capability_name | PCC | mapped_parcel_count | coverage_ratio | selected_top_k_max/min |")
    lines.append("|------|------------------|-----|---------------------|----------------|------------------------|")

    if len(near_zero_top) == 0:
        lines.append("| - | - | - | 无明显接近零相关 capability | - | - |")
    else:
        for rank, row in enumerate(near_zero_top.itertuples(index=False), start=1):
            capability_name = str(getattr(row, "capability_name"))
            pcc_val = float(getattr(row, "pcc_with_train_aic"))
            abs_pcc = float(getattr(row, "abs_pcc_with_train_aic"))
            mapped_count = getattr(row, "mapped_parcel_count", "-")
            coverage = getattr(row, "weight_coverage_ratio", "-")
            top_k_max = getattr(row, "selected_top_k_max", "-")
            top_k_min = getattr(row, "selected_top_k_min", "-")
            coverage_str = f"{float(coverage):.3f}" if coverage != "-" else "-"

            lines.append(
                f"| {rank} | {capability_name} | {pcc_val:.3f} (abs={abs_pcc:.3f}) | "
                f"{mapped_count} | {coverage_str} | {top_k_max}/{top_k_min} |"
            )

    lines.append("")

    total_caps = len(df_mapped)
    valid_caps = len(df_mapped.dropna(subset=["pcc_with_train_aic"]))
    positive_count = len(df_mapped[(df_mapped["pcc_with_train_aic"] > 0) & (~df_mapped["pcc_with_train_aic"].isna())])
    negative_count = len(df_mapped[(df_mapped["pcc_with_train_aic"] < 0) & (~df_mapped["pcc_with_train_aic"].isna())])
    near_zero_count = len(df_mapped[(df_mapped["abs_pcc_with_train_aic"] < 0.1) & (~df_mapped["abs_pcc_with_train_aic"].isna())])

    lines.append("**统计信息**")
    lines.append("")
    lines.append(f"- 总 capability 数：{total_caps}")
    lines.append(f"- 有效 capability 数（非NaN）：{valid_caps}")
    lines.append(f"- 正相关 capability 数：{positive_count}")
    lines.append(f"- 负相关 capability 数：{negative_count}")
    lines.append(f"- 接近零相关 capability 数（|PCC| < 0.1）：{near_zero_count}")
    if valid_caps > 0:
        max_abs_pcc = df_mapped["abs_pcc_with_train_aic"].max()
        lines.append(f"- 最大绝对相关系数：{max_abs_pcc:.3f}")
    lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main(top_k: int = 10) -> None:
    """生成 Step1 直接相关性分析报告。"""
    parcel_map = load_parcel_map()
    exp_report_map = load_experiment_report()
    
    dataset_keys = get_dataset_dirs()
    
    sections: List[str] = []
    stats_by_dataset: Dict[str, Any] = {}
    sections.append("# Step1 直接相关性分析报告（原始 Parcel 激活 vs AIC）")
    sections.append("")
    sections.append("## 说明")
    sections.append("")
    sections.append(
        "本报告基于 **Step1：直接相关性分析**，即在训练集上直接计算每个 parcel 特征激活值与真实 AIC 的 Pearson 相关系数（PCC），"
        "**不经过任何线性映射或模型训练**。"
    )
    sections.append("")
    sections.append("### 关键区别")
    sections.append("")
    sections.append("- **负相关（PCC < 0）**：parcel 激活越高 → AIC 越低 → 决策质量越好，表示该认知模块对任务有**正向贡献**")
    sections.append("- **正相关（PCC > 0）**：parcel 激活越高 → AIC 越高 → 决策质量越差，可能表示该模块与任务**不匹配或产生干扰**")
    sections.append("- **接近零相关（|PCC| ≈ 0）**：该模块与任务决策质量**无明显关联**，可能是背景通路")
    sections.append("")
    sections.append("### 分析方法")
    sections.append("")
    sections.append(
        f"对每个数据集，分别提取："
    )
    sections.append(f"- **负相关特征 Top-{top_k}**：按绝对值排序，找出激活升高时能降低 AIC 的有益通路")
    sections.append(f"- **正相关特征 Top-{top_k}**：按绝对值排序，找出激活升高时会导致 AIC 升高的干扰通路")
    sections.append("")
    sections.append("---")
    sections.append("")
    
    for dataset_key in dataset_keys:
        try:
            section = build_single_dataset_section(
                dataset_key=dataset_key,
                parcel_map=parcel_map,
                exp_report_map=exp_report_map,
                top_k=top_k,
            )
            sections.append(section)

            # 同步提取 JSON 统计（纯数据，不写任何文字介绍）
            df_step1 = load_step1_results(dataset_key)
            stats_by_dataset[dataset_key] = extract_step1_stats_for_dataset(
                dataset_key=dataset_key,
                df_step1=df_step1,
                parcel_map=parcel_map,
                top_k=top_k,
            )
        except Exception as e:
            print(f"[ERROR] 处理 {dataset_key} 时出错: {e}")
            sections.append(f"### 数据集：`{dataset_key}`")
            sections.append("")
            sections.append(f"**错误**：{e}")
            sections.append("")
            sections.append("---")
            sections.append("")
            # 避免 JSON 缺失该 dataset_key，记录为空三类
            stats_by_dataset[dataset_key] = {"positive": [], "zero": [], "negative": []}
    
    content = "\n".join(sections)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Step1 直接相关性分析报告已写入: {OUTPUT_MD}")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats_by_dataset, f, ensure_ascii=False, indent=2)
    print(f"Step1 直接相关性统计 JSON 已写入: {OUTPUT_JSON}")

    mapped_sections: List[str] = []
    mapped_stats_by_dataset: Dict[str, Any] = {}
    mapped_sections.append("# Step1 映射能力相关性分析报告（Mapped Capability vs AIC）")
    mapped_sections.append("")
    mapped_sections.append("## 说明")
    mapped_sections.append("")
    mapped_sections.append(
        "本报告基于 `step1_train_feature_pcc_mapped_capability_by_aic.csv`，"
        "在训练集上计算映射后的 capability 指标与真实 AIC 的 Pearson 相关系数（PCC）。"
    )
    mapped_sections.append("")
    mapped_sections.append("### 分析方法")
    mapped_sections.append("")
    mapped_sections.append(f"- **负相关 capability Top-{top_k}**：按绝对值排序，找出激活升高时能降低 AIC 的有益能力")
    mapped_sections.append(f"- **正相关 capability Top-{top_k}**：按绝对值排序，找出激活升高时会导致 AIC 升高的干扰能力")
    mapped_sections.append(f"- **接近零相关 capability Top-{top_k}**：按绝对值从小到大排序，辅助识别背景能力")
    mapped_sections.append("")
    mapped_sections.append("---")
    mapped_sections.append("")

    for dataset_key in dataset_keys:
        try:
            mapped_section = build_single_dataset_mapped_capability_section(
                dataset_key=dataset_key,
                exp_report_map=exp_report_map,
                top_k=top_k,
            )
            mapped_sections.append(mapped_section)

            df_mapped = load_step1_mapped_capability_results(dataset_key)
            mapped_stats_by_dataset[dataset_key] = extract_step1_mapped_capability_stats_for_dataset(
                dataset_key=dataset_key,
                df_step1_mapped=df_mapped,
                top_k=top_k,
            )
        except Exception as e:
            print(f"[ERROR] 处理 mapped capability {dataset_key} 时出错: {e}")
            mapped_sections.append(f"### 数据集：`{dataset_key}`")
            mapped_sections.append("")
            mapped_sections.append(f"**错误**：{e}")
            mapped_sections.append("")
            mapped_sections.append("---")
            mapped_sections.append("")
            mapped_stats_by_dataset[dataset_key] = {"positive": [], "zero": [], "negative": []}

    mapped_content = "\n".join(mapped_sections)
    with open(OUTPUT_MAPPED_MD, "w", encoding="utf-8") as f:
        f.write(mapped_content)
    print(f"Step1 mapped capability 相关性分析报告已写入: {OUTPUT_MAPPED_MD}")
    with open(OUTPUT_MAPPED_JSON, "w", encoding="utf-8") as f:
        json.dump(mapped_stats_by_dataset, f, ensure_ascii=False, indent=2)
    print(f"Step1 mapped capability 相关性统计 JSON 已写入: {OUTPUT_MAPPED_JSON}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成 Step1 直接相关性分析报告")
    parser.add_argument("--top-k", type=int, default=10, help="每个方向（正/负）提取的 Top-K 特征数")
    args = parser.parse_args()
    main(top_k=args.top_k)
