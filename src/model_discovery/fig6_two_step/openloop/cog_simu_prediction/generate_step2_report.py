#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成各数据集的 Step2A 单特征预测分析报告（基于一元线性回归模型的预测性能）。

Step2A 关注的是：
- 在训练集上对每个特征单独训练一元线性回归模型
- 在测试集上评估预测值与真实 AIC 的 Pearson 相关系数
- 关注的是特征的泛化预测能力（经过线性映射后的预测性能）
- 按绝对值排序提取 Top-K 和 Bottom-K 特征
"""

import json
import os
from typing import Dict, Any, List

import pandas as pd


BASE_DIR = os.path.dirname(__file__)
FEATURE_ANALYSIS_DIR = os.path.join(BASE_DIR, "results", "feature_analysis")
PARCEL_JSON = "/path/to/project_root/neurocogmap/2b_model_data/parcel.json"
EXPERIMENT_REPORT_JSON = "/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/dataset/experiment_analysis_report.json"
OUTPUT_MD = os.path.join(FEATURE_ANALYSIS_DIR, "Nature_Report_step2_detail.md")


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
    # 原始键形如 "badham2017deficits/exp1.csv"，我们做一个简易映射到 dataset_key 形式
    mapped: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        # 例如 badham2017deficits/exp1.csv -> badham2017deficits_exp1_csv
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


def load_step2a_df(dataset_key: str, by: str = "nll") -> pd.DataFrame:
    """读取指定数据集的 Step2A 结果，返回完整 DataFrame。"""
    csv_name = f"step2A_univariate_train_test_ranked_by_{by}.csv"
    csv_path = os.path.join(FEATURE_ANALYSIS_DIR, dataset_key, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 Step2A 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def safe_get(d: Dict[str, Any], key: str, default: str = "") -> str:
    """从字典中安全取字符串字段。"""
    val = d.get(key, default)
    if val is None:
        return default
    return str(val)


def build_single_dataset_section(
    dataset_key: str,
    parcel_map: Dict[int, Dict[str, Any]],
    exp_report_map: Dict[str, Dict[str, Any]],
    top_k: int = 5,
) -> str:
    """构建某个数据集的单特征分析 markdown 段落。"""
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

    # 单特征 Top-K / Bottom-K（NLL 和 AIC）
    for by in ("nll", "aic"):
        try:
            df = load_step2a_df(dataset_key, by=by)
        except FileNotFoundError as e:
            lines.append(f"未找到 Step2A 文件（{by}）：{e}")
            lines.append("")
            continue

        metric_name = "NLL" if by == "nll" else "AIC"
        score_col = "test_pred_pcc_with_nll" if by == "nll" else "test_pred_pcc_with_aic"
        abs_col = "abs_test_pred_pcc_with_nll" if by == "nll" else "abs_test_pred_pcc_with_aic"

        # 过滤 NaN，避免无效特征进入 Top/Bottom 排序
        df_valid = df.dropna(subset=[score_col, abs_col]).copy()

        # Top-K：按绝对 PCC 从大到小
        df_top = df_valid.sort_values(abs_col, ascending=False).head(top_k)

        lines.append(f"**Top-{top_k} 特征（按测试集 {metric_name} 预测 PCC 绝对值排序）**")
        lines.append("")
        lines.append("| 排名 | feature_idx | PCC | |parcel function_name|  | 功能简述 |")
        lines.append("|------|-------------|-----|----------------------|----------|")

        for rank, row in enumerate(df_top.itertuples(index=False), start=1):
            feature_idx = int(getattr(row, "feature_idx"))
            pcc_val = float(getattr(row, score_col))
            abs_pcc = float(getattr(row, abs_col))

            parcel_info = parcel_map.get(feature_idx, {})
            fname = safe_get(parcel_info, "function_name", "").strip()
            fdesc = safe_get(parcel_info, "function_description", "").strip()

            # function_name 里有 markdown 粗体前缀，简单清洗掉开头的 "** "
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

        # Bottom-K：按绝对 PCC 从小到大，挑选“几乎不相关”的特征
        df_bottom = df_valid.sort_values(abs_col, ascending=True).head(top_k)

        lines.append(f"**Bottom-{top_k} 特征（按测试集 {metric_name} 预测 PCC 绝对值从低到高排序）**")
        lines.append("")
        lines.append("| 排名 | feature_idx | PCC | |parcel function_name|  | 功能简述 |")
        lines.append("|------|-------------|-----|----------------------|----------|")

        for rank, row in enumerate(df_bottom.itertuples(index=False), start=1):
            feature_idx = int(getattr(row, "feature_idx"))
            pcc_val = float(getattr(row, score_col))
            abs_pcc = float(getattr(row, abs_col))

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

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def main(top_k: int = 10) -> None:
    parcel_map = load_parcel_map()
    exp_report_map = load_experiment_report()

    dataset_keys = get_dataset_dirs()

    sections: List[str] = []
    sections.append("## Step2A 单特征预测分析报告（基于一元线性回归模型）")
    sections.append("")
    sections.append(
        "本报告基于 **Step2A：单特征一元线性回归预测**，即在训练集上对每个特征单独训练线性回归模型，"
        "然后在测试集上评估预测值与真实 AIC 的 Pearson 相关系数。"
        "关注的是特征的**泛化预测能力**（经过线性映射后的预测性能），而非原始激活值的直接相关性。"
    )
    sections.append("")
    sections.append(
        "对每个数据集，分别提取："
    )
    sections.append(f"- **Top-{top_k} 特征**：按预测 PCC 绝对值排序，找出预测能力最强的特征")
    sections.append(f"- **Bottom-{top_k} 特征**：按预测 PCC 绝对值排序，找出预测能力最弱的特征")
    sections.append("")

    for dataset_key in dataset_keys:
        sections.append(
            build_single_dataset_section(
                dataset_key=dataset_key,
                parcel_map=parcel_map,
                exp_report_map=exp_report_map,
                top_k=top_k,
            )
        )

    content = "\n".join(sections)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Step2A 单特征预测分析报告已写入: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

