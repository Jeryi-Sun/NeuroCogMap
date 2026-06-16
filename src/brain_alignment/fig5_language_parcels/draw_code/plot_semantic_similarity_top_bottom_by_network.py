#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 7Networks 绘制 semantic_similarity：
1) 对每个 human parcel，分别计算其 top-k / bottom-k LLM 匹配的 semantic_similarity 均值；
2) 按 Yeo7 network 聚合 parcel 均值，得到 network 级别的均值和 SEM；
3) 绘制每个 network 两个柱子（Top vs Bottom），并叠加 parcel 散点（Nature 风格小图）。
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy import stats

from common import RESULT_DIR, ensure_output_dir, should_skip

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_DISPLAY_NAMES: Dict[str, str] = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "DorsalAttn",
    "SalVentAttn": "Salience",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}


def configure_nature_style_small_figure() -> None:
    """设置 1/4 A4 小图的 Nature 风格全局参数。"""
    try:
        fm.fontManager = fm.FontManager()
    except Exception as exc:  # noqa: BLE001
        logger.warning("刷新字体管理器失败: %s", exc)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.linewidth"] = 0.6
    plt.rcParams["xtick.major.width"] = 0.6
    plt.rcParams["ytick.major.width"] = 0.6
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def _darken_color(hex_color: str, factor: float = 0.7) -> Tuple[float, float, float]:
    """将颜色加深，用于散点与柱子的同色系区分。"""
    rgb = np.array(to_rgb(hex_color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def extract_network_from_parcel_name(parcel_name: str) -> Optional[str]:
    """
    从 human_parcel_name / parcel_name 中解析 Yeo7 network 名称。
    例如: 7Networks_LH_Vis_1 -> Vis
    """
    parts = parcel_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


@dataclass
class NetworkConditionSummary:
    network: str
    condition: str  # "top" 或 "bottom"
    mean: float
    sem: float
    n_parcels: int


def infer_subject_id_from_csv_path(csv_path: Path) -> str:
    """从输入 CSV 路径中推断被试 ID（如 uts02 / uts03），失败时返回 unknown_subject。"""
    parts = list(csv_path.resolve().parts)
    if "draw_result" in parts:
        idx = parts.index("draw_result")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    logger.warning("无法从路径推断被试 ID，使用 unknown_subject: %s", csv_path)
    return "unknown_subject"


def load_top_bottom_pairs(csv_path: Path, top_k: int, bottom_k: int) -> pd.DataFrame:
    """
    读取由 export_top_human_matches.py 导出的 top/bottom 匹配关系表，
    并根据传入的 top_k / bottom_k 只保留对应 rank_by_acc 范围内的配对。
    """
    logger.info("加载 top/bottom 匹配 CSV: %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到输入 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {
        "human_parcel_name",
        "selection_type",  # "top" / "bottom" / "random"
        "semantic_similarity",
        "rank_by_acc",
        "prediction_accuracy",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {missing}")

    # 仅保留核心列，避免后续 groupby 受无关列影响
    df = df[
        [
            "human_parcel_name",
            "selection_type",
            "semantic_similarity",
            "rank_by_acc",
            "prediction_accuracy",
        ]
    ].copy()

    # 根据 rank_by_acc 和 selection_type 进行动态 K 筛选：
    # - top:  保留 rank_by_acc <= top_k
    # - bottom: 保留 rank_by_acc <= bottom_k
    # 这样可以在同一个 CSV 中根据命令行参数灵活选择使用 top-1/top-3/... 等不同范围。
    cond_top = (df["selection_type"] == "top") & df["rank_by_acc"].notna()
    cond_bottom = (df["selection_type"] == "bottom") & df["rank_by_acc"].notna()

    if top_k > 0:
        cond_top = cond_top & (df["rank_by_acc"] <= top_k)
    else:
        # top_k <= 0 时，不使用任何 top 行
        cond_top = df["selection_type"] == "__never_match__"

    if bottom_k > 0:
        cond_bottom = cond_bottom & (df["rank_by_acc"] <= bottom_k)
    else:
        # bottom_k <= 0 时，不使用任何 bottom 行
        cond_bottom = df["selection_type"] == "__never_match__"

    df_filtered = df[cond_top | cond_bottom].copy()
    if df_filtered.empty:
        raise ValueError(
            f"在输入文件中根据 top_k={top_k}, bottom_k={bottom_k} 筛选后数据为空，"
            "请检查 CSV 是否包含相应 rank_by_acc 的 top/bottom 记录。"
        )

    return df_filtered[
        ["human_parcel_name", "selection_type", "semantic_similarity", "prediction_accuracy"]
    ]


def filter_parcels_by_network_top_percent(
    df: pd.DataFrame,
    top_percent: float,
) -> pd.DataFrame:
    """
    仅保留「每个 network 内，最高预测准确率位于前 top_percent%」的人脑 parcel。

    实现思路：
      1) 对于每个 human_parcel，先基于 prediction_accuracy 求其在 CSV 中出现过的
         所有配对里的最大预测准确率（通常是 rank_by_acc=1 对应的那一行）；
      2) 将 human_parcel 按 Yeo7 network 归组，在每个 network 内计算
         max_prediction_accuracy 的分位数阈值；
      3) 只保留 max_prediction_accuracy 高于该阈值的 parcels 所在的所有行。
    """
    if top_percent >= 100.0:
        # 100% 表示不过滤
        return df
    if top_percent <= 0.0:
        raise ValueError(f"top_percent 必须在 (0, 100] 区间内，当前为 {top_percent}.")

    work_df = df.copy()
    work_df["network"] = (
        work_df["human_parcel_name"].astype(str).apply(extract_network_from_parcel_name)
    )
    before_drop = len(work_df)
    work_df = work_df.dropna(subset=["network"])
    if len(work_df) < before_drop:
        logger.warning("在按 network 过滤时，有 %d 条记录无法解析 network，已丢弃。", before_drop - len(work_df))

    # 每个 parcel 在各自 network 内的「最高预测准确率」
    parcel_max = (
        work_df.groupby(["human_parcel_name", "network"])["prediction_accuracy"]
        .max()
        .reset_index()
        .rename(columns={"prediction_accuracy": "max_prediction_accuracy"})
    )

    # 在每个 network 内求出阈值：位于 top_percent% 之上的 parcels 会被保留
    # 例如 top_percent=20 → 保留最高的 20%，对应的分位数为 1 - 0.2 = 0.8
    quantile = 1.0 - top_percent / 100.0
    thresholds = (
        parcel_max.groupby("network")["max_prediction_accuracy"]
        .quantile(quantile)
        .rename("threshold")
        .reset_index()
    )

    parcel_with_thr = parcel_max.merge(thresholds, on="network", how="left")
    keep_parcels = parcel_with_thr.loc[
        parcel_with_thr["max_prediction_accuracy"] >= parcel_with_thr["threshold"],
        "human_parcel_name",
    ].unique()

    df_filtered = df[df["human_parcel_name"].isin(keep_parcels)].copy()
    if df_filtered.empty:
        raise ValueError(
            "根据每个 network 的最高预测准确率前 "
            f"{top_percent}% 进行过滤后，没有任何 parcel 被保留，请检查参数设置。"
        )

    logger.info(
        "根据每个 network 最高预测准确率前 %.1f%% 过滤后，保留 %d / %d 条记录，涉及 %d 个 parcel。",
        top_percent,
        len(df_filtered),
        len(df),
        len(keep_parcels),
    )
    return df_filtered


def compute_parcel_level_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 human parcel、每种 selection_type（top/bottom），
    计算其 semantic_similarity 均值，并附上所属 Yeo7 network。
    """
    df = df.copy()
    df["network"] = df["human_parcel_name"].astype(str).apply(extract_network_from_parcel_name)
    before_drop = len(df)
    df = df.dropna(subset=["network"])
    if len(df) < before_drop:
        logger.warning("有 %d 条记录无法解析 network，已丢弃。", before_drop - len(df))

    # 只保留我们关心的两类
    df = df[df["selection_type"].isin(["top", "bottom"])].copy()
    if df.empty:
        raise ValueError("筛选 selection_type in ['top', 'bottom'] 后数据为空，请检查输入文件。")

    if "subject_id" not in df.columns:
        raise ValueError("输入数据缺少 subject_id 列，无法进行跨被试融合统计。")

    grouped = (
        df.groupby(["subject_id", "human_parcel_name", "network", "selection_type"])["semantic_similarity"]
        .mean()
        .reset_index()
        .rename(columns={"semantic_similarity": "parcel_mean_similarity"})
    )

    logger.info(
        "共得到 %d 个 (parcel, network, selection_type) 组合。",
        len(grouped),
    )
    return grouped


def compute_network_condition_summary(parcel_means: pd.DataFrame) -> List[NetworkConditionSummary]:
    """
    输入 parcel 层面的均值表，按 (network, selection_type) 聚合，得到 network 级别均值和 SEM。
    """
    summaries: List[NetworkConditionSummary] = []
    for network in NETWORK_ORDER:
        for condition in ("top", "bottom"):
            mask = (parcel_means["network"] == network) & (parcel_means["selection_type"] == condition)
            values = parcel_means.loc[mask, "parcel_mean_similarity"].to_numpy(dtype=float)
            if values.size == 0:
                continue
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            sem = std / np.sqrt(values.size) if values.size > 0 else 0.0
            summaries.append(
                NetworkConditionSummary(
                    network=network,
                    condition=condition,
                    mean=mean,
                    sem=sem,
                    n_parcels=values.size,
                )
            )

    if not summaries:
        raise ValueError("按 network 聚合后为空，请检查输入数据与 NETWORK_ORDER 是否一致。")

    return summaries


def compute_top_bottom_significance(parcel_means: pd.DataFrame) -> pd.DataFrame:
    """
    按 network 计算 top vs bottom 的显著性（配对 t 检验）。

    对于每个 network：
      - 以 human_parcel_name 为单位，构造 top/bottom 成对的 parcel_mean_similarity
      - 使用 scipy.stats.ttest_rel 进行配对 t 检验
    返回包含每个 network 的 t 统计量、p 值、样本数等信息的 DataFrame。
    """
    records = []
    for network in NETWORK_ORDER:
        df_net = parcel_means.loc[parcel_means["network"] == network].copy()
        if df_net.empty:
            continue
        pivot = df_net.pivot_table(
            index=["subject_id", "human_parcel_name"],
            columns="selection_type",
            values="parcel_mean_similarity",
            aggfunc="mean",
        )
        if "top" not in pivot.columns or "bottom" not in pivot.columns:
            logger.warning("network=%s 缺少 top 或 bottom 数据，跳过显著性计算。", network)
            continue
        pivot = pivot.dropna(subset=["top", "bottom"])
        if pivot.empty:
            logger.warning("network=%s 在配对后没有有效的 (top, bottom) 成对样本，跳过。", network)
            continue

        top_vals = pivot["top"].to_numpy(dtype=float)
        bottom_vals = pivot["bottom"].to_numpy(dtype=float)
        if top_vals.size < 2:
            logger.warning("network=%s 有效成对样本数 < 2 (n=%d)，不进行 t 检验。", network, top_vals.size)
            continue

        t_stat, p_val = stats.ttest_rel(top_vals, bottom_vals, nan_policy="omit")
        diff = top_vals - bottom_vals
        mean_diff = float(np.nanmean(diff))
        std_diff = float(np.nanstd(diff, ddof=1)) if diff.size > 1 else np.nan
        cohen_d = float(mean_diff / std_diff) if std_diff not in (0.0, np.nan) else np.nan

        records.append(
            {
                "network": network,
                "n_pairs": int(top_vals.size),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "mean_top": float(np.nanmean(top_vals)),
                "mean_bottom": float(np.nanmean(bottom_vals)),
                "mean_diff_top_minus_bottom": mean_diff,
                "cohen_d_paired": cohen_d,
            }
        )

    if not records:
        raise ValueError("未能为任何 network 计算出有效的 top vs bottom 显著性，请检查输入数据。")

    sig_df = pd.DataFrame.from_records(records)
    logger.info("已为 %d 个 network 计算 top vs bottom 显著性。", len(sig_df))
    return sig_df


def compute_network_mean_variance_summary(parcel_means: pd.DataFrame) -> pd.DataFrame:
    """
    先在每个被试内按 (network, selection_type) 求均值，再跨被试计算均值与方差。

    输出字段：
      - network, selection_type
      - n_subjects
      - mean_across_subjects
      - variance_across_subjects
      - sem_across_subjects
    """
    if "subject_id" not in parcel_means.columns:
        raise ValueError("parcel_means 缺少 subject_id 列，无法计算跨被试均值/方差。")

    subject_level = (
        parcel_means.groupby(["subject_id", "network", "selection_type"])["parcel_mean_similarity"]
        .mean()
        .reset_index()
        .rename(columns={"parcel_mean_similarity": "subject_network_mean"})
    )
    if subject_level.empty:
        raise ValueError("subject-level network 汇总为空，无法计算跨被试均值/方差。")

    summary = (
        subject_level.groupby(["network", "selection_type"])["subject_network_mean"]
        .agg(["count", "mean", "var", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_subjects",
                "mean": "mean_across_subjects",
                "var": "variance_across_subjects",
                "std": "std_across_subjects",
            }
        )
    )
    summary["sem_across_subjects"] = summary["std_across_subjects"] / np.sqrt(summary["n_subjects"])
    summary = summary.drop(columns=["std_across_subjects"])
    logger.info("已生成 network 跨被试均值/方差汇总，共 %d 行。", len(summary))
    return summary


def save_network_mean_variance_summary(
    summary_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    bottom_k: int,
) -> Path:
    """保存融合被试后按 network 的均值/方差统计表。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"semantic_similarity_network_mean_variance_top{top_k}_bottom{bottom_k}.csv"
    summary_df.to_csv(out_path, index=False)
    logger.info("network 均值/方差统计表已保存: %s", out_path)
    return out_path


def save_significance_table(
    sig_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    bottom_k: int,
) -> Path:
    """
    将 top vs bottom 的显著性结果保存为 CSV。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"semantic_similarity_top_vs_bottom_significance_top{top_k}_bottom{bottom_k}.csv"
    sig_df.to_csv(out_path, index=False)
    logger.info("显著性结果表已保存: %s", out_path)
    return out_path


def _pvalue_to_stars(p_value: float) -> str:
    """将 p 值转换为星号标记。"""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_semantic_similarity_by_network(
    parcel_means: pd.DataFrame,
    summaries: List[NetworkConditionSummary],
    output_path: Path,
    top_k: int,
    bottom_k: int,
    overwrite: bool,
    sig_df: Optional[pd.DataFrame] = None,
) -> None:
    """绘制每个 Yeo7 network 的 Top vs Bottom semantic_similarity 柱状图（带散点和 SEM）。"""
    if should_skip(output_path, overwrite):
        return

    configure_nature_style_small_figure()
    fig, ax = plt.subplots(figsize=(1.8, 1.4))

    # 仅保留在 summaries 中实际出现的 network 顺序
    present_networks = sorted({s.network for s in summaries}, key=lambda n: NETWORK_ORDER.index(n))
    x = np.arange(len(present_networks), dtype=float)

    # 颜色：Top 使用蓝色，Bottom 使用黄色（在所有 network 上保持一致，便于条件对比）
    color_top = "#579FCA"   # medium_blue
    color_bottom = "#F7DC7C"  # warm_yellow
    scatter_top = _darken_color(color_top, factor=0.7)
    scatter_bottom = _darken_color(color_bottom, factor=0.7)

    bar_width = 0.32
    offsets = {"top": -bar_width / 2.0, "bottom": bar_width / 2.0}

    means_top, sems_top, means_bottom, sems_bottom = [], [], [], []
    for network in present_networks:
        # 方便后面统一算 y 轴范围
        s_top = next((s for s in summaries if s.network == network and s.condition == "top"), None)
        s_bottom = next((s for s in summaries if s.network == network and s.condition == "bottom"), None)
        means_top.append(s_top.mean if s_top is not None else np.nan)
        sems_top.append(s_top.sem if s_top is not None else 0.0)
        means_bottom.append(s_bottom.mean if s_bottom is not None else np.nan)
        sems_bottom.append(s_bottom.sem if s_bottom is not None else 0.0)

    means_top_arr = np.array(means_top, dtype=float)
    sems_top_arr = np.array(sems_top, dtype=float)
    means_bottom_arr = np.array(means_bottom, dtype=float)
    sems_bottom_arr = np.array(sems_bottom, dtype=float)

    # 绘制 Top 柱子
    ax.bar(
        x + offsets["top"],
        means_top_arr,
        yerr=sems_top_arr,
        width=bar_width,
        color=color_top,
        edgecolor="#333333",
        linewidth=0.6,
        error_kw={"elinewidth": 0.6, "ecolor": "#333333", "capsize": 1.6, "capthick": 0.6},
        label=f"Top-{top_k}",
        zorder=2,
    )
    # 绘制 Bottom 柱子
    ax.bar(
        x + offsets["bottom"],
        means_bottom_arr,
        yerr=sems_bottom_arr,
        width=bar_width,
        color=color_bottom,
        edgecolor="#333333",
        linewidth=0.6,
        error_kw={"elinewidth": 0.6, "ecolor": "#333333", "capsize": 1.6, "capthick": 0.6},
        label=f"Bottom-{bottom_k}",
        zorder=2,
    )

    # 叠加 parcel-level scatter（jitter），便于看到每个 network 内部的分布
    rng = np.random.default_rng(42)
    for i, network in enumerate(present_networks):
        for condition, offset, color in (
            ("top", offsets["top"], scatter_top),
            ("bottom", offsets["bottom"], scatter_bottom),
        ):
            mask = (parcel_means["network"] == network) & (parcel_means["selection_type"] == condition)
            vals = parcel_means.loc[mask, "parcel_mean_similarity"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.09, 0.09, size=vals.size)
            ax.scatter(
                np.full(vals.size, x[i] + offset) + jitter,
                vals,
                s=4,
                color=color,
                edgecolors="#222222",
                linewidths=0.3,
                alpha=0.6,
                zorder=3,
            )

    display_names = [NETWORK_DISPLAY_NAMES.get(n, n) for n in present_networks]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_xlabel("Yeo7 Network", fontsize=5, fontweight="bold")
    ax.set_ylabel("Semantic similarity (mean over parcels)", fontsize=5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=5, width=0.6, length=2.5)

    # y 轴范围和少量 padding
    all_values = np.concatenate(
        [
            means_top_arr + sems_top_arr,
            means_bottom_arr + sems_bottom_arr,
            parcel_means["parcel_mean_similarity"].to_numpy(dtype=float),
        ]
    )
    y_max = float(np.nanmax(all_values))
    y_min = float(np.nanmin(all_values))
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1e-6) * 0.1
    ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.12 * y_range)

    # 若提供显著性结果表，则在对应 network 上绘制显著性标记（星号）
    if sig_df is not None and not sig_df.empty:
        for i, network in enumerate(present_networks):
            row = sig_df.loc[sig_df["network"] == network]
            if row.empty:
                continue
            p_val = float(row["p_value"].iloc[0])
            stars = _pvalue_to_stars(p_val)
            if not stars:
                continue

            bar_top_y = max(
                means_top_arr[i] + sems_top_arr[i],
                means_bottom_arr[i] + sems_bottom_arr[i],
            )
            bar_height = y_max - y_min
            if bar_height <= 0:
                bar_height = max(abs(bar_top_y), 1e-6) * 0.1
            y_sig = bar_top_y + 0.05 * bar_height

            x_left = x[i] + offsets["top"]
            x_right = x[i] + offsets["bottom"]
            ax.plot(
                [x_left, x_left, x_right, x_right],
                [y_sig - 0.01 * bar_height, y_sig, y_sig, y_sig - 0.01 * bar_height],
                color="#333333",
                linewidth=0.5,
                zorder=4,
            )
            ax.text(
                x[i],
                y_sig + 0.01 * bar_height,
                stars,
                ha="center",
                va="bottom",
                fontsize=5,
                color="#222222",
                zorder=5,
            )

    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    # 简洁图例
    ax.legend(
        loc="upper right",
        fontsize=5,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "图片已保存: %s, %s, %s",
        output_path,
        output_path.with_suffix(".svg"),
        output_path.with_suffix(".pdf"),
    )


def plot_semantic_similarity_by_network_paired_lines(
    parcel_means: pd.DataFrame,
    summaries: List[NetworkConditionSummary],
    output_path: Path,
    top_k: int,
    bottom_k: int,
    overwrite: bool,
    sig_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    绘制无散点版本柱状图：
    1) 仅显示 Top/Bottom 柱子与误差线（去除误差线横帽）；
    2) 用细线连接同一 parcel 的 top 与 bottom（paired lines）。
    """
    if should_skip(output_path, overwrite):
        return

    configure_nature_style_small_figure()
    # 1/4 A4 小图宽度（~45 mm）
    fig, ax = plt.subplots(figsize=(1.8, 1.4))

    present_networks = sorted({s.network for s in summaries}, key=lambda n: NETWORK_ORDER.index(n))
    x = np.arange(len(present_networks), dtype=float)

    color_top = "#579FCA"
    color_bottom = "#F7DC7C"
    paired_line_color = "#9A9A9A"

    bar_width = 0.32
    offsets = {"top": -bar_width / 2.0, "bottom": bar_width / 2.0}

    means_top, sems_top, means_bottom, sems_bottom = [], [], [], []
    for network in present_networks:
        s_top = next((s for s in summaries if s.network == network and s.condition == "top"), None)
        s_bottom = next((s for s in summaries if s.network == network and s.condition == "bottom"), None)
        means_top.append(s_top.mean if s_top is not None else np.nan)
        sems_top.append(s_top.sem if s_top is not None else 0.0)
        means_bottom.append(s_bottom.mean if s_bottom is not None else np.nan)
        sems_bottom.append(s_bottom.sem if s_bottom is not None else 0.0)

    means_top_arr = np.array(means_top, dtype=float)
    sems_top_arr = np.array(sems_top, dtype=float)
    means_bottom_arr = np.array(means_bottom, dtype=float)
    sems_bottom_arr = np.array(sems_bottom, dtype=float)

    ax.bar(
        x + offsets["top"],
        means_top_arr,
        yerr=sems_top_arr,
        width=bar_width,
        color=color_top,
        edgecolor="#333333",
        linewidth=0.5,
        error_kw={"elinewidth": 0.5, "ecolor": "#333333", "capsize": 0},
        label=f"Top-{top_k}",
        zorder=2,
    )
    ax.bar(
        x + offsets["bottom"],
        means_bottom_arr,
        yerr=sems_bottom_arr,
        width=bar_width,
        color=color_bottom,
        edgecolor="#333333",
        linewidth=0.5,
        error_kw={"elinewidth": 0.5, "ecolor": "#333333", "capsize": 0},
        label=f"Bottom-{bottom_k}",
        zorder=2,
    )

    # 同一 (subject_id, human_parcel_name, network) 的 top/bottom 成对连线
    pair_table = parcel_means.pivot_table(
        index=["subject_id", "human_parcel_name", "network"],
        columns="selection_type",
        values="parcel_mean_similarity",
        aggfunc="mean",
    ).reset_index()
    pair_table = pair_table.dropna(subset=["top", "bottom"])

    for i, network in enumerate(present_networks):
        rows = pair_table[pair_table["network"] == network]
        if rows.empty:
            continue
        x_left = x[i] + offsets["top"]
        x_right = x[i] + offsets["bottom"]
        for _, row in rows.iterrows():
            ax.plot(
                [x_left, x_right],
                [float(row["top"]), float(row["bottom"])],
                color=paired_line_color,
                linewidth=0.3,
                alpha=0.4,
                zorder=3.2,
            )

    display_names = [NETWORK_DISPLAY_NAMES.get(n, n) for n in present_networks]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_xlabel("Yeo7 Network", fontsize=5, fontweight="bold")
    ax.set_ylabel("Semantic similarity (mean over parcels)", fontsize=5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=5, width=0.5, length=2.2)

    all_values = np.concatenate(
        [
            means_top_arr + sems_top_arr,
            means_bottom_arr + sems_bottom_arr,
            pair_table["top"].to_numpy(dtype=float) if not pair_table.empty else np.array([], dtype=float),
            pair_table["bottom"].to_numpy(dtype=float) if not pair_table.empty else np.array([], dtype=float),
        ]
    )
    y_max = float(np.nanmax(all_values))
    y_min = float(np.nanmin(all_values))
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1e-6) * 0.1
    ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.12 * y_range)

    if sig_df is not None and not sig_df.empty:
        for i, network in enumerate(present_networks):
            row = sig_df.loc[sig_df["network"] == network]
            if row.empty:
                continue
            p_val = float(row["p_value"].iloc[0])
            stars = _pvalue_to_stars(p_val)
            if not stars:
                continue

            bar_top_y = max(
                means_top_arr[i] + sems_top_arr[i],
                means_bottom_arr[i] + sems_bottom_arr[i],
            )
            y_sig = bar_top_y + 0.05 * y_range
            x_left = x[i] + offsets["top"]
            x_right = x[i] + offsets["bottom"]
            ax.plot(
                [x_left, x_left, x_right, x_right],
                [y_sig - 0.01 * y_range, y_sig, y_sig, y_sig - 0.01 * y_range],
                color="#333333",
                linewidth=0.45,
                zorder=4,
            )
            ax.text(
                x[i],
                y_sig + 0.01 * y_range,
                stars,
                ha="center",
                va="bottom",
                fontsize=5,
                color="#222222",
                zorder=5,
            )

    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax.legend(
        loc="upper right",
        fontsize=5,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.4,
        borderpad=0.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "无散点 paired-lines 柱状图已保存: %s, %s, %s",
        output_path,
        output_path.with_suffix(".svg"),
        output_path.with_suffix(".pdf"),
    )


def plot_semantic_similarity_violin_by_network(
    parcel_means: pd.DataFrame,
    output_path: Path,
    top_k: int,
    bottom_k: int,
    overwrite: bool,
    sig_df: Optional[pd.DataFrame] = None,
) -> None:
    """绘制按 network 分组的 Top/Bottom 小提琴图。"""
    if should_skip(output_path, overwrite):
        return

    configure_nature_style_small_figure()
    fig, ax = plt.subplots(figsize=(3.5, 1.8))

    df_plot = parcel_means.copy()
    df_plot["network"] = pd.Categorical(df_plot["network"], categories=NETWORK_ORDER, ordered=True)
    df_plot = df_plot[df_plot["selection_type"].isin(["top", "bottom"])].copy()
    if df_plot.empty:
        raise ValueError("用于小提琴图的数据为空。")

    # 统一配色：Top=蓝，Bottom=黄
    violin_colors = {"top": "#579FCA", "bottom": "#F7DC7C"}
    condition_order = ["top", "bottom"]

    positions = {net: i for i, net in enumerate(NETWORK_ORDER)}
    bar_width = 0.35
    offsets = {"top": -bar_width / 2.0, "bottom": bar_width / 2.0}
    group_stats: Dict[Tuple[str, str], Dict[str, float]] = {}

    for network in NETWORK_ORDER:
        sub_net = df_plot[df_plot["network"] == network]
        if sub_net.empty:
            continue
        for cond in condition_order:
            vals = sub_net.loc[sub_net["selection_type"] == cond, "parcel_mean_similarity"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            x_pos = positions[network] + offsets[cond]
            group_stats[(network, cond)] = {
                "x": float(x_pos),
                "mean": float(np.nanmean(vals)),
                "max": float(np.nanmax(vals)),
            }
            parts = ax.violinplot(
                dataset=vals,
                positions=[x_pos],
                widths=0.28,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in parts["bodies"]:
                body.set_facecolor(violin_colors[cond])
                body.set_edgecolor("#333333")
                body.set_linewidth(0.6)
                body.set_alpha(0.75)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#222222")
                parts["cmedians"].set_linewidth(0.6)

    y_all = df_plot["parcel_mean_similarity"].to_numpy(dtype=float)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1e-6) * 0.1

    # 每个小提琴顶部标注均值
    for (_, _), stat in group_stats.items():
        text_y = stat["max"] + 0.02 * y_range
        ax.text(
            stat["x"],
            text_y,
            f"{stat['mean']:.3f}",
            ha="center",
            va="bottom",
            fontsize=5,
            color="#222222",
            zorder=5,
        )

    # 显著性标注（每个 network 的 top vs bottom）
    if sig_df is not None and not sig_df.empty:
        for network in NETWORK_ORDER:
            row = sig_df.loc[sig_df["network"] == network]
            if row.empty:
                continue
            stars = _pvalue_to_stars(float(row["p_value"].iloc[0]))
            if not stars:
                continue
            key_top = (network, "top")
            key_bottom = (network, "bottom")
            if key_top not in group_stats or key_bottom not in group_stats:
                continue

            x_left = group_stats[key_top]["x"]
            x_right = group_stats[key_bottom]["x"]
            local_max = max(group_stats[key_top]["max"], group_stats[key_bottom]["max"])
            y_sig = local_max + 0.08 * y_range

            ax.plot(
                [x_left, x_left, x_right, x_right],
                [y_sig - 0.01 * y_range, y_sig, y_sig, y_sig - 0.01 * y_range],
                color="#333333",
                linewidth=0.5,
                zorder=4,
            )
            ax.text(
                (x_left + x_right) / 2.0,
                y_sig + 0.01 * y_range,
                stars,
                ha="center",
                va="bottom",
                fontsize=5,
                color="#222222",
                zorder=5,
            )

    display_names = [NETWORK_DISPLAY_NAMES.get(n, n) for n in NETWORK_ORDER]
    ax.set_xticks(np.arange(len(NETWORK_ORDER), dtype=float))
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_xlabel("Yeo7 Network", fontsize=5, fontweight="bold")
    ax.set_ylabel("Semantic similarity (parcel mean)", fontsize=5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=5, width=0.6, length=2.5)
    ax.set_ylim(y_min - 0.08 * y_range, y_max + 0.18 * y_range)
    ax.grid(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 手工图例，避免重复 label 导致图例项过多
    handles = [
        plt.Line2D([0], [0], color=violin_colors["top"], lw=3, label=f"Top-{top_k}"),
        plt.Line2D([0], [0], color=violin_colors["bottom"], lw=3, label=f"Bottom-{bottom_k}"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=5,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "小提琴图已保存: %s, %s, %s",
        output_path,
        output_path.with_suffix(".svg"),
        output_path.with_suffix(".pdf"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "基于 export_top_human_matches.py 生成的 top/bottom 匹配 CSV，"
            "按 Yeo7 network 绘制 semantic_similarity 的 Top vs Bottom 对比图。"
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="单个被试输入 CSV（兼容旧用法）。",
    )
    parser.add_argument(
        "--input-csvs",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "多个被试输入 CSV 路径（例如 uts02 与 uts03 的 top_human_parcels_per_llm.csv）；"
            "提供后将进行融合统计。"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_DIR,
        help="输出目录（默认使用全局 draw_result 根目录下的子目录）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K 的 LLM parcels 数量，仅用于图例/文件名标注（默认 10）",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=10,
        help="Bottom-K 的 LLM parcels 数量，仅用于图例/文件名标注（默认 10）",
    )
    parser.add_argument(
        "--network-top-percent",
        type=float,
        default=100.0,
        help=(
            "在每个 Yeo7 network 内，仅保留最高预测准确率位于前 top_percent%% 的 human parcels；"
            "默认 100 表示不过滤。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定则覆盖已有输出文件，否则若检测到目标文件存在会跳过。",
    )

    args = parser.parse_args()
    if args.input_csv is None and not args.input_csvs:
        raise ValueError("必须提供 --input-csv 或 --input-csvs。")
    if args.input_csv is not None and args.input_csvs:
        raise ValueError("--input-csv 与 --input-csvs 不能同时使用，请二选一。")

    ensure_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_csv_paths: List[Path]
    if args.input_csv is not None:
        input_csv_paths = [args.input_csv]
    else:
        input_csv_paths = list(args.input_csvs)

    # 支持多被试融合：分别加载并打上 subject_id，然后合并。
    all_pairs: List[pd.DataFrame] = []
    for csv_path in input_csv_paths:
        # 根据命令行传入的 top_k / bottom_k 动态选择要使用的配对数量
        df_one = load_top_bottom_pairs(csv_path, top_k=args.top_k, bottom_k=args.bottom_k)

        # 可选：在每个 network 内，仅保留「最高预测准确率处于前 top_percent%」的人脑 parcels
        if args.network_top_percent is not None and args.network_top_percent < 100.0:
            df_one = filter_parcels_by_network_top_percent(
                df_one,
                top_percent=args.network_top_percent,
            )

        df_one = df_one.copy()
        df_one["subject_id"] = infer_subject_id_from_csv_path(csv_path)
        all_pairs.append(df_one)

    if not all_pairs:
        raise ValueError("没有可用输入数据。")
    df_pairs = pd.concat(all_pairs, ignore_index=True)
    logger.info("已融合 %d 个被试输入文件，总记录数: %d", len(input_csv_paths), len(df_pairs))

    parcel_means = compute_parcel_level_means(df_pairs)
    summaries = compute_network_condition_summary(parcel_means)

    # 额外输出：融合被试后的 network 均值/方差统计表（先被试内均值，再跨被试方差）
    network_mv_df = compute_network_mean_variance_summary(parcel_means)
    save_network_mean_variance_summary(network_mv_df, args.output_dir, top_k=args.top_k, bottom_k=args.bottom_k)

    # 计算每个 network 上 Top vs Bottom 的显著性，并保存结果表
    sig_df = compute_top_bottom_significance(parcel_means)
    save_significance_table(sig_df, args.output_dir, top_k=args.top_k, bottom_k=args.bottom_k)

    # 默认输出文件名中带上 top/bottom K 以便区分不同配置
    base_name = f"semantic_similarity_by_network_top{args.top_k}_bottom{args.bottom_k}.png"
    output_png = args.output_dir / base_name

    plot_semantic_similarity_by_network(
        parcel_means=parcel_means,
        summaries=summaries,
        output_path=output_png,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        overwrite=args.overwrite,
        sig_df=sig_df,
    )
    paired_name = f"semantic_similarity_by_network_no_scatter_paired_top{args.top_k}_bottom{args.bottom_k}.png"
    plot_semantic_similarity_by_network_paired_lines(
        parcel_means=parcel_means,
        summaries=summaries,
        output_path=args.output_dir / paired_name,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        overwrite=args.overwrite,
        sig_df=sig_df,
    )

    violin_name = f"semantic_similarity_violin_by_network_top{args.top_k}_bottom{args.bottom_k}.png"
    plot_semantic_similarity_violin_by_network(
        parcel_means=parcel_means,
        output_path=args.output_dir / violin_name,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        overwrite=args.overwrite,
        sig_df=sig_df,
    )


if __name__ == "__main__":
    main()

