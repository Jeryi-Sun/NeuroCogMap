#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 7Networks 绘制预测准确度图：
1) 柱状图（network 均值）+ parcel 散点 + 误差线（SEM）+ 趋势折线
2) 小提琴图（parcel top-k 均值分布）+ 均值误差线 + 趋势折线
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from common import DATA_DIR, RESULT_DIR, ensure_output_dir, should_skip

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NETWORK_DISPLAY_NAMES = {
    "Vis": "Visual",
    "SomMot": "Somatomotor",
    "DorsAttn": "DorsalAttn",
    "SalVentAttn": "Salience",
    "Limbic": "Limbic",
    "Cont": "Control",
    "Default": "Default",
}

NATURE_PALETTE = [
    "#FAE6D7",
    "#F3C7BF",
    "#F0BBC1",
    "#F4E4B0",
    "#F7DC7C",
    "#B4DDF4",
    "#579FCA",
]
TREND_LINE_COLOR = "#0E7D82"


def configure_nature_style_small_figure() -> None:
    """设置 1/4 A4 小图的 Nature 风格全局参数。"""
    try:
        fm.fontManager = fm.FontManager()
    except Exception as exc:
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


def darken_color(hex_color: str, factor: float = 0.7) -> Tuple[float, float, float]:
    """将颜色加深，用于散点与柱子的同色系区分。"""
    rgb = np.array(to_rgb(hex_color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def extract_network_from_parcel_name(parcel_name: str) -> Optional[str]:
    parts = parcel_name.split("_")
    if len(parts) >= 3:
        return parts[2]
    return None


def extract_hemisphere_from_parcel_name(parcel_name: str) -> Optional[str]:
    parts = parcel_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def load_parcel_descriptions(parcel_desc_path: Path) -> Dict[int, Dict]:
    logger.info("加载 parcel 描述文件: %s", parcel_desc_path)
    if not parcel_desc_path.exists():
        raise FileNotFoundError(f"找不到 parcel 描述文件: {parcel_desc_path}")

    with parcel_desc_path.open("r", encoding="utf-8") as f:
        parcel_descriptions = json.load(f)

    parcel_dict = {}
    for parcel in parcel_descriptions:
        parcel_id = parcel.get("parcel_id")
        if parcel_id is not None:
            parcel_dict[parcel_id] = parcel

    logger.info("加载了 %d 个 parcel 描述", len(parcel_dict))
    return parcel_dict


def load_prediction_matrix(csv_path: Path) -> pd.DataFrame:
    logger.info("加载预测矩阵: %s", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到预测矩阵文件: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)
    logger.info("预测矩阵形状: %s", df.shape)
    return df


def compute_parcel_topk_mean(df: pd.DataFrame, parcel_dict: Dict[int, Dict], top_k: int) -> pd.DataFrame:
    """对每个 parcel 计算其 top-k 准确度均值。"""
    rows = []
    for idx in df.index:
        try:
            parcel_id = int(str(idx).split("_")[-1])
        except (ValueError, IndexError):
            logger.warning("无法从索引提取 parcel_id: %s", idx)
            continue

        parcel_info = parcel_dict.get(parcel_id)
        if parcel_info is None:
            logger.warning("未找到 parcel_id=%s 的描述信息", parcel_id)
            continue

        parcel_name = parcel_info.get("parcel_name", "")
        network = extract_network_from_parcel_name(parcel_name)
        if network is None:
            logger.warning("无法解析 network，parcel_name=%s", parcel_name)
            continue
        if network not in NETWORK_ORDER:
            logger.warning("network=%s 不在预定义顺序中，parcel_name=%s", network, parcel_name)
            continue

        row_values = df.loc[idx].to_numpy(dtype=float)
        sorted_values = np.sort(row_values)[::-1]
        current_top_k = min(top_k, sorted_values.size)
        if current_top_k <= 0:
            raise ValueError(f"top_k 无效: {top_k}")

        parcel_topk_mean = float(np.mean(sorted_values[:current_top_k]))
        rows.append(
            {
                "parcel_id": parcel_id,
                "parcel_name": parcel_name,
                "network": network,
                "hemisphere": extract_hemisphere_from_parcel_name(parcel_name),
                "parcel_topk_mean": parcel_topk_mean,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise ValueError("未计算出任何 parcel 统计，请检查输入数据和 parcel 描述映射。")
    logger.info("计算完成：%d 个 parcel 的 top-%d 均值", len(result_df), top_k)
    return result_df


def compute_network_summary(parcel_df: pd.DataFrame) -> pd.DataFrame:
    """按 network 聚合 parcel_topk_mean，计算均值与 SEM。"""
    summary = (
        parcel_df.groupby("network")["parcel_topk_mean"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "network_mean", "std": "network_std", "count": "n"})
    )
    summary["sem"] = summary["network_std"] / np.sqrt(summary["n"])
    summary["sem"] = summary["sem"].fillna(0.0)
    summary = summary.reindex([n for n in NETWORK_ORDER if n in summary.index])
    return summary.reset_index()


def plot_network_bar_with_parcel_scatter(
    parcel_df: pd.DataFrame,
    network_summary: pd.DataFrame,
    output_path: Path,
    top_k: int,
    overwrite: bool,
) -> None:
    """绘制柱状图 + 散点 + 误差线（SEM）。"""
    if should_skip(output_path, overwrite):
        return

    configure_nature_style_small_figure()
    fig, ax = plt.subplots(figsize=(1.8, 1.4))

    networks = network_summary["network"].tolist()
    x = np.arange(len(networks), dtype=float)
    means = network_summary["network_mean"].to_numpy(dtype=float)
    sems = network_summary["sem"].to_numpy(dtype=float)

    bar_colors = [NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(networks))]
    scatter_colors = [darken_color(c, factor=0.68) for c in bar_colors]

    ax.bar(
        x,
        means,
        yerr=sems,
        width=0.62,
        color=bar_colors,
        edgecolor="#333333",
        linewidth=0.6,
        error_kw={"elinewidth": 0.6, "ecolor": "#333333", "capsize": 1.8, "capthick": 0.6},
        zorder=2,
    )

    # 在柱状图上叠加更清晰的 network 均值趋势折线。
    ax.plot(
        x,
        means,
        color=TREND_LINE_COLOR,
        linewidth=1.2,
        alpha=0.55,
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgewidth=0.6,
        markeredgecolor=TREND_LINE_COLOR,
        zorder=4,
    )

    rng = np.random.default_rng(42)
    for i, network in enumerate(networks):
        vals = parcel_df.loc[parcel_df["network"] == network, "parcel_topk_mean"].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(
            np.full(vals.size, x[i]) + jitter,
            vals,
            s=5,
            color=scatter_colors[i],
            edgecolors="#222222",
            linewidths=0.3,
            alpha=0.55,
            zorder=3,
        )

    display_names = [NETWORK_DISPLAY_NAMES.get(n, n) for n in networks]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_xlabel("Yeo7 Network", fontsize=5, fontweight="bold")
    ax.set_ylabel(f"Parcel Top-{top_k} Mean Accuracy", fontsize=5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=5, width=0.6, length=2.5)

    y_max_candidates = [np.max(means + sems), parcel_df["parcel_topk_mean"].max()]
    y_min_candidates = [np.min(means - sems), parcel_df["parcel_topk_mean"].min()]
    y_max = max(y_max_candidates)
    y_min = min(y_min_candidates)
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1e-6) * 0.1
    label_offset = 0.03 * y_range
    for i, mean_val in enumerate(means):
        ax.text(
            x[i],
            mean_val + sems[i] + label_offset,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=5,
            color="#222222",
        )
    ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.12 * y_range)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info("图片已保存: %s, %s", output_path, output_path.with_suffix(".svg"))


def plot_network_violin(
    parcel_df: pd.DataFrame,
    network_summary: pd.DataFrame,
    output_path: Path,
    top_k: int,
    overwrite: bool,
) -> None:
    """绘制 network 小提琴图 + 均值误差线 + 趋势折线。"""
    if should_skip(output_path, overwrite):
        return

    configure_nature_style_small_figure()
    fig, ax = plt.subplots(figsize=(1.8, 1.4))

    networks = network_summary["network"].tolist()
    x = np.arange(len(networks), dtype=float)
    means = network_summary["network_mean"].to_numpy(dtype=float)
    sems = network_summary["sem"].to_numpy(dtype=float)

    distribution_data = [
        parcel_df.loc[parcel_df["network"] == network, "parcel_topk_mean"].to_numpy(dtype=float)
        for network in networks
    ]
    violin_colors = [NATURE_PALETTE[i % len(NATURE_PALETTE)] for i in range(len(networks))]

    violin_parts = ax.violinplot(
        distribution_data,
        positions=x,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for idx, body in enumerate(violin_parts["bodies"]):
        body.set_facecolor(violin_colors[idx])
        body.set_edgecolor("#333333")
        body.set_alpha(0.78)
        body.set_linewidth(0.6)

    ax.errorbar(
        x,
        means,
        yerr=sems,
        fmt="o",
        color="#333333",
        ecolor="#333333",
        elinewidth=0.6,
        capsize=1.8,
        markersize=2.6,
        markerfacecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )
    ax.plot(
        x,
        means,
        color=TREND_LINE_COLOR,
        linewidth=1.1,
        alpha=0.55,
        marker="o",
        markersize=2.4,
        markerfacecolor="white",
        markeredgecolor=TREND_LINE_COLOR,
        markeredgewidth=0.5,
        zorder=4,
    )

    display_names = [NETWORK_DISPLAY_NAMES.get(n, n) for n in networks]
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=30, ha="right")
    ax.set_xlabel("Yeo7 Network", fontsize=5, fontweight="bold")
    ax.set_ylabel(f"Parcel Top-{top_k} Mean Accuracy", fontsize=5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=5, width=0.6, length=2.5)

    y_max = parcel_df["parcel_topk_mean"].max()
    y_min = parcel_df["parcel_topk_mean"].min()
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1e-6) * 0.1
    label_offset = 0.03 * y_range
    for i, mean_val in enumerate(means):
        violin_top = float(np.max(distribution_data[i])) if distribution_data[i].size > 0 else mean_val
        ax.text(
            x[i],
            violin_top + label_offset,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=5,
            color="#222222",
        )
    ax.set_ylim(y_min - 0.14 * y_range, y_max + 0.14 * y_range)

    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=450, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), dpi=450, bbox_inches="tight")
    plt.close(fig)
    logger.info("图片已保存: %s, %s", output_path, output_path.with_suffix(".svg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="按 7Networks 绘制 top-k 准确度柱状图+散点图（Nature 风格）")
    parser.add_argument(
        "--prediction-matrix",
        type=Path,
        default=DATA_DIR / "wheretheressmoke" / "prediction_matrix_gemma2_2b.csv",
        help="预测矩阵 CSV 文件路径",
    )
    parser.add_argument(
        "--parcel-descriptions",
        type=Path,
        default=Path(
            "/path/to/project_root/Human_LLM_align/"
            "litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
        ),
        help="parcel 描述 JSON 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULT_DIR / "wheretheressmoke",
        help="输出目录",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K 的 LLM Parcels 数量（默认 20）",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有文件")
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError(f"--top-k 必须是正整数，当前为 {args.top_k}")

    ensure_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_matrix = load_prediction_matrix(args.prediction_matrix)
    parcel_dict = load_parcel_descriptions(args.parcel_descriptions)

    parcel_df = compute_parcel_topk_mean(prediction_matrix, parcel_dict, top_k=args.top_k)
    network_summary = compute_network_summary(parcel_df)
    if network_summary.empty:
        raise ValueError("network 汇总结果为空，无法绘图。")

    # 清理旧版脚本输出的额外图片，确保目录中只保留当前目标图。
    legacy_images = [
        args.output_dir / "prediction_accuracy_bubble_plot.png",
        args.output_dir / "mean_accuracy_by_network_lineplot.png",
    ]
    for legacy_image in legacy_images:
        if legacy_image.exists():
            legacy_image.unlink()
            logger.info("已删除旧版图片: %s", legacy_image)

    output_png = args.output_dir / f"prediction_accuracy_bubble_plot_top_{args.top_k}.png"
    plot_network_bar_with_parcel_scatter(
        parcel_df=parcel_df,
        network_summary=network_summary,
        output_path=output_png,
        top_k=args.top_k,
        overwrite=args.overwrite,
    )
    violin_png = args.output_dir / f"prediction_accuracy_violin_plot_top_{args.top_k}.png"
    plot_network_violin(
        parcel_df=parcel_df,
        network_summary=network_summary,
        output_path=violin_png,
        top_k=args.top_k,
        overwrite=args.overwrite,
    )

    logger.info("分析完成，输出图片: %s, %s", output_png, violin_png)


if __name__ == "__main__":
    main()

