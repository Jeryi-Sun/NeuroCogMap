#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 feature_analysis 输出，绘制“不同认知实验(=dataset_key) 的 PCC”柱状图（Nature 风格）。

默认输出两张 PDF（不拼成大图）：
1) Step1：每个数据集在 `step1_train_feature_pcc_ranked_by_aic.csv` 中“绝对相关性最强(排序第1行)”的 train-AIC PCC
2) Step2B：`summary_metrics.csv` 里的多元线性模型 test-AIC PCC（step2B_test_pcc_aic）

用法示例：
  python plot_experiment_pcc_bars.py \
    --feature-analysis-dir /abs/path/to/results/feature_analysis \
    --out-dir /abs/path/to/results/feature_analysis/plots \
    --metric aic \
    --use-abs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

try:
    from scipy import stats as sp_stats
except Exception as e:  # noqa: BLE001
    # 显著性计算依赖 scipy，如无则直接报错提醒用户安装
    raise ImportError("需要 scipy 才能计算 PCC 的显著性，请先安装 scipy：pip install scipy") from e


PALETTE = [
    "#FAE6D7",
    "#F3C7BF",
    "#F0BBC1",
    "#F4E4B0",
    "#F7DC7C",
    "#B4DDF4",
    "#DE7D82",
    "#579FCA",
]

DATASET_KEY_TO_TASK_NAME = {
    # 来自 openloop/dataset/data_desription.md
    "badham2017deficits_exp1_csv": "Shepard categorization",
    "bahrami2020four_exp_csv": "Drifting four-armed bandit",
    "hilbig2014generalized_exp1_csv": "Multi-attribute decision-making",
    "popov2023intent_exp1_csv": "Episodic long-term memory",
    "ruggeri2022globalizability_exp1_csv": "Intertemporal choice",
}


def _setup_mpl():
    # 重新加载字体管理器以确保识别最新字体
    try:
        fm.fontManager = fm.FontManager()
    except Exception:
        pass

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def _save_bar_pdf(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_pdf: Path,
    *,
    title: str | None = None,
    ylabel: str = "PCC",
    sig_col: str | None = None,
):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_svg = out_pdf.with_suffix(".svg")

    fig_w = max(3.2, 0.22 * len(df))  # 数据集多时自动加宽
    fig_h = 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    bars = ax.bar(df[x_col].tolist(), df[y_col].tolist(), color=colors, edgecolor="black", linewidth=0.4)

    if title:
        ax.set_title(title, fontsize=7, fontweight="bold")

    ax.set_xlabel("Experiment", fontsize=8, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6, rotation=45, labelrotation=45)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # 在柱顶标注数值（pcc）与显著性星号（只对 p<0.05 的情况）
    value_offset_ratio = 0.02
    star_offset_ratio = 0.06

    sig_values = df[sig_col].tolist() if (sig_col is not None and sig_col in df.columns) else [None] * len(df)

    y_min = None
    y_max = None

    for bar, sig in zip(bars, sig_values):
        height = float(bar.get_height())
        x_center = bar.get_x() + bar.get_width() / 2.0

        # 相关性数值（保留 3 位小数；abs 图会是非负）
        value_text = f"{height:.3f}"
        if height >= 0:
            value_y = height * (1.0 + value_offset_ratio)
            value_va = "bottom"
        else:
            value_y = height * (1.0 + value_offset_ratio)
            value_va = "top"
        ax.text(
            x_center,
            value_y,
            value_text,
            ha="center",
            va=value_va,
            fontsize=6,
            fontweight="bold",
        )
        y_min = value_y if y_min is None else min(y_min, value_y)
        y_max = value_y if y_max is None else max(y_max, value_y)

        # 显著性星号
        if sig in {"*", "**", "***"}:
            if height >= 0:
                star_y = height * (1.0 + star_offset_ratio)
                star_va = "bottom"
            else:
                star_y = height * (1.0 + star_offset_ratio)
                star_va = "top"
            ax.text(
                x_center,
                star_y,
                sig,
                ha="center",
                va=star_va,
                fontsize=7,
                fontweight="bold",
            )
            y_min = star_y if y_min is None else min(y_min, star_y)
            y_max = star_y if y_max is None else max(y_max, star_y)

    # 防止柱顶/星号/数值裁切
    cur_bottom, cur_top = ax.get_ylim()
    new_bottom = cur_bottom
    new_top = cur_top
    if y_min is not None and y_min < 0:
        new_bottom = min(cur_bottom, y_min * 1.08)
    if y_max is not None and y_max > 0:
        new_top = max(cur_top, y_max * 1.08)
    ax.set_ylim(new_bottom, new_top)

    # 让标签不被裁切
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=450, bbox_inches="tight", transparent=True)
    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _add_p_values(df: pd.DataFrame, pcc_col: str, n_col: str) -> pd.DataFrame:
    """
    基于 PCC 与样本量 n 计算 p 值与显著性星号。

    t = r * sqrt((n-2)/(1-r^2)), df = n-2, p = 2 * (1 - CDF_t(|t|))
    """
    from math import sqrt

    rs = df[pcc_col].astype(float).to_numpy()
    ns = df[n_col].astype(int).to_numpy()

    p_vals = []
    t_vals = []
    stars = []
    for r, n in zip(rs, ns):
        if n <= 2:
            p_vals.append(float("nan"))
            t_vals.append(float("nan"))
            stars.append("n/a")
            continue
        if abs(r) >= 1.0:
            # 完全相关，理论上 p -> 0
            p_vals.append(0.0)
            # t 理论上 -> inf，这里用一个很大的数表示
            t_vals.append(float("inf"))
            stars.append("***")
            continue
        t = r * sqrt((n - 2.0) / max(1e-12, 1.0 - r * r))
        dfree = max(1, n - 2)
        p = 2.0 * sp_stats.t.sf(abs(t), df=dfree)
        p_vals.append(float(p))
        t_vals.append(float(t))
        if p < 0.001:
            s = "***"
        elif p < 0.01:
            s = "**"
        elif p < 0.05:
            s = "*"
        else:
            s = "ns"
        stars.append(s)

    df = df.copy()
    df["p_value"] = p_vals
    df["t_value"] = t_vals
    df["significance"] = stars
    return df


def _load_summary_metrics(feature_analysis_dir: Path) -> pd.DataFrame:
    p = feature_analysis_dir / "summary_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 summary_metrics.csv：{p}")
    return pd.read_csv(p)


def _extract_step1_top_pcc(feature_analysis_dir: Path, dataset_key: str, metric: str, use_abs: bool) -> float:
    p = feature_analysis_dir / dataset_key / f"step1_train_feature_pcc_ranked_by_{metric}.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 Step1(train) 文件：{p}")
    df = pd.read_csv(p)
    col = f"pcc_with_train_{metric}"
    abs_col = f"abs_pcc_with_train_{metric}"
    if col not in df.columns or abs_col not in df.columns:
        raise KeyError(f"Step1(train) 列缺失：需要 {col} 与 {abs_col}，实际列={list(df.columns)}（文件：{p}）")
    top_row = df.iloc[0]
    return float(top_row[abs_col] if use_abs else top_row[col])


def _extract_step2a_top_pcc(feature_analysis_dir: Path, dataset_key: str, metric: str, use_abs: bool) -> float:
    """
    Step2A：单特征一元线性回归（train 拟合），test 上预测值与真实值的 PCC。
    读取 ranked_by_{metric}.csv（已按 abs_test_pred_pcc_* 降序），取第 1 行作为“该实验最强单特征预测能力”。
    """
    p = feature_analysis_dir / dataset_key / f"step2A_univariate_train_test_ranked_by_{metric}.csv"
    if not p.exists():
        raise FileNotFoundError(f"找不到 Step2A 文件：{p}")
    df = pd.read_csv(p)
    col = f"test_pred_pcc_with_{metric}"
    abs_col = f"abs_test_pred_pcc_with_{metric}"
    if col not in df.columns or abs_col not in df.columns:
        raise KeyError(f"Step2A 列缺失：需要 {col} 与 {abs_col}，实际列={list(df.columns)}（文件：{p}）")
    top_row = df.iloc[0]
    return float(top_row[abs_col] if use_abs else top_row[col])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-analysis-dir", type=str, required=True, help="results/feature_analysis 目录（绝对路径）")
    parser.add_argument("--out-dir", type=str, required=True, help="输出目录（绝对路径）")
    parser.add_argument("--metric", type=str, default="aic", choices=["aic", "nll"], help="选择 aic 或 nll")
    parser.add_argument("--use-abs", action="store_true", help="使用绝对值 PCC（更适合比较强弱）")
    parser.add_argument("--skip-existing", action="store_true", help="若输出 PDF 已存在则跳过")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="pretty",
        choices=["dataset_key", "pretty"],
        help="x 轴显示名：dataset_key 原样 / pretty（优先替换为 Task 名；找不到则回退去掉 _csv 并替换下划线）",
    )
    args = parser.parse_args()

    feature_analysis_dir = Path(args.feature_analysis_dir)
    out_dir = Path(args.out_dir)
    metric = args.metric

    _setup_mpl()

    summary = _load_summary_metrics(feature_analysis_dir)
    if "dataset_key" not in summary.columns:
        raise KeyError(f"summary_metrics.csv 缺少 dataset_key 列：{list(summary.columns)}")

    dataset_keys = [
        k for k in summary["dataset_key"].astype(str).tolist() if k in DATASET_KEY_TO_TASK_NAME
    ]
    if len(dataset_keys) == 0:
        raise ValueError("summary_metrics.csv 中没有任何 dataset_key")

    def _pretty_label(k: str) -> str:
        s = k
        if s.endswith("_csv"):
            s = s[: -len("_csv")]
        s = s.replace("_", " ")
        return s

    def _task_or_pretty_label(k: str) -> str:
        # 优先显示 data_desription.md 里定义的具体任务名；找不到时再退回原始 pretty 逻辑
        if k in DATASET_KEY_TO_TASK_NAME:
            return DATASET_KEY_TO_TASK_NAME[k]
        return _pretty_label(k)

    if args.label_mode == "dataset_key":
        labels = dataset_keys
    else:
        labels = [_task_or_pretty_label(k) for k in dataset_keys]

    # --- Step1: top feature pcc (rank 1 row in ranked-by-metric csv) ---
    step1_vals = []
    for k in dataset_keys:
        step1_vals.append(_extract_step1_top_pcc(feature_analysis_dir, k, metric=metric, use_abs=args.use_abs))
    df_step1 = pd.DataFrame(
        {
            "dataset_key": dataset_keys,
            "experiment": labels,
            "pcc": step1_vals,
        }
    )

    # Step1 的样本量：summary_metrics 里的 num_train_participants
    if "num_train_participants" not in summary.columns:
        raise KeyError(f"summary_metrics.csv 缺少 num_train_participants 列：{list(summary.columns)}")
    n_map = summary.set_index("dataset_key")["num_train_participants"].to_dict()
    df_step1["num_train_participants"] = df_step1["dataset_key"].map(n_map)
    df_step1 = _add_p_values(df_step1, pcc_col="pcc", n_col="num_train_participants")
    df_step1 = df_step1.sort_values("pcc", ascending=False)

    step1_out = out_dir / f"bar_step1_top_feature_train_{metric}_pcc{'_abs' if args.use_abs else ''}.pdf"
    step1_svg_out = step1_out.with_suffix(".svg")
    step1_table_out = out_dir / f"bar_step1_top_feature_train_{metric}_pcc{'_abs' if args.use_abs else ''}_with_pvals.csv"
    step1_both_exist = step1_out.exists() and step1_svg_out.exists()
    if not (args.skip_existing and step1_both_exist):
        _save_bar_pdf(
            df_step1,
            x_col="experiment",
            y_col="pcc",
            out_pdf=step1_out,
            title=None,
            ylabel="Pearson Correlation Coefficient",
            sig_col="significance",
        )
    # 始终更新统计表（包括 p 值）
    df_step1.to_csv(step1_table_out, index=False)

    # --- Step2B: overall multivariate test pcc from summary_metrics.csv ---
    step2b_col = f"step2B_test_pcc_{metric}"
    if step2b_col not in summary.columns:
        raise KeyError(f"summary_metrics.csv 缺少 {step2b_col} 列：{list(summary.columns)}")

    df_step2b = summary[["dataset_key", step2b_col, "num_test_participants"]].copy()
    df_step2b = df_step2b[df_step2b["dataset_key"].astype(str).isin(dataset_keys)].copy()
    df_step2b.rename(columns={step2b_col: "pcc"}, inplace=True)
    df_step2b["experiment"] = df_step2b["dataset_key"].map(
        {k: lab for k, lab in zip(dataset_keys, labels)}
    )
    df_step2b["pcc"] = pd.to_numeric(df_step2b["pcc"], errors="raise")
    if args.use_abs:
        df_step2b["pcc"] = df_step2b["pcc"].abs()
    df_step2b = _add_p_values(df_step2b, pcc_col="pcc", n_col="num_test_participants")
    df_step2b = df_step2b.sort_values("pcc", ascending=False)

    step2b_out = out_dir / f"bar_step2B_multivariate_test_{metric}_pcc{'_abs' if args.use_abs else ''}.pdf"
    step2b_svg_out = step2b_out.with_suffix(".svg")
    step2b_table_out = out_dir / f"bar_step2B_multivariate_test_{metric}_pcc{'_abs' if args.use_abs else ''}_with_pvals.csv"
    step2b_both_exist = step2b_out.exists() and step2b_svg_out.exists()
    if not (args.skip_existing and step2b_both_exist):
        _save_bar_pdf(
            df_step2b,
            x_col="experiment",
            y_col="pcc",
            out_pdf=step2b_out,
            title=f"Step2B: Multivariate |test {metric.upper()} PCC| across tasks" if args.use_abs else f"Step2B: Multivariate test {metric.upper()} PCC across tasks",
            ylabel="|PCC|" if args.use_abs else "PCC",
            sig_col="significance",
        )
    df_step2b.to_csv(step2b_table_out, index=False)

    # --- Step2A: top single-feature predictive pcc (rank 1 row in ranked-by-metric csv) ---
    step2a_vals = []
    for k in dataset_keys:
        step2a_vals.append(_extract_step2a_top_pcc(feature_analysis_dir, k, metric=metric, use_abs=args.use_abs))

    df_step2a = pd.DataFrame(
        {
            "dataset_key": dataset_keys,
            "experiment": labels,
            "pcc": step2a_vals,
        }
    )
    df_step2a["num_test_participants"] = df_step2a["dataset_key"].map(n_map)
    df_step2a = _add_p_values(df_step2a, pcc_col="pcc", n_col="num_test_participants")
    df_step2a = df_step2a.sort_values("pcc", ascending=False)

    step2a_out = out_dir / f"bar_step2A_top_feature_test_{metric}_pcc{'_abs' if args.use_abs else ''}.pdf"
    step2a_svg_out = step2a_out.with_suffix(".svg")
    step2a_table_out = out_dir / f"bar_step2A_top_feature_test_{metric}_pcc{'_abs' if args.use_abs else ''}_with_pvals.csv"

    step2a_both_exist = step2a_out.exists() and step2a_svg_out.exists()
    if not (args.skip_existing and step2a_both_exist):
        _save_bar_pdf(
            df_step2a,
            x_col="experiment",
            y_col="pcc",
            out_pdf=step2a_out,
            title=f"Step2A: Top-1 |test {metric.upper()} PCC| across tasks" if args.use_abs else f"Step2A: Top-1 test {metric.upper()} PCC across tasks",
            ylabel="|PCC|" if args.use_abs else "PCC",
            sig_col="significance",
        )
    df_step2a.to_csv(step2a_table_out, index=False)

    print(
        "[OK] Wrote:\n"
        f"- {step1_out}\n"
        f"- {step1_table_out}\n"
        f"- {step2a_out}\n"
        f"- {step2a_table_out}\n"
        f"- {step2b_out}\n"
        f"- {step2b_table_out}"
    )


if __name__ == "__main__":
    main()

