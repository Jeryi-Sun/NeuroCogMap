#!/usr/bin/env python3
"""
基于 neural/analysis_results/result/panel_a 下的 *_summary.json：
- 绘制 Nature 风格横向柱状图（均值 + 95% bootstrap CI）
- 计算与 reference 方法的配对显著性（paired t-test），并做 BH-FDR 校正
- 导出 PDF/SVG/PNG 以及统计表 TSV
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# 重新加载字体管理器以确保识别最新字体（用户规则）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass


A4_WIDTH_IN = 210.0 / 25.4
HALF_A4_WIDTH_IN = A4_WIDTH_IN / 2.0
FIGSIZE = (HALF_A4_WIDTH_IN, 2.55)

# 按用户要求：统一显示名与配色（Nature palette）
# 顺序也按该列表固定（不按均值排序）
DISPLAY_LABELS = {
    "embeddings_general_roi": "Word2Vec",
    "bert_model_layer=12_roi": "BERT",
    "language_model_attention_layer=12_roi": "Language_Context",
    "language_model_layer=12_roi": "Language_Standard",
    "saeact_model_all_parcels_n=270_roi": "NeuroCogMap",
}

METHOD_COLORS = {
    "Word2Vec": "#B4DDF4",
    "BERT": "#579FCA",
    "Language_Context": "#F4E4B0",
    "Language_Standard": "#F7DC7C",
    "NeuroCogMap": "#DE7D82",
}

PLOT_ORDER = ["Word2Vec", "BERT", "Language_Context", "Language_Standard", "NeuroCogMap"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel-a-dir",
        type=Path,
        default=Path(
            "/path/to/project_root/"
            "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/analysis_results/result/panel_a"
        ),
        help="panel_a 结果目录（包含 *_summary.json）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/path/to/project_root/"
            "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/analysis_results/result/panel_a"
        ),
        help="输出目录（默认写回 panel_a 目录）",
    )
    p.add_argument(
        "--output-stem",
        type=str,
        default="panel_a_method_comparison",
        help="输出文件名 stem",
    )
    p.add_argument(
        "--reference-model",
        type=str,
        default="saeact_model_all_parcels_n=270_roi",
        help="reference 方法名（用于配对检验与 q 值标注）",
    )
    p.add_argument("--dpi", type=int, default=450, help="PNG 分辨率（默认 450）")
    return p.parse_args()


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 5,
            "axes.titlesize": 5,
            "axes.labelsize": 5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def fisher_z(r: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    r = np.clip(r, -1.0 + eps, 1.0 - eps)
    return np.arctanh(r)


def fisher_mean(r: np.ndarray) -> float:
    z = fisher_z(r)
    return float(np.tanh(np.mean(z)))


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR for 1D array; returns q-values (same shape)."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    pv = p[finite]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    n = ranked.size
    adjusted = np.empty_like(ranked)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative = min(cumulative, value)
        adjusted[i] = cumulative
    qv = np.clip(adjusted, 0.0, 1.0)
    q[finite] = qv[np.argsort(order)]
    return q


def format_q(q: float, is_ref: bool) -> str:
    if is_ref:
        return "reference"
    if not np.isfinite(q):
        return "BH q = NA"
    if q < 0.001:
        return "BH q < 0.001"
    return f"BH q = {q:.3f}"


def method_color(name: str) -> str:
    # name 是 display label（如 Word2Vec/BERT/...）
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    palette = ["#FAE6D7", "#F3C7BF", "#F0BBC1", "#F4E4B0", "#F7DC7C", "#B4DDF4", "#579FCA"]
    return palette[hash(name) % len(palette)]


def display_label(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)


def load_panel_a_summaries(panel_a_dir: Path) -> list[dict]:
    summaries = []
    for path in sorted(panel_a_dir.glob("*_summary.json")):
        with path.open("r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    if not summaries:
        raise FileNotFoundError(f"在 {panel_a_dir} 下未找到 *_summary.json")
    return summaries


def extract_scores_by_pid(summary: dict) -> dict[str, float]:
    by_id: dict[str, float] = {}
    for p in summary.get("participants", []):
        pid = p.get("participant_id")
        if pid is None:
            continue
        by_id[str(pid)] = float(p["mean_score"])
    return by_id


def bootstrap_ci_fisher_mean(r: np.ndarray, n_bootstrap: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = np.asarray(r, dtype=float)
    n = r.size
    if n == 0:
        raise ValueError("bootstrap 输入为空")
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    samples = r[idx]
    # Fisher 聚合后回变换
    z = fisher_z(samples)
    means = np.tanh(np.mean(z, axis=1))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def build_dataframe(summaries: list[dict], reference_model: str) -> pd.DataFrame:
    rows = []
    models = []
    by_model_scores: dict[str, dict[str, float]] = {}
    meta = {}

    for s in summaries:
        model = str(s.get("model_name") or s.get("model") or "unknown")
        models.append(model)
        by_model_scores[model] = extract_scores_by_pid(s)
        meta[model] = {
            "n_bootstrap": s.get("n_bootstrap"),
            "bootstrap_seed": s.get("bootstrap_seed"),
        }

    if reference_model not in set(models):
        raise ValueError(f"reference_model='{reference_model}' 不在 panel_a summaries 中。可选: {sorted(set(models))}")

    # 取所有方法之间的共同 participant_id，保证配对检验严格可比
    pid_sets = [set(d.keys()) for d in by_model_scores.values()]
    common_pids = set.intersection(*pid_sets) if pid_sets else set()
    if not common_pids:
        raise ValueError("不同方法之间没有共同的 participant_id，无法做配对统计")

    common_pids = sorted(common_pids, key=lambda x: int(x) if x.isdigit() else x)

    ref = np.asarray([by_model_scores[reference_model][pid] for pid in common_pids], dtype=float)

    for model in sorted(set(models)):
        scores = np.asarray([by_model_scores[model][pid] for pid in common_pids], dtype=float)
        mean_raw = float(np.mean(scores))
        std_raw = float(np.std(scores, ddof=1)) if scores.size >= 2 else float("nan")
        mean_f = fisher_mean(scores)
        ci_low, ci_high = bootstrap_ci_fisher_mean(scores, n_bootstrap=5000, seed=42)
        ci_half = (ci_high - ci_low) / 2.0

        if model == reference_model:
            p = float("nan")
        else:
            # 配对 t-test：用 Fisher Z 空间更接近正态
            p = float(stats.ttest_rel(fisher_z(scores), fisher_z(ref)).pvalue)

        rows.append(
            {
                "model": model,
                "n_participants": int(scores.size),
                "corr_mean_raw": mean_raw,
                "corr_std_raw": std_raw,
                "corr_mean_fisher": mean_f,
                "ci95_low_fisher": ci_low,
                "ci95_high_fisher": ci_high,
                "ci95_halfwidth_fisher": ci_half,
                "p_vs_reference": p,
                "color": method_color(model),
            }
        )

    df = pd.DataFrame(rows)
    df["q_vs_reference"] = bh_fdr(df["p_vs_reference"].to_numpy(dtype=float))
    df["significance_label"] = [
        format_q(q, is_ref=(m == reference_model))
        for m, q in zip(df["model"].tolist(), df["q_vs_reference"].tolist())
    ]
    df["display_model"] = [display_label(m) for m in df["model"].tolist()]
    df["color"] = [method_color(x) for x in df["display_model"].tolist()]

    # 固定顺序：Word2Vec, BERT, Language_Context, Language_Standard, NeuroCogMap
    order_map = {name: i for i, name in enumerate(PLOT_ORDER)}
    df["plot_order"] = [order_map.get(x, 10**9) for x in df["display_model"].tolist()]
    df = df.sort_values(["plot_order", "corr_mean_fisher"], ascending=[True, True]).reset_index(drop=True)
    return df


def save_stats_table(df: pd.DataFrame, out_tsv: Path) -> None:
    export = df[
        [
            "model",
            "n_participants",
            "corr_mean_fisher",
            "ci95_low_fisher",
            "ci95_high_fisher",
            "corr_mean_raw",
            "corr_std_raw",
            "p_vs_reference",
            "q_vs_reference",
        ]
    ].copy()
    export.to_csv(out_tsv, sep="\t", index=False, float_format="%.6f")


def plot_bar(df: pd.DataFrame, out_pdf: Path, out_svg: Path, out_png: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
    ax.set_facecolor("white")

    y = np.arange(len(df))
    means = df["corr_mean_fisher"].to_numpy(dtype=float)
    ci = df["ci95_halfwidth_fisher"].to_numpy(dtype=float)
    colors = df["color"].tolist()

    ax.barh(
        y,
        means,
        xerr=ci,
        height=0.68,
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=0.6,
        error_kw={"elinewidth": 0.8, "ecolor": "#4A4A4A", "capsize": 2.0, "capthick": 0.8},
        zorder=3,
    )

    xmax = max(0.65, float(np.nanmax(means + ci)) + 0.25)
    ax.set_xlim(0.0, xmax)
    ax.set_xlabel("Mean participant-wise correlation (r)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["display_model"].tolist())
    ax.invert_yaxis()

    # 细虚线网格（可读性）
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2.5, width=0.7, labelsize=5)

    # 文本标注：均值与 q
    mean_label_x = means + ci + 0.012
    q_label_x = min(xmax - 0.02, float(np.nanmax(means + ci)) + 0.10)
    for i, row in df.iterrows():
        ax.text(
            mean_label_x[i],
            y[i],
            f"r = {row['corr_mean_fisher']:.3f}",
            va="center",
            ha="left",
            fontsize=5,
            color="#222222",
        )
        ax.text(
            q_label_x,
            y[i],
            str(row["significance_label"]),
            va="center",
            ha="left",
            fontsize=5,
            color="#222222",
        )

    fig.text(
        0.02,
        0.985,
        "Bars: Fisher-Z mean across participants; error bars: 95% bootstrap CI; labels: BH-FDR vs reference",
        ha="left",
        va="top",
        fontsize=5,
        color="#444444",
    )
    # 正常绘制：仅用固定边距，不使用 bbox_inches='tight' / pad_inches
    fig.subplots_adjust(left=0.22, right=0.99, top=0.93, bottom=0.18)
    fig.savefig(out_pdf, facecolor="white")
    fig.savefig(out_svg, facecolor="white")
    fig.savefig(out_png, dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_matplotlib()

    panel_a_dir = args.panel_a_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = load_panel_a_summaries(panel_a_dir)
    df = build_dataframe(summaries, reference_model=args.reference_model)

    out_pdf = out_dir / f"{args.output_stem}.pdf"
    out_svg = out_dir / f"{args.output_stem}.svg"
    out_png = out_dir / f"{args.output_stem}.png"
    out_tsv = out_dir / f"{args.output_stem}_stats.tsv"

    save_stats_table(df, out_tsv)
    plot_bar(df, out_pdf, out_svg, out_png, dpi=args.dpi)

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_tsv}")


if __name__ == "__main__":
    main()

