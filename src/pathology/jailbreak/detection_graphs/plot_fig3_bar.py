"""
Jailbreak 数据常规柱状图（Nature 风格）。
使用 AdvBench 与 JBB-Behaviors 数据集的检测结果。
"""
import argparse
import ast
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 重新加载字体管理器以确保识别最新字体
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass


CONFIG = {
    "rcparams": {
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": 5.0,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 450,
        "savefig.dpi": 450,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    },
    "bar": {
        "figsize": (2.6, 1.8),
        "xlabel_size": 5.0,
        "ylabel_size": 5.0,
        "tick_size": 5.0,
        "legend_size": 5.0,
        "bar_cluster_width": 0.82,
        "bar_edge_width": 0.45,
        "bar_edge_width_ours": 0.6,
        "error_line_width": 0.55,
        "capsize": 2.4,
        "ylim": (0.0, 1.0),
        "x_margin": 0.06,
        "legend_ncol": 2,
        "bar_width_factor": 0.65,
        "scatter_size": 8.0,
        "scatter_alpha": 0.6,
        "scatter_jitter": 0.15,
        "value_label_size": 5.0,
        "value_label_offset": 0.02,
        "method_colors": [
            "#579FCA",
            "#F7DC7C",
            "#F0BBC1",
            "#DE7D82",
            "#B4DDF4",
            "#FAE6D7",
            "#F4E4B0",
            "#F3C7BF",
        ],
    },
}

BASE_DIR = Path("/path/to/project_root/safety_explanation/jailbreak")
CSV_PATH = BASE_DIR / "results/detection/all_results/all_metrics_plot_auroc.csv"
OUT_DIR = BASE_DIR / "detection_graphs/output"
DEFAULT_METRIC = "auroc"
JAILBREAK_DATASETS = {"AdvBench", "JBB-Behaviors"}


def apply_style():
    plt.rcParams.update(CONFIG["rcparams"])


def normalize_model_name(name: str) -> str:
    mapping = {"gemma-2-2b": "Gemma-2-2B", "gemma-2-9b-it": "Gemma-2-9B-IT"}
    return mapping.get(str(name).strip().lower(), str(name))


def normalize_method_name(name: str) -> str:
    mapping = {
        "our_method": "NeuroCogMap",
        "ours": "NeuroCogMap",
        "ppl": "Perplexity",
        "logits_svm": "Logits SVM",
        "smoothllm": "SmoothLLM",
    }
    return mapping.get(str(name).strip().lower(), str(name))


def normalize_dataset_name(name: str) -> str:
    mapping = {"advbench": "AdvBench", "jbb-behaviors": "JBB-Behaviors"}
    return mapping.get(str(name).strip().lower(), str(name))


def dataset_sort_key(name: str) -> tuple:
    order = {"AdvBench": 0, "JBB-Behaviors": 1}
    return (order.get(name, 999), name)


def format_p_and_stars(p: float) -> tuple:
    if p < 0.001:
        return "p<0.001", "***"
    if p < 0.01:
        return "p<0.01", "**"
    if p < 0.05:
        return "p<0.05", "*"
    return f"p={p:.3f}", "ns"


def parse_values(x) -> np.ndarray:
    # 容错解析：列表中可能含缺失的交叉验证折（NaN）。json.loads 原生支持 NaN，
    # 这里解析后丢弃 NaN 折，仅用有效折参与后续均值/方差计算。
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.asarray([], dtype=float)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.asarray([], dtype=float)
    arr = np.asarray(json.loads(s), dtype=float)
    return arr[~np.isnan(arr)]


def load_df(csv_path=None, metric=DEFAULT_METRIC):
    path = Path(csv_path) if csv_path is not None else CSV_PATH
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise ValueError(f"Unknown metric: {metric}")

    missing_metric = df[metric].isna() | (df[metric].astype(str).str.strip() == "")
    if missing_metric.any():
        print(f'[Info] Skip {int(missing_metric.sum())} rows with empty metric "{metric}".')
    df = df.loc[~missing_metric].copy()
    if df.empty:
        raise ValueError(f'No valid rows left for metric "{metric}" in CSV: {path}')

    vals = [parse_values(v) for v in df[metric]]
    out = df[["base_model", "dataset", "method"]].copy()
    out["score_values"] = vals
    out["score_mean"] = [float(v.mean()) for v in vals]
    out["score_std"] = [float(v.std(ddof=1)) if len(v) > 1 else 0.0 for v in vals]

    out["base_model"] = out["base_model"].map(normalize_model_name)
    out["dataset"] = out["dataset"].map(normalize_dataset_name)
    out["method_raw"] = out["method"]
    out["method"] = out["method"].map(normalize_method_name)

    out = out[~out["method_raw"].isin(["llm_detector", "llm_detector_simple", "user_attention"])].copy()
    out = out[out["dataset"].isin(JAILBREAK_DATASETS)].copy()
    if out.empty:
        raise ValueError("No jailbreak datasets (AdvBench, JBB-Behaviors) found in CSV.")
    return out


def build_matrix(df_model):
    datasets = sorted(df_model["dataset"].unique(), key=dataset_sort_key)
    method_order = ["ppl", "logits_svm", "smoothllm", "our_method"]
    methods = [m for m in method_order if m in set(df_model["method_raw"])]

    complete_methods = []
    for m in methods:
        sub = df_model[df_model["method_raw"] == m]
        if set(sub["dataset"]) == set(datasets):
            complete_methods.append(m)
    methods = complete_methods

    d_count, m_count = len(datasets), len(methods)
    if d_count == 0 or m_count == 0:
        raise ValueError("No complete data to plot.")

    means = np.zeros((d_count, m_count), dtype=float)
    stds = np.zeros((d_count, m_count), dtype=float)
    score_values_list = []

    ours_idx = methods.index("our_method") if "our_method" in methods else -1
    highest_baseline_indices = []

    for i, ds in enumerate(datasets):
        values_row = []
        for j, m in enumerate(methods):
            row = df_model[(df_model["dataset"] == ds) & (df_model["method_raw"] == m)]
            means[i, j] = float(row["score_mean"].iloc[0])
            stds[i, j] = float(row["score_std"].iloc[0])
            values_row.append(row["score_values"].iloc[0])
        score_values_list.append(values_row)

        baseline_means_with_idx = [(means[i, j], j) for j, m in enumerate(methods) if m != "our_method"]
        if baseline_means_with_idx:
            _, highest_idx = max(baseline_means_with_idx, key=lambda x: x[0])
            highest_baseline_indices.append(highest_idx)
        else:
            highest_baseline_indices.append(-1)

    labels = [normalize_method_name(m) for m in methods]
    return datasets, labels, means, stds, score_values_list, ours_idx, highest_baseline_indices


def plot_model(df_model, model_name, out_dir, metric=DEFAULT_METRIC, skip_existing=False):
    apply_style()
    c = CONFIG["bar"]
    datasets, methods, means, stds, score_values_list, ours_idx, highest_baseline_indices = build_matrix(df_model)
    d_count, m_count = means.shape

    slug = model_name.lower().replace(" ", "_").replace("-", "_")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"fig3_bar_{slug}_{metric.lower()}.png"
    pdf = out_dir / f"fig3_bar_{slug}_{metric.lower()}.pdf"
    svg = out_dir / f"fig3_bar_{slug}_{metric.lower()}.svg"
    if skip_existing and png.exists() and pdf.exists() and svg.exists():
        print(f"[Skip] Existing outputs found for {model_name}: {png.name}, {pdf.name}, {svg.name}")
        return [png, pdf, svg]

    fig, ax = plt.subplots(figsize=c["figsize"], dpi=CONFIG["rcparams"]["figure.dpi"])
    x = np.arange(d_count, dtype=float)
    cluster_w = c["bar_cluster_width"]
    bar_w = cluster_w / max(1, m_count)
    y_min = float(c["ylim"][0])

    colors = c["method_colors"]
    method_xpos = np.zeros((d_count, m_count), dtype=float)
    legend_handles = {}
    np.random.seed(42)

    # 每个数据集内按均值从小到大排序柱子
    for i, _ in enumerate(datasets):
        sorted_idx = np.argsort(means[i, :])
        for rank, j in enumerate(sorted_idx):
            x_pos = x[i] - cluster_w / 2 + (rank + 0.5) * bar_w
            method_xpos[i, j] = x_pos

            lw = c["bar_edge_width_ours"] if j == ours_idx else c["bar_edge_width"]
            height = means[i, j]
            std_val = stds[i, j]
            bar = ax.bar(
                [x_pos],
                [max(height - y_min, 0.0)],
                width=bar_w * c["bar_width_factor"],
                bottom=[y_min],
                color=colors[j % len(colors)],
                edgecolor="#2b2b2b",
                linewidth=lw,
                yerr=[std_val],
                capsize=c["capsize"],
                error_kw={"elinewidth": c["error_line_width"], "ecolor": "#2b2b2b", "capthick": c["error_line_width"]},
                zorder=3,
            )
            bar[0].set_clip_on(False)
            if j not in legend_handles:
                legend_handles[j] = bar[0]

            y_pos = height + std_val + c["value_label_offset"] * (c["ylim"][1] - c["ylim"][0])
            ax.text(
                x_pos,
                y_pos,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=c["value_label_size"],
                color="#2b2b2b",
                zorder=5,
                clip_on=False,
            )

            values = score_values_list[i][j]
            if len(values) > 0:
                n_points = len(values)
                jitter = np.linspace(-c["scatter_jitter"] * bar_w, c["scatter_jitter"] * bar_w, n_points)
                if n_points > 1:
                    jitter += np.random.normal(0, c["scatter_jitter"] * bar_w * 0.3, size=n_points)
                x_scatter = x_pos + jitter
                ax.scatter(
                    x_scatter,
                    values,
                    s=c["scatter_size"],
                    alpha=c["scatter_alpha"],
                    color=colors[j % len(colors)],
                    edgecolors="#2b2b2b",
                    linewidths=0.3,
                    zorder=4,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=c["tick_size"])
    ax.set_xlabel("Dataset (category)", fontsize=c["xlabel_size"])
    ax.set_ylabel(f"{metric.upper()} (score)", fontsize=c["ylabel_size"])
    ax.tick_params(axis="both", labelsize=c["tick_size"], width=0.6, length=2.6)
    ax.set_ylim(c["ylim"][0], c["ylim"][1])
    ax.margins(x=c["x_margin"])
    ax.grid(False)

    if ours_idx >= 0:
        y_max = c["ylim"][1]
        y_axis_min = c["ylim"][0]
        y_range = y_max - y_axis_min

        for i, ds in enumerate(datasets):
            baseline_idx = highest_baseline_indices[i]
            if baseline_idx >= 0:
                ours_values = score_values_list[i][ours_idx]
                baseline_values = score_values_list[i][baseline_idx]

                if len(ours_values) >= 2 and len(baseline_values) >= 2:
                    try:
                        ours_arr = np.asarray(ours_values, dtype=float)
                        base_arr = np.asarray(baseline_values, dtype=float)
                        if len(ours_arr) != len(base_arr):
                            raise ValueError(f"fold 数量不一致: ours={len(ours_arr)}, baseline={len(base_arr)}")
                        if np.any(~np.isfinite(ours_arr)) or np.any(~np.isfinite(base_arr)):
                            raise ValueError("存在 NaN/Inf，无法做显著性检验")

                        t_stat, p_val = stats.ttest_rel(ours_arr, base_arr)
                        p_txt, stars = format_p_and_stars(p_val)

                        x_ours = method_xpos[i, ours_idx]
                        x_baseline = method_xpos[i, baseline_idx]

                        local_top = float(np.max(means[i, :] + stds[i, :]))
                        y_sig = min(y_max - 0.02 * y_range, local_top + 0.04 * y_range)
                        if y_sig <= y_axis_min:
                            y_sig = y_axis_min + 0.1 * y_range

                        sig_line_width = 0.4
                        sig_line_color = "#2b2b2b"
                        ax.plot([x_baseline, x_ours], [y_sig, y_sig], color=sig_line_color, linewidth=sig_line_width, zorder=5)
                        ax.plot(
                            [x_baseline, x_baseline],
                            [y_sig - 0.015 * y_range, y_sig],
                            color=sig_line_color,
                            linewidth=sig_line_width,
                            zorder=5,
                        )
                        ax.plot(
                            [x_ours, x_ours],
                            [y_sig - 0.015 * y_range, y_sig],
                            color=sig_line_color,
                            linewidth=sig_line_width,
                            zorder=5,
                        )

                        x_center = (x_baseline + x_ours) / 2
                        text_y = min(y_max - 0.005 * y_range, y_sig + 0.01 * y_range)
                        ax.text(
                            x_center,
                            text_y,
                            f"{p_txt} {stars}",
                            ha="center",
                            va="bottom",
                            fontsize=c["tick_size"],
                            color="#2b2b2b",
                            zorder=6,
                        )

                        print(f"  [{model_name}][{ds}] NeuroCogMap vs {methods[baseline_idx]}: t={t_stat:.4f}, {p_txt} {stars}")
                    except Exception as e:
                        print(f"  [Warning] 统计检验失败 ({model_name}, {ds}): {e}")

        ax.set_ylim(c["ylim"][0], min(1.0, c["ylim"][1]))

    legend_order = [j for j in range(m_count) if j in legend_handles]
    ax.legend(
        [legend_handles[j] for j in legend_order],
        [methods[j] for j in legend_order],
        frameon=False,
        fontsize=c["legend_size"],
        ncol=c["legend_ncol"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        columnspacing=0.9,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.17, right=0.98, top=0.72, bottom=0.26)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, facecolor="white", bbox_inches=None)
    plt.close(fig)
    return [png, pdf, svg]


def main(csv_path=None, out_dir=None, metric=DEFAULT_METRIC, skip_existing=False):
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_df(csv_path=csv_path, metric=metric)

    files = []
    for model in sorted(df["base_model"].unique()):
        files.extend(plot_model(df[df["base_model"] == model], model, out_dir, metric=metric, skip_existing=skip_existing))

    print("Generated files:")
    for f in files:
        print(str(f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot normal bar detection chart for jailbreak data")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to input CSV")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--metric", type=str, default=DEFAULT_METRIC, help="Metric to plot")
    parser.add_argument("--skip-existing", action="store_true", help="Skip plotting if png/pdf/svg already exist for a model")
    args = parser.parse_args()
    main(csv_path=args.csv_path, out_dir=args.out_dir, metric=args.metric, skip_existing=args.skip_existing)
