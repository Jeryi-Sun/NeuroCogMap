"""
将各实验目录下的 `step1_train_feature_pcc_ranked_by_aic.csv` 中 feature 的 PCC
（使用列 `pcc_with_train_aic`）聚合到 motif 维度，并绘制 Task × Motif mean PCC heatmap。

替换内容：
- 不再读取 `Nature_Report_step1_direct_correlation_stats.json`
- 改为直接从原始 CSV 读取，并复刻 `generate_step1_report.py` 的桶划分逻辑：
  - positive：PCC > 0，按 abs(PCC) 取 Top-K
  - negative：PCC < 0，按 abs(PCC) 取 Top-K
  - zero：|PCC| < near_zero_abs_threshold，按 abs(PCC) 从小到大取 Top-K
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


# --- Nature/Arial font setup ---
try:
    # 重新加载字体管理器以确保识别最新字体
    fm.fontManager = fm.FontManager()
except Exception:
    pass

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


PRIMARY_BLUE = "#579FCA"
LIGHT_BLUE = "#B4DDF4"
WARM_YELLOW = "#F7DC7C"
ROSE_PINK = "#F0BBC1"
LIGHT_PEACH = "#FAE6D7"
PREFERRED_COLORS = [LIGHT_BLUE, PRIMARY_BLUE, LIGHT_PEACH, ROSE_PINK, WARM_YELLOW]
NATURE_CMAP = LinearSegmentedColormap.from_list("nature_div", PREFERRED_COLORS)


FEATURE_ANALYSIS_DIR = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/feature_analysis"
)
MOTIF_CLASSIFICATION_CSV = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "neurocogmap_step1_motif_classification.csv"
)
LAYER_CLASSIFICATION_CSV = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "neurocogmap_step1_layer_classification.csv"
)
PARCEL_JSON = (
    "/path/to/project_root/neurocogmap/2b_model_data/parcel.json"
)

MATRIX_CSV = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "task_motif_mean_pcc_heatmap.csv"
)
TASK_LABEL_MAP: Dict[str, str] = {
    "badham2017deficits_exp1_csv": "Shepard categorization",
    "bahrami2020four_exp_csv": "Drifting four-armed bandit",
    "hilbig2014generalized_exp1_csv": "Multi-attribute decision-making",
    "popov2023intent_exp1_csv": "Episodic long-term memory",
    "ruggeri2022globalizability_exp1_csv": "Intertemporal choice",
}
EXCLUDED_DATASETS = {"collsiöö2023MCPL_exp1_csv"}
LAYER_LABEL_MAP: Dict[str, str] = {
    "A": "Perceptual Access and Attentional Gating Layer",
    "B": "Semantic Representation and Knowledge Integration Layer",
    "C": "Abstract Reasoning and Meta-Cognitive Control Layer",
    "D": "Situated Application and Social Interaction Layer",
}
DETAIL_CSV = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "task_motif_mean_pcc_longform.csv"
)
IMG_PNG = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "task_motif_mean_pcc_heatmap.png"
)
IMG_PDF = IMG_PNG.replace(".png", ".pdf")
IMG_SVG = IMG_PNG.replace(".png", ".svg")

LAYER_MATRIX_CSV = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "task_layer_mean_pcc_heatmap.csv"
)
LAYER_DETAIL_CSV = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "task_layer_mean_pcc_longform.csv"
)
LAYER_IMG_PNG = (
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/NCM_analysis/"
    "task_layer_mean_pcc_heatmap.png"
)
LAYER_IMG_PDF = LAYER_IMG_PNG.replace(".png", ".pdf")
LAYER_IMG_SVG = LAYER_IMG_PNG.replace(".png", ".svg")


def get_dataset_dirs(feature_analysis_dir: str) -> List[str]:
    base = Path(feature_analysis_dir)
    if not base.exists():
        raise FileNotFoundError(f"feature_analysis 目录不存在: {base}")
    dataset_keys = [p.name for p in base.iterdir() if p.is_dir() and p.name.endswith("_csv")]
    return sorted(dataset_keys)


def load_parcel_map(parcel_json_path: str) -> Dict[int, Dict[str, object]]:
    path = Path(parcel_json_path)
    if not path.exists():
        raise FileNotFoundError(f"parcel.json 不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    parcels = data.get("parcel_summaries", [])
    parcel_map: Dict[int, Dict[str, object]] = {}
    for p in parcels:
        pid = p.get("parcel_id")
        if isinstance(pid, int):
            parcel_map[pid] = p
    return parcel_map


def clean_function_name(name: str) -> str:
    name = str(name).strip()
    if name.startswith("**"):
        return name.lstrip("* ").strip()
    return name


def split_by_buckets(
    df: pd.DataFrame,
    top_k: int = 10,
    near_zero_abs_threshold: float = 0.1,
) -> Dict[str, pd.DataFrame]:
    df_clean = df.dropna(subset=["pcc_with_train_aic", "abs_pcc_with_train_aic"]).copy()

    positive = df_clean[df_clean["pcc_with_train_aic"] > 0].copy()
    positive = positive.sort_values("abs_pcc_with_train_aic", ascending=False).head(top_k)

    negative = df_clean[df_clean["pcc_with_train_aic"] < 0].copy()
    negative = negative.sort_values("abs_pcc_with_train_aic", ascending=False).head(top_k)

    near_zero = df_clean[df_clean["abs_pcc_with_train_aic"] < near_zero_abs_threshold].copy()
    near_zero = near_zero.sort_values("abs_pcc_with_train_aic", ascending=True).head(top_k)

    return {"positive": positive, "zero": near_zero, "negative": negative}


def build_group_outputs(
    records_df: pd.DataFrame,
    group_col: str,
    matrix_csv: str,
    detail_csv: str,
    img_png: str,
    img_pdf: str,
    img_svg: str,
    xlabel: str,
    title: str,
    preferred_order: List[str] | None = None,
    skip_existing: bool = False,
) -> None:
    if group_col not in records_df.columns:
        raise KeyError(f"records_df 缺少分组列: {group_col}")

    agg = (
        records_df.groupby(["task", group_col], as_index=False)
        .agg(
            mean_pcc=("pcc", "mean"),
            mean_abs_pcc=("pcc", lambda s: float(np.mean(np.abs(s)))),
            n_entries=("pcc", "size"),
            positive_count=("pcc", lambda s: int((s > 0).sum())),
            negative_count=("pcc", lambda s: int((s < 0).sum())),
            zero_like_count=("bucket", lambda s: int((s == "zero").sum())),
        )
    )

    heatmap = agg.pivot(index="task", columns=group_col, values="mean_pcc").fillna(0)
    if preferred_order:
        ordered_cols = [c for c in preferred_order if c in heatmap.columns] + [
            c for c in heatmap.columns if c not in preferred_order
        ]
        heatmap = heatmap[ordered_cols]

    matrix_out = Path(matrix_csv)
    if skip_existing and matrix_out.exists():
        print(f"[SKIP] CSV 已存在: {matrix_out}")
    else:
        matrix_out.parent.mkdir(parents=True, exist_ok=True)
        matrix_export_df = heatmap.copy()
        matrix_export_df.insert(0, "task_type", [TASK_LABEL_MAP.get(str(t), str(t)) for t in heatmap.index])
        matrix_export_df.insert(0, "task", [str(t) for t in heatmap.index])
        matrix_export_df.to_csv(matrix_out, index=False)
        print(f"[OK] 写出 CSV: {matrix_out}")

    maybe_write_csv(detail_csv, agg.sort_values(["task", group_col]), skip_existing=skip_existing)

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    im = ax.imshow(heatmap.values, aspect="auto", cmap=NATURE_CMAP, norm=norm)

    wrapped_xlabels = []
    for col_name in heatmap.columns:
        label = str(col_name).strip()
        if len(label) > 18:
            label = "\n".join(textwrap.wrap(label, width=22, break_long_words=False))
        wrapped_xlabels.append(label)

    ax.set_xlabel(xlabel, fontsize=8, fontweight="bold")
    ax.set_ylabel("Task", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels(wrapped_xlabels, rotation=0, ha="center", fontsize=7)
    ax.set_yticks(range(len(heatmap.index)))
    y_labels = [TASK_LABEL_MAP.get(str(task), str(task)) for task in heatmap.index]
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.tick_params(axis="both", labelsize=8, width=0.6)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=6)

    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            val = float(heatmap.iat[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean PCC", fontsize=8)
    cbar.ax.tick_params(labelsize=8, width=0.6)
    plt.tight_layout()

    maybe_write_figure(img_png, fig, skip_existing=skip_existing)
    maybe_write_figure(img_pdf, fig, skip_existing=skip_existing)
    maybe_write_figure(img_svg, fig, skip_existing=skip_existing)
    plt.close(fig)


def maybe_write_csv(path: str, df: pd.DataFrame, skip_existing: bool) -> None:
    out = Path(path)
    if skip_existing and out.exists():
        print(f"[SKIP] CSV 已存在: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] 写出 CSV: {out}")


def maybe_write_figure(save_path: str, fig: plt.Figure, skip_existing: bool) -> None:
    out = Path(save_path)
    if skip_existing and out.exists():
        print(f"[SKIP] Figure 已存在: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=450, bbox_inches="tight", transparent=True)
    print(f"[OK] 写出 Figure: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 Task × Motif mean PCC heatmap（直接读取原始 CSV）")
    parser.add_argument("--top-k", type=int, default=10, help="每个方向（正/负/近零）取 Top-K 特征")
    parser.add_argument(
        "--near-zero-abs-threshold",
        type=float,
        default=0.1,
        help="近零桶阈值：|pcc_with_train_aic| < 阈值",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若输出文件已存在则跳过写入（CSV / 图像分别跳过）",
    )
    args = parser.parse_args()

    try:
        motif_df = pd.read_csv(MOTIF_CLASSIFICATION_CSV)
        if "feature_idx" not in motif_df.columns or "motif_main_6class" not in motif_df.columns:
            raise KeyError("motif classification CSV 缺少必需列: feature_idx / motif_main_6class")
        motif_map = dict(zip(motif_df["feature_idx"].astype(int), motif_df["motif_main_6class"]))

        layer_df = pd.read_csv(LAYER_CLASSIFICATION_CSV)
        if "feature_idx" not in layer_df.columns or "mapped_layer" not in layer_df.columns:
            raise KeyError("layer classification CSV 缺少必需列: feature_idx / mapped_layer")
        layer_df = layer_df.copy()
        layer_df["mapped_layer"] = (
            layer_df["mapped_layer"].astype(str).str.strip().map(LAYER_LABEL_MAP).fillna(layer_df["mapped_layer"].astype(str).str.strip())
        )
        layer_map = dict(zip(layer_df["feature_idx"].astype(int), layer_df["mapped_layer"]))

        parcel_map = load_parcel_map(PARCEL_JSON)
        dataset_keys = [
            k for k in get_dataset_dirs(FEATURE_ANALYSIS_DIR) if k not in EXCLUDED_DATASETS
        ]
        if not dataset_keys:
            raise ValueError(f"在 {FEATURE_ANALYSIS_DIR} 下未找到任何 *_csv 数据集目录")

        records: List[Dict[str, object]] = []
        for task in dataset_keys:
            input_csv = Path(FEATURE_ANALYSIS_DIR) / task / "step1_train_feature_pcc_ranked_by_aic.csv"
            if not input_csv.exists():
                raise FileNotFoundError(f"缺少输入 CSV: {input_csv}")

            df_step1 = pd.read_csv(input_csv)
            expected_cols = {"feature_idx", "pcc_with_train_aic", "abs_pcc_with_train_aic"}
            missing = expected_cols - set(df_step1.columns)
            if missing:
                raise KeyError(f"{task} 输入 CSV 缺少列: {sorted(missing)}")

            buckets = split_by_buckets(
                df=df_step1,
                top_k=args.top_k,
                near_zero_abs_threshold=args.near_zero_abs_threshold,
            )

            for bucket_name, bucket_df in buckets.items():
                for row in bucket_df.itertuples(index=False):
                    feature_idx = int(getattr(row, "feature_idx"))
                    pcc_val = float(getattr(row, "pcc_with_train_aic"))
                    motif = motif_map.get(feature_idx, "Unknown")
                    parcel_info = parcel_map.get(feature_idx, {})
                    parcel_fn_name = clean_function_name(parcel_info.get("function_name", ""))

                    records.append(
                        {
                            "task": task,
                            "bucket": bucket_name,
                            "feature_idx": feature_idx,
                            "pcc": pcc_val,
                            "motif": motif,
                            "mapped_layer": layer_map.get(feature_idx, "Unknown"),
                            "parcel_function_name": parcel_fn_name,
                        }
                    )

        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("汇总得到的 records 为空，请检查输入数据与阈值参数。")

        preferred_order = [
            "Structured reasoning",
            "Verification / control",
            "Entity retrieval",
            "Direct answer",
            "Narrative / concrete processing",
            "Social / affective modulation",
        ]
        build_group_outputs(
            records_df=df,
            group_col="motif",
            matrix_csv=MATRIX_CSV,
            detail_csv=DETAIL_CSV,
            img_png=IMG_PNG,
            img_pdf=IMG_PDF,
            img_svg=IMG_SVG,
            xlabel="Motif class",
            title="Task × Motif mean PCC heatmap",
            preferred_order=preferred_order,
            skip_existing=args.skip_existing,
        )

        build_group_outputs(
            records_df=df,
            group_col="mapped_layer",
            matrix_csv=LAYER_MATRIX_CSV,
            detail_csv=LAYER_DETAIL_CSV,
            img_png=LAYER_IMG_PNG,
            img_pdf=LAYER_IMG_PDF,
            img_svg=LAYER_IMG_SVG,
            xlabel="Layer class",
            title="Task × Layer mean PCC heatmap",
            skip_existing=args.skip_existing,
        )

        print(MATRIX_CSV)
        print(DETAIL_CSV)
        print(IMG_PNG)
        print(IMG_PDF)
        print(IMG_SVG)
        print(LAYER_MATRIX_CSV)
        print(LAYER_DETAIL_CSV)
        print(LAYER_IMG_PNG)
        print(LAYER_IMG_PDF)
        print(LAYER_IMG_SVG)

    except Exception as e:
        print(f"[ERROR] plot_feature_motif 失败: {e}")
        raise


if __name__ == "__main__":
    main()
