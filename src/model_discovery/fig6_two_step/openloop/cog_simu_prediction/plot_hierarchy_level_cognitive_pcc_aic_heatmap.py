#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 heatmap：
- 横坐标：hierarchy_level（A/B/C/D）
- 纵坐标：认知数据（即 6 个实验目录）
- 每个格子的值：pcc_with_train_aic

输出：
- PNG / PDF / SVG：写到 results/feature_analysis/plots/
- CSV 矩阵：写到同目录，方便复现/校验
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

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


FEATURE_ANALYSIS_DIR = Path(
    "/path/to/project_root/Human_LLM_align/"
    "Llama-3.1-Centaur-70B-main/openloop/cog_simu_prediction/results/feature_analysis"
)

DEFAULT_EXPERIMENT_DIRS: List[str] = [
    "badham2017deficits_exp1_csv",
    "bahrami2020four_exp_csv",
    "collsiöö2023MCPL_exp1_csv",
    "hilbig2014generalized_exp1_csv",
    "popov2023intent_exp1_csv",
    "ruggeri2022globalizability_exp1_csv",
]


def _clean_dataset_label(exp_dir: str) -> str:
    # 尽量压缩显示，避免 tick 过长；不改变主要可辨识信息
    return (
        exp_dir.replace("_exp1_csv", "")
        .replace("_exp1", "")
        .replace("_exp_csv", "")
        .replace("exp", "")
    )


def _load_hierarchy_aic_value(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少输入层级 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"hierarchy_level", "pcc_with_train_aic"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path} 缺少列: {sorted(missing)}")

    # 转成 dict：A/B/C/D -> value
    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        hl = str(row["hierarchy_level"]).strip().upper()
        if hl not in {"A", "B", "C", "D"}:
            continue
        out[hl] = float(row["pcc_with_train_aic"])

    # 必须 4 层都存在
    missing_levels = [x for x in ["A", "B", "C", "D"] if x not in out]
    if missing_levels:
        raise ValueError(f"{csv_path} 层级缺失: {missing_levels}")
    return out


def _maybe_save_figure(save_path: Path, fig: plt.Figure, skip_existing: bool) -> None:
    if skip_existing and save_path.exists():
        print(f"[SKIP] Figure 已存在: {save_path}")
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=450, bbox_inches="tight", transparent=True)
    print(f"[OK] 写出 Figure: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 hierarchy_level × cognitive data 的 pcc_with_train_aic heatmap")
    parser.add_argument(
        "--experiment-dirs",
        type=str,
        default=",".join(DEFAULT_EXPERIMENT_DIRS),
        help="逗号分隔的 6 个或多个实验目录名（位于 results/feature_analysis/ 下）",
    )
    parser.add_argument("--skip-existing", action="store_true", help="输出已存在则跳过写入")
    args = parser.parse_args()

    experiment_dirs = [x.strip() for x in args.experiment_dirs.split(",") if x.strip()]
    if not experiment_dirs:
        raise ValueError("experiment-dirs 为空")

    hierarchy_order = ["A", "B", "C", "D"]
    y_labels = [_clean_dataset_label(x) for x in experiment_dirs]

    matrix: List[List[float]] = []
    for exp_dir in experiment_dirs:
        csv_path = (
            FEATURE_ANALYSIS_DIR
            / exp_dir
            / "step1_train_feature_pcc_mapped_hierarchy_by_aic.csv"
        )
        hv = _load_hierarchy_aic_value(csv_path)
        matrix.append([hv[h] for h in hierarchy_order])

    df_mat = pd.DataFrame(matrix, index=y_labels, columns=hierarchy_order)

    plots_dir = FEATURE_ANALYSIS_DIR / "plots"
    matrix_csv = plots_dir / "hierarchy_level_x_cognitive_pcc_with_train_aic_heatmap.csv"
    if args.skip_existing and matrix_csv.exists():
        print(f"[SKIP] CSV 已存在: {matrix_csv}")
    else:
        matrix_csv.parent.mkdir(parents=True, exist_ok=True)
        df_mat.to_csv(matrix_csv)
        print(f"[OK] 写出矩阵 CSV: {matrix_csv}")

    # --- plot ---
    max_abs = float(pd.DataFrame(matrix).abs().to_numpy().max())
    if not math.isfinite(max_abs):
        raise ValueError("矩阵包含非有限值，无法绘图")
    if max_abs <= 0:
        # 极端情况下：全部为 0，TwoSlopeNorm 无法提供有效范围
        max_abs = 1.0

    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    im = ax.imshow(df_mat.values, aspect="auto", cmap=NATURE_CMAP, norm=norm)

    ax.set_xlabel("Hierarchy level", fontsize=8, fontweight="bold")
    ax.set_ylabel("Cognitive data", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(hierarchy_order)))
    ax.set_xticklabels(hierarchy_order, fontsize=8)
    ax.set_yticks(range(len(df_mat.index)))
    ax.set_yticklabels(df_mat.index, fontsize=8)
    ax.tick_params(axis="both", labelsize=8, width=0.6)
    ax.set_title("Hierarchy level × cognitive data (pcc_with_train_aic)", fontsize=8, fontweight="bold", pad=6)

    # 标注格子值（6×4 共 24 个）
    for i in range(df_mat.shape[0]):
        for j in range(df_mat.shape[1]):
            val = float(df_mat.iat[i, j])
            if math.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pcc_with_train_aic", fontsize=8)
    cbar.ax.tick_params(labelsize=8, width=0.6)

    plt.tight_layout()

    img_png = plots_dir / "hierarchy_level_x_cognitive_pcc_with_train_aic_heatmap.png"
    img_pdf = plots_dir / "hierarchy_level_x_cognitive_pcc_with_train_aic_heatmap.pdf"
    img_svg = plots_dir / "hierarchy_level_x_cognitive_pcc_with_train_aic_heatmap.svg"
    _maybe_save_figure(img_png, fig, skip_existing=args.skip_existing)
    _maybe_save_figure(img_pdf, fig, skip_existing=args.skip_existing)
    _maybe_save_figure(img_svg, fig, skip_existing=args.skip_existing)
    plt.close(fig)


if __name__ == "__main__":
    main()

