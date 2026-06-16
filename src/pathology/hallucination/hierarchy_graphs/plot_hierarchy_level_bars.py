#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 hierarchy_level_all_models.json 绘制各模型在认知层级 A/B/C/D 上的平均激活差异柱状图。

数据来源:
  /path/to/project_root/safety_explanation/hallucination/hierarchy_graphs/data/hierarchy_level_all_models.json

绘图规范:
  - 每个模型单独一个 PDF 图 (Nature 风格，A4 兼容)
  - Y 轴: 层级平均激活差异 (在绘图阶段将 Hallucination-Truthfulness 反号为 Truthfulness-Hallucination)
          并基于每层的样本计算均值与 95% CI
  - X 轴顺序固定: A (Perception Layer) -> B (Representation Layer) ->
                  C (Abstract Layer) -> D (Application Layer)
  - 字体: Arial，字号满足用户指定范围
  - 颜色: 使用项目提供的调色板 (4 种颜色对应 4 个层级)
"""

import json
from pathlib import Path
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import numpy as np

# 重新加载字体管理器以确保识别最新字体（按用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass


def load_all_models(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"输入 JSON 不存在: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("hierarchy_level_all_models.json 顶层必须是 dict")
    return data


def configure_matplotlib():
    # 全局字体和样式设置 (Nature 风格 + Arial)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    # 字号设置 (pt) - 统一为 5pt（1/4 A4 宽度小图规范）
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.labelsize"] = 5     # 坐标轴 label: 5 pt
    plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["ytick.labelsize"] = 5
    plt.rcParams["legend.fontsize"] = 5    # 图例: 5 pt
    plt.rcParams["figure.dpi"] = 450

    # 坐标轴与刻度线宽、边框 (Nature 风格)
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

    # 矢量图字体嵌入设置，保证 PDF/SVG 文字可编辑且符合 Nature 规范
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"


def plot_model_bars(model_name: str, model_data: dict, output_dir: Path, group_high_label: str = "Hallucination", group_low_label: str = "Truthfulness"):
    # 固定顺序和人类可读 label
    layer_order = ["A", "B", "C", "D"]
    layer_labels = {
        "A": "Perception Layer",
        "B": "Representation Layer",
        "C": "Abstract Layer",
        "D": "Application Layer",
    }

    # 使用每层原始 activation_diff 列表来计算 Truthfulness-Hallucination 的均值和95% CI
    values = []
    ci_err = []  # 对称误差条
    for lid in layer_order:
        if lid not in model_data:
            raise KeyError(f"模型 {model_name} 数据中缺少层级 {lid} 的原始 activation_diff 列表")
        diffs = model_data[lid]
        if not isinstance(diffs, list) or len(diffs) == 0:
            raise ValueError(f"模型 {model_name} 层级 {lid} 的数据为空或格式错误")

        # 反号: Hallucination-Truthfulness -> Truthfulness-Hallucination
        diffs_arr = -np.asarray(diffs, dtype=float)

        mean_val = float(diffs_arr.mean())
        # 使用无偏标准差计算标准误，然后乘以 1.96 作为近似 95% CI
        if diffs_arr.size > 1:
            std_val = float(diffs_arr.std(ddof=1))
            se = std_val / sqrt(diffs_arr.size)
            ci = 1.96 * se
        else:
            # 只有一个样本时，无法估计方差，这里设为 0 并显式提醒
            print(f"[WARN] 模型 {model_name} 层级 {lid} 仅有一个样本，95% CI 设为 0")
            ci = 0.0

        values.append(mean_val)
        ci_err.append(ci)

    # 颜色方案 (A/B/C/D 分别使用 475f65, d09b4a, a55650, 828369)
    color_map = {
        "A": "#475f65",
        "B": "#d09b4a",
        "C": "#a55650",
        "D": "#828369",
    }
    colors = [color_map[lid] for lid in layer_order]

    fig, ax = plt.subplots(figsize=(1.8, 1.4))  # 1/4 A4 宽度（约 45mm）

    x = np.arange(len(layer_order))
    ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        yerr=ci_err,
        capsize=3,
        ecolor="black",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([layer_labels[lid] for lid in layer_order], rotation=30, ha="right", fontsize=5)
    ax.set_ylabel(f"Mean activation difference\n({group_low_label} - {group_high_label})", fontsize=5)

    # 零线
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)

    # 简单网格 (Y 方向)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    ax.tick_params(axis="both", labelsize=5)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path_pdf = output_dir / f"{model_name}_hierarchy_level_bar.pdf"
    out_path_svg = output_dir / f"{model_name}_hierarchy_level_bar.svg"
    fig.savefig(out_path_pdf, format="pdf")
    fig.savefig(out_path_svg, format="svg")
    plt.close(fig)

    print(f"[OK] 保存柱状图: {out_path_pdf} 和 {out_path_svg}")


def plot_model_violins(model_name: str, model_data: dict, output_dir: Path, group_high_label: str = "Hallucination", group_low_label: str = "Truthfulness"):
    """
    使用小提琴图展示每个层级 Truthfulness-Hallucination 的分布，作为补充参考。
    """
    layer_order = ["A", "B", "C", "D"]
    layer_labels = {
        "A": "Perception Layer",
        "B": "Representation Layer",
        "C": "Abstract Layer",
        "D": "Application Layer",
    }

    # 收集每层的反号后样本
    layer_samples = []
    for lid in layer_order:
        if lid not in model_data:
            raise KeyError(f"模型 {model_name} 数据中缺少层级 {lid} 的原始 activation_diff 列表")
        diffs = model_data[lid]
        if not isinstance(diffs, list) or len(diffs) == 0:
            raise ValueError(f"模型 {model_name} 层级 {lid} 的数据为空或格式错误")
        diffs_arr = -np.asarray(diffs, dtype=float)  # Truthfulness - Hallucination
        layer_samples.append(diffs_arr)

    fig, ax = plt.subplots(figsize=(1.8, 1.4))  # 1/4 A4 宽度（约 45mm）

    x = np.arange(1, len(layer_order) + 1)

    parts = ax.violinplot(
        layer_samples,
        positions=x,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        widths=0.6,
    )
    # 使用 Nature 风格调色板（每个层级一个颜色）
    # PRIMARY_COLORS: medium_blue, warm_yellow, rose_pink
    # EXTRA_COLORS:  light_blue（补足第四个颜色）
    nature_palette = {
        "A": "#579FCA",  # medium_blue
        "B": "#F7DC7C",  # warm_yellow
        "C": "#F0BBC1",  # rose_pink
        "D": "#B4DDF4",  # light_blue
    }
    for i, pc in enumerate(parts["bodies"]):
        lid = layer_order[i]
        base_color = nature_palette.get(lid, "#666666")
        pc.set_facecolor(base_color)
        pc.set_edgecolor("#000000")
        pc.set_alpha(0.35)

    # 叠加每个样本的散点 (轻微抖动) + 更简洁的均值水平线
    rng = np.random.default_rng(42)
    for i, samples in enumerate(layer_samples):
        lid = layer_order[i]
        base_color = nature_palette.get(lid, "#666666")
        # 稍微变深一点的颜色，用于散点主体颜色
        rgb = np.array(mcolors.to_rgb(base_color))
        darker_rgb = np.clip(rgb * 0.8, 0, 1)

        x_jitter = x[i] + rng.normal(0, 0.03, size=samples.size)
        ax.scatter(
            x_jitter,
            samples,
            c=[darker_rgb],
            s=3,
            alpha=0.7,
            linewidths=0.3,
            edgecolors="black",
        )
        # 用一条短水平线标记均值，替代黑色实心点
        mean_val = float(samples.mean())
        ax.hlines(
            y=mean_val,
            xmin=x[i] - 0.15,
            xmax=x[i] + 0.15,
            colors="#666666",  # 灰色虚线均值标记
            linewidth=1.0,
            linestyles="--",
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([layer_labels[lid] for lid in layer_order], rotation=30, ha="right", fontsize=5)
    ax.set_ylabel(f"Activation difference\n({group_low_label} - {group_high_label})", fontsize=5)

    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    ax.tick_params(axis="both", labelsize=5)

    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path_pdf = output_dir / f"{model_name}_hierarchy_level_violin.pdf"
    out_path_svg = output_dir / f"{model_name}_hierarchy_level_violin.svg"
    fig.savefig(out_path_pdf, format="pdf")
    fig.savefig(out_path_svg, format="svg")
    plt.close(fig)

    print(f"[OK] 保存小提琴图: {out_path_pdf} 和 {out_path_svg}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="绘制层级激活差异柱状图和小提琴图")
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="输入 JSON 路径（默认: hallucination/hierarchy_graphs/data/hierarchy_level_all_models.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: hallucination/hierarchy_graphs/figures_hierarchy_level_bars）",
    )
    parser.add_argument(
        "--project-type",
        type=str,
        default="hallucination",
        choices=["hallucination", "fairness_bias"],
        help="项目类型：hallucination 或 fairness_bias（用于自动设置默认标签）",
    )
    parser.add_argument(
        "--group-high-label",
        type=str,
        default=None,
        help="高激活组标签（如 Hallucination / Incorrect）。如未指定，将根据 project-type 自动设置",
    )
    parser.add_argument(
        "--group-low-label",
        type=str,
        default=None,
        help="低激活组标签（如 Truthfulness / Correct）。如未指定，将根据 project-type 自动设置",
    )
    args = parser.parse_args()
    
    # 如果未指定标签，根据 project_type 设置默认值
    # argparse 会将 --project-type 转换为 project_type 属性
    project_type = getattr(args, 'project_type', 'hallucination')
    if args.group_high_label is None:
        if project_type == "fairness_bias":
            args.group_high_label = "Incorrect"
        else:
            args.group_high_label = "Hallucination"
    
    if args.group_low_label is None:
        if project_type == "fairness_bias":
            args.group_low_label = "Correct"
        else:
            args.group_low_label = "Truthfulness"

    base_dir = Path("/path/to/project_root")
    if args.json_path is None:
        json_path = (
            base_dir
            / "safety_explanation"
            / "hallucination"
            / "hierarchy_graphs"
            / "data"
            / "hierarchy_level_all_models.json"
        )
    else:
        json_path = Path(args.json_path)

    if args.output_dir is None:
        output_dir = (
            base_dir
            / "safety_explanation"
            / "hallucination"
            / "hierarchy_graphs"
            / "figures_hierarchy_level_bars"
        )
    else:
        output_dir = Path(args.output_dir)

    configure_matplotlib()
    all_models = load_all_models(json_path)

    for model_name, model_data in all_models.items():
        try:
            plot_model_bars(model_name, model_data, output_dir, args.group_high_label, args.group_low_label)
            plot_model_violins(model_name, model_data, output_dir, args.group_high_label, args.group_low_label)
        except Exception as e:
            print(f"[ERROR] 绘制模型 {model_name} 时出错: {e}")


if __name__ == "__main__":
    main()

