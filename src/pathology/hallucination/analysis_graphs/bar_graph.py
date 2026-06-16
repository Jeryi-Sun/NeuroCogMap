#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Parcel/Capability 级 top_anomalous_*.json 绘制 Nature 风格条形图。

功能：
1. 从 top_anomalous_parcels.json 或 top_anomalous_capabilities.json 中读取各条目的 activation_diff 及 95% CI；
2. 按 activation_diff 从高到低排序；
3. 正值（Higher in Hallucinated）用暖色系填充，负值（Higher in Truthful）用蓝色填充；
4. 误差线来自 activation_diff_ci_lower ~ activation_diff_ci_upper（如果存在）；
5. 背景使用渐变色带（绿色=High / 黄色=中间 / 红色=Low）表示激活水平区域；
6. 显著性标注（p < 0.05 的条目在圆点旁标 *）；
7. 输出 PDF 和 SVG 格式，Nature 风格，Arial 字体，≥450 dpi。

支持两种级别：
- parcel: 使用 parcel_id, function_name, parcel_level 路径
- capability: 使用 capability_id, capability_name, capability_level 路径
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import matplotlib
import matplotlib.font_manager as fm

# 重新加载字体管理器以确保识别最新字体（用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# =====================
# 全局绘图配置
# =====================
CONFIG: Dict[str, Any] = {
    # 版式（1/4 A4 宽度，约 45 mm）
    "figure_width_inch": 1.8,
    "figure_height_inch": 1.4,
    "dpi": 450,
    # 字体（Nature 要求 Arial，小图统一 5 pt）
    "font_family": "Arial",
    "font_title": 5.0,
    "font_axis_label": 5.0,
    "font_tick": 5.0,
    "font_legend": 5.0,
    "font_annotation": 5.0,
    # 标记（scatter 节点大小，原为 25，略显偏大，这里调小一点）
    "marker_size": 15,
    "marker_edge_lw": 0.4,
    "errorbar_lw": 0.7,
    "errorbar_capsize": 1.8,
    # 统计
    "p_threshold": 0.05,
}

# Nature 风格基础设置
plt.rcParams["font.family"] = CONFIG["font_family"]
plt.rcParams["font.size"] = 5
plt.rcParams["axes.labelsize"] = 5
plt.rcParams["axes.titlesize"] = 5
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["legend.fontsize"] = 5
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.unicode_minus"] = False

# 颜色方案（来自用户调色板）
COLOR_POS = "#F7DC7C"       # 正 activation_diff（Higher in Hallucinated Group）
COLOR_NEG = "#579FCA"       # 负 activation_diff（Higher in Truthful Group）
COLOR_POS_EDGE = "#D4A843"  # 正值圆点边框色
COLOR_NEG_EDGE = "#3A6F93"  # 负值圆点边框色
COLOR_ERRORBAR = "#333333"  # 误差线颜色

# 背景渐变色带
BAND_HIGH_COLOR = "#D5EDCA"   # 上方绿色（High activation）
BAND_MID_COLOR = "#FFF8DC"    # 中间淡黄色
BAND_LOW_COLOR = "#F8D7D7"    # 下方红色（Low activation）


def sanitize_text(text: str) -> str:
    """清理文本中的特殊字符，避免 Arial 缺字。"""
    if not isinstance(text, str):
        text = str(text)
    for ch in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        text = text.replace(ch, "-")
    # 去除开头的 **
    text = text.strip().lstrip("*").strip()
    return text


def load_json(path: Path) -> Any:
    """加载 JSON 文件。"""
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_bar_graph(
    top_items_path: Path,
    output_fig: Path,
    level: str = "parcel",
    p_threshold: float = CONFIG["p_threshold"],
    title: str = "",
    only_significant: bool = False,
    extra_non_top20_k: int = 0,
    top_k: int = 0,
    top_k_pos: int = 0,
    top_k_neg: int = 0,
    group_high_label: str = "Hallucinated",
    group_low_label: str = "Truthful",
    use_cached_data: bool = True,
    cached_data_path: Path = None,
    save_cached_data: bool = True,
    force_recompute: bool = False,
) -> None:
    """
    绘制 Nature 风格激活差异条形图（scatter + errorbar + 背景色带）。
    
    Args:
        top_items_path: top_anomalous_parcels.json 或 top_anomalous_capabilities.json 路径
        output_fig: 输出图像路径
        level: "parcel" 或 "capability"
        p_threshold: 显著性阈值
        title: 图形标题
        only_significant: 是否仅绘制显著条目
        extra_non_top20_k: 额外包含多少条 `in_top_20=false` 的条目（默认 0，表示完全不包含）
        top_k: 在已筛选的条目中，分别在「Truthful > Hallucinated」（正值）和
               「Hallucinated > Truthful」（负值）两组中各自保留前 k 个（0 或小于等于 0 表示不截断）
        group_high_label: 高激活组标签（如 Hallucinated / Incorrect），用于图例和 Y 轴
        group_low_label: 低激活组标签（如 Truthful / Correct）
        use_cached_data: 是否使用缓存的中间数据文件（默认 True，如果缓存存在则使用）
        cached_data_path: 缓存数据文件路径（如果为 None，将根据 output_fig 自动生成）
        save_cached_data: 是否保存中间数据到缓存文件（默认 True）
        force_recompute: 是否强制重新计算（忽略缓存，默认 False）
    """
    # 根据 level 确定字段名和术语
    if level == "capability":
        id_field = "capability_id"
        name_field = "capability_name"
        item_term = "Capability"
        item_term_lower = "capability"
    else:  # parcel
        id_field = "parcel_id"
        name_field = "function_name"
        item_term = "Parcel"
        item_term_lower = "parcel"
    
    # 确定 level_name（用于构建缓存路径）
    level_name = f"{item_term_lower}_level"
    
    # 确定缓存数据路径
    if cached_data_path is None:
        # 根据 output_fig 自动生成缓存路径
        # 例如：figures/dataset/parcel_level/bar_graph.pdf -> data/dataset/parcel_level/plot_data.json
        fig_parts = output_fig.parts
        if "figures" in fig_parts:
            fig_idx = fig_parts.index("figures")
            dataset_name = fig_parts[fig_idx + 1] if fig_idx + 1 < len(fig_parts) else "unknown"
            level_name_from_path = fig_parts[fig_idx + 2] if fig_idx + 2 < len(fig_parts) else level_name
            # 构建缓存路径：将 figures 替换为 data
            cache_dir = output_fig.parent.parent.parent / "data" / dataset_name / level_name_from_path
            cached_data_path = cache_dir / "plot_data.json"
        else:
            # 如果无法从路径推断，使用默认路径
            cache_dir = output_fig.parent.parent.parent / "data" / level_name
            cached_data_path = cache_dir / "plot_data.json"
    
    # ---- 1. 加载数据 ----
    # 如果强制重写，则忽略缓存；否则如果使用缓存且缓存存在，则从缓存读取
    should_use_cache = use_cached_data and not force_recompute and cached_data_path.exists()
    
    if should_use_cache:
        # 从缓存文件读取
        print(f"[INFO] 从缓存文件读取数据: {cached_data_path}")
        try:
            cached_data = load_json(cached_data_path)
            records = cached_data.get("records", [])
            if not records:
                raise ValueError(f"缓存文件格式异常或为空: {cached_data_path}")
            print(f"[INFO] 成功从缓存加载 {len(records)} 条记录")
        except Exception as e:
            print(f"[WARN] 读取缓存文件失败: {e}，将从原始文件重新计算")
            should_use_cache = False
    
    if not should_use_cache:
        # 从原始文件读取并处理
        if top_items_path.name == "dummy" or not top_items_path.exists():
            raise ValueError(f"缓存文件不存在 ({cached_data_path})，必须提供有效的 --top_parcels_json 参数")
        data = load_json(top_items_path)
        if not isinstance(data, list):
            raise ValueError(f"top_anomalous_{item_term_lower}s.json 格式异常（应为 list）：{top_items_path}")

        # 提取有效记录，并将差值从
        #   (Hallucinated - Truthful)
        # 转换为
        #   (Truthful - Hallucinated)
        # 同时对 CI 做相应变换：
        #   new_diff     = -diff
        #   new_ci_lower = -ci_upper
        #   new_ci_upper = -ci_lower
        # 同时根据 in_top_20 标签控制是否选择 `in_top_20 = False` 的条目
        # 默认行为：仅使用 in_top_20=True（或未提供该字段）的条目；
        # 如果 extra_non_top20_k > 0，则按 |activation_diff| 大小选取指定数量的 in_top_20=False 条目一同绘制。
        main_records: List[Dict[str, Any]] = []        # in_top_20=True 或缺失
        extra_candidates: List[Dict[str, Any]] = []    # in_top_20=False
        for item in data:
            act_diff_raw = float(item.get("activation_diff", 0.0))
            ci_lower_raw = item.get("activation_diff_ci_lower", None)
            ci_upper_raw = item.get("activation_diff_ci_upper", None)
            p_val = float(item.get("p_value", 1.0))
            is_sig = bool(item.get("is_significant", False))
            t_stat = item.get("t_stat", None)
            # in_top_20 标记：缺失时视为 True（兼容 capability 等无此字段的情况）
            in_top_20 = bool(item.get("in_top_20", True))

            # 如果没有 CI，尝试从 t_stat 估算（假设 t_stat = diff / SE，SE = diff / t_stat）
            # 95% CI 约等于 diff ± 1.96 * SE
            if ci_lower_raw is None or ci_upper_raw is None:
                if t_stat is not None and abs(float(t_stat)) > 1e-6:
                    se_estimate = abs(act_diff_raw / float(t_stat))
                    ci_lower_raw = act_diff_raw - 1.96 * se_estimate
                    ci_upper_raw = act_diff_raw + 1.96 * se_estimate
                    print(f"[INFO] {item_term} {item.get(id_field, '?')} 使用估算 CI（基于 t_stat）")
                else:
                    # 如果连 t_stat 都没有，使用 activation_diff 的 20% 作为误差范围
                    margin = abs(act_diff_raw) * 0.2
                    ci_lower_raw = act_diff_raw - margin
                    ci_upper_raw = act_diff_raw + margin
                    print(f"[WARN] {item_term} {item.get(id_field, '?')} 缺少 CI 和 t_stat，使用估算值")

            if only_significant and (not is_sig or p_val >= p_threshold):
                continue

            # 差值和 CI 翻转符号（Truthful - Hallucinated）
            act_diff = -act_diff_raw
            ci_lower = -float(ci_upper_raw)
            ci_upper = -float(ci_lower_raw)

            raw_label = str(item.get(name_field, f"{item_term} {item[id_field]}"))
            # 不再对名称进行长度截断，完整展示标签
            clean_label = sanitize_text(raw_label)

            record = {
                "item_id": int(item[id_field]),
                "label": clean_label,
                "activation_diff": act_diff,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "p_value": p_val,
                "is_significant": is_sig,
                "in_top_20": in_top_20,
            }

            # 分类存放：默认仅使用 in_top_20=True 或缺失；in_top_20=False 作为候选
            if in_top_20:
                main_records.append(record)
            else:
                extra_candidates.append(record)

        # 额外选取若干 in_top_20=False 的条目（按 |activation_diff| 从大到小）
        if extra_non_top20_k > 0 and extra_candidates:
            extra_sorted = sorted(
                extra_candidates,
                key=lambda x: abs(x["activation_diff"]),
                reverse=True,
            )
            selected_extra = extra_sorted[:extra_non_top20_k]
            main_records.extend(selected_extra)

        records = main_records
        if not records:
            raise ValueError(f"没有可绘制的 {item_term} 记录。")
        
        # 按 |activation_diff| 从高到低排序（Truthful - Hallucinated），但保留原始符号和值
        records.sort(key=lambda x: abs(x["activation_diff"]), reverse=True)
        
        # 保存中间数据到缓存文件
        if save_cached_data:
            cached_data_path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "records": records,
                "metadata": {
                    "level": level,
                    "item_term": item_term,
                    "p_threshold": p_threshold,
                    "only_significant": only_significant,
                    "extra_non_top20_k": extra_non_top20_k,
                    "top_k_used_for_plot_per_sign": top_k,
                    "top_k_pos": top_k_pos,
                    "top_k_neg": top_k_neg,
                    "group_high_label": group_high_label,
                    "group_low_label": group_low_label,
                    "source_file": str(top_items_path),
                }
            }
            with cached_data_path.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 中间数据已保存到: {cached_data_path}")

    # 如果指定 top_k / top_k_pos / top_k_neg，则在排序后的记录中分别对正负两组各自保留前 k 个
    # 优先使用 top_k_pos / top_k_neg；如果为 0，则回退到统一的 top_k
    k_pos = int(top_k_pos) if isinstance(top_k_pos, int) else 0
    k_neg = int(top_k_neg) if isinstance(top_k_neg, int) else 0
    if (not isinstance(top_k, int)) or top_k < 0:
        top_k = 0
    if k_pos <= 0:
        k_pos = top_k
    if k_neg <= 0:
        k_neg = top_k

    if (isinstance(k_pos, int) and k_pos > 0) or (isinstance(k_neg, int) and k_neg > 0):
        pos_records = [r for r in records if r["activation_diff"] >= 0]
        neg_records = [r for r in records if r["activation_diff"] < 0]
        orig_pos = len(pos_records)
        orig_neg = len(neg_records)
        if isinstance(k_pos, int) and k_pos > 0 and orig_pos > k_pos:
            pos_records = pos_records[:k_pos]
        if isinstance(k_neg, int) and k_neg > 0 and orig_neg > k_neg:
            # 保留负值部分前 k_neg 个，并按你之前的需求反转顺序
            neg_records = neg_records[:k_neg][::-1]
        print(
            f"[INFO] 按符号分别截断 top_k: top_k={top_k}, "
            f"top_k_pos={k_pos}, top_k_neg={k_neg}, "
            f"原始正/负记录数=({orig_pos}, {orig_neg})，"
            f"截断后=({len(pos_records)}, {len(neg_records)})"
        )
        # 保持「正值在前、负值在后」的顺序
        records = pos_records + neg_records

    n = len(records)
    labels = [r["label"] for r in records]
    act_diffs = np.array([r["activation_diff"] for r in records])
    ci_lowers = np.array([r["ci_lower"] for r in records])
    ci_uppers = np.array([r["ci_upper"] for r in records])
    p_values = np.array([r["p_value"] for r in records])
    is_sigs = np.array([r["is_significant"] for r in records])

    # 误差线：从 activation_diff 到 CI 的距离
    err_lower = act_diffs - ci_lowers  # 下方距离
    err_upper = ci_uppers - act_diffs  # 上方距离
    yerr = np.array([err_lower, err_upper])

    # ---- 2. 创建图形 ----
    fig_w = float(CONFIG["figure_width_inch"])
    fig_h = float(CONFIG["figure_height_inch"])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_pos = np.arange(n)

    # ---- 3. 绘制背景渐变色带 ----
    # 为 capability 级单独调整 y 轴范围，避免极端高值拉伸导致其余点挤在一起
    if level == "capability":
        data_min = float(min(ci_lowers.min(), act_diffs.min(), 0.0))
        data_max = float(max(ci_uppers.max(), act_diffs.max(), 0.0))
        span = data_max - data_min
        if span <= 0:
            span = max(abs(data_min), abs(data_max), 1.0)
            data_min = -span * 0.5
            data_max = span * 0.5
        pad = span * 0.1
        y_min = data_min - pad
        y_max = data_max + pad
    else:
        y_min = min(ci_lowers.min(), act_diffs.min()) * 1.3
        y_max = max(ci_uppers.max(), act_diffs.max()) * 1.3
        # 确保包含 0
        if y_min > 0:
            y_min = -abs(y_max) * 0.2
        if y_max < 0:
            y_max = abs(y_min) * 0.2

    # 创建渐变背景
    gradient_resolution = 256
    gradient = np.linspace(0, 1, gradient_resolution).reshape(-1, 1)

    # 自定义颜色映射：红(底) -> 黄(中) -> 绿(顶)
    cmap_colors = [
        mcolors.to_rgba(BAND_LOW_COLOR),
        mcolors.to_rgba(BAND_MID_COLOR),
        mcolors.to_rgba(BAND_HIGH_COLOR),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_bg", cmap_colors, N=gradient_resolution)

    ax.imshow(
        gradient[::-1],
        extent=[-0.7, n - 0.3, y_min, y_max],
        aspect="auto",
        cmap=cmap,
        alpha=0.35,
        zorder=0,
    )

    # ---- 4. 绘制零线 ----
    ax.axhline(y=0, color="#888888", linewidth=0.5, linestyle="-", zorder=1)

    # ---- 5. 绘制误差线 + scatter 散点 ----
    colors = []
    edge_colors = []
    for r in records:
        # 现在正值表示 Truthful > Hallucinated
        if r["activation_diff"] >= 0:
            colors.append(COLOR_NEG)       # Truthful 更高 -> 蓝色
            edge_colors.append(COLOR_NEG_EDGE)
        else:
            colors.append(COLOR_POS)       # Hallucinated 更高 -> 暖色
            edge_colors.append(COLOR_POS_EDGE)

    # 误差线
    ax.errorbar(
        x_pos, act_diffs,
        yerr=yerr,
        fmt="none",
        ecolor=COLOR_ERRORBAR,
        elinewidth=float(CONFIG["errorbar_lw"]),
        capsize=float(CONFIG["errorbar_capsize"]),
        capthick=0.8,
        zorder=2,
    )

    # 散点
    ax.scatter(
        x_pos, act_diffs,
        s=float(CONFIG["marker_size"]),
        c=colors,
        edgecolors=edge_colors,
        linewidths=float(CONFIG["marker_edge_lw"]),
        zorder=3,
        clip_on=False,
    )

    # ---- 6. 显著性标注 ----
    for i, r in enumerate(records):
        if r["is_significant"] and r["p_value"] < p_threshold:
            # 在圆点上方标注 *
            y_offset = ci_uppers[i] + (y_max - y_min) * 0.02
            if r["activation_diff"] < 0:
                y_offset = ci_lowers[i] - (y_max - y_min) * 0.02
            # 使用更细致的标注
            sig_marker = "*"
            if r["p_value"] < 0.001:
                sig_marker = "***"
            elif r["p_value"] < 0.01:
                sig_marker = "**"
            ax.text(
                i, y_offset, sig_marker,
                ha="center", va="bottom" if r["activation_diff"] >= 0 else "top",
                fontsize=float(CONFIG["font_annotation"]),
                fontweight="bold",
                color="#333333",
                zorder=5,
            )

    # ---- 7. 坐标轴与标签 ----
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        fontsize=float(CONFIG["font_tick"]),
        fontfamily=CONFIG["font_family"],
    )
    ax.set_ylabel(
        f"Activation difference\n({group_low_label} - {group_high_label})",
        fontsize=float(CONFIG["font_axis_label"]),
        fontfamily=CONFIG["font_family"],
    )
    ax.set_xlabel(
        f"{item_term} (functional module)" if level == "parcel" else f"{item_term}",
        fontsize=float(CONFIG["font_axis_label"]),
        fontfamily=CONFIG["font_family"],
    )

    # Y 轴 tick
    ax.tick_params(axis="y", labelsize=float(CONFIG["font_tick"]))
    ax.tick_params(axis="x", length=2, width=0.4)
    ax.tick_params(axis="y", length=2, width=0.4)

    # 边框样式（Nature 风格：仅保留左和下）
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(y_min, y_max)

    # ---- 8. 右侧标注色带标签 ----
    ax.text(
        n - 0.1, y_max * 0.85, "High",
        fontsize=float(CONFIG["font_annotation"]),
        fontfamily=CONFIG["font_family"],
        color="#4A8C3F",
        fontweight="bold",
        ha="left", va="center",
        zorder=5,
    )
    ax.text(
        n - 0.1, y_min * 0.85, "Low",
        fontsize=float(CONFIG["font_annotation"]),
        fontfamily=CONFIG["font_family"],
        color="#C25A5A",
        fontweight="bold",
        ha="left", va="center",
        zorder=5,
    )

    # ---- 9. 标题 ----
    if title:
        ax.set_title(
            title,
            fontsize=float(CONFIG["font_title"]),
            fontfamily=CONFIG["font_family"],
            fontweight="bold",
            pad=6,
        )

    # ---- 10. 图例（紧凑，放在图内右上角） ----
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=COLOR_NEG,
            markeredgecolor=COLOR_NEG_EDGE,
            markeredgewidth=0.4,
            markersize=3.5,
            label=f"{group_low_label} > {group_high_label}",
        ),
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=COLOR_POS,
            markeredgecolor=COLOR_POS_EDGE,
            markeredgewidth=0.4,
            markersize=3.5,
            label=f"{group_high_label} > {group_low_label}",
        ),
        Line2D(
            [0], [0],
            color=COLOR_ERRORBAR,
            linewidth=0.7,
            label="95% CI",
        ),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        framealpha=0.92,
        fontsize=float(CONFIG["font_legend"]),
        handlelength=1.2,
        handletextpad=0.3,
        labelspacing=0.25,
        borderpad=0.3,
        columnspacing=0.8,
    )
    leg.get_frame().set_linewidth(0.3)

    # ---- 12. 保存（PDF 和 SVG） ----
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存 PDF 格式
    fig.savefig(
        output_fig,
        format="pdf",
        dpi=int(CONFIG["dpi"]),
        bbox_inches="tight",
        pad_inches=0.05,
    )
    print(f"[INFO] PDF 条形图已保存到: {output_fig}")
    
    # 保存 SVG 格式（将 .pdf 扩展名替换为 .svg）
    output_svg = output_fig.with_suffix(".svg")
    fig.savefig(
        output_svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.05,
    )
    print(f"[INFO] SVG 条形图已保存到: {output_svg}")
    
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 Parcel/Capability 级 activation_diff 条形图（Nature 风格）"
    )
    parser.add_argument(
        "--top_parcels_json",
        type=str,
        required=False,
        default=None,
        help="top_anomalous_parcels.json 或 top_anomalous_capabilities.json 路径（使用缓存数据时可省略）",
    )
    parser.add_argument(
        "--output_fig",
        type=str,
        required=True,
        help="图像输出路径（.pdf，将自动生成同名的 .svg 文件）",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="parcel",
        choices=["parcel", "capability"],
        help="级别类型：parcel 或 capability（默认: parcel）",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="图形标题（可选）",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=CONFIG["p_threshold"],
        help=f"显著性阈值（默认: {CONFIG['p_threshold']}）",
    )
    parser.add_argument(
        "--only_significant",
        action="store_true",
        help="仅绘制显著的条目（is_significant=true 且 p < p_threshold）",
    )
    parser.add_argument(
        "--extra_non_top20_k",
        type=int,
        default=0,
        help="额外包含多少条 `in_top_20=false` 的条目（默认 0，表示不包含）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="在已筛选的条目中仅保留前 k 个用于绘图（0 或小于等于 0 表示不截断）",
    )
    parser.add_argument(
        "--top_k_pos",
        type=int,
        default=0,
        help="正 activation_diff（Truthful > Hallucinated）一侧保留的前 k 个（0 表示回退到 --top_k 设置）",
    )
    parser.add_argument(
        "--top_k_neg",
        type=int,
        default=0,
        help="负 activation_diff（Hallucinated > Truthful）一侧保留的前 k 个（0 表示回退到 --top_k 设置）",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="若 output_fig 已存在则跳过绘图（适合批量处理）",
    )
    parser.add_argument(
        "--project_type",
        type=str,
        default="hallucination",
        choices=["hallucination", "fairness_bias"],
        help="项目类型：hallucination 或 fairness_bias（用于自动设置默认标签）",
    )
    parser.add_argument(
        "--group_high_label",
        type=str,
        default=None,
        help="高激活组标签，用于 Y 轴和图例（如 Hallucinated / Incorrect）。如未指定，将根据 project_type 自动设置",
    )
    parser.add_argument(
        "--group_low_label",
        type=str,
        default=None,
        help="低激活组标签，用于 Y 轴和图例（如 Truthful / Correct）。如未指定，将根据 project_type 自动设置",
    )
    parser.add_argument(
        "--no_use_cache",
        action="store_true",
        help="不使用缓存的中间数据文件（默认使用缓存，如果存在）",
    )
    parser.add_argument(
        "--cached_data_path",
        type=str,
        default=None,
        help="缓存数据文件路径（如果未指定，将根据 output_fig 自动生成）",
    )
    parser.add_argument(
        "--no_save_cache",
        action="store_true",
        help="不保存中间数据到缓存文件",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="强制重新计算（忽略缓存，从原始文件重新读取并处理）",
    )
    args = parser.parse_args()
    
    # 如果未指定标签，根据 project_type 设置默认值
    if args.group_high_label is None:
        if args.project_type == "fairness_bias":
            args.group_high_label = "Incorrect"
        else:
            args.group_high_label = "Hallucinated"
    
    if args.group_low_label is None:
        if args.project_type == "fairness_bias":
            args.group_low_label = "Correct"
        else:
            args.group_low_label = "Truthful"
    
    return args


def main() -> None:
    args = parse_args()

    # 验证参数：如果强制重写或不使用缓存，则需要原始文件
    use_cached_data = not args.no_use_cache  # 默认 True
    if args.force_recompute or args.no_use_cache:
        if not args.top_parcels_json:
            raise ValueError("必须提供 --top_parcels_json 参数（强制重写或不使用缓存时需要原始文件）")

    top_items_path = Path(args.top_parcels_json).resolve() if args.top_parcels_json else None
    output_fig = Path(args.output_fig).resolve()
    cached_data_path = Path(args.cached_data_path).resolve() if args.cached_data_path else None

    # 检测结果文件是否已存在（用户要求的功能）
    output_svg = output_fig.with_suffix(".svg")
    if args.skip_if_exists and output_fig.exists() and output_svg.exists():
        print(f"[INFO] 结果已存在，跳过绘图: {output_fig} 和 {output_svg}")
        return

    # 如果使用缓存数据但未指定 top_parcels_json，检查缓存是否存在
    # 如果缓存存在，使用占位符；如果缓存不存在，需要原始文件
    if use_cached_data and not args.force_recompute and top_items_path is None:
        if cached_data_path and cached_data_path.exists():
            top_items_path = Path("dummy")  # 占位符，不会被使用
        else:
            raise ValueError(f"缓存文件不存在 ({cached_data_path})，必须提供 --top_parcels_json 参数")

    try:
        plot_bar_graph(
            top_items_path=top_items_path,
            output_fig=output_fig,
            level=args.level,
            p_threshold=args.p_threshold,
            title=args.title,
            only_significant=args.only_significant,
            extra_non_top20_k=args.extra_non_top20_k,
            top_k=args.top_k,
            top_k_pos=args.top_k_pos,
            top_k_neg=args.top_k_neg,
            group_high_label=args.group_high_label,
            group_low_label=args.group_low_label,
            use_cached_data=use_cached_data,
            cached_data_path=cached_data_path,
            save_cached_data=not args.no_save_cache,
            force_recompute=args.force_recompute,
        )
    except Exception as e:
        print(f"[ERROR] 绘图过程中出现异常: {e}")
        raise


if __name__ == "__main__":
    main()
