#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Parcel 级结果绘制单圈网络图（circle graph）— Nature 风格。

功能：
1. 从 top_anomalous_parcels.json 中读取显著异常的 Parcel 作为节点；
2. 从 parcel_level_analysis_complete.json 中读取连接差异矩阵作为边；
3. 支持 traditional / pca_concate 两种连接来源；
4. 将可直接绘图的中间数据保存到 graphs/data 下；
5. 将最终图像（PDF）保存到 graphs/figures 下。

Nature 风格改进：
- 使用引导线（leader line）从节点明确指向标签，消除对应关系模糊
- 力导向标签防重叠算法
- 更精致的线条、配色和图例
- Arial 字体，≥5pt，450 dpi
"""

import argparse
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# 重新加载字体管理器以确保识别最新字体（用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass

# ------------------------
# 全局绘图配置（便于集中修改）
# ------------------------
CONFIG: Dict[str, Any] = {
    # 版式相关
    "figure_width_inch": 7.0,
    "figure_height_inch": 6.0,
    "dpi": 450,
    # 字体与字号（符合 Nature 要求）
    "font_family": "Arial",
    "font_title": 9.0,
    "font_node_label": 7.0,
    "font_legend": 7.0,
    # 节点视觉编码
    "node_size_min": 100.0,
    "node_size_max": 600.0,
    "node_edge_lw": 0.8,
    # 边视觉编码
    "edge_width_min": 0.5,
    "edge_width_max": 2.5,
    "edge_alpha_min": 0.25,
    "edge_alpha_max": 0.65,
    # 统计与筛选
    "default_p_threshold": 0.05,
    "default_max_edges": 200,
    # 标签布局
    "label_radius": 1.00,         # 标签起始径向距离
    "leader_line_lw": 0.5,        # 引导线宽度
    "leader_line_color": "#888888",
    "leader_dot_size": 2.0,       # 引导线末端小点大小
    # 圆环
    "circle_lw": 1.2,
    "circle_color": "#CCCCCC",
}

# Nature 风格相关基础设置
plt.rcParams["font.family"] = CONFIG["font_family"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False


def sanitize_text(text: str) -> str:
    """
    清理文本中的特殊连字符等，避免 Arial 缺字导致的警告。
    """
    if not isinstance(text, str):
        text = str(text)
    for ch in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        text = text.replace(ch, "-")
    return text


# 颜色方案（来自用户给定调色板）
COLOR_NODE_POS = "#F7DC7C"   # 正 activation_diff（Higher in hallucination）
COLOR_NODE_NEG = "#579FCA"   # 负 activation_diff（Lower in hallucination）
# 改成灰色
COLOR_EDGE_POS = "#B3B3B3"   # 深灰 —— 正 connectivity_diff（Stronger）
COLOR_EDGE_NEG = "#4D4D4D"   # 浅灰 —— 负 connectivity_diff（Weaker）



def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_with_clipping(
    values: np.ndarray,
    out_min: float,
    out_max: float,
    clip_low_pct: float = 5.0,
    clip_high_pct: float = 95.0,
) -> np.ndarray:
    """对绝对值进行百分位裁剪后映射到 [out_min, out_max]。"""
    if values.size == 0:
        return values
    abs_vals = np.abs(values.astype(float))
    vmin = np.percentile(abs_vals, clip_low_pct)
    vmax = np.percentile(abs_vals, clip_high_pct)
    if vmax <= vmin:
        return np.full_like(abs_vals, (out_min + out_max) / 2.0)
    clipped = np.clip(abs_vals, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)
    return out_min + norm * (out_max - out_min)


def build_graph_data(
    top_parcels_path: Path,
    analysis_complete_path: Path,
    method: str = "traditional",
    p_threshold: float = 0.05,
    max_edges: int = 300,
) -> Dict[str, Any]:
    """
    从分析结果中构建可视化所需的节点和边数据。
    """
    top_data = _load_json(top_parcels_path)
    if not isinstance(top_data, list):
        raise ValueError(f"top_anomalous_parcels.json 格式异常（应为 list）：{top_parcels_path}")

    nodes_raw = []
    for item in top_data:
        if not bool(item.get("is_significant", False)):
            continue
        if float(item.get("p_value", 1.0)) >= p_threshold:
            continue
        raw_label = str(item.get("function_name", f"Parcel {item['parcel_id']}")).replace("**", "").strip()
        clean_label = sanitize_text(raw_label)
        nodes_raw.append(
            {
                "parcel_id": int(item["parcel_id"]),
                "label": clean_label,
                "activation_diff": float(item.get("activation_diff", 0.0)),
                "p_value": float(item.get("p_value", 1.0)),
                "is_significant": True,
            }
        )

    if not nodes_raw:
        raise ValueError("在 top_anomalous_parcels.json 中未找到任何显著节点（is_significant 且 p < 阈值）。")

    complete = _load_json(analysis_complete_path)
    try:
        conn_root = complete["connectivity_analysis"][method]
    except KeyError:
        raise KeyError(f"parcel_level_analysis_complete.json 中未找到 connectivity_analysis['{method}']")

    conn_diff = np.array(conn_root["connectivity_diff"], dtype=float)
    if conn_diff.ndim != 2 or conn_diff.shape[0] != conn_diff.shape[1]:
        raise ValueError("connectivity_diff 不是方阵，无法作为连接矩阵使用。")

    conn_p_values = None
    if method == "traditional":
        conn_p_values = np.array(conn_root["connectivity_p_values"], dtype=float)
        if conn_p_values.shape != conn_diff.shape:
            raise ValueError("connectivity_p_values 形状与 connectivity_diff 不一致。")

    parcel_ids = [n["parcel_id"] for n in nodes_raw]
    n_nodes = len(parcel_ids)

    edges: List[Dict[str, Any]] = []
    for i in range(n_nodes):
        pid_i = parcel_ids[i]
        for j in range(i + 1, n_nodes):
            pid_j = parcel_ids[j]
            if pid_i >= conn_diff.shape[0] or pid_j >= conn_diff.shape[0]:
                continue
            w = float(conn_diff[pid_i, pid_j])
            if w == 0.0:
                continue

            p_val = 1.0
            is_sig = True
            if conn_p_values is not None:
                p_val = float(conn_p_values[pid_i, pid_j])
                is_sig = p_val < p_threshold
                if not is_sig:
                    continue

            edges.append(
                {
                    "src_parcel": int(pid_i),
                    "dst_parcel": int(pid_j),
                    "weight": w,
                    "p_value": p_val,
                    "is_significant": is_sig,
                }
            )

    if not edges:
        raise ValueError("在显著节点之间未找到任何符合条件的连接边。")

    edges = sorted(edges, key=lambda e: abs(e["weight"]), reverse=True)[:max_edges]

    graph_data = {
        "nodes": nodes_raw,
        "edges": edges,
        "meta": {
            "method": method,
            "p_threshold": p_threshold,
            "max_edges": max_edges,
            "n_nodes": len(nodes_raw),
            "n_edges": len(edges),
        },
    }
    return graph_data


def draw_chord(ax, x0: float, y0: float, x1: float, y1: float,
               lw: float, color: str, alpha: float, w_sign: float) -> None:
    """绘制一条从 (x0,y0) 到 (x1,y1) 的贝塞尔曲线边。"""
    cx, cy = 0.0, 0.0
    curvature = 0.55
    c1 = (x0 * (1 - curvature) + cx * curvature, y0 * (1 - curvature) + cy * curvature)
    c2 = (x1 * (1 - curvature) + cx * curvature, y1 * (1 - curvature) + cy * curvature)
    path = MplPath(
        [(x0, y0), c1, c2, (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    linestyle = (0, (3, 3)) if w_sign > 0 else "-"
    ax.add_patch(
        patches.PathPatch(
            path,
            fill=False,
            lw=float(lw),
            alpha=float(alpha),
            linestyle=linestyle,
            edgecolor=color,
            capstyle="round",
            joinstyle="round",
        )
    )


def _repel_labels(
    angles: np.ndarray,
    label_r: float,
    min_angular_gap: float = 0.18,
    n_iter: int = 80,
    push_strength: float = 0.3,
) -> np.ndarray:
    """
    简单的力导向算法：在圆周外的标签角度上施加斥力，防止标签重叠。
    返回调整后的角度数组。
    """
    adjusted = angles.copy().astype(float)
    n = len(adjusted)
    if n <= 1:
        return adjusted

    for _ in range(n_iter):
        # 按角度排序
        order = np.argsort(adjusted)
        for k in range(n):
            i = order[k]
            j = order[(k + 1) % n]
            # 计算角度差（考虑周期性）
            diff = (adjusted[j] - adjusted[i]) % (2 * math.pi)
            if diff < min_angular_gap:
                push = (min_angular_gap - diff) / 2.0 * push_strength
                adjusted[i] -= push
                adjusted[j] += push
        # 保持 [0, 2*pi) 范围
        adjusted = adjusted % (2 * math.pi)

    return adjusted


def plot_circle_graph(
    graph_data: Dict[str, Any],
    output_fig: Path,
    title: str = "",
    node_size_min: float = CONFIG["node_size_min"],
    node_size_max: float = CONFIG["node_size_max"],
    edge_width_min: float = CONFIG["edge_width_min"],
    edge_width_max: float = CONFIG["edge_width_max"],
    group_high_label: str = "Hallucinated",
    group_low_label: str = "Truthful",
) -> None:
    """根据预处理好的 graph_data 绘制 Nature 风格单圈网络图，并保存为 PDF。"""
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # 环形布局
    n = len(nodes)
    R = 1.0
    thetas = np.linspace(0, 2 * math.pi, n, endpoint=False)

    # 从顶部开始 (pi/2)，顺时针排列，使布局更美观
    thetas = np.array([math.pi / 2 - i * 2 * math.pi / n for i in range(n)])

    for idx, node in enumerate(nodes):
        node["theta"] = float(thetas[idx])
        node["x"] = float(R * math.cos(thetas[idx]))
        node["y"] = float(R * math.sin(thetas[idx]))

    # 归一化节点大小
    act_diffs = np.array([nd["activation_diff"] for nd in nodes], dtype=float)
    node_sizes = normalize_with_clipping(act_diffs, node_size_min, node_size_max)

    # 归一化边宽与透明度
    weights = np.array([e["weight"] for e in edges], dtype=float)
    edge_widths = normalize_with_clipping(weights, edge_width_min, edge_width_max)
    edge_alphas = normalize_with_clipping(
        weights,
        float(CONFIG["edge_alpha_min"]),
        float(CONFIG["edge_alpha_max"]),
    )

    # parcel_id -> index 映射
    id2idx = {int(nd["parcel_id"]): i for i, nd in enumerate(nodes)}

    # 创建图形
    fig_w = float(CONFIG["figure_width_inch"])
    fig_h = float(CONFIG["figure_height_inch"])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    # --- 绘制外圈 ---
    circle = plt.Circle(
        (0, 0), R,
        fill=False,
        linewidth=float(CONFIG["circle_lw"]),
        edgecolor=CONFIG["circle_color"],
        zorder=1,
    )
    ax.add_patch(circle)

    # --- 绘制边 ---
    for e, lw, alpha in zip(edges, edge_widths, edge_alphas):
        pid_i = int(e["src_parcel"])
        pid_j = int(e["dst_parcel"])
        if pid_i not in id2idx or pid_j not in id2idx:
            continue
        ni = nodes[id2idx[pid_i]]
        nj = nodes[id2idx[pid_j]]
        x0, y0 = ni["x"], ni["y"]
        x1, y1 = nj["x"], nj["y"]
        w = float(e["weight"])
        color = COLOR_EDGE_POS if w > 0 else COLOR_EDGE_NEG
        draw_chord(ax, x0, y0, x1, y1, lw=lw, color=color, alpha=float(alpha), w_sign=w)

    # --- 绘制节点 ---
    xs = np.array([nd["x"] for nd in nodes], dtype=float)
    ys = np.array([nd["y"] for nd in nodes], dtype=float)
    pos_mask = act_diffs >= 0

    ax.scatter(
        xs[pos_mask], ys[pos_mask],
        s=node_sizes[pos_mask],
        alpha=0.92,
        linewidths=float(CONFIG["node_edge_lw"]),
        edgecolors="#555555",
        color=COLOR_NODE_POS,
        zorder=4,
    )
    ax.scatter(
        xs[~pos_mask], ys[~pos_mask],
        s=node_sizes[~pos_mask],
        alpha=0.92,
        linewidths=float(CONFIG["node_edge_lw"]),
        edgecolors="#555555",
        color=COLOR_NODE_NEG,
        zorder=4,
    )

    # --- 标签布局：使用力导向防重叠 + 引导线 ---
    label_r = float(CONFIG["label_radius"])

    # 计算防重叠后的标签角度
    original_angles = np.array([nd["theta"] for nd in nodes], dtype=float)
    # 归一化到 [0, 2*pi)
    original_angles_norm = original_angles % (2 * math.pi)
    label_angles = _repel_labels(original_angles_norm, label_r, min_angular_gap=0.22, n_iter=100)

    for idx, nd in enumerate(nodes):
        # 节点在圆上的位置
        nx, ny = nd["x"], nd["y"]

        # 标签位置（调整后的角度，更远的半径）
        la = label_angles[idx]
        lx = label_r * math.cos(la)
        ly = label_r * math.sin(la)

        # 确定文本对齐方向
        cos_val = math.cos(la)
        if cos_val > 0.15:
            ha = "left"
        elif cos_val < -0.15:
            ha = "right"
        else:
            ha = "center"

        # 标签文本（温和换行）
        label_text = sanitize_text(nd["label"]).strip()
        if len(label_text) > 30:
            wrapped = textwrap.wrap(label_text, width=18)
            label_text = "\n".join(wrapped[:2])
            if len(wrapped) > 2:
                label_text = label_text.rstrip() + "..."

        # 绘制引导线：从节点边缘到标签位置
        # 引导线起点稍微在节点外侧
        node_r_visual = math.sqrt(node_sizes[idx] / math.pi) / 72.0 * 2.0  # 近似半径
        guide_start_r = R + 0.03
        gsx = guide_start_r * math.cos(nd["theta"])
        gsy = guide_start_r * math.sin(nd["theta"])

        # 引导线终点稍微在标签内侧
        guide_end_r = label_r - 0.03
        gex = guide_end_r * math.cos(la)
        gey = guide_end_r * math.sin(la)

        ax.plot(
            [gsx, gex], [gsy, gey],
            color=CONFIG["leader_line_color"],
            linewidth=float(CONFIG["leader_line_lw"]),
            alpha=0.5,
            zorder=2,
            clip_on=False,
        )
        # 引导线末端小点
        ax.scatter(
            [gsx], [gsy],
            s=float(CONFIG["leader_dot_size"]),
            color=CONFIG["leader_line_color"],
            alpha=0.6,
            zorder=2,
            clip_on=False,
        )

        # 绘制标签文字
        ax.text(
            lx, ly,
            label_text,
            ha=ha,
            va="center",
            fontsize=float(CONFIG["font_node_label"]),
            fontfamily=CONFIG["font_family"],
            zorder=6,
            clip_on=False,
        )

    # ========== 图例 ==========
    legend_elements = [
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=COLOR_NODE_POS,
            markeredgecolor="#555555",
            markeredgewidth=0.5,
            markersize=5.0,
            label=f"Higher Activation in {group_high_label} Group",
        ),
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=COLOR_NODE_NEG,
            markeredgecolor="#555555",
            markeredgewidth=0.5,
            markersize=5.0,
            label=f"Higher Activation in {group_low_label} Group",
        ),
        Line2D(
            [0], [0],
            color=COLOR_EDGE_NEG,
            linewidth=1.0,
            linestyle="-",
            label=f"Higher {group_low_label} connectivity",
        ),
        Line2D(
            [0], [0],
            color=COLOR_EDGE_POS,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            label=f"Higher {group_high_label} connectivity",
        ),
    ]

    leg = ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        framealpha=0.95,
        fontsize=float(CONFIG["font_legend"]),
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.4,
        columnspacing=1.5,
        borderpad=0.5,
    )
    leg.get_frame().set_linewidth(0.4)

    # --- 视野 ---
    margin = 0.55
    ax.set_xlim(-label_r - margin, label_r + margin)
    ax.set_ylim(-label_r - margin, label_r + margin)

    # --- 保存 ---
    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_fig,
        format="pdf",
        dpi=int(CONFIG["dpi"]),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def save_graph_data(graph_data: Dict[str, Any], output_data: Path) -> None:
    """将可直接绘图的节点/边数据保存为 JSON。"""
    output_data.parent.mkdir(parents=True, exist_ok=True)
    with output_data.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 Parcel 级 circle graph 网络图")
    parser.add_argument(
        "--top_parcels_json",
        type=str,
        required=True,
        help="top_anomalous_parcels.json 路径",
    )
    parser.add_argument(
        "--analysis_complete_json",
        type=str,
        required=True,
        help="parcel_level_analysis_complete.json 路径",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="中间数据输出路径（.json），位于 graphs/data 下",
    )
    parser.add_argument(
        "--output_fig",
        type=str,
        required=True,
        help="图像输出路径（.pdf），位于 graphs/figures 下",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="traditional",
        choices=["traditional", "pca_concate"],
        help="连接来源：traditional 或 pca_concate（默认 traditional）",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=CONFIG["default_p_threshold"],
        help=f"显著性阈值，用于节点和 traditional 边 (默认: {CONFIG['default_p_threshold']})",
    )
    parser.add_argument(
        "--max_edges",
        type=int,
        default=CONFIG["default_max_edges"],
        help=f"最多绘制的边数量（按 |connectivity_diff| 排序后截断，默认: {CONFIG['default_max_edges']}）",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="若 output_fig 已存在则跳过绘图（适合批量处理）",
    )
    parser.add_argument(
        "--refresh_data",
        action="store_true",
        help="若指定，则强制从原始 JSON 重建中间数据；否则优先复用已存在的中间数据",
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
        help="高激活组标签（如 Hallucinated / Incorrect）。如未指定，将根据 project_type 自动设置",
    )
    parser.add_argument(
        "--group_low_label",
        type=str,
        default=None,
        help="低激活组标签（如 Truthful / Correct）。如未指定，将根据 project_type 自动设置",
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

    top_parcels_path = Path(args.top_parcels_json).resolve()
    analysis_complete_path = Path(args.analysis_complete_json).resolve()
    output_data = Path(args.output_data).resolve()
    output_fig = Path(args.output_fig).resolve()

    try:
        # 1) 若开启 skip_if_exists 且图像与中间数据都已存在，直接跳过
        if args.skip_if_exists and output_fig.exists() and output_data.exists():
            print(f"[INFO] 结果已存在，跳过绘图: {output_fig}")
            return

        # 2) 优先复用已有中间数据（除非强制 refresh_data）
        if (not args.refresh_data) and output_data.exists():
            print(f"[INFO] 检测到已有中间数据，直接加载: {output_data}")
            with output_data.open("r", encoding="utf-8") as f:
                graph_data = json.load(f)
        else:
            print("[INFO] 构建新的中间数据（graph_data）...")
            graph_data = build_graph_data(
                top_parcels_path=top_parcels_path,
                analysis_complete_path=analysis_complete_path,
                method=args.method,
                p_threshold=args.p_threshold,
                max_edges=args.max_edges,
            )
            save_graph_data(graph_data, output_data)

        plot_circle_graph(
            graph_data,
            output_fig=output_fig,
            group_high_label=args.group_high_label,
            group_low_label=args.group_low_label,
        )
        print(f"[INFO] 中间数据已保存到: {output_data}")
        print(f"[INFO] 图像已保存到: {output_fig}")
    except Exception as e:
        # 用户要求：异常需要报告，不能静默吞掉
        print(f"[ERROR] 绘图过程中出现异常: {e}")
        raise


if __name__ == "__main__":
    main()
