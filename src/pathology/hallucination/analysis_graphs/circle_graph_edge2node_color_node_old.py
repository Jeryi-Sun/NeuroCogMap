#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel-level circle graph (edge → node) – Nature-ish layout.

根据异常 edge 选择节点：
- edge 来自 anomalous_connections.json；
- node 的 Δactivation 来自 parcel_level_analysis_complete.json.activation_analysis.activation_diff；
- 所有点按单圈布局绘制，并带有名称（外部射线 + 防重叠标签）。

改动：
- 可控圆环尺寸（ring_radius）
- 外部射线（leader line / radial ray）指向标签
- 简单力导向防重叠（角度 repel）
"""

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# 重新加载字体管理器以确保识别最新字体（用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass

CONFIG: Dict[str, Any] = {
    "figure_width_inch": 7.0,
    "figure_height_inch": 6.0,
    "dpi": 450,
    "font_family": "Arial",
    "font_title": 9.0,
    "font_node_label": 7.0,
    "font_legend": 7.0,
    "node_size_min": 80.0,
    "node_size_max": 500.0,
    "node_edge_lw": 0.8,
    "edge_width_min": 0.5,
    "edge_width_max": 2.5,
    "edge_alpha_min": 0.25,
    "edge_alpha_max": 0.65,
    "default_p_threshold": 0.05,
    "default_max_edges": 200,

    # ===== 新增：圆环/标签/射线 =====
    "ring_radius": 1.1,          # 圆环半径（原来 1.0），调小给外部标签空间
    "circle_lw": 1.2,
    "circle_color": "#CCCCCC",

    "label_radius": 1.28,         # 标签放置半径（> ring_radius）
    "leader_line_lw": 0.55,
    "leader_line_color": "#888888",
    "leader_line_alpha": 0.55,
    "leader_dot_size": 6.0,       # 射线起点小点（散点 size）
    "leader_start_pad": 0.03,     # 射线从 ring 外再往外一点开始
    "leader_end_pad": 0.03,       # 射线到 label 半径内侧一点结束

    "label_min_angular_gap": 0.22,
    "label_repel_iter": 110,
    "label_repel_strength": 0.32,

    "wrap_width": 18,             # 标签自动换行宽度
    "wrap_max_lines": 2,
}

plt.rcParams["font.family"] = CONFIG["font_family"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False

COLOR_NODE_POS = "#F7DC7C"
COLOR_NODE_NEG = "#579FCA"
COLOR_EDGE_POS = "#B3B3B3"
COLOR_EDGE_NEG = "#4D4D4D"


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ch in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        text = text.replace(ch, "-")
    return text


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_with_clipping(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    if values.size == 0:
        return values
    abs_vals = np.abs(values.astype(float))
    vmin = np.percentile(abs_vals, 5.0)
    vmax = np.percentile(abs_vals, 95.0)
    if vmax <= vmin:
        return np.full_like(abs_vals, (out_min + out_max) / 2.0)
    clipped = np.clip(abs_vals, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin)
    return out_min + norm * (out_max - out_min)


def build_graph_data_edge2node(
    anomalous_edges_path: Path,
    analysis_complete_path: Path,
    p_threshold: float,
    max_edges: int,
    edge_selection: str = "abs",
    level: str = "parcel",
) -> Dict[str, Any]:
    """
    构建 edge→node 圆图所需的数据。

    支持两种 level：
    - parcel: 使用 anomalous_connections 中的 parcel_i / parcel_j，标签字段为 function_name
    - capability: 使用 capability_i / capability_j，标签字段为 capability_name
    """
    # 根据 level 选择字段与术语
    # capability 级别：anomalous_capability_connections 中节点使用 "name"（与 bar_graph 一致的可读名称）
    if level == "capability":
        node_i_key = "capability_i"
        node_j_key = "capability_j"
        label_field = "capability_name"
        label_fallback_field = "name"  # 实际 JSON 中多为 "name"
        item_term = "Capability"
    else:
        node_i_key = "parcel_i"
        node_j_key = "parcel_j"
        label_field = "function_name"
        label_fallback_field = None
        item_term = "Parcel"

    data = _load_json(anomalous_edges_path)
    root = data.get("anomalous_connections", {})
    pos_list = root.get("pos_connections", [])
    neg_list = root.get("neg_connections", [])

    edges_raw: List[Dict[str, Any]] = []
    for item in pos_list + neg_list:
        try:
            node_i = item[node_i_key]
            node_j = item[node_j_key]
            nid_i = int(node_i["id"])
            nid_j = int(node_j["id"])
            w = float(item["connectivity_diff"])
            p_val = float(item["p_value"])
            is_sig = bool(item.get("is_significant", False))
        except Exception as e:
            print(f"[WARN] 解析 {item_term} edge 失败: {e}")
            continue
        if (not is_sig) or p_val >= p_threshold:
            continue
        edges_raw.append(
            {
                "src_id": nid_i,
                "dst_id": nid_j,
                "weight": w,
                "p_value": p_val,
                "is_significant": is_sig,
                "abs_diff": abs(w),
                "node_i": node_i,
                "node_j": node_j,
            }
        )

    if not edges_raw:
        raise ValueError("没有满足显著性阈值的异常连接。")

    # 根据 edge_selection 超参数选择 edge：
    # - "abs": 按绝对值整体排序，取前 max_edges（默认行为）
    # - "half_signed": 分别从最正向与最负向中各取约一半
    edges_sorted_abs = sorted(edges_raw, key=lambda e: e["abs_diff"], reverse=True)
    if edge_selection == "half_signed":
        pos_edges = [e for e in edges_raw if e["weight"] > 0]
        neg_edges = [e for e in edges_raw if e["weight"] < 0]
        pos_sorted = sorted(pos_edges, key=lambda e: e["abs_diff"], reverse=True)
        neg_sorted = sorted(neg_edges, key=lambda e: e["abs_diff"], reverse=True)

        half = max_edges // 2
        pos_pick = min(half, len(pos_sorted))
        neg_pick = min(half, len(neg_sorted))

        selected: List[Dict[str, Any]] = []
        selected.extend(pos_sorted[:pos_pick])
        selected.extend(neg_sorted[:neg_pick])

        # 如果一侧数量不足，剩余 quota 用整体绝对值最大的 edge 补齐
        if len(selected) < max_edges:
            remaining = [e for e in edges_sorted_abs if e not in selected]
            selected.extend(remaining[: max_edges - len(selected)])

        edges_raw = selected
    else:
        # 默认：整体绝对值最大的 max_edges
        edges_raw = edges_sorted_abs[:max_edges]

    node_ids = sorted(
        {int(e["src_id"]) for e in edges_raw}
        | {int(e["dst_id"]) for e in edges_raw}
    )

    complete = _load_json(analysis_complete_path)
    act = np.array(complete["activation_analysis"]["activation_diff"], dtype=float)

    # 与 bar_graph 一致：上游为 (High - Low)，取负为 (Low - High) 再绘图，使正值表示「在 Correct/Truthful 侧更高」
    act_display = -act

    nodes: List[Dict[str, Any]] = []
    for nid in node_ids:
        if nid < 0 or nid >= act.shape[0]:
            continue
        label_raw = None
        for e in edges_raw:
            node_obj = None
            if int(e["node_i"]["id"]) == nid:
                node_obj = e["node_i"]
            elif int(e["node_j"]["id"]) == nid:
                node_obj = e["node_j"]
            if node_obj is not None:
                label_raw = node_obj.get(label_field)
                if (label_raw is None or (isinstance(label_raw, str) and label_raw.strip() == "")) and label_fallback_field:
                    label_raw = node_obj.get(label_fallback_field)
                break
        if label_raw is None or (isinstance(label_raw, str) and label_raw.strip() == ""):
            label_raw = f"{item_term} {nid}"
        label = sanitize_text(str(label_raw).replace("**", "").strip())
        nodes.append(
            {
                "node_id": int(nid),
                "label": label,
                "activation_diff": float(act_display[nid]),
            }
        )

    # 边权同样取负：上游 connectivity_diff = (High - Low)，取负后正值表示「在 Correct/Truthful 侧连接更强」
    edges = [
        {
            "src_id": int(e["src_id"]),
            "dst_id": int(e["dst_id"]),
            "weight": -float(e["weight"]),
            "p_value": float(e["p_value"]),
            "is_significant": bool(e["is_significant"]),
        }
        for e in edges_raw
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "level": level,
            "item_term": item_term,
        },
    }


def draw_chord(ax, x0, y0, x1, y1, lw, color, alpha, w_sign):
    cx, cy = 0.0, 0.0
    curvature = 0.55
    c1 = (x0 * (1 - curvature) + cx * curvature, y0 * (1 - curvature) + cy * curvature)
    c2 = (x1 * (1 - curvature) + cx * curvature, y1 * (1 - curvature) + cy * curvature)
    path = MplPath(
        [(x0, y0), c1, c2, (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    # 取负后：正=Correct 更强，负=Incorrect 更强；实线=Correct，虚线=Incorrect
    linestyle = "-" if w_sign > 0 else (0, (3, 3))
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


def _repel_angles(
    angles: np.ndarray,
    min_angular_gap: float,
    n_iter: int,
    push_strength: float,
) -> np.ndarray:
    """在圆周角度上做简单斥力，减少标签重叠（角度域）。"""
    adjusted = (angles.copy().astype(float)) % (2 * math.pi)
    n = len(adjusted)
    if n <= 1:
        return adjusted

    for _ in range(int(n_iter)):
        order = np.argsort(adjusted)
        for k in range(n):
            i = order[k]
            j = order[(k + 1) % n]
            diff = (adjusted[j] - adjusted[i]) % (2 * math.pi)
            if diff < min_angular_gap:
                push = (min_angular_gap - diff) / 2.0 * float(push_strength)
                adjusted[i] -= push
                adjusted[j] += push
        adjusted = adjusted % (2 * math.pi)

    return adjusted


def _wrap_label(text: str) -> str:
    t = sanitize_text(text).strip()
    if len(t) <= 26:
        return t
    wrapped = textwrap.wrap(t, width=int(CONFIG["wrap_width"]))
    wrapped = wrapped[: int(CONFIG["wrap_max_lines"])]
    if len(wrapped) >= int(CONFIG["wrap_max_lines"]) and len(textwrap.wrap(t, width=int(CONFIG["wrap_width"]))) > int(CONFIG["wrap_max_lines"]):
        wrapped[-1] = wrapped[-1].rstrip() + "..."
    return "\n".join(wrapped)


def plot_circle_graph(
    graph_data: Dict[str, Any],
    output_fig: Path,
    title: str = "",
    group_high_label: str = "Hallucinated",
    group_low_label: str = "Truthful",
) -> None:
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    meta = graph_data.get("meta", {})
    level = meta.get("level", "parcel")
    item_term = meta.get("item_term", "Parcel" if level == "parcel" else "Capability")
    n = len(nodes)
    if n == 0:
        raise ValueError("nodes 为空。")

    # ===== 环形布局（可控半径）=====
    R = float(CONFIG["ring_radius"])
    thetas = np.array([math.pi / 2 - i * 2 * math.pi / n for i in range(n)])

    for idx, node in enumerate(nodes):
        th = float(thetas[idx])
        node["theta"] = th
        node["x"] = float(R * math.cos(th))
        node["y"] = float(R * math.sin(th))

    # ===== 视觉映射 =====
    act_diffs = np.array([nd["activation_diff"] for nd in nodes], dtype=float)
    node_sizes = normalize_with_clipping(act_diffs, CONFIG["node_size_min"], CONFIG["node_size_max"])

    weights = np.array([e["weight"] for e in edges], dtype=float)
    edge_widths = normalize_with_clipping(weights, CONFIG["edge_width_min"], CONFIG["edge_width_max"])
    edge_alphas = normalize_with_clipping(weights, CONFIG["edge_alpha_min"], CONFIG["edge_alpha_max"])

    id2idx = {int(nd["node_id"]): i for i, nd in enumerate(nodes)}

    fig, ax = plt.subplots(figsize=(CONFIG["figure_width_inch"], CONFIG["figure_height_inch"]))
    ax.set_aspect("equal")
    ax.axis("off")

    # ===== 外圈 =====
    ax.add_patch(
        plt.Circle(
            (0, 0),
            R,
            fill=False,
            linewidth=float(CONFIG["circle_lw"]),
            edgecolor=str(CONFIG["circle_color"]),
            zorder=1,
        )
    )

    # ===== 边 =====
    for e, lw, alpha in zip(edges, edge_widths, edge_alphas):
        pid_i = int(e["src_id"])
        pid_j = int(e["dst_id"])
        if pid_i not in id2idx or pid_j not in id2idx:
            continue
        ni = nodes[id2idx[pid_i]]
        nj = nodes[id2idx[pid_j]]
        x0, y0 = ni["x"], ni["y"]
        x1, y1 = nj["x"], nj["y"]
        w = float(e["weight"])
        color = COLOR_EDGE_POS if w > 0 else COLOR_EDGE_NEG
        draw_chord(ax, x0, y0, x1, y1, lw, color, float(alpha), w)

    # ===== 节点 =====
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

    # ===== 标签角度 repel + 外部射线 =====
    label_r = float(CONFIG["label_radius"])
    original_angles = (np.array([nd["theta"] for nd in nodes], dtype=float)) % (2 * math.pi)
    label_angles = _repel_angles(
        original_angles,
        min_angular_gap=float(CONFIG["label_min_angular_gap"]),
        n_iter=int(CONFIG["label_repel_iter"]),
        push_strength=float(CONFIG["label_repel_strength"]),
    )

    start_r = R + float(CONFIG["leader_start_pad"])
    end_r = label_r - float(CONFIG["leader_end_pad"])

    for idx, nd in enumerate(nodes):
        la = float(label_angles[idx])

        # 射线起点：沿着节点原角度，从 ring 外侧出发（更像“外部射线”）
        th = float(nd["theta"])
        gsx = start_r * math.cos(th)
        gsy = start_r * math.sin(th)

        # 射线终点：沿着 repel 后的标签角度，指向标签半径
        gex = end_r * math.cos(la)
        gey = end_r * math.sin(la)

        # 标签位置
        lx = label_r * math.cos(la)
        ly = label_r * math.sin(la)

        cos_val = math.cos(la)
        if cos_val > 0.15:
            ha = "left"
        elif cos_val < -0.15:
            ha = "right"
        else:
            ha = "center"

        label_text = _wrap_label(nd["label"])

        # 外部射线
        ax.plot(
            [gsx, gex], [gsy, gey],
            color=str(CONFIG["leader_line_color"]),
            linewidth=float(CONFIG["leader_line_lw"]),
            alpha=float(CONFIG["leader_line_alpha"]),
            zorder=2,
            clip_on=False,
        )
        # 起点小点（靠近节点）
        ax.scatter(
            [gsx], [gsy],
            s=float(CONFIG["leader_dot_size"]),
            color=str(CONFIG["leader_line_color"]),
            alpha=min(0.9, float(CONFIG["leader_line_alpha"]) + 0.15),
            zorder=3,
            clip_on=False,
        )

        ax.text(
            lx, ly,
            label_text,
            ha=ha, va="center",
            fontsize=float(CONFIG["font_node_label"]),
            fontfamily=str(CONFIG["font_family"]),
            zorder=6,
            clip_on=False,
        )

    # ===== 图例：仅数值取负（Correct−Incorrect），颜色与 group 对应关系由取值决定，不额外翻转图例文字 =====
    # 取负后 正值=group_low(Correct)、负值=group_high(Incorrect)，黄=正、蓝=负，故黄=group_low、蓝=group_high
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_NODE_POS,
            markeredgecolor="#555555",
            markeredgewidth=0.5,
            markersize=5.0,
            label=f"Higher activated {item_term.lower()}s in {group_low_label.lower()} examples",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_NODE_NEG,
            markeredgecolor="#555555",
            markeredgewidth=0.5,
            markersize=5.0,
            label=f"Higher activated {item_term.lower()}s in {group_high_label.lower()} examples",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_EDGE_POS,
            linewidth=1.0,
            linestyle="-",
            label=f"Stronger connectivity in {group_low_label.lower()} examples",
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_EDGE_NEG,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            label=f"Stronger connectivity in {group_high_label.lower()} examples",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=float(CONFIG["font_legend"]),
        handlelength=1.8,
        handletextpad=0.4,
        labelspacing=0.4,
        columnspacing=1.4,
    )

    if title:
        ax.set_title(sanitize_text(title), fontsize=float(CONFIG["font_title"]), pad=1)

    # ===== 视野（避免裁剪）=====
    margin = 0.55
    ax.set_xlim(-label_r - margin, label_r + margin)
    ax.set_ylim(-label_r - margin, label_r + margin)

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, format="pdf", dpi=int(CONFIG["dpi"]), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_graph_data(graph_data: Dict[str, Any], output_data: Path) -> None:
    output_data.parent.mkdir(parents=True, exist_ok=True)
    with output_data.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parcel/Capability-level circle graph (edge2node)")
    parser.add_argument("--anomalous_edges_json", type=str, required=True)
    parser.add_argument("--analysis_complete_json", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--output_fig", type=str, required=True)
    parser.add_argument("--p_threshold", type=float, default=CONFIG["default_p_threshold"])
    parser.add_argument("--max_edges", type=int, default=CONFIG["default_max_edges"])
    parser.add_argument(
        "--edge_selection",
        type=str,
        choices=["abs", "half_signed"],
        default="abs",
        help=(
            "edge 选择策略："
            "'abs' 按 |weight| 整体排序取前 max_edges（默认）；"
            "'half_signed' 从最正向和最负向中各取约一半，再按绝对值补齐。"
        ),
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["parcel", "capability"],
        default="parcel",
        help="级别类型：parcel 或 capability（默认 parcel）",
    )
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--refresh_data", action="store_true")
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
    anomalous_edges_path = Path(args.anomalous_edges_json).resolve()
    analysis_complete_path = Path(args.analysis_complete_json).resolve()
    output_data = Path(args.output_data).resolve()
    output_fig = Path(args.output_fig).resolve()

    if args.skip_if_exists and output_fig.exists() and output_data.exists():
        print(f"[INFO] 结果已存在，跳过绘图: {output_fig}")
        return

    try:
        if (not args.refresh_data) and output_data.exists():
            print(f"[INFO] 使用已有中间数据: {output_data}")
            with output_data.open("r", encoding="utf-8") as f:
                graph_data = json.load(f)
        else:
            print("[INFO] 从异常 edge 与 activation_diff 构建 edge2node 中间数据...")
            graph_data = build_graph_data_edge2node(
                anomalous_edges_path,
                analysis_complete_path,
                p_threshold=args.p_threshold,
                max_edges=args.max_edges,
                edge_selection=args.edge_selection,
                level=args.level,
            )
            save_graph_data(graph_data, output_data)

        title = (
            "Parcel-level connectome (edge-driven)"
            if args.level == "parcel"
            else "Capability-level connectome (edge-driven)"
        )
        plot_circle_graph(
            graph_data,
            output_fig,
            title=title,
            group_high_label=args.group_high_label,
            group_low_label=args.group_low_label,
        )
        print(f"[INFO] 中间数据已保存到: {output_data}")
        print(f"[INFO] 图像已保存到: {output_fig}")
    except Exception as e:
        print(f"[ERROR] 绘图过程中出现异常: {e}")
        raise


if __name__ == "__main__":
    main()
