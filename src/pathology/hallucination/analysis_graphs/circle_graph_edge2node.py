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

# A4 宽度 1/3：210 mm / 3 = 70 mm
A4_ONE_THIRD_WIDTH_INCH = (210.0 / 3.0) / 25.4

# 重新加载字体管理器以确保识别最新字体（用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass

CONFIG: Dict[str, Any] = {
    "figure_width_inch": A4_ONE_THIRD_WIDTH_INCH,
    "figure_height_inch": A4_ONE_THIRD_WIDTH_INCH,
    "dpi": 450,
    "font_family": "Arial",
    "font_title": 5.0,
    "font_node_label": 5.0,
    "font_legend": 4.5,
    "node_size": 70.0,           # 统一节点大小（Nature 风格不按数值区分）
    "node_edge_lw": 0.65,       # 空心节点边框 0.5–0.8 pt
    "node_fill_color": "white",  # 空心节点填充：白或极浅灰
    "node_edge_color": "#C8C8C8",
    "edge_width_min": 0.5,
    "edge_width_max": 2.0,
    "edge_alpha_min": 0.35,
    "edge_alpha_max": 0.75,
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

    # ===== 连通子图凸包强调（受 --show_hull 控制）=====
    "hull_alpha": 0.10,           # 凸包填充透明度（非常克制，Nature 风格）
    "hull_edge_lw": 0.5,          # 凸包边框线宽
    "hull_edge_alpha": 0.25,      # 凸包边框透明度
    "hull_edge_color": "#555555", # 凸包边缘：深灰虚线
    "hull_pad": 0.08,             # 凸包膨胀半径（避免贴着节点）
    "hull_min_nodes": 3,          # 最少 3 个节点才画（2 节点跳过）
}

plt.rcParams["font.family"] = CONFIG["font_family"]
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"

# 节点：浅灰空心（fill 白/极浅灰，edge #C8C8C8，无阴影/渐变）
COLOR_NODE_FILL = "white"
COLOR_NODE_EDGE = "#C8C8C8"
# 边方向（Truthful/Hallucination）用颜色；结构（M>0/M<0）用线型
COLOR_STRONGER_TRUTHFUL = "#579FCA"   # Stronger in truthful examples
COLOR_STRONGER_HALLUCINATED = "#F7DC7C"  # Pale Yellow (Nature palette), stronger in hallucinated
# 非结构模式回退：方向用灰度
COLOR_EDGE_DIRECTION_SOLID = "#4D4D4D"
COLOR_EDGE_DIRECTION_DASHED = "#888888"


def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ch in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        text = text.replace(ch, "-")
    return text


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_parcel_matrix(csv_path: Path) -> np.ndarray:
    """
    从 parcel_connection_matrix.csv 读取结构连接矩阵 M。

    约定格式：
    - 首行：空单元 + 列索引 0..N-1
    - 首列：行索引 0..N-1
    - 其余为数值矩阵
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Parcel connection matrix CSV 不存在: {csv_path}")

    try:
        raw = np.genfromtxt(str(csv_path), delimiter=",")
    except Exception as e:
        print(f"[ERROR] 读取 parcel 结构矩阵失败: {csv_path} ({e})")
        raise

    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError(
            f"parcel 结构矩阵形状异常: {raw.shape}，期望 >= (2, 2)。"
        )

    # 去掉首行首列索引，只保留数值矩阵
    mat = raw[1:, 1:]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(
            f"parcel 结构矩阵不是方阵: {mat.shape}。"
        )
    return mat.astype(float)


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


def reorder_nodes_by_edge_category(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    按边类别重排节点，便于环图分区显示：
    - positive-only: 仅连接到 weight > 0 的边
    - negative-only: 仅连接到 weight < 0 的边
    - mixed: 同时连接到正负两类边（放在两类分区交界处）
    """
    stats: Dict[int, Dict[str, float]] = {}
    for nd in nodes:
        nid = int(nd["node_id"])
        stats[nid] = {"pos_count": 0.0, "neg_count": 0.0, "strength": 0.0}

    for e in edges:
        src_id = int(e["src_id"])
        dst_id = int(e["dst_id"])
        w = float(e["weight"])
        abs_w = abs(w)

        if w > 0:
            sign_key = "pos_count"
        elif w < 0:
            sign_key = "neg_count"
        else:
            sign_key = None

        for nid in (src_id, dst_id):
            if nid not in stats:
                continue
            stats[nid]["strength"] += abs_w
            if sign_key is not None:
                stats[nid][sign_key] += 1.0

    def sort_key(nd: Dict[str, Any]) -> tuple:
        nid = int(nd["node_id"])
        st = stats[nid]
        return (-st["strength"], -(st["pos_count"] + st["neg_count"]), nid)

    pos_only: List[Dict[str, Any]] = []
    neg_only: List[Dict[str, Any]] = []
    mixed: List[Dict[str, Any]] = []
    neutral: List[Dict[str, Any]] = []

    for nd in nodes:
        nid = int(nd["node_id"])
        st = stats[nid]
        has_pos = st["pos_count"] > 0
        has_neg = st["neg_count"] > 0
        if has_pos and has_neg:
            mixed.append(nd)
        elif has_pos:
            pos_only.append(nd)
        elif has_neg:
            neg_only.append(nd)
        else:
            neutral.append(nd)

    pos_only = sorted(pos_only, key=sort_key)
    neg_only = sorted(neg_only, key=sort_key)
    mixed = sorted(mixed, key=sort_key)
    neutral = sorted(neutral, key=sort_key)

    # mixed 节点拆成两段，放在两类分区的交界处（顶部与底部）
    split_idx = (len(mixed) + 1) // 2
    mixed_top = mixed[:split_idx]
    mixed_bottom = mixed[split_idx:]

    ordered = mixed_top + pos_only + mixed_bottom + neg_only + neutral
    if len(ordered) != len(nodes):
        raise ValueError("节点重排后数量不一致，请检查分类逻辑。")
    return ordered


def build_graph_data_edge2node(
    anomalous_edges_path: Path,
    analysis_complete_path: Path,
    p_threshold: float,
    max_edges: int | None = None,
    edge_selection: str = "abs",
    level: str = "parcel",
    pos_edges: int | None = None,
    neg_edges: int | None = None,
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
    # 如果提供了 pos_edges 和 neg_edges 参数，则直接使用这些数量（忽略 max_edges）
    edges_sorted_abs = sorted(edges_raw, key=lambda e: e["abs_diff"], reverse=True)
    
    # 如果提供了 pos_edges 和 neg_edges，直接使用这些数量
    if pos_edges is not None and neg_edges is not None:
        pos_edge_list = [e for e in edges_raw if e["weight"] > 0]
        neg_edge_list = [e for e in edges_raw if e["weight"] < 0]
        pos_sorted = sorted(pos_edge_list, key=lambda e: e["abs_diff"], reverse=True)
        neg_sorted = sorted(neg_edge_list, key=lambda e: e["abs_diff"], reverse=True)
        
        pos_pick = min(pos_edges, len(pos_sorted))
        neg_pick = min(neg_edges, len(neg_sorted))
        
        selected: List[Dict[str, Any]] = []
        selected.extend(pos_sorted[:pos_pick])
        selected.extend(neg_sorted[:neg_pick])
        
        edges_raw = selected
    elif edge_selection == "half_signed":
        if max_edges is None:
            raise ValueError("edge_selection='half_signed' 时必须提供 max_edges 参数")
        pos_edge_list = [e for e in edges_raw if e["weight"] > 0]
        neg_edge_list = [e for e in edges_raw if e["weight"] < 0]
        pos_sorted = sorted(pos_edge_list, key=lambda e: e["abs_diff"], reverse=True)
        neg_sorted = sorted(neg_edge_list, key=lambda e: e["abs_diff"], reverse=True)

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
        if max_edges is None:
            raise ValueError("edge_selection='abs' 时必须提供 max_edges 参数")
        edges_raw = edges_sorted_abs[:max_edges]

    node_ids = sorted(
        {int(e["src_id"]) for e in edges_raw}
        | {int(e["dst_id"]) for e in edges_raw}
    )

    complete = _load_json(analysis_complete_path)
    act = np.array(complete["activation_analysis"]["activation_diff"], dtype=float)

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
        # activation_diff 仅用于上游分析，这里不再写入中间 JSON，避免冗余字段
        nodes.append(
            {
                "node_id": int(nid),
                "label": label,
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


def draw_chord(ax, x0, y0, x1, y1, lw, color, alpha, linestyle):
    """绘制弦边。linestyle: '-' 实线 或 (0, (3, 3)) 虚线。"""
    cx, cy = 0.0, 0.0
    curvature = 0.55
    c1 = (x0 * (1 - curvature) + cx * curvature, y0 * (1 - curvature) + cy * curvature)
    c2 = (x1 * (1 - curvature) + cx * curvature, y1 * (1 - curvature) + cy * curvature)
    path = MplPath(
        [(x0, y0), c1, c2, (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
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


def _find_connected_components(
    edge_pairs: List[tuple],
    node_ids: set,
) -> List[set]:
    """BFS 提取连通分量，返回 list[set[int]]，仅含 >= 2 个节点的分量。"""
    from collections import defaultdict, deque

    adj: Dict[int, set] = defaultdict(set)
    for src, dst in edge_pairs:
        adj[src].add(dst)
        adj[dst].add(src)

    visited: set = set()
    components: List[set] = []
    for nid in node_ids:
        if nid in visited:
            continue
        comp: set = set()
        q: deque = deque([nid])
        while q:
            cur = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for nb in adj[cur]:
                if nb not in visited and nb in node_ids:
                    q.append(nb)
        if len(comp) >= 2:
            components.append(comp)
    return components


def _draw_hull(
    ax,
    points: np.ndarray,
    color: str,
    alpha: float,
    edge_lw: float,
    edge_alpha: float,
    edge_color: str,
    pad: float,
) -> None:
    """
    对 points(Nx2) 计算凸包并绘制边缘虚线（不做内部填充）。
    要求 points >= 3 个（调用方保证）。
    zorder=0，在圆环/边/节点之下。
    """
    from scipy.spatial import ConvexHull

    # 膨胀：每个点从质心方向向外偏移 pad
    cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())
    expanded = []
    for px, py in points:
        dx, dy = float(px) - cx, float(py) - cy
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-9:
            expanded.append((float(px), float(py)))
        else:
            expanded.append((float(px) + pad * dx / d, float(py) + pad * dy / d))
    expanded_arr = np.array(expanded, dtype=float)

    try:
        hull = ConvexHull(expanded_arr)
    except Exception as exc:
        print(f"[WARN] ConvexHull 计算失败（点共线或退化）: {exc}")
        return

    verts = expanded_arr[hull.vertices]
    poly = plt.Polygon(
        verts,
        closed=True,
        facecolor=color,
        alpha=0.1,
        edgecolor=edge_color,
        linewidth=edge_lw,
        linestyle=(0, (4, 2)),  # 虚线
        zorder=0,
    )
    ax.add_patch(poly)


def plot_circle_graph(
    graph_data: Dict[str, Any],
    output_fig: Path,
    title: str = "",
    group_high_label: str = "Hallucinated",
    group_low_label: str = "Truthful",
    parcel_matrix_path: Path | None = None,
    use_struct_matrix_color: bool = True,
    show_hull: bool = False,
) -> None:
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    meta = graph_data.get("meta", {})
    level = meta.get("level", "parcel")
    item_term = meta.get("item_term", "Parcel" if level == "parcel" else "Capability")
    n = len(nodes)
    if n == 0:
        raise ValueError("nodes 为空。")

    # 根据边类别重排节点：正负边分区，mixed 节点放在分区交界处
    nodes = reorder_nodes_by_edge_category(nodes, edges)

    # ===== 可选：加载 parcel 结构连接矩阵，用于边上色 =====
    parcel_matrix: np.ndarray | None = None
    use_struct_color = False
    if level == "parcel" and use_struct_matrix_color:
        if parcel_matrix_path is None:
            raise ValueError(
                "level='parcel' 且 use_struct_matrix_color=True 时必须提供 parcel_matrix_path。"
            )
        parcel_matrix = _load_parcel_matrix(parcel_matrix_path)
        use_struct_color = True

    # ===== 环形布局（可控半径）=====
    R = float(CONFIG["ring_radius"])
    thetas = np.array([math.pi / 2 - i * 2 * math.pi / n for i in range(n)])

    for idx, node in enumerate(nodes):
        th = float(thetas[idx])
        node["theta"] = th
        node["x"] = float(R * math.cos(th))
        node["y"] = float(R * math.sin(th))

    # ===== 视觉映射（仅边按权重映射粗细/透明度，节点统一大小与颜色）=====
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
    max_node_id = max(int(nd["node_id"]) for nd in nodes)
    if parcel_matrix is not None and parcel_matrix.shape[0] <= max_node_id:
        raise ValueError(
            f"parcel 结构矩阵尺寸 {parcel_matrix.shape} 不足以覆盖最大节点 id={max_node_id}。"
        )

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

        if use_struct_color and parcel_matrix is not None:
            try:
                m_val = float(parcel_matrix[pid_i, pid_j])
            except Exception as exc:
                print(
                    f"[ERROR] 访问 parcel 结构矩阵索引失败: M[{pid_i}, {pid_j}] ({exc})"
                )
                raise
            if m_val == 0:
                continue  # M==0 不绘制
            # 结构用线型：M>0 实线，M<0 虚线
            edge_linestyle = "-" if m_val > 0 else (0, (3, 3))
            # 方向用颜色：truthful 更强=蓝，hallucinated 更强=黄
            edge_color = (
                COLOR_STRONGER_TRUTHFUL if w > 0 else COLOR_STRONGER_HALLUCINATED
            )
        else:
            edge_color = (
                COLOR_EDGE_DIRECTION_SOLID if w > 0 else COLOR_EDGE_DIRECTION_DASHED
            )
            edge_linestyle = "-" if w > 0 else (0, (3, 3))

        draw_chord(ax, x0, y0, x1, y1, lw, edge_color, float(alpha), edge_linestyle)

    # ===== 连通子图凸包强调（仅 show_hull=True 且 use_struct_color=True 时执行）=====
    if show_hull and use_struct_color:
        truthful_edge_pairs: List[tuple] = []
        hallucinated_edge_pairs: List[tuple] = []
        for e in edges:
            pid_i = int(e["src_id"])
            pid_j = int(e["dst_id"])
            if pid_i not in id2idx or pid_j not in id2idx:
                continue
            w = float(e["weight"])
            if w > 0:
                truthful_edge_pairs.append((pid_i, pid_j))
            else:
                hallucinated_edge_pairs.append((pid_i, pid_j))

        all_nids = {int(nd["node_id"]) for nd in nodes}
        min_n = int(CONFIG.get("hull_min_nodes", 2))

        for edge_pairs, hull_color in [
            (truthful_edge_pairs, COLOR_STRONGER_TRUTHFUL),
            (hallucinated_edge_pairs, COLOR_STRONGER_HALLUCINATED),
        ]:
            comps = _find_connected_components(edge_pairs, all_nids)
            for comp in comps:
                if len(comp) < min_n:
                    continue
                pts = np.array(
                    [[nodes[id2idx[nid]]["x"], nodes[id2idx[nid]]["y"]]
                     for nid in comp],
                    dtype=float,
                )
                _draw_hull(
                    ax, pts, hull_color,
                    alpha=float(CONFIG.get("hull_alpha", 0.10)),
                    edge_lw=float(CONFIG.get("hull_edge_lw", 0.5)),
                    edge_alpha=float(CONFIG.get("hull_edge_alpha", 0.25)),
                    edge_color=str(CONFIG.get("hull_edge_color", "#555555")),
                    pad=float(CONFIG.get("hull_pad", 0.08)),
                )

    # ===== 节点：浅灰空心（fill 白/极浅灰，edge #C8C8C8，线宽 0.5–0.8 pt，无阴影/渐变）=====
    xs = np.array([nd["x"] for nd in nodes], dtype=float)
    ys = np.array([nd["y"] for nd in nodes], dtype=float)
    node_size = float(CONFIG["node_size"])
    ax.scatter(
        xs, ys,
        s=node_size,
        c=CONFIG.get("node_fill_color", COLOR_NODE_FILL),
        edgecolors=CONFIG.get("node_edge_color", COLOR_NODE_EDGE),
        linewidths=float(CONFIG["node_edge_lw"]),
        alpha=1.0,
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

    # ===== 图例：方向用颜色，结构用线型 =====
    if level == "parcel" and use_struct_color:
        # 方向（Truthful/Hallucination）用颜色
        legend_elements = [
            Line2D(
                [0], [0],
                color=COLOR_STRONGER_TRUTHFUL,
                linewidth=1.5,
                linestyle="-",
                label=f"{group_low_label} > {group_high_label}",
            ),
            Line2D(
                [0], [0],
                color=COLOR_STRONGER_HALLUCINATED,
                linewidth=1.5,
                linestyle="-",
                label=f"{group_high_label} > {group_low_label}",
            ),
            # 结构（M>0/M<0）用线型，中性灰
            Line2D(
                [0], [0],
                color="#777777",
                linewidth=1.2,
                linestyle="-",
                label="Structural connectivity: Activation",
            ),
            Line2D(
                [0], [0],
                color="#777777",
                linewidth=1.2,
                linestyle=(0, (3, 3)),
                label="Structural connectivity: Inhibition",
            ),
        ]
    else:
        legend_elements = [
            Line2D(
                [0], [0],
                color="#777777",
                linewidth=1.2,
                linestyle="-",
                label=f"{group_low_label} > {group_high_label} (Δconn > 0)",
            ),
            Line2D(
                [0], [0],
                color="#777777",
                linewidth=1.2,
                linestyle=(0, (3, 3)),
                label=f"{group_high_label} > {group_low_label} (Δconn < 0)",
            ),
        ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=float(CONFIG["font_legend"]),
        handlelength=1.4,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.9,
    )

    if title:
        ax.set_title(sanitize_text(title), fontsize=float(CONFIG["font_title"]), pad=1)

    # ===== 视野（避免裁剪）=====
    margin = 0.55
    ax.set_xlim(-label_r - margin, label_r + margin)
    ax.set_ylim(-label_r - margin, label_r + margin)

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_fig,
        format="pdf",
        dpi=int(CONFIG["dpi"]),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    output_svg = output_fig.with_suffix(".svg")
    fig.savefig(
        output_svg,
        format="svg",
        dpi=int(CONFIG["dpi"]),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


def save_graph_data(graph_data: Dict[str, Any], output_data: Path) -> None:
    output_data.parent.mkdir(parents=True, exist_ok=True)
    with output_data.open("w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)


def enrich_edge_annotations(
    graph_data: Dict[str, Any],
    level: str,
    parcel_matrix_path: Path,
    use_struct_matrix_color: bool,
) -> bool:
    """
    为中间数据中的 edges 增加可解释标签：
    - group_direction: 哪一组连接更强（由 edge weight 符号决定）
    - connection_effect: 结构连接是增强/抑制（仅 parcel + 结构矩阵模式）
    """
    edges = graph_data.get("edges", [])
    if not isinstance(edges, list):
        raise ValueError("graph_data['edges'] 不是 list，无法补充连接注释。")

    need_struct_effect = (level == "parcel") and bool(use_struct_matrix_color)
    parcel_matrix: np.ndarray | None = None
    if need_struct_effect:
        parcel_matrix = _load_parcel_matrix(parcel_matrix_path)

    changed = False
    for e in edges:
        src_id = int(e["src_id"])
        dst_id = int(e["dst_id"])
        weight = float(e["weight"])

        # 与图例语义一致：weight > 0 表示 low 组更强，weight < 0 表示 high 组更强
        direction = "group_low_stronger" if weight > 0 else "group_high_stronger"
        if e.get("group_direction") != direction:
            e["group_direction"] = direction
            changed = True

        if need_struct_effect:
            assert parcel_matrix is not None
            if (
                src_id < 0
                or dst_id < 0
                or src_id >= parcel_matrix.shape[0]
                or dst_id >= parcel_matrix.shape[1]
            ):
                raise ValueError(
                    f"结构矩阵索引越界: M[{src_id}, {dst_id}], 矩阵形状={parcel_matrix.shape}"
                )
            m_val = float(parcel_matrix[src_id, dst_id])
            if m_val > 0:
                effect = "enhancement"
            elif m_val < 0:
                effect = "inhibition"
            else:
                effect = "none"

            if e.get("connection_effect") != effect:
                e["connection_effect"] = effect
                changed = True
            if e.get("struct_connectivity_value") != m_val:
                e["struct_connectivity_value"] = m_val
                changed = True

    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parcel/Capability-level circle graph (edge2node)")
    parser.add_argument("--anomalous_edges_json", type=str, required=True)
    parser.add_argument("--analysis_complete_json", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--output_fig", type=str, required=True)
    parser.add_argument("--p_threshold", type=float, default=CONFIG["default_p_threshold"])
    parser.add_argument("--max_edges", type=int, default=None, help="最大边数量（与 --pos_edges/--neg_edges 互斥）")
    parser.add_argument(
        "--pos_edges",
        type=int,
        default=None,
        help="正边数量（必须与 --neg_edges 同时使用，且与 --max_edges 互斥）",
    )
    parser.add_argument(
        "--neg_edges",
        type=int,
        default=None,
        help="负边数量（必须与 --pos_edges 同时使用，且与 --max_edges 互斥）",
    )
    parser.add_argument(
        "--edge_selection",
        type=str,
        choices=["abs", "half_signed"],
        default="abs",
        help=(
            "edge 选择策略："
            "'abs' 按 |weight| 整体排序取前 max_edges（默认）；"
            "'half_signed' 从最正向和最负向中各取约一半，再按绝对值补齐。"
            "如果提供了 --pos_edges 和 --neg_edges，则忽略此参数。"
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
    parser.add_argument(
        "--parcel_matrix_csv",
        type=str,
        default="/path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv",
        help="Parcel-level 结构连接矩阵 CSV 路径（仅在 level=parcel 且启用结构上色时使用）",
    )
    parser.add_argument(
        "--use_struct_matrix_color",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否使用 parcel 结构矩阵 M[i,j] 的正负为边上色（1=启用，0=关闭；仅 level=parcel 生效）",
    )
    parser.add_argument(
        "--show_hull",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否为同色连通子图绘制凸包强调（1=开启，0=关闭；默认关闭）",
    )
    args = parser.parse_args()
    
    # 检查 max_edges 和 pos_edges/neg_edges 的互斥
    has_pos_neg = (args.pos_edges is not None) or (args.neg_edges is not None)
    has_max_edges = args.max_edges is not None
    
    if has_pos_neg and has_max_edges:
        parser.error("--max_edges 与 --pos_edges/--neg_edges 不能同时使用")
    
    if (args.pos_edges is not None) != (args.neg_edges is not None):
        parser.error("--pos_edges 和 --neg_edges 必须同时提供或都不提供")
    
    # 如果都没有提供，使用默认的 max_edges
    if not has_pos_neg and not has_max_edges:
        args.max_edges = CONFIG["default_max_edges"]
    
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

    # 标准化成 bool
    args.use_struct_matrix_color = bool(args.use_struct_matrix_color)
    
    return args


def main() -> None:
    args = parse_args()
    anomalous_edges_path = Path(args.anomalous_edges_json).resolve()
    analysis_complete_path = Path(args.analysis_complete_json).resolve()
    output_data = Path(args.output_data).resolve()
    output_fig = Path(args.output_fig).resolve()
    parcel_matrix_path = Path(args.parcel_matrix_csv).resolve()

    if args.skip_if_exists and output_fig.exists() and output_data.exists():
        print(f"[INFO] 检测到已有结果，检查中间数据注释完整性: {output_data}")
        with output_data.open("r", encoding="utf-8") as f:
            graph_data = json.load(f)
        changed = enrich_edge_annotations(
            graph_data=graph_data,
            level=args.level,
            parcel_matrix_path=parcel_matrix_path,
            use_struct_matrix_color=bool(args.use_struct_matrix_color),
        )
        if changed:
            save_graph_data(graph_data, output_data)
            print("[INFO] 已为已有中间数据补充 connection_effect/group_direction 注释。")
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
                pos_edges=args.pos_edges,
                neg_edges=args.neg_edges,
            )

        changed = enrich_edge_annotations(
            graph_data=graph_data,
            level=args.level,
            parcel_matrix_path=parcel_matrix_path,
            use_struct_matrix_color=bool(args.use_struct_matrix_color),
        )
        if changed or (args.refresh_data or (not output_data.exists())):
            save_graph_data(graph_data, output_data)

        title = ""
        plot_circle_graph(
            graph_data,
            output_fig,
            title=title,
            group_high_label=args.group_high_label,
            group_low_label=args.group_low_label,
            parcel_matrix_path=parcel_matrix_path,
            use_struct_matrix_color=bool(args.use_struct_matrix_color),
            show_hull=bool(args.show_hull),
        )
        print(f"[INFO] 中间数据已保存到: {output_data}")
        print(f"[INFO] 图像已保存到: {output_fig}")
    except Exception as e:
        print(f"[ERROR] 绘图过程中出现异常: {e}")
        raise


if __name__ == "__main__":
    main()
