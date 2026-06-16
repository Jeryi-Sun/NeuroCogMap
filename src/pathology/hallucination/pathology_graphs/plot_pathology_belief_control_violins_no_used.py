#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于病理分类（Belief-related vs Control-related），绘制 activation difference 的小提琴图。

输出为两个独立 PDF（capability 与 parcel 各一个），保存到指定 graph 目录。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from math import sqrt

try:
    from scipy import stats as _stats  # 可选：用于精确 p 值计算
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# 重新加载字体管理器以确保识别最新字体（按用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass

# Belief 组仅保留置信度较高的样本，避免数量远大于 Control
MIN_BELIEF_CONFIDENCE = 8


def configure_matplotlib():
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.labelsize"] = 5
    plt.rcParams["xtick.labelsize"] = 5
    plt.rcParams["ytick.labelsize"] = 5
    plt.rcParams["legend.fontsize"] = 5
    plt.rcParams["figure.dpi"] = 450


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_groups_from_capability_classification(
    cls_path: Path,
) -> Tuple[List[str], List[str]]:
    data = load_json(cls_path)
    belief, control = [], []
    for name, info in data.items():
        if not isinstance(info, dict):
            continue
        cat = info.get("category")
        parsing_error = info.get("parsing_error", False)
        if parsing_error:
            continue
        if cat == "Belief-related":
            belief.append(name)
        elif cat == "Control-related":
            control.append(name)
    return belief, control


def parse_parcel_id(label: str) -> int:
    m = re.match(r"^parcel_(\d+)\s*:", label.strip())
    if not m:
        raise ValueError(f"无法从 label 解析 parcel_id: {label}")
    return int(m.group(1))


def load_groups_from_parcel_classification(
    cls_path: Path,
    min_conf_belief: int = MIN_BELIEF_CONFIDENCE,
) -> Tuple[List[int], List[int]]:
    data = load_json(cls_path)
    belief, control = [], []
    for label, info in data.items():
        if not isinstance(info, dict):
            continue
        cat = info.get("category")
        conf = info.get("confidence", 0)
        parsing_error = info.get("parsing_error", False)
        try:
            pid = parse_parcel_id(label)
        except Exception:
            continue
        if cat == "Belief-related":
            if (not parsing_error) and conf >= min_conf_belief:
                belief.append(pid)
        elif cat == "Control-related":
            if not parsing_error:
                control.append(pid)
    return sorted(set(belief)), sorted(set(control))


def load_capability_activation_diff(analysis_path: Path) -> Dict[str, float]:
    data = load_json(analysis_path)
    activation = data.get("activation_analysis", {})
    names = activation.get("capability_names")
    diffs = activation.get("activation_diff")
    if not isinstance(names, list) or not isinstance(diffs, list):
        raise KeyError("capability activation_analysis 中缺少 capability_names 或 activation_diff")
    if len(names) != len(diffs):
        raise ValueError(f"capability_names 与 activation_diff 长度不一致: {len(names)} vs {len(diffs)}")
    # 先从 Hallucination - Truthfulness 反号为 Truthfulness - Hallucination，再取绝对值，
    # 避免正负抵消，只关心差异幅度。
    diffs_arr = -np.asarray(diffs, dtype=float)
    diffs_arr = np.abs(diffs_arr)
    return {str(n): float(d) for n, d in zip(names, diffs_arr.tolist())}


def load_parcel_activation_diff(analysis_path: Path) -> Dict[int, float]:
    data = load_json(analysis_path)
    activation = data.get("activation_analysis", {})
    diffs = activation.get("activation_diff")
    if not isinstance(diffs, list):
        raise KeyError("parcel activation_analysis 中缺少 activation_diff")
    diffs_arr = -np.asarray(diffs, dtype=float)  # Truthfulness - Hallucination
    diffs_arr = np.abs(diffs_arr)
    return {pid: float(diffs_arr[pid]) for pid in range(diffs_arr.size)}


def match_capability_names(group_names: List[str], diff_map: Dict[str, float]) -> Tuple[List[float], List[str]]:
    norm_to_real = {normalize_name(k): k for k in diff_map.keys()}
    values, unmatched = [], []
    for n in group_names:
        real = norm_to_real.get(normalize_name(n))
        if real is None:
            unmatched.append(n)
            continue
        values.append(diff_map[real])
    return values, unmatched


def values_for_parcels(parcel_ids: List[int], diff_map: Dict[int, float]) -> Tuple[List[float], List[int]]:
    values, missing = [], []
    for pid in parcel_ids:
        if pid not in diff_map:
            missing.append(pid)
            continue
        values.append(diff_map[pid])
    return values, missing


def two_sample_t_test(belief_vals: List[float], control_vals: List[float]) -> Tuple[float, float]:
    """
    比较 Belief 组与 Control 组是否有显著差异（双样本 t 检验，Welch 近似），返回 (t_stat, p_value)。
    """
    b = np.asarray(belief_vals, dtype=float)
    c = np.asarray(control_vals, dtype=float)
    if b.size < 2 or c.size < 2:
        return float("nan"), float("nan")
    if _HAS_SCIPY:
        t_stat, p_val = _stats.ttest_ind(b, c, equal_var=False)
        return float(t_stat), float(p_val)
    # 无 SciPy 时，手动近似 Welch t（p 记为 NaN）
    mb, mc = float(b.mean()), float(c.mean())
    sb, sc = float(b.std(ddof=1)), float(c.std(ddof=1))
    nb, nc = b.size, c.size
    num = mb - mc
    den = sqrt(sb**2 / nb + sc**2 / nc) if sb > 0 or sc > 0 else 0.0
    if den == 0.0:
        return (float("inf") if num > 0 else float("-inf")), float("nan")
    t_stat = num / den
    return float(t_stat), float("nan")


def plot_violin_two_groups(
    out_path: Path,
    model_name: str,
    title: str,
    ylabel: str,
    belief_vals: List[float],
    control_vals: List[float],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 颜色：使用用户优先色板（蓝 vs 高对比红）
    belief_color = "#579FCA"
    control_color = "#DE7D82"

    # 1/4 A4 宽度：约 45 mm，对应 figsize=(1.8, 1.4)
    fig, ax = plt.subplots(figsize=(1.8, 1.4))

    data = [np.asarray(belief_vals, dtype=float), np.asarray(control_vals, dtype=float)]
    positions = [1, 2]

    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        widths=0.65,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_facecolor(belief_color if i == 0 else control_color)
        body.set_alpha(0.35)

    # 叠加灰色半透明散点表示单个样本分布（自然风格）
    rng = np.random.default_rng(42)
    for x, vals in zip(positions, data):
        if vals.size == 0:
            continue
        x_jitter = x + rng.normal(0, 0.04, size=vals.size)
        ax.scatter(
            x_jitter,
            vals,
            color="#555555",
            s=5,
            alpha=0.4,
            linewidths=0,
        )

    # 中位数短线
    for x, vals in zip(positions, data):
        if vals.size == 0:
            continue
        med = float(np.median(vals))
        ax.hlines(med, x - 0.18, x + 0.18, colors="#444444", linewidth=1.0)

    # 显著性标注：比较 Belief vs Control 是否存在显著差异（双样本 t 检验）
    if all(v.size > 1 for v in data):
        all_vals = np.concatenate(data)
        global_min, global_max = float(all_vals.min()), float(all_vals.max())
        span = global_max - global_min if global_max > global_min else 1.0
        y_offset = 0.08 * span

        t_stat, p_val = two_sample_t_test(belief_vals, control_vals)
        if np.isnan(p_val):
            label_text = "n/a"
        else:
            if p_val < 1e-3:
                stars = "***"
            elif p_val < 1e-2:
                stars = "**"
            elif p_val < 5e-2:
                stars = "*"
            else:
                stars = "n.s."
            label_text = stars

        y_pos = float(max(global_max, 0.0)) + y_offset
        ax.text(
            1.5,
            y_pos,
            label_text,
            ha="center",
            va="bottom",
            fontsize=5,
            color="#333333",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Belief-related", "Control-related"])
    ax.set_ylabel(ylabel, fontsize=5, fontweight='bold')
    ax.set_xlabel("", fontsize=5, fontweight='bold')
    ax.tick_params(axis="both", labelsize=5)
    ax.set_title(f"{model_name}\n{title}", fontsize=5)

    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--capability_cls_v2", type=str, required=True)
    parser.add_argument("--parcel_cls_v2", type=str, required=True)
    parser.add_argument("--capability_analysis_complete", type=str, required=True)
    parser.add_argument("--parcel_analysis_complete", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    configure_matplotlib()
    out_dir = Path(args.output_dir)
    out_cap = out_dir / f"{args.model_name}_capability_belief_vs_control_violin.pdf"
    out_par = out_dir / f"{args.model_name}_parcel_belief_vs_control_violin.pdf"

    if args.skip_existing and out_cap.exists() and out_par.exists():
        print(f"[SKIP] 输出已存在: {out_cap} and {out_par}")
        return

    belief_caps, control_caps = load_groups_from_capability_classification(
        Path(args.capability_cls_v2)
    )
    belief_parcels, control_parcels = load_groups_from_parcel_classification(
        Path(args.parcel_cls_v2), min_conf_belief=MIN_BELIEF_CONFIDENCE
    )

    cap_diff_map = load_capability_activation_diff(Path(args.capability_analysis_complete))
    par_diff_map = load_parcel_activation_diff(Path(args.parcel_analysis_complete))

    belief_cap_vals, belief_cap_unmatched = match_capability_names(belief_caps, cap_diff_map)
    control_cap_vals, control_cap_unmatched = match_capability_names(control_caps, cap_diff_map)

    belief_par_vals, belief_par_missing = values_for_parcels(belief_parcels, par_diff_map)
    control_par_vals, control_par_missing = values_for_parcels(control_parcels, par_diff_map)

    if belief_cap_unmatched or control_cap_unmatched:
        print("[WARN] capability 名称未匹配到（将忽略）：")
        for n in sorted(set(belief_cap_unmatched + control_cap_unmatched)):
            print("  -", n)

    if belief_par_missing or control_par_missing:
        print("[WARN] parcel_id 未找到（将忽略）：")
        for pid in sorted(set(belief_par_missing + control_par_missing)):
            print("  -", pid)

    if not (args.skip_existing and out_cap.exists()):
        plot_violin_two_groups(
            out_path=out_cap,
            model_name=args.model_name,
            title="Capability activation difference",
            ylabel="Absolute activation difference\n|Truthfulness - Hallucination|",
            belief_vals=belief_cap_vals,
            control_vals=control_cap_vals,
        )
        print(f"[OK] 保存: {out_cap}")

    if not (args.skip_existing and out_par.exists()):
        plot_violin_two_groups(
            out_path=out_par,
            model_name=args.model_name,
            title="Parcel activation difference",
            ylabel="Absolute activation difference\n|Truthfulness - Hallucination|",
            belief_vals=belief_par_vals,
            control_vals=control_par_vals,
        )
        print(f"[OK] 保存: {out_par}")

    print("\n=== Count summary ===")
    print(f"Capabilities: belief={len(belief_cap_vals)} control={len(control_cap_vals)}")
    print(f"Parcels:      belief={len(belief_par_vals)} control={len(control_par_vals)}")

    # 额外输出：Belief vs Control 的双样本 t 检验结果
    print("\n=== Two-sample t-test: Belief vs Control ===")
    for label, b_vals, c_vals in [
        ("capability", belief_cap_vals, control_cap_vals),
        ("parcel", belief_par_vals, control_par_vals),
    ]:
        if len(b_vals) < 2 or len(c_vals) < 2:
            print(f"{label}: belief_n={len(b_vals)}, control_n={len(c_vals)} (too few samples for t-test)")
            continue
        t_stat, p_val = two_sample_t_test(b_vals, c_vals)
        if np.isnan(p_val):
            p_str = "NaN (SciPy not available)"
        else:
            p_str = f"{p_val:.3g}"
        print(f"{label}: belief_n={len(b_vals)}, control_n={len(c_vals)}, t={t_stat:.3f}, p={p_str}")


if __name__ == "__main__":
    main()

