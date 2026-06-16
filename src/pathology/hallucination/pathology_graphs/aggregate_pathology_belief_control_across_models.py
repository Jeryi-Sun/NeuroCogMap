#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨多个数据集/任务，对同一模型系列的 Belief-related / Control-related capability 与 parcel
的 activation difference 进行聚合（对每个 capability / parcel 先跨任务取平均），
然后将 capability 与 parcel 合并在一起，绘制一张 Belief vs Control 的小提琴图。

示例用法（9B 系列，多数据集聚合）：

  python aggregate_pathology_belief_control_across_models.py \
    --aggregated_model_name "gemma-2-9b-it" \
    --model_data_list MedHallu_gemma-2-9b-it HaluEval_gemma-2-9b-it ... \
    --base_dir /path/to/project_root \
    --capability_cls_v2 /.../capability_descriptions_run2_pathology_classification_v2.json \
    --parcel_cls_v2 /.../latent_parcel_topsamples_functionality_summary_pathology_classification_v2.json \
    --output_dir /.../pathology_graphs/graph

  默认对 activation difference 取绝对值；不取绝对值时加 --no_abs（或 bash 中 USE_ABS=0）。
"""

import argparse
import json
import re
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import numpy as np

try:
    from scipy import stats as _stats

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# 重新加载字体管理器以确保识别最新字体（按用户要求）
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass

def configure_matplotlib():
    """
    统一设置为 Nature 风格、小图 5pt 字号。
    """
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 5
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.labelsize"] = 5
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


def build_model_data_suffix(aggregated_model_name: str, model_data_list: List[str]) -> str:
    """
    从 model_data_list 中提取“数据后缀”，用于输出文件名。
    规则（尽量稳健，兼容不同命名）：
    - 若条目形如 "<dataTag>_<aggregated_model_name>" 或 "<dataTag>-<aggregated_model_name>"，则 dataTag 作为后缀；
    - 否则退化为条目自身（取 basename）。
    - 多个条目时按出现顺序去重并用 "_" 连接。
    """
    pat = re.compile(rf"^(?P<prefix>.+?)(?:[_-]){re.escape(aggregated_model_name)}$")
    tags: List[str] = []
    for raw in model_data_list:
        base = Path(raw).name.strip()
        if not base:
            continue
        m = pat.match(base)
        tag = (m.group("prefix") if m else base).strip()
        # 文件名安全化：保留字母数字，其余统一转为 "-"
        tag = re.sub(r"\s+", "-", tag)
        tag = re.sub(r"[^A-Za-z0-9]+", "-", tag)
        tag = re.sub(r"-{2,}", "-", tag).strip("-")
        if tag and tag not in tags:
            tags.append(tag)
    return "_".join(tags)


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


MIN_BELIEF_CONFIDENCE = 9


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


def load_capability_activation_diff_map(
    analysis_path: Path,
    use_abs: bool = True,
) -> Dict[str, float]:
    data = load_json(analysis_path)
    activation = data.get("activation_analysis", {})
    names = activation.get("capability_names")
    diffs = activation.get("activation_diff")
    if not isinstance(names, list) or not isinstance(diffs, list):
        raise KeyError("capability activation_analysis 中缺少 capability_names 或 activation_diff")
    if len(names) != len(diffs):
        raise ValueError(f"capability_names 与 activation_diff 长度不一致: {len(names)} vs {len(diffs)}")
    diffs_arr = -np.asarray(diffs, dtype=float)  # Truthfulness - Hallucination
    if use_abs:
        diffs_arr = np.abs(diffs_arr)
    return {str(n): float(d) for n, d in zip(names, diffs_arr.tolist())}


def load_parcel_activation_diff_map(
    analysis_path: Path,
    use_abs: bool = True,
) -> Dict[int, float]:
    data = load_json(analysis_path)
    activation = data.get("activation_analysis", {})
    diffs = activation.get("activation_diff")
    if not isinstance(diffs, list):
        raise KeyError("parcel activation_analysis 中缺少 activation_diff")
    diffs_arr = -np.asarray(diffs, dtype=float)  # Truthfulness - Hallucination
    if use_abs:
        diffs_arr = np.abs(diffs_arr)
    return {pid: float(diffs_arr[pid]) for pid in range(diffs_arr.size)}


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
    model_label: str,
    ylabel: str,
    belief_vals: List[float],
    control_vals: List[float],
    group_label: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 Nature palette 颜色：蓝色 vs 暖黄色
    belief_color = "#579FCA"   # medium blue
    control_color = "#F7DC7C"  # warm yellow

    # 1/5 A4 宽度 ≈ 1.42 inch，这里取近似方形小图
    fig, ax = plt.subplots(figsize=(1.42, 1.42))

    data = [np.asarray(belief_vals, dtype=float), np.asarray(control_vals, dtype=float)]
    positions = [1, 2]

    parts = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        widths=0.5,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_facecolor(belief_color if i == 0 else control_color)
        body.set_alpha(0.35)

    # 散点（与各自小提琴颜色相近、半透明，小点，自然风格），参考层级小提琴图实现
    rng = np.random.default_rng(42)
    for x, vals, color in zip(positions, data, [belief_color, control_color]):
        if vals.size == 0:
            continue
        # 稍微变深一点的颜色，用于散点主体颜色
        rgb = np.array(mcolors.to_rgb(color))
        darker_rgb = np.clip(rgb * 0.8, 0, 1)
        x_jitter = x + rng.normal(0, 0.0, size=vals.size)
        ax.scatter(
            x_jitter,
            vals,
            c=[darker_rgb],
            s=3,
            alpha=0.6,
            linewidths=0.3,
            edgecolors="black",
        )

    # 均值虚线短线，参考层级小提琴图实现
    for x, vals in zip(positions, data):
        if vals.size == 0:
            continue
        mean_val = float(np.mean(vals))
        ax.hlines(
            y=mean_val,
            xmin=x - 0.18,
            xmax=x + 0.18,
            colors="#666666",
            linewidth=1.0,
            linestyles="--",
            zorder=5,
        )

    # 显著性标注：比较 Belief vs Control 是否存在显著差异（双样本 t 检验）
    if all(v.size > 1 for v in data):
        all_vals = np.concatenate(data)
        global_min, global_max = float(all_vals.min()), float(all_vals.max())
        span = global_max - global_min if global_max > global_min else 1.0
        y_offset = 0.08 * span

        t_stat, p_val = two_sample_t_test(belief_vals, control_vals)
        if np.isnan(p_val):
            label_text = "n/a"
            p_str = "NaN (SciPy not available)"
        else:
            # 星号表示显著性等级
            if p_val < 1e-3:
                stars = "***"
            elif p_val < 1e-2:
                stars = "**"
            elif p_val < 5e-2:
                stars = "*"
            else:
                stars = "n.s."

            # 文本同时包含 P 值和星号（更接近 Nature 标注风格）
            if p_val < 1e-3:
                label_text = rf"$P < 0.001$ {stars}"
            else:
                label_text = rf"$P = {p_val:.3f}$ {stars}"
            p_str = f"{p_val:.3g}"

        # 在终端打印 t / p，便于记录
        print(
            f"[agg] {model_label} {group_label}: "
            f"belief_n={len(belief_vals)}, control_n={len(control_vals)}, "
            f"t={t_stat:.3f}, p={p_str}"
        )

        # 画一条“括号”线连接 Belief 与 Control，并在中间标注 P 值（Nature 风格）
        y_pos = float(max(global_max, 0.0)) + y_offset
        tick = 0.02 * span
        # 竖线小脚
        ax.plot(
            [positions[0], positions[0]],
            [y_pos - tick, y_pos],
            color="#333333",
            linewidth=0.8,
        )
        ax.plot(
            [positions[1], positions[1]],
            [y_pos - tick, y_pos],
            color="#333333",
            linewidth=0.8,
        )
        # 顶部横线
        ax.plot(
            [positions[0], positions[1]],
            [y_pos, y_pos],
            color="#333333",
            linewidth=0.8,
        )
        ax.text(
            0.5 * (positions[0] + positions[1]),
            y_pos + tick,
            label_text,
            ha="center",
            va="bottom",
            fontsize=5,
            color="#333333",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Belief-related", "Control-related"])
    ax.set_ylabel(ylabel)

    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    # Nature 规范中建议避免背景网格线，这里不再开启 grid

    fig.tight_layout()
    # 同时导出 PDF 和 SVG 两种矢量格式
    fig.savefig(out_path, format="pdf")
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)


def aggregate_across_models(
    base_dir: Path,
    model_data_list: List[str],
    capability_cls_v2: Path,
    parcel_cls_v2: Path,
    project_type: str = "hallucination",
    use_abs: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    对给定的一组 model_data（不同数据集上的同一模型系列）：
    - 对每个 capability 取 activation_diff 跨模型平均；
    - 对每个 parcel_id 取 activation_diff 跨模型平均；
    - capability 与 parcel 分开，根据病理分类分到 Belief / Control 组。
    返回 (cap_belief, cap_control, parcel_belief, parcel_control) 四个列表。
    """
    # 聚合 maps: name -> [diffs], parcel_id -> [diffs]
    cap_values: Dict[str, List[float]] = {}
    parcel_values: Dict[int, List[float]] = {}

    for model_name in model_data_list:
        cap_path = (
            base_dir
            / "safety_explanation"
            / project_type
            / "results"
            / "analysis_output"
            / model_name
            / "capability_level"
            / "capability_level_analysis_complete.json"
        )
        par_path = (
            base_dir
            / "safety_explanation"
            / project_type
            / "results"
            / "analysis_output"
            / model_name
            / "parcel_level"
            / "parcel_level_analysis_complete.json"
        )

        if not cap_path.exists() or not par_path.exists():
            print(f"[WARN] 分析结果缺失，跳过模型: {model_name}")
            continue

        cap_map = load_capability_activation_diff_map(cap_path, use_abs=use_abs)
        par_map = load_parcel_activation_diff_map(par_path, use_abs=use_abs)

        for name, diff in cap_map.items():
            cap_values.setdefault(name, []).append(diff)
        for pid, diff in par_map.items():
            parcel_values.setdefault(pid, []).append(diff)
    # 跨模型平均
    cap_mean: Dict[str, float] = {k: float(np.max(v)) for k, v in cap_values.items() if v}
    parcel_mean: Dict[int, float] = {k: float(np.max(v)) for k, v in parcel_values.items() if v}

    # 加载病理分类：capability 不筛选置信度，parcel 对 Belief 施加置信度过滤
    belief_caps, control_caps = load_groups_from_capability_classification(
        capability_cls_v2
    )
    belief_parcels, control_parcels = load_groups_from_parcel_classification(
        parcel_cls_v2, min_conf_belief=MIN_BELIEF_CONFIDENCE
    )

    # 归一化 capability 名称匹配
    norm_to_real = {normalize_name(k): k for k in cap_mean.keys()}

    cap_belief: List[float] = []
    cap_control: List[float] = []

    for n in belief_caps:
        real = norm_to_real.get(normalize_name(n))
        if real is not None:
            cap_belief.append(cap_mean[real])
    for n in control_caps:
        real = norm_to_real.get(normalize_name(n))
        if real is not None:
            cap_control.append(cap_mean[real])

    parcel_belief: List[float] = []
    parcel_control: List[float] = []

    for pid in belief_parcels:
        if pid in parcel_mean:
            parcel_belief.append(parcel_mean[pid])
    for pid in control_parcels:
        if pid in parcel_mean:
            parcel_control.append(parcel_mean[pid])

    return cap_belief, cap_control, parcel_belief, parcel_control


def main():
    parser = argparse.ArgumentParser(
        description="跨多个模型数据集聚合 Belief/Control activation difference，并绘制一张小提琴图（capability+parcel 合并）。"
    )
    parser.add_argument(
        "--aggregated_model_name",
        type=str,
        required=True,
        help="聚合后图中展示的模型系列名称（用于标题和输出文件名）",
    )
    parser.add_argument(
        "--model_data_list",
        type=str,
        nargs="+",
        required=True,
        help="需要聚合的模型数据名列表（与 analysis_output 子目录名一致）",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/path/to/project_root",
        help="项目根目录（包含 safety_explanation/...）",
    )
    parser.add_argument(
        "--capability_cls_v2",
        type=str,
        required=True,
        help="capability 病理分类 v2 JSON 路径",
    )
    parser.add_argument(
        "--parcel_cls_v2",
        type=str,
        required=True,
        help="parcel 病理分类 v2 JSON 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录（通常为 pathology_graphs/graph）",
    )
    parser.add_argument(
        "--append_model_data_suffix",
        action="store_true",
        help="将 --model_data_list 中解析出的数据后缀追加到输出 PDF 文件名（多数据会全部追加）",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果目标 PDF 已存在则跳过",
    )
    parser.add_argument(
        "--project_type",
        type=str,
        default="hallucination",
        choices=["hallucination", "fairness_bias", "jailbreak", "sycophancy"],
        help="项目类型：hallucination / fairness_bias / jailbreak / sycophancy（默认: hallucination）",
    )
    parser.add_argument(
        "--group_high_label",
        type=str,
        default=None,
        help="高激活组标签（如 Hallucination / Incorrect），默认根据 project_type 自动设置",
    )
    parser.add_argument(
        "--group_low_label",
        type=str,
        default=None,
        help="低激活组标签（如 Truthfulness / Correct），默认根据 project_type 自动设置",
    )
    parser.add_argument(
        "--no_abs",
        action="store_true",
        help="不对 activation difference 取绝对值（默认取绝对值）",
    )

    args = parser.parse_args()
    use_abs = not args.no_abs
    configure_matplotlib()

    # 根据 project_type 设置默认标签（如果未显式指定）
    if args.group_high_label is None:
        if args.project_type == "fairness_bias":
            args.group_high_label = "Incorrect"
        elif args.project_type == "jailbreak":
            args.group_high_label = "Unsafe"
        elif args.project_type == "sycophancy":
            args.group_high_label = "Sycophantic"
        else:
            args.group_high_label = "Hallucination"
    if args.group_low_label is None:
        if args.project_type == "fairness_bias":
            args.group_low_label = "Correct"
        elif args.project_type == "jailbreak":
            args.group_low_label = "Safe"
        elif args.project_type == "sycophancy":
            args.group_low_label = "Non-sycophantic"
        else:
            args.group_low_label = "Truthfulness"

    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    suffix = ""
    if args.append_model_data_suffix:
        suffix = build_model_data_suffix(args.aggregated_model_name, args.model_data_list)
        suffix = f"_{suffix}" if suffix else ""

    out_cap = out_dir / f"{args.aggregated_model_name}_capability_belief_vs_control_violin_agg{suffix}.pdf"
    out_par = out_dir / f"{args.aggregated_model_name}_parcel_belief_vs_control_violin_agg{suffix}.pdf"

    if args.skip_existing and out_cap.exists() and out_par.exists():
        print(f"[SKIP] 输出已存在: {out_cap} and {out_par}")
        return

    cap_belief, cap_control, parcel_belief, parcel_control = aggregate_across_models(
        base_dir=base_dir,
        model_data_list=args.model_data_list,
        capability_cls_v2=Path(args.capability_cls_v2),
        parcel_cls_v2=Path(args.parcel_cls_v2),
        project_type=args.project_type,
        use_abs=use_abs,
    )

    print(f"Aggregated counts (capability): belief={len(cap_belief)}, control={len(cap_control)}")
    print(f"Aggregated counts (parcel):     belief={len(parcel_belief)}, control={len(parcel_control)}")

    diff_ylabel = (
        f"Absolute activation difference\n|{args.group_low_label} - {args.group_high_label}|"
        if use_abs
        else f"Activation difference\n({args.group_low_label} - {args.group_high_label})"
    )
    # capability 聚合图
    if not (args.skip_existing and out_cap.exists()):
        plot_violin_two_groups(
            out_path=out_cap,
            model_label=args.aggregated_model_name,
            ylabel=diff_ylabel,
            belief_vals=cap_belief,
            control_vals=cap_control,
            group_label="capability",
        )
        print(f"[OK] 保存聚合小提琴图 (capability): {out_cap}")

    # parcel 聚合图
    if not (args.skip_existing and out_par.exists()):
        plot_violin_two_groups(
            out_path=out_par,
            model_label=args.aggregated_model_name,
            ylabel=diff_ylabel,
            belief_vals=parcel_belief,
            control_vals=parcel_control,
            group_label="parcel",
        )
        print(f"[OK] 保存聚合小提琴图 (parcel): {out_par}")


if __name__ == "__main__":
    main()

