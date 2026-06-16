#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 aggregate_intervention_sycophancy_table 生成的汇总 CSV，绘制干预强度柱状图。

汇总 CSV（如 intervention_sycophancy_table_strengths.csv）中：
  baseline_accuracy / intervention_accuracy 与 fairness 对齐语义，表示 **非谄媚率**
  （见 code/detection/aggregate_intervention_sycophancy_table.py）。

本图不绘制 CSV 中的 accuracy 列本身，而是分别换算为 **谄媚率**（accuracy 列为非谄媚率）：
  baseline 柱：sycophancy = 1 - baseline_accuracy
  干预柱：    sycophancy = 1 - intervention_accuracy

聚合方式与 hallucination/intervention_graph/plot_intervention_strength_bars.py 一致：
  - baseline：每个 dataset 在按 strength 排序后 **第一个 strength** 对应的行
  - intervention：各 strength 上 intervention_accuracy 最大者对应的干预结果
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 重新加载字体管理器以确保识别最新字体
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass


CONFIG = {
    'rcparams': {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 5.0,
        'axes.linewidth': 0.8,
        'axes.labelsize': 5.0,
        'axes.titlesize': 5.0,
        'xtick.labelsize': 5.0,
        'ytick.labelsize': 5.0,
        'xtick.major.width': 0.7,
        'ytick.major.width': 0.7,
        'figure.dpi': 450,
        'savefig.dpi': 450,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    },
    'plot': {
        'figsize': (1.8, 1.4),
        'bar_width': 0.28,
        'capsize': 2.4,
        'bar_edge_width': 0.8,
        'y_pad_ratio': 0.10,
        'y_step': 0.05,
        'legend_fontsize': 5.0,
        'x_label_size': 5.0,
        'value_fontsize': 5.0,
        'value_weight': 'normal',
        'label_offset_ratio': 0.012,
        'grid_alpha': 0.18,
        'grid_line_width': 0.5,
        'colors': {
            'baseline': '#B4DDF4',
            'intervention': '#DE7D82',
            'edge': '#2B2B2B',
            'text': '#1F1F1F',
        },
        'legend_labels': {
            'baseline': 'Baseline',
            'intervention': 'With Intervention',
        },
    },
}

BASE = '/path/to/project_root'
DEFAULT_CSV_PATH = (
    f'{BASE}/safety_explanation/sycophancy/results/intervention/aggregate/'
    'intervention_sycophancy_table_strengths.csv'
)
DEFAULT_OUT_DIR = f'{BASE}/safety_explanation/sycophancy/intervention_graph/output'


def apply_style():
    plt.rcParams.update(CONFIG['rcparams'])


def norm_model(x: str) -> str:
    key = str(x).strip().lower()
    return {
        'gemma-2-2b': 'Gemma-2-2B',
        'gemma-2-9b-it': 'Gemma-2-9B-IT',
    }.get(key, x)


def norm_dataset_sycophancy(x: str) -> str:
    """将 dataset 列映射为 Answer / Feedback（与 hallucination 脚本 sycophancy 分支一致）。"""
    k = str(x).strip().lower()
    if 'feedback' in k:
        return 'Feedback'
    if 'answer' in k:
        return 'Answer'
    return x


def dataset_order_key(ds: str) -> int:
    order = {'Answer': 0, 'Feedback': 1}
    return order.get(ds, 999)


def sycophancy_from_non_sycophantic_rate(acc: float) -> float:
    """谄媚率 = 1 − accuracy；CSV 中 accuracy 表示非谄媚率。"""
    return 1.0 - float(acc)


def nice_limits(vmin: float, vmax: float, pad_ratio=0.10, step=0.05):
    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    lo = vmin - pad_ratio * span
    hi = vmax + pad_ratio * span
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    if hi - lo < 0.25 * max(1.0, abs(hi)):
        mid = 0.5 * (hi + lo)
        lo = mid - 0.125
        hi = mid + 0.125
    return float(lo), float(hi)


def annotate_bar_values(ax, bars, values, bottoms, fontsize, weight, color, offset_ratio):
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin if ymax > ymin else 1.0
    for rect, v, bottom in zip(bars, values, bottoms):
        if not np.isfinite(v):
            continue
        x = rect.get_x() + rect.get_width() / 2
        y = bottom + rect.get_height()
        ax.text(
            x,
            y + offset_ratio * yr,
            f'{v * 100:.1f}%',
            ha='center',
            va='bottom',
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=0.15),
            zorder=6,
            clip_on=False,
        )


def plot_model(df_model: pd.DataFrame, model_name: str, out_dir: Path):
    apply_style()
    c = CONFIG['plot']
    pad_ratio = c['y_pad_ratio']

    required = {'strength', 'dataset', 'baseline_accuracy', 'intervention_accuracy'}
    missing = required - set(df_model.columns)
    if missing:
        raise ValueError(f'CSV 缺少必需列: {sorted(missing)}')

    df_model = df_model.sort_values('strength').reset_index(drop=True)
    df_model['dataset'] = df_model['dataset'].map(norm_dataset_sycophancy)

    datasets = sorted(df_model['dataset'].unique(), key=dataset_order_key)

    baseline_agg = (
        df_model.groupby('dataset', as_index=False)
        .agg(baseline_acc=('baseline_accuracy', 'first'))
    )

    best_idx = df_model.groupby('dataset')['intervention_accuracy'].idxmax()
    intervention_best = (
        df_model.loc[best_idx, ['dataset', 'intervention_accuracy', 'strength']]
        .rename(columns={'intervention_accuracy': 'intervention_acc_best', 'strength': 'best_strength'})
        .reset_index(drop=True)
    )

    agg = baseline_agg.merge(intervention_best, on='dataset')
    agg['__order'] = agg['dataset'].map(dataset_order_key)
    agg = agg.sort_values(['__order', 'dataset']).drop(columns='__order').reset_index(drop=True)

    agg['baseline_syc'] = agg['baseline_acc'].map(sycophancy_from_non_sycophantic_rate)
    agg['intervention_syc'] = agg['intervention_acc_best'].map(sycophancy_from_non_sycophantic_rate)

    print(f'  [{model_name}] Best intervention strength per dataset (by max intervention_accuracy / non-sycophantic rate):')
    for _, row in agg.iterrows():
        ib = row['intervention_acc_best']
        print(
            f'    {row["dataset"]}: strength={row["best_strength"]} -> '
            f'intervention_acc={ib * 100:.2f}% (non-syc) | '
            f'sycophancy_after={row["intervention_syc"] * 100:.2f}% | '
            f'sycophancy_baseline={row["baseline_syc"] * 100:.2f}%'
        )

    x = np.arange(len(datasets), dtype=float)
    w = c['bar_width']

    fig, ax_left = plt.subplots(1, 1, figsize=c['figsize'], dpi=CONFIG['rcparams']['figure.dpi'])
    ax_right = ax_left.twinx()

    left_datasets = ['Answer']
    left_mask = agg['dataset'].isin(left_datasets).to_numpy()
    right_mask = ~left_mask

    x_left = x[left_mask]
    b_left = agg.loc[left_mask, 'baseline_syc'].to_numpy(float)
    i_left = agg.loc[left_mask, 'intervention_syc'].to_numpy(float)

    x_right = x[right_mask]
    b_right = agg.loc[right_mask, 'baseline_syc'].to_numpy(float)
    i_right = agg.loc[right_mask, 'intervention_syc'].to_numpy(float)

    def finite_vals(arr):
        a = arr[np.isfinite(arr)]
        return a

    left_vals = finite_vals(np.r_[b_left, i_left])
    right_vals = finite_vals(np.r_[b_right, i_right])

    if left_vals.size > 0:
        yl_min, yl_max = nice_limits(
            float(np.min(left_vals)),
            float(np.max(left_vals)),
            pad_ratio=pad_ratio,
            step=c['y_step'],
        )
    else:
        yl_min, yl_max = 0.0, 1.0
    if right_vals.size > 0:
        yr_min, yr_max = nice_limits(
            float(np.min(right_vals)),
            float(np.max(right_vals)),
            pad_ratio=pad_ratio,
            step=c['y_step'],
        )
    else:
        yr_min, yr_max = 0.0, 1.0

    ax_left.set_ylim(yl_min, yl_max)
    ax_right.set_ylim(yr_min, yr_max)

    b_left_heights = np.nan_to_num(np.maximum(b_left - yl_min, 0.0), nan=0.0)
    i_left_heights = np.nan_to_num(np.maximum(i_left - yl_min, 0.0), nan=0.0)
    b_left_bottoms = np.full_like(b_left, yl_min, dtype=float)
    i_left_bottoms = np.full_like(i_left, yl_min, dtype=float)

    b_right_heights = np.nan_to_num(np.maximum(b_right - yr_min, 0.0), nan=0.0)
    i_right_heights = np.nan_to_num(np.maximum(i_right - yr_min, 0.0), nan=0.0)
    b_right_bottoms = np.full_like(b_right, yr_min, dtype=float)
    i_right_bottoms = np.full_like(i_right, yr_min, dtype=float)

    bars_bl_left = ax_left.bar(
        x_left - w / 2, b_left_heights, width=w, bottom=b_left_bottoms,
        color=c['colors']['baseline'], edgecolor=c['colors']['edge'],
        linewidth=c['bar_edge_width'], zorder=3, label=c['legend_labels']['baseline']
    )
    bars_it_left = ax_left.bar(
        x_left + w / 2, i_left_heights, width=w, bottom=i_left_bottoms,
        color=c['colors']['intervention'], edgecolor=c['colors']['edge'],
        linewidth=c['bar_edge_width'], zorder=3, label=c['legend_labels']['intervention']
    )

    bars_bl_right = ax_right.bar(
        x_right - w / 2, b_right_heights, width=w, bottom=b_right_bottoms,
        color=c['colors']['baseline'], edgecolor=c['colors']['edge'],
        linewidth=c['bar_edge_width'], zorder=3
    )
    bars_it_right = ax_right.bar(
        x_right + w / 2, i_right_heights, width=w, bottom=i_right_bottoms,
        color=c['colors']['intervention'], edgecolor=c['colors']['edge'],
        linewidth=c['bar_edge_width'], zorder=3
    )

    for bar in bars_bl_left:
        bar.set_clip_on(False)
    for bar in bars_it_left:
        bar.set_clip_on(False)
    for bar in bars_bl_right:
        bar.set_clip_on(False)
    for bar in bars_it_right:
        bar.set_clip_on(False)

    ax_left.set_xlim(-0.5, len(datasets) - 0.5)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(datasets, fontsize=5.0)

    ax_left.set_xlabel('Dataset', fontsize=5.0)
    ax_left.set_ylabel('Sycophancy Rate', fontsize=5.0)
    ax_right.set_ylabel('')

    ax_left.tick_params(axis='both', labelsize=5.0)
    ax_right.tick_params(axis='y', labelsize=5.0)

    ax_left.spines['top'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_left.grid(axis='y', alpha=c['grid_alpha'], linewidth=c['grid_line_width'])
    ax_right.grid(False)

    annotate_bar_values(
        ax_left, bars_bl_left, b_left, b_left_bottoms,
        fontsize=c['value_fontsize'], weight=c['value_weight'],
        color=c['colors']['text'], offset_ratio=c['label_offset_ratio'],
    )
    annotate_bar_values(
        ax_left, bars_it_left, i_left, i_left_bottoms,
        fontsize=c['value_fontsize'], weight=c['value_weight'],
        color=c['colors']['text'], offset_ratio=c['label_offset_ratio'],
    )
    annotate_bar_values(
        ax_right, bars_bl_right, b_right, b_right_bottoms,
        fontsize=c['value_fontsize'], weight=c['value_weight'],
        color=c['colors']['text'], offset_ratio=c['label_offset_ratio'],
    )
    annotate_bar_values(
        ax_right, bars_it_right, i_right, i_right_bottoms,
        fontsize=c['value_fontsize'], weight=c['value_weight'],
        color=c['colors']['text'], offset_ratio=c['label_offset_ratio'],
    )

    handles, labels = ax_left.get_legend_handles_labels()
    ax_left.legend(
        handles,
        labels,
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        fontsize=c['legend_fontsize'],
    )

    fig.tight_layout()

    fig.text(
        0.5,
        0.97,
        model_name,
        ha='center',
        va='top',
        fontsize=5.0,
    )

    slug = model_name.lower().replace(' ', '_').replace('-', '_')
    out_dir = Path(out_dir)
    png = out_dir / f'fig3_intervention_sycophancy_strength_{slug}.png'
    pdf = out_dir / f'fig3_intervention_sycophancy_strength_{slug}.pdf'
    svg = out_dir / f'fig3_intervention_sycophancy_strength_{slug}.svg'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, facecolor='white', bbox_inches=None)
    plt.close(fig)
    return [png, pdf, svg]


def main(csv_path: str, out_dir: str) -> None:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['model'] = df['model'].map(norm_model)

    files = []
    for model in sorted(df['model'].unique()):
        files.extend(plot_model(df[df['model'] == model].copy(), model, out_p))

    print('Generated files:')
    for f in files:
        print(str(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot sycophancy intervention strength bars from aggregate CSV')
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH, help='汇总 CSV 路径')
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT_DIR, help='图像输出目录')
    args = parser.parse_args()
    main(csv_path=args.csv_path, out_dir=args.out_dir)
