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
        'axes.linewidth': 0.6,
        'axes.labelsize': 5.0,
        'axes.titlesize': 5.0,
        'xtick.labelsize': 5.0,
        'ytick.labelsize': 5.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
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
        'bar_edge_width': 0.6,
        'y_pad': 0.03,
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

DEFAULT_CSV_PATH = '/path/to/project_root/safety_explanation/hallucination/results/intervention/aggregate/intervention_accuracy_table_strength_0.1_0.3_0.5.csv'
DEFAULT_OUT_DIR = '/path/to/project_root/safety_explanation/hallucination/intervention_graph/output'


def apply_style():
    plt.rcParams.update(CONFIG['rcparams'])


def norm_model(x: str) -> str:
    key = str(x).strip().lower()
    return {
        'gemma-2-2b': 'Gemma-2-2B',
        'gemma-2-9b-it': 'Gemma-2-9B-IT',
    }.get(key, x)


def norm_dataset(x: str, project_type: str = 'hallucination') -> str:
    key = str(x).strip().lower()
    if project_type == 'fairness_bias':
        mapping = {
            'bbq_age': 'BBQ-Age',
            'bbq_nationality': 'BBQ-Nationality',
            'bbq_gender_identity': 'BBQ-Gender',
            'bbq_disability_status': 'BBQ-Disability',
        }
    elif project_type == 'jailbreak':
        mapping = {
            'jbb-behaviors': 'JBB-Behaviors',
            'advbench': 'AdvBench',
        }
    elif project_type == 'sycophancy':
        mapping = {
            'answer': 'Answer',
            'feedback': 'Feedback',
        }
    else:  # hallucination
        mapping = {
            'medhallu': 'MedHallu',
            'nq_open': 'NQ-Open',
            'truthfulqa': 'TruthfulQA',
        }
    return mapping.get(key, x)


def dataset_order_key(ds: str, project_type: str = 'hallucination') -> int:
    if project_type == 'fairness_bias':
        order = {'BBQ-Age': 0, 'BBQ-Nationality': 1, 'BBQ-Gender': 2, 'BBQ-Disability': 3}
    elif project_type == 'jailbreak':
        order = {'JBB-Behaviors': 0, 'AdvBench': 1}
    elif project_type == 'sycophancy':
        order = {'Answer': 0, 'Feedback': 1}
    else:  # hallucination
        order = {'MedHallu': 0, 'NQ-Open': 1, 'TruthfulQA': 2}
    return order.get(ds, 999)


def nice_limits(vmin: float, vmax: float, pad=0.03, step=0.05):
    lo = max(0.0, vmin - pad)
    hi = min(1.0, vmax + pad)
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step
    if hi - lo < 0.15:
        hi = min(1.0, lo + 0.15)
    return float(lo), float(hi)


def annotate_bar_values(ax, bars, values, bottoms, fontsize, weight, color, offset_ratio):
    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    for rect, v, bottom in zip(bars, values, bottoms):
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


def plot_model(df_model: pd.DataFrame, model_name: str, out_dir, project_type: str = 'hallucination'):
    apply_style()
    c = CONFIG['plot']
    slug = model_name.lower().replace(' ', '_').replace('-', '_')

    # 按 strength 排序，确保「第一个强度」定义明确
    df_model = df_model.sort_values('strength').reset_index(drop=True)

    datasets = sorted(df_model['dataset'].unique(), key=lambda x: dataset_order_key(x, project_type))

    # Baseline: 取第一个强度下的 baseline_accuracy
    baseline_agg = (
        df_model.groupby('dataset', as_index=False)
        .agg(baseline_mean=('baseline_accuracy', 'first'))
    )

    # Intervention: 取各强度下准确率最高的值，并记录对应的 best_strength
    best_idx = df_model.groupby('dataset')['intervention_accuracy'].idxmax()
    intervention_agg = (
        df_model.loc[best_idx, ['dataset', 'intervention_accuracy', 'strength']]
        .rename(columns={'intervention_accuracy': 'intervention_mean', 'strength': 'best_strength'})
        .reset_index(drop=True)
    )
    agg = baseline_agg.merge(intervention_agg, on='dataset')

    agg['__order'] = agg['dataset'].map(lambda x: dataset_order_key(x, project_type))
    agg = agg.sort_values(['__order', 'dataset']).drop(columns='__order').reset_index(drop=True)

    # 保存绘图所需的中间结果，方便后续直接复用而无需再次从原始表重算
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    mid_csv = out_dir_path / f'fig3_intervention_strength_{slug}_agg.csv'
    agg.to_csv(mid_csv, index=False)

    # 打印每个 dataset 选择的最佳干预强度
    print(f'  [{model_name}] Best intervention strength per dataset:')
    for _, row in agg.iterrows():
        print(f'    {row["dataset"]}: strength={row["best_strength"]} -> acc={row["intervention_mean"]*100:.2f}%')

    x = np.arange(len(datasets), dtype=float)
    w = c['bar_width']

    fig, ax = plt.subplots(1, 1, figsize=c['figsize'], dpi=CONFIG['rcparams']['figure.dpi'])

    # 根据项目类型设置单一 y 轴标签
    if project_type == 'fairness_bias':
        y_label = 'Accuracy'
    elif project_type == 'jailbreak':
        y_label = 'Accuracy'
    elif project_type == 'sycophancy':
        y_label = 'Accuracy'
    else:  # hallucination
        y_label = 'Accuracy'

    # 所有数据在同一个坐标轴上绘制（baseline 与 intervention 两组柱）
    b_vals = agg['baseline_mean'].to_numpy(float)
    i_vals = agg['intervention_mean'].to_numpy(float)

    # Axis limits（无方差，直接用 bar 值）；单一 y 轴范围
    all_vals = np.r_[b_vals, i_vals]
    if all_vals.size > 0:
        y_min, y_max = nice_limits(float(np.min(all_vals)), float(np.max(all_vals)), pad=c['y_pad'], step=c['y_step'])
    else:
        y_min, y_max = 0.0, 1.0

    ax.set_ylim(y_min, y_max)

    b_heights = np.maximum(b_vals - y_min, 0.0)
    i_heights = np.maximum(i_vals - y_min, 0.0)
    b_bottoms = np.full_like(b_vals, y_min, dtype=float)
    i_bottoms = np.full_like(i_vals, y_min, dtype=float)

    bars_bl = ax.bar(
        x - w / 2, b_heights, width=w, bottom=b_bottoms,
        color=c['colors']['baseline'], edgecolor=c['colors']['edge'],
        linewidth=c['bar_edge_width'], zorder=3, label=c['legend_labels']['baseline']
    )
    bars_it = ax.bar(
        x + w / 2, i_heights, width=w, bottom=i_bottoms,
        color=c['colors']['intervention'], edgecolor=c['colors']['edge'],
        linewidth=c['bar_edge_width'], zorder=3, label=c['legend_labels']['intervention']
    )

    for bar in bars_bl:
        bar.set_clip_on(False)
    for bar in bars_it:
        bar.set_clip_on(False)

    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=5.0, rotation=30, ha='right')

    ax.set_xlabel('Dataset', fontsize=5.0)
    ax.set_ylabel(y_label, fontsize=5.0)

    ax.tick_params(axis='both', labelsize=5.0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=c['grid_alpha'], linewidth=c['grid_line_width'])

    # Value labels (accuracy) on each bar.
    annotate_bar_values(
        ax,
        bars_bl,
        b_vals,
        b_bottoms,
        fontsize=c['value_fontsize'],
        weight=c['value_weight'],
        color=c['colors']['text'],
        offset_ratio=c['label_offset_ratio'],
    )
    annotate_bar_values(
        ax,
        bars_it,
        i_vals,
        i_bottoms,
        fontsize=c['value_fontsize'],
        weight=c['value_weight'],
        color=c['colors']['text'],
        offset_ratio=c['label_offset_ratio'],
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        fontsize=c['legend_fontsize'],
    )

    fig.tight_layout()

    # 将模型名称文字放在图例正下方，作为整体标题
    fig.text(
        0.5,
        0.97,
        model_name,
        ha='center',
        va='top',
        fontsize=5.0,
    )

    out_dir = Path(out_dir)
    png = out_dir / f'fig3_intervention_strength_{slug}.png'
    pdf = out_dir / f'fig3_intervention_strength_{slug}.pdf'
    svg = out_dir / f'fig3_intervention_strength_{slug}.svg'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, facecolor='white', bbox_inches=None)
    plt.close(fig)
    return [png, pdf, svg]


def main(csv_path, out_dir, project_type: str = 'hallucination'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['model'] = df['model'].map(norm_model)
    df['dataset'] = df['dataset'].map(lambda x: norm_dataset(x, project_type))

    files = []
    for model in sorted(df['model'].unique()):
        files.extend(plot_model(df[df['model'] == model].copy(), model, out_dir, project_type))

    print('Generated files:')
    for f in files:
        print(str(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot intervention strength bars')
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH, help=f'Path to input CSV file (default: {DEFAULT_CSV_PATH})')
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT_DIR, help=f'Output directory for plots (default: {DEFAULT_OUT_DIR})')
    parser.add_argument('--project-type', type=str, default='hallucination',
                        choices=['hallucination', 'fairness_bias', 'jailbreak', 'sycophancy'],
                        help='Project type: hallucination, fairness_bias, jailbreak, or sycophancy (default: hallucination)')
    args = parser.parse_args()
    main(csv_path=args.csv_path, out_dir=args.out_dir, project_type=args.project_type)
