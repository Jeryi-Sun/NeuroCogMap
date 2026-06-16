"""
Jailbreak 数据极坐标检测性能图（1/2 圆版本）。
使用 AdvBench 与 JBB-Behaviors 数据集的检测结果。
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

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
        'legend.fontsize': 5.0,
        'figure.dpi': 450,
        'savefig.dpi': 450,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    },
    'polar': {
        'figsize': (2.36, 1.70),  # 与 polar_only 统一宽度，半圆图压缩高度
        'dataset_labelsize': 5.0,
        'radial_ticksize': 5.0,
        'legend_size': 5.0,
        'legend_ncol': 2,
        'bg_edge_width': 0.5,
        'bar_edge_width': 0.55,
        'bar_edge_width_ours': 1.0,
        'error_line_width': 0.5,
        'show_grid': True,
        'grid_alpha': 0.22,
        'grid_line_width': 0.40,
        'grid_linestyle': '--',
        'outer_spine_width': 0.7,
        'center_circle_width': 0.7,
        'legend_anchor': (1.18, 1.08),
        'inner_blank_ratio': 0.38,
        'gap_ratio': 0.18,
        'theta_start': -np.pi / 2,
        'theta_span': np.pi,
        'sector_bg_alpha': 0.5,
        'sector_bg_colors': ['#FDF3EC', '#FDF1EE', '#FCEFF1', '#FCF6E6', '#FDF8E8', '#EEF6FC', '#FCEFF0'],
        # 方法颜色按冷暖交替和高对比顺序排列，提升相邻柱子的区分度，同时使用统一 Nature palette
        # 1) 主色优先: medium_blue, warm_yellow, rose_pink
        # 2) 冷暖交替: 蓝 → 黄 → 红 → 浅蓝 → 浅粉等
        'method_colors': [
            '#579FCA',  # medium_blue
            '#F7DC7C',  # warm_yellow
            '#DE7D82',  # teal_red (高对比强调色)
            '#B4DDF4',  # light_blue
            '#F0BBC1',  # rose_pink
            '#FAE6D7',  # light_peach
            '#F4E4B0',  # pale_yellow
            '#F3C7BF',  # soft_pink
        ],
    },
}

BASE_DIR = Path('/path/to/project_root/safety_explanation/jailbreak')
CSV_PATH = Path('/path/to/project_root/safety_explanation/jailbreak/results/detection/all_results/all_metrics_plot_auroc.csv')
OUT_DIR = BASE_DIR / 'detection_graphs/output'
DEFAULT_METRIC = 'auroc'
JBREAK_DATASETS = {'AdvBench', 'JBB-Behaviors'}


def apply_style():
    plt.rcParams.update(CONFIG['rcparams'])


def normalize_model_name(name: str) -> str:
    mapping = {'gemma-2-2b': 'Gemma-2-2B', 'gemma-2-9b-it': 'Gemma-2-9B-IT'}
    return mapping.get(str(name).strip().lower(), str(name))


def normalize_method_name(name: str) -> str:
    mapping = {
        'our_method': 'NeuroCogMap',
        'ours': 'NeuroCogMap',
        'smoothllm': 'SmoothLLM',
        'logits_svm': 'Logits SVM',
        'attention': 'Attention Probing',
        'hidden': 'Hidden Probing',
        'linear_probing': 'Hidden Probing',
        'eigenscore': 'EigenScore',
        'entropy': 'Entropy',
        'ln_entropy': 'LN Entropy',
        'ppl': 'Perplexity',
        'selfcheckgpt': 'SelfCheckGPT',
        'semantic_entropy': 'Semantic Entropy',
    }
    return mapping.get(str(name).strip().lower(), str(name))


def normalize_dataset_name(name: str) -> str:
    mapping = {
        'halueval': 'HaluEval',
        'medhallu': 'MedHallu',
        'dolly_close': 'Dolly-Closed',
        'nq_open': 'NQ-Open',
        'sciq': 'SciQ',
        'triviaqa': 'TriviaQA',
        'truthfulqa': 'TruthfulQA',
        'bbq_age': 'BBQ-Age',
        'bbq_nationality': 'BBQ-Nationality',
        'bbq_gender_identity': 'BBQ-Gender',
        'bbq_disability_status': 'BBQ-Disability',
    }
    return mapping.get(str(name).strip().lower(), str(name))


def dataset_sort_key(name: str) -> tuple:
    order = {
        'BBQ-Age': 0, 'BBQ-Nationality': 1, 'BBQ-Gender': 2, 'BBQ-Disability': 3,
        'HaluEval': 10, 'MedHallu': 11, 'Dolly-Closed': 12, 'NQ-Open': 13, 'SciQ': 14, 'TruthfulQA': 15,
    }
    return (order.get(name, 999), name)


def parse_values(x):
    # 容错解析：列表中可能含缺失的交叉验证折（NaN）。json.loads 原生支持 NaN，
    # 这里解析后丢弃 NaN 折，仅用有效折参与后续均值/方差计算。
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.asarray([], dtype=float)
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.asarray([], dtype=float)
    arr = np.asarray(json.loads(s), dtype=float)
    return arr[~np.isnan(arr)]


def load_df(csv_path=None, metric=DEFAULT_METRIC):
    path = Path(csv_path) if csv_path is not None else CSV_PATH
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise ValueError(f'Unknown metric: {metric}')

    vals = [parse_values(v) for v in df[metric]]
    out = df[['base_model', 'dataset', 'method']].copy()
    out['score_values'] = vals
    out['score_mean'] = [float(v.mean()) for v in vals]
    out['score_std'] = [float(v.std(ddof=1)) if len(v) > 1 else 0.0 for v in vals]

    out['base_model'] = out['base_model'].map(normalize_model_name)
    out['dataset'] = out['dataset'].map(normalize_dataset_name)
    out['method_raw'] = out['method']
    out['method'] = out['method'].map(normalize_method_name)

    out = out[out['dataset'] != 'TriviaQA'].copy()
    out = out[~out['method_raw'].isin(['llm_detector', 'llm_detector_simple'])].copy()
    out = out[out['dataset'].isin(JBREAK_DATASETS)].copy()
    if out.empty:
        raise ValueError('No jailbreak datasets (AdvBench, JBB-Behaviors) found in CSV.')
    return out


def build_matrix(df_model):
    datasets = sorted(df_model['dataset'].unique(), key=dataset_sort_key)
    # 优先展示 jailbreak 任务中的几种方法，NeuroCogMap 放在最后以便突出显示
    method_order = [
        'ppl',
        'logits_svm',
        'smoothllm',
        'attention',
        'entropy',
        'linear_probing',
        'ln_entropy',
        'hidden',
        'selfcheckgpt',
        'semantic_entropy',
        'our_method',
    ]

    methods = [m for m in method_order if m in set(df_model['method_raw'])]
    complete_methods = []
    for m in methods:
        sub = df_model[df_model['method_raw'] == m]
        if set(sub['dataset']) == set(datasets):
            complete_methods.append(m)
    methods = complete_methods

    D, M = len(datasets), len(methods)
    if D == 0 or M == 0:
        raise ValueError('No complete data to plot.')

    means = np.zeros((D, M), dtype=float)
    stds = np.zeros((D, M), dtype=float)
    for i, ds in enumerate(datasets):
        for j, m in enumerate(methods):
            row = df_model[(df_model['dataset'] == ds) & (df_model['method_raw'] == m)]
            means[i, j] = float(row['score_mean'].iloc[0])
            stds[i, j] = float(row['score_std'].iloc[0])

    labels = [normalize_method_name(m) for m in methods]
    return datasets, labels, means, stds


def plot_model(df_model, model_name, out_dir, metric=DEFAULT_METRIC):
    apply_style()
    p = CONFIG['polar']

    datasets, methods, means, stds = build_matrix(df_model)
    D, M = means.shape

    theta_start = p['theta_start']
    theta_span = p['theta_span']
    sector = theta_span / D
    thetas = theta_start + (np.arange(D) + 0.5) * sector
    w = sector * (1 - 2 * p['gap_ratio']) / M

    fig = plt.figure(figsize=p['figsize'], dpi=CONFIG['rcparams']['figure.dpi'])
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetalim(theta_start, theta_start + theta_span)

    vmax = max(1.0, float(np.max(means + stds)))
    vals = np.clip(means / vmax, 0, 1)
    ers = np.clip(stds / vmax, 0, 1)

    inner = p['inner_blank_ratio']
    span = 1.0 - inner

    sector_bg = p['sector_bg_colors']
    for d in range(D):
        ax.bar(
            thetas[d],
            span,
            width=sector * 0.98,
            bottom=inner,
            color=sector_bg[d % len(sector_bg)],
            edgecolor='#d7d7d7',
            linewidth=p['bg_edge_width'],
            alpha=p['sector_bg_alpha'],
            zorder=0,
            align='center',
        )

    colors = p['method_colors']
    ours_idx = methods.index('NeuroCogMap') if 'NeuroCogMap' in methods else -1

    for d in range(D):
        start = thetas[d] - sector / 2 + sector * p['gap_ratio']
        for m in range(M):
            theta = start + (m + 0.5) * w
            r = float(vals[d, m])
            e = float(ers[d, m])
            lw = p['bar_edge_width_ours'] if m == ours_idx else p['bar_edge_width']
            bar_color = colors[m % len(colors)]

            ax.bar(theta, r * span, width=w * 0.88, bottom=inner, linewidth=lw, edgecolor='#2b2b2b', color=bar_color, alpha=1.0, zorder=3)

            low = max(0.0, r - e)
            high = min(1.0, r + e)
            y0 = inner + low * span
            y1 = inner + high * span
            ax.plot([theta, theta], [y0, y1], linewidth=p['error_line_width'], color='#2b2b2b', zorder=4)
            ax.plot([theta - w * 0.14, theta + w * 0.14], [y1, y1], linewidth=p['error_line_width'], color='#2b2b2b', zorder=4)

    ax.set_ylim(0, 1.08)
    ax.bar(theta_start + theta_span / 2, inner - 0.005, width=theta_span, bottom=0, color='white', edgecolor='none', zorder=5)
    circle = plt.Circle((0, 0), inner, transform=ax.transData._b, color='white', ec='#d0d0d0', lw=p['center_circle_width'], zorder=6)
    ax.add_artist(circle)

    ax.set_xticks(thetas)
    ax.set_xticklabels(datasets, fontsize=p['dataset_labelsize'])
    yt = np.linspace(inner, 1.0, 4)
    ax.set_yticks(yt)
    radial_labels = np.linspace(0.25 * vmax, vmax, 4)
    radial_labels[-1] = 1.0
    ax.set_yticklabels([f'{v:.2f}' for v in radial_labels], fontsize=p['radial_ticksize'])
    ax.set_rlabel_position(0)

    ax.tick_params(axis='x', which='major', length=2.8, width=0.6, pad=2, color='#666666')
    ax.tick_params(axis='y', which='major', length=2.3, width=0.6, pad=1, color='#666666')
    if p['show_grid']:
        ax.grid(alpha=p['grid_alpha'], color='#9a9a9a', linewidth=p['grid_line_width'], linestyle=p['grid_linestyle'])
    else:
        ax.grid(False)
    ax.spines['polar'].set_linewidth(p['outer_spine_width'])
    ax.spines['polar'].set_color('#444444')

    handles = []
    for m in range(M):
        lw = p['bar_edge_width_ours'] if m == ours_idx else p['bar_edge_width']
        bar_color = colors[m % len(colors)]
        handles.append(Rectangle((0, 0), 1, 1, facecolor=bar_color, edgecolor='#2b2b2b', linewidth=lw))

    ax.legend(
        handles,
        methods,
        frameon=False,
        fontsize=p['legend_size'],
        loc='upper right',
        bbox_to_anchor=p['legend_anchor'],
        borderaxespad=0,
        ncol=p['legend_ncol'],
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.6,
        labelspacing=0.4,
    )

    fig.subplots_adjust(left=0.03, right=0.86, top=0.89, bottom=0.03)
    slug = model_name.lower().replace(' ', '_').replace('-', '_')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f'fig3_polar_half_{slug}_{metric.lower()}.png'
    pdf = out_dir / f'fig3_polar_half_{slug}_{metric.lower()}.pdf'
    svg = out_dir / f'fig3_polar_half_{slug}_{metric.lower()}.svg'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    return [png, pdf, svg]


def main(csv_path=None, out_dir=None, metric=DEFAULT_METRIC):
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_df(csv_path=csv_path, metric=metric)
    files = []
    for model in sorted(df['base_model'].unique()):
        files.extend(plot_model(df[df['base_model'] == model], model, out_dir, metric=metric))
    print('Generated files:')
    for f in files:
        print(str(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot polar detection (half circle) for bias data')
    parser.add_argument('--csv-path', type=str, default=None, help='Path to input CSV (default: fairness_bias all_metrics_flat.csv)')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory (default: fairness_bias/detection_graphs/output)')
    parser.add_argument('--metric', type=str, default=DEFAULT_METRIC, help='Metric to plot')
    args = parser.parse_args()
    main(csv_path=args.csv_path, out_dir=args.out_dir, metric=args.metric)
