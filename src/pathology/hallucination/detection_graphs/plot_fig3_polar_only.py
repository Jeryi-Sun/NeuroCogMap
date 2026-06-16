import argparse
import ast
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
        'font.family': 'sans-serif',  # 全局字体族，配合下面 Arial 使用
        'font.sans-serif': ['Arial'],  # 强制优先使用 Arial
        'font.size': 5.0,  # 全局基础字号
        'axes.linewidth': 0.8,  # 坐标轴边框线宽
        'axes.labelsize': 5.0,  # 坐标轴标签字号
        'axes.titlesize': 5.0,  # 标题字号（当前图未显示标题）
        'xtick.labelsize': 5.0,  # x 轴刻度标签字号
        'ytick.labelsize': 5.0,  # y 轴刻度标签字号
        'xtick.major.width': 0.7,  # x 轴主刻度线宽
        'ytick.major.width': 0.7,  # y 轴主刻度线宽
        'legend.fontsize': 5.0,  # 图例文字字号
        'figure.dpi': 450,  # 画布显示分辨率
        'savefig.dpi': 450,  # 导出分辨率
    },
    'polar': {
        'figsize': (2.36, 2.36),  # 图尺寸（英寸），约 60mm * 60mm
        'dataset_labelsize': 5.0,  # 数据集名称标签字号
        'radial_ticksize': 5.0,  # 径向刻度标签字号
        'legend_size': 5.0,  # 图例字号
        'legend_ncol': 2,  # 图例列数
        'title_size': 5.0,  # 标题字号（保留参数，当前不显示标题）
        'title_pad': 16,  # 标题与图面间距（保留参数，当前不显示标题）
        'bg_edge_width': 0.5,  # 扇区背景边框线宽
        'bar_edge_width': 0.55,  # 常规方法柱子边框线宽
        'bar_edge_width_ours': 1.0,  # NeuroCogMap 柱子边框线宽（强调）
        'error_line_width': 0.5,  # 误差线线宽
        'show_grid': True,  # 是否显示极坐标网格
        'grid_alpha': 0.22,  # 网格透明度
        'grid_line_width': 0.40,  # 网格线宽
        'grid_linestyle': '--',  # 网格线样式
        'outer_spine_width': 0.7,  # 最外圈边框线宽
        'center_circle_width': 0.7,  # 中心白色圆环边框线宽
        'legend_anchor': (1.18, 1.08),  # 图例锚点位置（axes 坐标）
        'inner_blank_ratio': 0.38,  # 中心留白半径占比
        'gap_ratio': 0.18,  # 每个数据集扇区内左右留白占比
        'sector_bg_alpha': 0.5,  # 扇区背景透明度
        'sector_bg_colors': ['#FDF3EC', '#FDF1EE', '#FCEFF1', '#FCF6E6', '#FDF8E8', '#EEF6FC', '#FCEFF0'],  # 数据集扇区背景色
        'method_colors': ['#FAE6D7', '#F3C7BF', '#F0BBC1', '#F4E4B0', '#F7DC7C', '#B4DDF4', '#DE7D82', '#579FCA'],  # 方法柱子配色
    },
}

DEFAULT_METRIC = 'auroc'  # 默认绘制指标
DEFAULT_CSV_PATH = '/path/to/project_root/safety_explanation/hallucination/results/detection/all_results/all_metrics_flat.csv'  # 默认输入 CSV
DEFAULT_OUT_DIR = '/path/to/project_root/safety_explanation/hallucination/detection_graphs/output'  # 默认输出目录


def apply_style():
    plt.rcParams.update(CONFIG['rcparams'])


def normalize_model_name(name: str) -> str:
    mapping = {'gemma-2-2b': 'Gemma-2-2B', 'gemma-2-9b-it': 'Gemma-2-9B-IT'}
    return mapping.get(str(name).strip().lower(), str(name))


def normalize_method_name(name: str) -> str:
    mapping = {
        'our_method': 'NeuroCogMap',
        'ours': 'NeuroCogMap',
        'attention': 'Attention Probing',
        'eigenscore': 'EigenScore',
        'entropy': 'Entropy',
        'hidden': 'Hidden Probing',
        'linear_probing': 'Hidden Probing',
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
        # fairness_bias datasets
        'bbq_age': 'BBQ-Age',
        'bbq_nationality': 'BBQ-Nationality',
        'bbq_gender_identity': 'BBQ-Gender',
        'bbq_disability_status': 'BBQ-Disability',
        # jailbreak datasets
        'jbb-behaviors': 'JBB-Behaviors',
        'advbench': 'AdvBench',
        # sycophancy datasets
        'answer': 'Answer',
        'feedback': 'Feedback',
    }
    return mapping.get(str(name).strip().lower(), str(name))


def dataset_sort_key(name: str) -> tuple:
    order = {
        'HaluEval': 0, 'MedHallu': 1, 'Dolly-Closed': 2, 'NQ-Open': 3, 'SciQ': 4, 'TruthfulQA': 5,
        # fairness_bias datasets
        'BBQ-Age': 6, 'BBQ-Nationality': 7, 'BBQ-Gender': 8, 'BBQ-Disability': 9,
        # jailbreak datasets
        'JBB-Behaviors': 10, 'AdvBench': 11,
        # sycophancy datasets
        'Answer': 12, 'Feedback': 13,
    }
    return (order.get(name, 999), name)


def parse_values(x):
    return np.asarray(ast.literal_eval(x), dtype=float)


def load_df(csv_path, metric=DEFAULT_METRIC):
    df = pd.read_csv(csv_path)
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
    return out


def build_matrix(df_model):
    datasets = sorted(df_model['dataset'].unique(), key=dataset_sort_key)
    # 包含 hallucination 与 fairness_bias 等任务的全部方案，绘图时只保留 CSV 中存在且数据完整的方法
    method_order = ['attention', 'entropy', 'linear_probing', 'ln_entropy', 'ppl', 'hidden', 'selfcheckgpt', 'semantic_entropy', 'our_method']

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

    thetas = np.linspace(0, 2 * np.pi, D, endpoint=False)
    sector = 2 * np.pi / D
    w = sector * (1 - 2 * p['gap_ratio']) / M

    fig = plt.figure(figsize=p['figsize'], dpi=CONFIG['rcparams']['figure.dpi'])
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

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

            ax.bar(theta, r * span, width=w * 0.92, bottom=inner, linewidth=lw, edgecolor='#2b2b2b', color=colors[m % len(colors)], alpha=1.0, zorder=3)

            low = max(0.0, r - e)
            high = min(1.0, r + e)
            y0 = inner + low * span
            y1 = inner + high * span
            ax.plot([theta, theta], [y0, y1], linewidth=p['error_line_width'], color='#2b2b2b', zorder=4)
            ax.plot([theta - w * 0.14, theta + w * 0.14], [y1, y1], linewidth=p['error_line_width'], color='#2b2b2b', zorder=4)

    ax.set_ylim(0, 1.08)
    ax.bar(0, inner - 0.005, width=2 * np.pi, bottom=0, color='white', edgecolor='none', zorder=5)
    circle = plt.Circle((0, 0), inner, transform=ax.transData._b, color='white', ec='#d0d0d0', lw=p['center_circle_width'], zorder=6)
    ax.add_artist(circle)

    ax.set_xticks(thetas)
    ax.set_xticklabels(datasets, fontsize=p['dataset_labelsize'])
    yt = np.linspace(inner, 1.0, 4)
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(0.25 * vmax, vmax, 4)], fontsize=p['radial_ticksize'])
    ax.set_rlabel_position(90)
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
        handles.append(Rectangle((0, 0), 1, 1, facecolor=colors[m % len(colors)], edgecolor='#2b2b2b', linewidth=lw))

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
    png = out_dir / f'fig3_polar_only_{slug}_{metric.lower()}.png'
    pdf = out_dir / f'fig3_polar_only_{slug}_{metric.lower()}.pdf'
    svg = out_dir / f'fig3_polar_only_{slug}_{metric.lower()}.svg'
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    return [png, pdf, svg]


def main(csv_path, out_dir, metric=DEFAULT_METRIC):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_df(csv_path, metric=metric)
    files = []
    for model in sorted(df['base_model'].unique()):
        files.extend(plot_model(df[df['base_model'] == model], model, out_dir, metric=metric))
    print('Generated files:')
    for f in files:
        print(str(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot polar detection performance')
    parser.add_argument('--csv-path', type=str, default=DEFAULT_CSV_PATH, help=f'Path to input CSV file (default: {DEFAULT_CSV_PATH})')  # 输入结果 CSV 路径
    parser.add_argument('--out-dir', type=str, default=DEFAULT_OUT_DIR, help=f'Output directory for plots (default: {DEFAULT_OUT_DIR})')  # 图像输出目录
    parser.add_argument('--metric', type=str, default=DEFAULT_METRIC, help=f'Metric to plot (default: {DEFAULT_METRIC})')  # 要绘制的指标名
    args = parser.parse_args()
    main(csv_path=args.csv_path, out_dir=args.out_dir, metric=args.metric)
