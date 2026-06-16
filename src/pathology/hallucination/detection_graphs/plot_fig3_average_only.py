import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import colors as mcolors
from scipy import stats as scipy_stats

# 重新加载字体管理器以确保识别最新字体
try:
    fm.fontManager = fm.FontManager()
except Exception:
    pass


CONFIG = {
    'rcparams': {
        'font.family': 'Arial',
        'font.sans-serif': ['Arial'],
        'font.size': 5.0,
        'axes.linewidth': 0.8,
        'axes.labelsize': 5.0,
        'axes.titlesize': 5.0,
        'xtick.labelsize': 5.0,
        'ytick.labelsize': 5.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'figure.dpi': 450,
        'savefig.dpi': 450,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    },
    'avg_bar': {
        'figsize': (1.42, 1.05),
        'inset_figsize': (1.42, 1.42),
        'title_size': 5.0,
        'inset_title_size': 5.0,
        'xlabel_size': 5.0,
        'ylabel_size': 5.0,
        'bar_width': 0.72,
        'bar_edge_width': 0.75,
        'capsize': 2.6,
        'error_line_width': 0.6,
        'error_cap_thick': 0.6,
        'grid_alpha': 0.3,
        'grid_line_width': 0.5,
        'grid_linestyle': '--',
        'ymin': None,
        'ymax': None,
        'y_pad': 0.03,
        'y_step': 0.05,
        'scatter_size': 1,
        'scatter_alpha': 0.70,
        'scatter_jitter': 0.05,
        'scatter_dense_threshold': 0.015,
        'scatter_darken_factor': 0.75,
        'rotation': 28,
        'method_colors': [
            '#FAE6D7', '#F3C7BF', '#F0BBC1', '#F4E4B0',
            '#F7DC7C', '#B4DDF4', '#DE7D82', '#579FCA'
        ],
    },
}

DEFAULT_METRIC = 'auroc'
DEFAULT_CSV_PATH = '/path/to/project_root/safety_explanation/hallucination/results/detection/all_results/all_metrics_flat.csv'
DEFAULT_OUT_DIR = '/path/to/project_root/safety_explanation/hallucination/detection_graphs/output'


def apply_style():
    plt.rcParams.update(CONFIG['rcparams'])


def normalize_model_name(name: str) -> str:
    mapping = {
        'gemma-2-2b': 'Gemma-2-2B',
        'gemma-2-9b-it': 'Gemma-2-9B-IT'
    }
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


def short_method_label(name: str) -> str:
    mapping = {
        'Attention': 'Attn',
        'EigenScore': 'Eigen',
        'Entropy': 'Ent',
        'Hidden': 'Hidden',
        'Linear Probing': 'LP',
        'LN Entropy': 'LN',
        'PPL': 'PPL',
        'SelfCheckGPT': 'SCG',
        'Semantic Entropy': 'SemE',
        'NeuroCogMap': 'NCM',
    }
    return mapping.get(name, name)


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
        'jbb-behaviors': 'JBB-Behaviors',
        'advbench': 'AdvBench',
        'answer': 'Answer',
        'feedback': 'Feedback',
    }
    return mapping.get(str(name).strip().lower(), str(name))


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

    out['base_model'] = out['base_model'].map(normalize_model_name)
    out['dataset'] = out['dataset'].map(normalize_dataset_name)
    out['method_raw'] = out['method']
    out['method'] = out['method'].map(normalize_method_name)

    out = out[out['dataset'] != 'TriviaQA'].copy()
    out = out[~out['method_raw'].isin(['llm_detector', 'llm_detector_simple'])].copy()
    return out


def _nice_limits(ymin_raw, ymax_raw, pad=0.03, step=0.05):
    ymin = max(0.0, ymin_raw - pad)
    ymax = min(1.0, ymax_raw + pad)
    ymin = np.floor(ymin / step) * step
    ymax = np.ceil(ymax / step) * step
    if ymax - ymin < 0.10:
        ymax = min(1.0, ymin + 0.10)
    return float(ymin), float(ymax)


def _compute_axis_limits(md, method_order, c):
    grp = md.groupby(['dataset', 'method_raw'])['score_mean'].mean().reset_index()
    ymins = []
    ymaxs = []
    for m in method_order:
        if m not in set(md['method_raw']):
            continue
        vals = grp[grp['method_raw'] == m]['score_mean'].values
        if len(vals) == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ymins.append(min(float(np.min(vals)), mean - std))
        ymaxs.append(max(float(np.max(vals)), mean + std))
    if not ymins:
        return 0.0, 1.0
    return _nice_limits(min(ymins), max(ymaxs), pad=c['y_pad'], step=c['y_step'])


def _p_to_stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _format_p_text(p):
    if p < 1e-4:
        return 'p<0.0001'
    return f'p={p:.4f}'


def _darken_color(color, factor=0.78):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    rgb = np.clip(rgb * float(factor), 0.0, 1.0)
    return mcolors.to_hex(rgb)


def _adaptive_scatter_offsets(values, max_jitter=0.1, dense_threshold=0.015):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    offsets = np.zeros(n, dtype=float)
    if n <= 1:
        return offsets

    order = np.argsort(vals)
    sorted_vals = vals[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and (sorted_vals[j] - sorted_vals[j - 1]) <= dense_threshold:
            j += 1
        group = order[i:j]
        k = len(group)
        if k > 1:
            offsets[group] = np.linspace(-max_jitter, max_jitter, k)
        i = j
    return offsets


def _paired_t_pvalue_all_folds(md, method_a_raw, method_b_raw):
    datasets = sorted(set(md['dataset']))
    a_all = []
    b_all = []
    for ds in datasets:
        ra = md[(md['dataset'] == ds) & (md['method_raw'] == method_a_raw)]
        rb = md[(md['dataset'] == ds) & (md['method_raw'] == method_b_raw)]
        if ra.empty or rb.empty:
            continue
        xa = np.asarray(ra['score_values'].iloc[0], dtype=float)
        xb = np.asarray(rb['score_values'].iloc[0], dtype=float)
        n = min(len(xa), len(xb))
        if n == 0:
            continue
        a_all.extend(xa[:n].tolist())
        b_all.extend(xb[:n].tolist())
    if len(a_all) < 2:
        return np.nan
    res = scipy_stats.ttest_rel(
        np.asarray(a_all, dtype=float),
        np.asarray(b_all, dtype=float),
        alternative='two-sided'
    )
    return float(res.pvalue)


def _annotate_significance(ax, x, means, errs, methods_raw, compare_indices=None, compact_label=False):
    if 'our_method' not in methods_raw:
        return

    ref_idx = methods_raw.index('our_method')
    comp_indices = [i for i in range(len(methods_raw)) if i != ref_idx]
    if compare_indices is not None:
        comp_indices = [i for i in comp_indices if i in set(compare_indices)]
    if not comp_indices:
        return

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin
    tops = np.asarray(means) + np.asarray(errs)
    y_start = float(np.max(tops) + 0.03 * yr)
    step = 0.06 * yr
    h = 0.020 * yr
    max_y = y_start + len(comp_indices) * step + 0.05 * yr
    ax.set_ylim(ymin, max(ymax, max_y))

    comp_indices = sorted(comp_indices, key=lambda i: x[i])

    for k, i in enumerate(comp_indices):
        p = _paired_t_pvalue_all_folds(_annotate_significance.md, 'our_method', methods_raw[i])
        if not np.isfinite(p):
            continue

        y = y_start + k * step
        x1, x2 = x[i], x[ref_idx]
        left, right = (x1, x2) if x1 < x2 else (x2, x1)

        ax.plot(
            [left, left, right, right],
            [y - h, y, y, y - h],
            color='#333333',
            linewidth=0.7,
            zorder=5,
            clip_on=False,
        )

        stars = _p_to_stars(p)
        if compact_label:
            label = _format_p_text(p) + (f' {stars}' if stars else ' ns')
            fs = 5.0
        else:
            label = _format_p_text(p) + (f' {stars}' if stars else '')
            fs = 5.0

        ax.text(
            (left + right) / 2,
            y + 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            label,
            ha='center',
            va='bottom',
            fontsize=fs,
            color='#111111',
            zorder=6,
            clip_on=False,
        )


def _draw_average_on_ax(ax, md, metric, c, title, compact=False, annotate_sig=False):
    method_order = [
        'attention', 'entropy', 'linear_probing', 'ln_entropy',
        'ppl', 'hidden', 'selfcheckgpt', 'semantic_entropy', 'our_method'
    ]
    methods = [m for m in method_order if m in set(md['method_raw'])]
    labels_full = [normalize_method_name(m) for m in methods]
    labels = labels_full

    grp = md.groupby(['dataset', 'method_raw'])['score_mean'].mean().reset_index()
    means, errs, scatters = [], [], []
    for m in methods:
        vals = grp[grp['method_raw'] == m]['score_mean'].values
        means.append(float(np.mean(vals)))
        errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        scatters.append(vals)

    x = np.arange(len(methods))
    colors = c['method_colors']

    # 先算显示范围，再让柱子真实从 ymin 开始画，避免依赖 axes clipping
    ymin, ymax = _compute_axis_limits(md, method_order, c)
    bar_heights = np.asarray(means, dtype=float) - ymin
    bar_heights = np.maximum(bar_heights, 0.0)

    bars = ax.bar(
        x,
        bar_heights,
        bottom=ymin,
        width=c['bar_width'],
        yerr=errs,
        capsize=c['capsize'],
        error_kw={
            'elinewidth': c['error_line_width'],
            'capthick': c['error_cap_thick'],
            'ecolor': '#2b2b2b'
        },
        color=[colors[i % len(colors)] for i in x],
        edgecolor='#2b2b2b',
        linewidth=c['bar_edge_width'],
        zorder=2,
    )

    for bar in bars:
        bar.set_clip_on(False)

    if 'NeuroCogMap' in labels_full:
        idx = labels_full.index('NeuroCogMap')
        bars[idx].set_linewidth(1.0)

    for i, vals in enumerate(scatters):
        jitter = _adaptive_scatter_offsets(
            vals,
            max_jitter=c['scatter_jitter'],
            dense_threshold=c['scatter_dense_threshold'],
        )
        scatter_color = _darken_color(
            colors[i % len(colors)],
            factor=c['scatter_darken_factor']
        )
        ax.scatter(
            np.full(len(vals), x[i]) + jitter,
            vals,
            s=c['scatter_size'],
            facecolors=scatter_color,
            edgecolors='black',
            alpha=c['scatter_alpha'],
            linewidths=0.4,
            zorder=3,
            clip_on=False,
        )

    ax.set_title(
        title,
        fontsize=c['inset_title_size'] if compact else c['title_size'],
        fontweight='bold'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=c['rotation'],
        ha='right',
        fontsize=c['xlabel_size']
    )
    ax.set_ylim(ymin, ymax)
    ax.grid(
        axis='y',
        alpha=c['grid_alpha'],
        linewidth=c['grid_line_width'],
        linestyle=c['grid_linestyle'],
        zorder=0,
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if annotate_sig:
        if 'our_method' in methods:
            ref_idx = methods.index('our_method')
            baseline_indices = [i for i in range(len(methods)) if i != ref_idx]
            top2 = sorted(baseline_indices, key=lambda i: means[i], reverse=True)[:2]
        else:
            top2 = []

        _annotate_significance.md = md
        _annotate_significance(
            ax,
            x,
            means,
            errs,
            methods,
            compare_indices=top2,
            compact_label=compact
        )


def plot_average(df, out_dir, metric=DEFAULT_METRIC):
    apply_style()
    c = CONFIG['avg_bar']
    files = []
    out_dir = Path(out_dir)

    for model in sorted(df['base_model'].unique()):
        md = df[df['base_model'] == model]
        fig, ax = plt.subplots(
            1, 1,
            figsize=c['figsize'],
            dpi=CONFIG['rcparams']['figure.dpi']
        )
        _draw_average_on_ax(ax, md, metric, c, title=model, compact=False, annotate_sig=True)
        ax.set_ylabel(
            f'Mean {metric.upper()} across datasets',
            fontsize=c['ylabel_size'],
            fontweight='bold'
        )
        fig.tight_layout()

        slug = model.lower().replace(' ', '_').replace('-', '_')
        png = out_dir / f'fig3_methods_average_only_{slug}_{metric.lower()}.png'
        pdf = out_dir / f'fig3_methods_average_only_{slug}_{metric.lower()}.pdf'

        fig.savefig(png, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)
        files.extend([png, pdf])

    return files


def plot_average_per_model(df, out_dir, metric=DEFAULT_METRIC):
    apply_style()
    c = CONFIG['avg_bar']
    files = []
    out_dir = Path(out_dir)

    for model in sorted(df['base_model'].unique()):
        md = df[df['base_model'] == model]
        fig, ax = plt.subplots(
            1, 1,
            figsize=c['inset_figsize'],
            dpi=CONFIG['rcparams']['figure.dpi']
        )
        _draw_average_on_ax(ax, md, metric, c, title=model, compact=True, annotate_sig=True)
        ax.set_ylabel(metric.upper(), fontsize=5.0, fontweight='bold')

        fig.subplots_adjust(left=0.22, right=0.96, bottom=0.30, top=0.86)

        slug = model.lower().replace(' ', '_').replace('-', '_')
        svg = out_dir / f'fig3_methods_average_inset_{slug}_{metric.lower()}.svg'
        png = out_dir / f'fig3_methods_average_inset_{slug}_{metric.lower()}.png'

        # 对 SVG 不用 tight bbox，减少导入 AI 后的边界/裁剪复杂度
        fig.savefig(svg, facecolor='white', bbox_inches=None)
        fig.savefig(png, facecolor='white')
        plt.close(fig)
        files.extend([svg, png])

    return files


def main(csv_path, out_dir, metric=DEFAULT_METRIC):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_df(csv_path, metric=metric)
    files = plot_average_per_model(df, out_dir, metric=metric)

    print('Generated files:')
    for f in files:
        print(str(f))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot average detection performance')
    parser.add_argument(
        '--csv-path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f'Path to input CSV file (default: {DEFAULT_CSV_PATH})'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f'Output directory for plots (default: {DEFAULT_OUT_DIR})'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=DEFAULT_METRIC,
        help=f'Metric to plot (default: {DEFAULT_METRIC})'
    )
    args = parser.parse_args()
    main(csv_path=args.csv_path, out_dir=args.out_dir, metric=args.metric)
