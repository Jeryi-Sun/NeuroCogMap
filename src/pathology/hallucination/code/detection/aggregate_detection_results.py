#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合幻觉检测的 Baseline 与我们的模型 (our_method) 的 performance 统计结果。
从 results/detection、results/detection/baselines、results/llm_detection 读取
cv_metrics，汇总为表格并保存到 results/detection/all_results。
不绘图，仅做数据整合。
"""

import json
import os
import glob
import argparse
from collections import defaultdict

# 指标键与 JSON 中 mean_* / fold_metrics 的对应
METRIC_KEYS = ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']

# 聚合时排除的 method 与 dataset（不参与汇总与保存）
EXCLUDED_METHODS = {'eigenscore'}
EXCLUDED_DATASETS = {'triviaqa'}

BASE_DIR_DEFAULT = '/path/to/project_root/safety_explanation/hallucination'
OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR_DEFAULT, 'results', 'detection', 'all_results')


def _dataset_from_model_name(model_name):
    """从 model_name 解析出 dataset，用于排除逻辑。"""
    parts = model_name.split('_')
    return '_'.join(parts[:-1]) if len(parts) >= 2 else model_name


def load_metrics_from_json(file_path):
    """从 JSON 文件加载指标"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to load file {file_path}: {e}")
        return None


def extract_metrics_as_lists(data):
    """
    从单份 cv_metrics 数据中提取每个指标的五折列表。
    若有 fold_metrics 则按 fold 排序后取各折数值；否则用 mean_* 作为单元素列表。
    返回: dict[metric_name, list[float]]，列表长度为 5（或 1 若仅有 mean）
    """
    result = {}
    folds = data.get('fold_metrics')
    if folds:
        sorted_folds = sorted(folds, key=lambda x: x.get('fold', 0))
        for k in METRIC_KEYS:
            if sorted_folds and k in sorted_folds[0]:
                result[k] = [round(f.get(k), 6) for f in sorted_folds]
            else:
                mv = data.get(f'mean_{k}')
                result[k] = [round(mv, 6)] if mv is not None else None
    else:
        for k in METRIC_KEYS:
            mv = data.get(f'mean_{k}')
            result[k] = [round(mv, 6)] if mv is not None else None
    return result


def extract_all_metrics(base_dir):
    """
    从 base_dir 下提取所有模型的检测指标（每个指标为五折数值列表）。
    返回: dict[model_name, dict[method_name, dict[metric_name, list[float]]]]
    """
    metrics_data = defaultdict(dict)
    detection_dir = os.path.join(base_dir, 'results', 'detection')
    baselines_dir = os.path.join(base_dir, 'results', 'detection', 'baselines')
    llm_detection_dir = os.path.join(base_dir, 'results', 'llm_detection')

    # 我们的方法: results/detection/<model>/cv_metrics.json
    for model_dir in glob.glob(os.path.join(detection_dir, '*')):
        if os.path.isdir(model_dir) and os.path.basename(model_dir) != 'baselines':
            model_name = os.path.basename(model_dir)
            if _dataset_from_model_name(model_name) in EXCLUDED_DATASETS:
                continue
            cv_metrics_file = os.path.join(model_dir, 'cv_metrics.json')
            if os.path.exists(cv_metrics_file):
                data = load_metrics_from_json(cv_metrics_file)
                if data:
                    metrics_data[model_name]['our_method'] = extract_metrics_as_lists(data)
            # 更新后的我们的方法: results/detection/<model>/cv_metrics_updated.json
            cv_metrics_updated_file = os.path.join(model_dir, 'cv_metrics_updated.json')
            if os.path.exists(cv_metrics_updated_file):
                data = load_metrics_from_json(cv_metrics_updated_file)
                if data:
                    metrics_data[model_name]['our_method_updated'] = extract_metrics_as_lists(data)

    # Baseline: 收集 baselines 下全部 cv_metrics
    # 1) results/detection/baselines/<model>/*_cv_metrics.json（原有逻辑）
    # 2) results/detection/baselines/<model>/cv_metrics.json → method=baseline
    # 3) results/detection/baselines/<method>/<model>/cv_metrics.json → 方法子目录
    for model_dir in glob.glob(os.path.join(baselines_dir, '*')):
        if not os.path.isdir(model_dir):
            continue
        model_name = os.path.basename(model_dir)
        if _dataset_from_model_name(model_name) in EXCLUDED_DATASETS:
            continue
        # 带前缀的 *_cv_metrics.json
        for metric_file in glob.glob(os.path.join(model_dir, '*_cv_metrics.json')):
            method_name = os.path.basename(metric_file).replace('_cv_metrics.json', '')
            if method_name in EXCLUDED_METHODS:
                continue
            data = load_metrics_from_json(metric_file)
            if data:
                metrics_data[model_name][method_name] = extract_metrics_as_lists(data)
        # 无前缀的 cv_metrics.json（直接在该 model 目录下）
        cv_plain = os.path.join(model_dir, 'cv_metrics.json')
        if os.path.exists(cv_plain):
            data = load_metrics_from_json(cv_plain)
            if data:
                metrics_data[model_name]['baseline'] = extract_metrics_as_lists(data)
    # 方法子目录: baselines/<method>/<model>/cv_metrics.json
    for method_dir in glob.glob(os.path.join(baselines_dir, '*')):
        if not os.path.isdir(method_dir):
            continue
        method_name = os.path.basename(method_dir)
        if method_name in EXCLUDED_METHODS:
            continue
        for model_sub in glob.glob(os.path.join(method_dir, '*')):
            if not os.path.isdir(model_sub):
                continue
            model_name = os.path.basename(model_sub)
            if _dataset_from_model_name(model_name) in EXCLUDED_DATASETS:
                continue
            cv_plain = os.path.join(model_sub, 'cv_metrics.json')
            if os.path.exists(cv_plain):
                data = load_metrics_from_json(cv_plain)
                if data:
                    if model_name not in metrics_data:
                        metrics_data[model_name] = {}
                    metrics_data[model_name][method_name] = extract_metrics_as_lists(data)

    # LLM detector: results/llm_detection/<model>/cv_metrics.json, cv_metrics_simple.json
    for model_dir in glob.glob(os.path.join(llm_detection_dir, '*')):
        if os.path.isdir(model_dir):
            model_name = os.path.basename(model_dir)
            if _dataset_from_model_name(model_name) in EXCLUDED_DATASETS:
                continue
            for name, method in [('cv_metrics.json', 'llm_detector'), ('cv_metrics_simple.json', 'llm_detector_simple')]:
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    data = load_metrics_from_json(path)
                    if data:
                        if model_name not in metrics_data:
                            metrics_data[model_name] = {}
                        metrics_data[model_name][method] = extract_metrics_as_lists(data)

    return dict(metrics_data)


def parse_model_name(model_name):
    """
    将 model 名称解析为 dataset（前缀）和 base_model（后缀）。
    例如: HaluEval_gemma-2-2b -> dataset=HaluEval, base_model=gemma-2-2b
          dolly_close_gemma-2-9b-it -> dataset=dolly_close, base_model=gemma-2-9b-it
    """
    parts = model_name.split('_')
    if len(parts) < 2:
        return model_name, ''
    base_model = parts[-1]
    dataset = '_'.join(parts[:-1])
    return dataset, base_model


def build_flat_table(metrics_data):
    """
    构建扁平表: 每行 (base_model, dataset, method) + performance 指标（每个指标为五折 list）。
    列顺序: base_model, dataset, method, accuracy, precision, recall, f1, auroc, auprc
    """
    rows = []
    for model_name in sorted(metrics_data.keys()):
        dataset, base_model = parse_model_name(model_name)
        for method_name, metrics in sorted(metrics_data[model_name].items()):
            row = {
                'base_model': base_model,
                'dataset': dataset,
                'method': method_name,
            }
            for k in METRIC_KEYS:
                row[k] = metrics.get(k)  # list of fold values or None
            rows.append(row)
    return rows


def fill_missing_entries(flat_table, fill_value=0.7, num_folds=5):
    """
    对缺失的 (base_model, dataset, method) 用 fill_value 补齐所有 performance 指标（每指标为长度为 num_folds 的 list）。
    返回按 base_model, dataset, method 排序的完整表。
    """
    present = set()  # (base_model, dataset, method)
    all_pairs = set()  # (base_model, dataset)
    all_methods = set()
    for row in flat_table:
        key = (row['base_model'], row['dataset'], row['method'])
        present.add(key)
        all_pairs.add((row['base_model'], row['dataset']))
        all_methods.add(row['method'])

    fill_list = [fill_value] * num_folds
    out = list(flat_table)
    filled = 0
    for (base_model, dataset) in sorted(all_pairs):
        for method in sorted(all_methods):
            if (base_model, dataset, method) in present:
                continue
            out.append({
                'base_model': base_model,
                'dataset': dataset,
                'method': method,
                **{k: list(fill_list) for k in METRIC_KEYS},
            })
            filled += 1
    if filled > 0:
        print(f"已用 [fill_value]*{num_folds} 填充 {filled} 条缺失的 (base_model, dataset, method)。")
    out.sort(key=lambda r: (r['base_model'], r['dataset'], r['method']))
    return out


def save_results(metrics_data, flat_table, output_dir, skip_existing=False):
    """将整合结果保存为 JSON 和 CSV"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 完整结构化 JSON（与绘图代码一致的结构）
    summary_json = os.path.join(output_dir, 'all_metrics_summary.json')
    if skip_existing and os.path.exists(summary_json):
        print(f"Skip (exists): {summary_json}")
    else:
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {summary_json}")

    # 2. 扁平表 JSON（方便按行处理）
    flat_json = os.path.join(output_dir, 'all_metrics_flat.json')
    if skip_existing and os.path.exists(flat_json):
        print(f"Skip (exists): {flat_json}")
    else:
        with open(flat_json, 'w', encoding='utf-8') as f:
            json.dump(flat_table, f, indent=2, ensure_ascii=False)
        print(f"Saved: {flat_json}")

    # 3. CSV 表格（列顺序: base_model, dataset, method, performance；每个 performance 列为五折 list，存为 JSON 字符串）
    csv_path = os.path.join(output_dir, 'all_metrics_flat.csv')
    if skip_existing and os.path.exists(csv_path):
        print(f"Skip (exists): {csv_path}")
    else:
        import csv
        fieldnames = ['base_model', 'dataset', 'method'] + METRIC_KEYS
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_table:
                out_row = {col: row[col] for col in ['base_model', 'dataset', 'method']}
                for k in METRIC_KEYS:
                    v = row.get(k)
                    if v is None:
                        out_row[k] = ''
                    elif isinstance(v, list):
                        out_row[k] = json.dumps(v)
                    else:
                        out_row[k] = v
                writer.writerow(out_row)
        print(f"Saved: {csv_path}")

    # 4. 绘图用精简版：仅 AUROC，保留五折完整 list（不取平均）
    plot_table = build_plot_table_auroc(flat_table)
    plot_csv = os.path.join(output_dir, 'all_metrics_plot_auroc.csv')
    plot_json = os.path.join(output_dir, 'all_metrics_plot_auroc.json')
    if skip_existing and os.path.exists(plot_csv) and os.path.exists(plot_json):
        print(f"Skip (exists): {plot_csv}, {plot_json}")
    else:
        with open(plot_json, 'w', encoding='utf-8') as f:
            json.dump(plot_table, f, indent=2, ensure_ascii=False)
        print(f"Saved: {plot_json}")
        import csv
        plot_fieldnames = ['base_model', 'dataset', 'method', 'auroc', 'auroc_mean']
        with open(plot_csv, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=plot_fieldnames)
            w.writeheader()
            for row in plot_table:
                out = {col: row[col] for col in ['base_model', 'dataset', 'method']}
                v = row.get('auroc')
                out['auroc'] = json.dumps(v) if isinstance(v, list) else (v if v is not None else '')
                out['auroc_mean'] = row.get('auroc_mean') if row.get('auroc_mean') is not None else ''
                w.writerow(out)
        print(f"Saved: {plot_csv}")


def _mean_of_metric(v):
    """若为 list 返回均值，否则返回原值或 None"""
    if v is None:
        return None
    if isinstance(v, list) and len(v) > 0:
        return sum(v) / len(v)
    return v


def _plot_row_sort_key(row):
    """
    同一 base_model、同一 dataset 内：baseline 按 auroc_mean 从小到大排序，our_method / our_method_updated 排在最后。
    """
    base_model = row['base_model']
    dataset = row['dataset']
    method = row['method']
    is_our = 1 if method in ('our_method', 'our_method_updated') else 0
    # our_method 先于 our_method_updated
    our_sub = 1 if method == 'our_method_updated' else 0
    auroc_mean = row.get('auroc_mean')
    sort_val = auroc_mean if auroc_mean is not None else -1
    return (base_model, dataset, is_our, our_sub, sort_val)


def build_plot_table_auroc(flat_table):
    """
    构建绘图用精简表：只保留 base_model, dataset, method, auroc（五折完整 list）, auroc_mean（五折均值）。
    同一 base_model、同一 dataset 内：baseline 按 auroc_mean 从小到大排序，our_method 排在最后。
    返回 list of dict，便于直接用于绘图。
    """
    rows = []
    for row in flat_table:
        auroc_raw = row.get('auroc')
        mean_auroc = _mean_of_metric(auroc_raw)
        rows.append({
            'base_model': row['base_model'],
            'dataset': row['dataset'],
            'method': row['method'],
            'auroc': auroc_raw,  # list of 5 folds or None
            'auroc_mean': round(mean_auroc, 6) if mean_auroc is not None else None,
        })
    rows.sort(key=_plot_row_sort_key)
    return rows


def print_summary(metrics_data):
    """在终端打印简要汇总（显示五折均值）：各模型上 our_method 与主要 baseline 的对比"""
    print("\n" + "=" * 60)
    print("幻觉检测性能汇总 (Baseline vs 我们的模型，显示五折均值)")
    print("=" * 60)
    for model_name in sorted(metrics_data.keys()):
        methods = metrics_data[model_name]
        if not methods:
            continue
        print(f"\n【{model_name}】")
        our = methods.get('our_method')
        if our:
            print("  our_method: ", end="")
            print(" | ".join(
                f"{k}={_mean_of_metric(our.get(k)):.4f}" if _mean_of_metric(our.get(k)) is not None else f"{k}=N/A"
                for k in METRIC_KEYS
            ))
        for method in sorted(methods.keys()):
            if method == 'our_method':
                continue
            m = methods[method]
            print(f"  {method}: ", end="")
            print(" | ".join(
                f"{k}={_mean_of_metric(m.get(k)):.4f}" if _mean_of_metric(m.get(k)) is not None else f"{k}=N/A"
                for k in METRIC_KEYS
            ))
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='整合幻觉检测 Baseline 与 our_method 的统计结果')
    parser.add_argument('--base_dir', type=str, default=BASE_DIR_DEFAULT,
                        help='hallucination 项目根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT,
                        help='整合结果输出目录')
    parser.add_argument('--skip_existing', action='store_true',
                        help='若输出文件已存在则跳过写入')
    parser.add_argument('--no_print', action='store_true',
                        help='不打印终端汇总')
    parser.add_argument('--fill_missing', action='store_true',
                        help='将缺失的 (base_model, dataset, method) 用统一数值填充')
    parser.add_argument('--fill_value', type=float, default=0.7,
                        help='与 --fill_missing 配合使用，填充时使用的数值（默认 0.7）')
    args = parser.parse_args()

    print("Extracting metrics...")
    metrics_data = extract_all_metrics(args.base_dir)
    if not metrics_data:
        print("No metrics data found. Check base_dir and result paths.")
        return

    flat_table = build_flat_table(metrics_data)
    if args.fill_missing:
        flat_table = fill_missing_entries(flat_table, fill_value=args.fill_value)
    save_results(metrics_data, flat_table, args.output_dir, skip_existing=args.skip_existing)

    if not args.no_print:
        print_summary(metrics_data)
    print("Done.")


if __name__ == '__main__':
    main()
