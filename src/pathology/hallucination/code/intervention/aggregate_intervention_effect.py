#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总干预效果统计：扫描 results/intervention 下各干预强度目录中的 *_intervention.json
（不读取带 analysis 后缀的文件），从 evaluation.baseline_evaluations 与
evaluation.intervention_evaluations 中的 is_correct 统计 baseline 正确/错误、
被干预改正/破坏数，并输出汇总表（JSON/CSV）。
"""

import argparse
import json
import os
import glob
from collections import defaultdict

BASE_DIR_DEFAULT = '/path/to/project_root/safety_explanation/hallucination'
INTERVENTION_RESULTS_DEFAULT = os.path.join(BASE_DIR_DEFAULT, 'results', 'intervention')
OUTPUT_DIR_DEFAULT = os.path.join(INTERVENTION_RESULTS_DEFAULT, 'aggregate')


def load_json(file_path):
    """从 JSON 文件加载数据。失败时打印异常并返回 None。"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"文件不存在: {file_path}. 详情: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {file_path}. 详情: {e}")
        return None
    except Exception as e:
        print(f"加载失败: {file_path}. 详情: {e}")
        return None


def extract_stats_from_intervention(data):
    """
    从 *_intervention.json 的 evaluation 中提取统计量。
    evaluation.baseline_evaluations[i].is_correct 为 baseline 准确性，
    evaluation.intervention_evaluations[i].is_correct 为干预结果（按索引一一对应）。
    返回: dict 含 baseline_correct, baseline_wrong, corrected_by_intervention,
          degraded_by_intervention, total_baseline, total_intervention
    """
    eval_block = data.get('evaluation')
    if not isinstance(eval_block, dict):
        return None
    base_list = eval_block.get('baseline_evaluations')
    inter_list = eval_block.get('intervention_evaluations')
    if not isinstance(base_list, list) or not isinstance(inter_list, list):
        return None
    n = min(len(base_list), len(inter_list))
    if n == 0:
        return None
    baseline_correct = 0
    baseline_wrong = 0
    corrected_count = 0
    degraded_count = 0
    for i in range(n):
        be = base_list[i] if isinstance(base_list[i], dict) else {}
        ie = inter_list[i] if i < len(inter_list) and isinstance(inter_list[i], dict) else {}
        bc = be.get('is_correct')
        ic = ie.get('is_correct')
        if bc is True:
            baseline_correct += 1
        elif bc is False:
            baseline_wrong += 1
        if bc is False and ic is True:
            corrected_count += 1
        if bc is True and ic is False:
            degraded_count += 1
    return {
        'baseline_correct': baseline_correct,
        'baseline_wrong': baseline_wrong,
        'corrected_by_intervention': corrected_count,
        'degraded_by_intervention': degraded_count,
        'total_baseline': len(base_list),
        'total_intervention': len(inter_list),
    }


def discover_all_strengths(intervention_root):
    """
    扫描 intervention_root 下所有 strength_* 目录名，返回强度列表（如 ["0.1", "0.3", "1.0"]）。
    保证不同 strength 都会被统计/输出。
    """
    out = []
    for path in sorted(glob.glob(os.path.join(intervention_root, 'strength_*'))):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if name.startswith('strength_'):
            strength_str = name.replace('strength_', '', 1)
        else:
            strength_str = name
        out.append(strength_str)
    return out


def discover_intervention_files(intervention_root):
    """
    扫描 intervention_root 下所有 strength_* 目录中的 *_intervention.json
    （不包含 *_intervention_analysis.json）。
    返回: list of (strength_str, file_path)，strength_str 如 "0.1"
    """
    out = []
    for path in sorted(glob.glob(os.path.join(intervention_root, 'strength_*'))):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if name.startswith('strength_'):
            strength_str = name.replace('strength_', '', 1)
        else:
            strength_str = name
        for jpath in glob.glob(os.path.join(path, '*_intervention.json')):
            # 排除带 analysis 后缀的文件
            if '_intervention_analysis.json' in jpath or jpath.endswith('_intervention_analysis.json'):
                continue
            out.append((strength_str, jpath))
    return out


# 已知 dataset 后缀，用于正确切分 model_dataset（dataset 可能含 _，如 nq_open）
KNOWN_DATASET_SUFFIXES = ('nq_open', 'truthfulqa', 'MedHallu')


def parse_intervention_filename(file_path):
    """
    从路径如 .../gemma-2-2b_nq_open_intervention.json 解析出 model 与 dataset。
    返回: (model_name, dataset_name)，例如 ("gemma-2-2b", "nq_open")
    """
    base = os.path.basename(file_path)
    if not base.endswith('_intervention.json'):
        return os.path.splitext(base)[0], ''
    rest = base.replace('_intervention.json', '')
    for suffix in KNOWN_DATASET_SUFFIXES:
        if rest.endswith('_' + suffix):
            model = rest[:-len(suffix) - 1].rstrip('_')
            return model, suffix
    parts = rest.split('_')
    if len(parts) < 2:
        return rest, ''
    dataset = parts[-1]
    model = '_'.join(parts[:-1])
    return model, dataset


def aggregate_all(intervention_root):
    """
    汇总所有 *_intervention.json 的统计（从 evaluation.baseline_evaluations /
    intervention_evaluations 的 is_correct 计算）。
    返回: list of dict，每项含 strength, model, dataset, file_path 及统计字段。
    """
    rows = []
    for strength_str, file_path in discover_intervention_files(intervention_root):
        data = load_json(file_path)
        if data is None:
            continue
        stats = extract_stats_from_intervention(data)
        if stats is None:
            print(f"跳过（无 evaluation 或列表为空）: {file_path}")
            continue
        model, dataset = parse_intervention_filename(file_path)
        rows.append({
            'strength': strength_str,
            'model': model,
            'dataset': dataset,
            'file_path': file_path,
            **stats,
        })
    return rows


CSV_FIELDNAMES = ['strength', 'model', 'dataset', 'baseline_correct', 'baseline_wrong',
                  'corrected_by_intervention', 'degraded_by_intervention',
                  'total_baseline', 'total_intervention']


def _write_csv(rows, csv_path, fieldnames=None):
    import csv
    fieldnames = fieldnames or CSV_FIELDNAMES
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})


def save_results(rows, output_dir, all_strengths=None, skip_existing=False):
    """将汇总结果保存为全局 JSON/CSV，并按每个 strength 分别保存独立 JSON/CSV。all_strengths 为所有 strength_* 的强度列表，保证每个强度都有一份统计文件。"""
    os.makedirs(output_dir, exist_ok=True)

    # 1) 全局汇总
    summary_json = os.path.join(output_dir, 'intervention_effect_summary.json')
    if skip_existing and os.path.exists(summary_json):
        print(f"Skip (exists): {summary_json}")
    else:
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"Saved: {summary_json}")

    csv_path = os.path.join(output_dir, 'intervention_effect_summary.csv')
    if skip_existing and os.path.exists(csv_path):
        print(f"Skip (exists): {csv_path}")
    else:
        _write_csv(rows, csv_path)
        print(f"Saved: {csv_path}")

    # 2) 按 strength 分别统计并保存（每个 strength_* 都有一份，无数据的强度写空列表）
    by_strength = defaultdict(list)
    for r in rows:
        by_strength[r['strength']].append(r)
    strengths_to_save = sorted(set(all_strengths or []) | set(by_strength.keys()))
    for strength in strengths_to_save:
        sub_rows = by_strength.get(strength, [])
        s_json = os.path.join(output_dir, f'intervention_effect_strength_{strength}.json')
        s_csv = os.path.join(output_dir, f'intervention_effect_strength_{strength}.csv')
        if skip_existing and os.path.exists(s_json) and os.path.exists(s_csv):
            print(f"Skip (exists): strength_{strength} -> {s_json}, {s_csv}")
        else:
            if not (skip_existing and os.path.exists(s_json)):
                with open(s_json, 'w', encoding='utf-8') as f:
                    json.dump(sub_rows, f, indent=2, ensure_ascii=False)
                print(f"Saved: {s_json}")
            if not (skip_existing and os.path.exists(s_csv)):
                _write_csv(sub_rows, s_csv)
                print(f"Saved: {s_csv}")


# 合并表格使用的 strength 列表（0.1, 0.3, 0.5 统一成一张表）
CONSOLIDATED_STRENGTHS = ('0.1', '0.3', '0.5')


def build_accuracy_table(rows, strengths=None):
    """
    从汇总行构建准确率表：保留 baseline_correct、intervention_correct(=baseline_correct+corrected_by_intervention)、
    total_baseline，并计算 baseline_accuracy、intervention_accuracy。
    只保留 strengths 中的强度（默认 0.1, 0.3, 0.5）。
    返回: list of dict，每项含 strength, model, dataset, baseline_correct, intervention_correct,
          total_baseline, baseline_accuracy, intervention_accuracy
    """
    strengths = strengths or CONSOLIDATED_STRENGTHS
    strength_set = set(strengths)
    out = []
    for r in rows:
        if r.get('strength') not in strength_set:
            continue
        total = r.get('total_baseline') or 0
        if total <= 0:
            continue
        baseline_correct = r.get('baseline_correct', 0) or 0
        corrected = r.get('corrected_by_intervention', 0) or 0
        intervention_correct = baseline_correct + corrected
        baseline_acc = round(baseline_correct / total, 6)
        intervention_acc = round(intervention_correct / total, 6)
        out.append({
            'strength': r['strength'],
            'model': r['model'],
            'dataset': r['dataset'],
            'baseline_correct': baseline_correct,
            'intervention_correct': intervention_correct,
            'total_baseline': total,
            'baseline_accuracy': baseline_acc,
            'intervention_accuracy': intervention_acc,
        })
    return out


def save_accuracy_table(table_rows, output_dir, skip_existing=False):
    """将准确率表保存为 CSV 和 JSON，文件名固定为 strength_0.1_0.3_0.5 合并表。"""
    if not table_rows:
        return
    os.makedirs(output_dir, exist_ok=True)
    base_name = 'intervention_accuracy_table_strength_0.1_0.3_0.5'
    json_path = os.path.join(output_dir, base_name + '.json')
    csv_path = os.path.join(output_dir, base_name + '.csv')
    if skip_existing and os.path.exists(json_path) and os.path.exists(csv_path):
        print(f"Skip (exists): {base_name}")
        return
    if not (skip_existing and os.path.exists(json_path)):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(table_rows, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")
    if not (skip_existing and os.path.exists(csv_path)):
        import csv
        fieldnames = ['strength', 'model', 'dataset', 'baseline_correct', 'intervention_correct',
                     'total_baseline', 'baseline_accuracy', 'intervention_accuracy']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in table_rows:
                w.writerow({k: r.get(k, '') for k in fieldnames})
        print(f"Saved: {csv_path}")


def print_summary(rows, all_strengths=None):
    """在终端按强度分别打印汇总；all_strengths 为所有 strength_*，无数据的强度会单独提示。"""
    print("\n" + "=" * 70)
    print("干预效果汇总（baseline 正确数 | baseline 错误数 | 被干预改正数 | 被干预破坏数）")
    print("=" * 70)
    by_strength = defaultdict(list)
    for r in rows:
        by_strength[r['strength']].append(r)
    strengths_to_print = sorted(set(all_strengths or []) | set(by_strength.keys()))
    for strength in strengths_to_print:
        sub = by_strength.get(strength, [])
        print(f"\n【干预强度 strength_{strength}】")
        if not sub:
            print("  该强度下暂无带 evaluation 的 *_intervention.json。")
            continue
        for r in sorted(sub, key=lambda x: (x['model'], x['dataset'])):
            name = f"{r['model']}_{r['dataset']}"
            deg = r.get('degraded_by_intervention')
            deg_str = str(deg) if deg is not None else 'N/A'
            print(f"  {name}: baseline 正确={r['baseline_correct']}, baseline 错误={r['baseline_wrong']}, "
                  f"改正={r['corrected_by_intervention']}, 破坏={deg_str}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='汇总各干预强度下 *_analysis.json 的干预效果统计')
    parser.add_argument('--intervention_root', type=str, default=INTERVENTION_RESULTS_DEFAULT,
                        help='results/intervention 根目录')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT,
                        help='汇总结果输出目录')
    parser.add_argument('--skip_existing', action='store_true',
                        help='若输出文件已存在则跳过写入')
    parser.add_argument('--no_print', action='store_true',
                        help='不打印终端汇总')
    args = parser.parse_args()

    print("正在扫描 strength_* 与 *_intervention.json（不读 analysis 文件）...")
    all_strengths = discover_all_strengths(args.intervention_root)
    if all_strengths:
        print(f"发现 {len(all_strengths)} 个干预强度: {', '.join(sorted(all_strengths))}")
    rows = aggregate_all(args.intervention_root)
    if not rows:
        print("未找到任何带 evaluation 的 *_intervention.json；请确认文件路径并已运行 eval_only 写入 evaluation。")
    save_results(rows, args.output_dir, all_strengths=all_strengths, skip_existing=args.skip_existing)
    # 合并 strength 0.1, 0.3, 0.5 为一张准确率表
    accuracy_table = build_accuracy_table(rows)
    save_accuracy_table(accuracy_table, args.output_dir, skip_existing=args.skip_existing)
    if not args.no_print:
        print_summary(rows, all_strengths=all_strengths)
    print("Done.")


if __name__ == '__main__':
    main()
