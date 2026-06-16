#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总干预效果统计：扫描 results/intervention/eval 下各评测结果文件
从 *_baseline_eval.jsonl 和 *_intervention_eval.jsonl 中的 eval_type 统计：
- baseline 正确/错误（bias 为错误，no_bias/antibias 为正确）
- 被干预改正/破坏数
并输出汇总表（JSON/CSV）。
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from pathlib import Path

BASE_DIR_DEFAULT = '/path/to/project_root/safety_explanation/fairness_bias'
INTERVENTION_RESULTS_DEFAULT = os.path.join(BASE_DIR_DEFAULT, 'results', 'intervention')
OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR_DEFAULT, 'results', 'intervention', 'aggregate')


def load_jsonl(file_path):
    """从 JSONL 文件加载数据，返回记录列表。失败时打印异常并返回 None。"""
    try:
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"JSONL 解析失败（跳过该行）: {file_path}. 详情: {e}")
                    continue
        return records
    except FileNotFoundError as e:
        print(f"文件不存在: {file_path}. 详情: {e}")
        return None
    except Exception as e:
        print(f"加载失败: {file_path}. 详情: {e}")
        return None


def is_correct(eval_record):
    """
    判断评测记录是否为正确（非偏见）。
    eval_type == "bias" 表示错误（偏见），其他（"no_bias", "antibias"）都是正确的。
    """
    if "error" in eval_record:
        return None  # 无法判断
    eval_type = eval_record.get("eval_type", "")
    if eval_type == "bias":
        return False  # 错误（偏见）
    elif eval_type in ("no_bias", "antibias"):
        return True  # 正确（无偏见）
    return None  # 未知类型


def extract_stats_from_eval_files(baseline_eval_file, intervention_eval_file):
    """
    从 baseline_eval.jsonl 和 intervention_eval.jsonl 中提取统计量。
    返回: dict 含 baseline_correct, baseline_wrong, corrected_by_intervention,
          degraded_by_intervention, total_baseline, total_intervention
    """
    baseline_records = load_jsonl(baseline_eval_file)
    intervention_records = load_jsonl(intervention_eval_file)
    
    if baseline_records is None or intervention_records is None:
        return None
    
    # 按 index 建立索引
    baseline_dict = {}
    for rec in baseline_records:
        idx = rec.get("index")
        if idx is not None:
            baseline_dict[idx] = rec
    
    intervention_dict = {}
    for rec in intervention_records:
        idx = rec.get("index")
        if idx is not None:
            intervention_dict[idx] = rec
    
    # 统计
    baseline_correct = 0
    baseline_wrong = 0
    corrected_count = 0
    degraded_count = 0
    
    # 获取所有共同的索引
    common_indices = set(baseline_dict.keys()) & set(intervention_dict.keys())
    
    for idx in common_indices:
        baseline_rec = baseline_dict[idx]
        intervention_rec = intervention_dict[idx]
        
        baseline_is_correct = is_correct(baseline_rec)
        intervention_is_correct = is_correct(intervention_rec)
        
        if baseline_is_correct is None or intervention_is_correct is None:
            continue  # 跳过无法判断的记录
        
        if baseline_is_correct:
            baseline_correct += 1
        else:
            baseline_wrong += 1
        
        # 被干预改正：baseline 错误 -> intervention 正确
        if not baseline_is_correct and intervention_is_correct:
            corrected_count += 1
        
        # 被干预破坏：baseline 正确 -> intervention 错误
        if baseline_is_correct and not intervention_is_correct:
            degraded_count += 1
    
    return {
        'baseline_correct': baseline_correct,
        'baseline_wrong': baseline_wrong,
        'corrected_by_intervention': corrected_count,
        'degraded_by_intervention': degraded_count,
        'total_baseline': len(baseline_dict),
        'total_intervention': len(intervention_dict),
    }


def normalize_model_dataset(model, dataset):
    """
    去掉 model 名称中重复的 dataset 后缀，保证 model 仅为模型名、dataset 仅为数据集名。
    例如 model=gemma-2-2b_bbq_nationality, dataset=bbq_nationality -> model=gemma-2-2b, dataset=bbq_nationality
    """
    if not model or not dataset:
        return model or '', dataset or ''
    suffix = '_' + dataset
    if model.endswith(suffix):
        return model[:-len(suffix)], dataset
    return model, dataset


def _parse_model_dataset_from_baseline_basename(basename):
    """
    从 *_baseline_eval.jsonl 的文件名解析 (model, dataset)。
    支持重复 dataset 的命名：{model}_{dataset}_{dataset}，例如
    gemma-2-2b_bbq_nationality_bbq_nationality -> model=gemma-2-2b, dataset=bbq_nationality
    """
    rest = basename.replace('_baseline_eval.jsonl', '')
    parts = rest.split('_')
    # 找到第一个 "bbq" 的索引，后面是 dataset 名（可能重复两次）
    dataset_start = None
    for i, p in enumerate(parts):
        if p == 'bbq':
            dataset_start = i
            break
    if dataset_start is None:
        if len(parts) >= 2:
            return '_'.join(parts[:-1]), parts[-1]
        return rest, ''
    n = len(parts) - dataset_start
    if n % 2 != 0:
        # 无法拆成两段重复，整段当作 dataset
        return '_'.join(parts[:dataset_start]), '_'.join(parts[dataset_start:])
    half = n // 2
    first = parts[dataset_start:dataset_start + half]
    second = parts[dataset_start + half:dataset_start + 2 * half]
    if first == second:
        dataset = '_'.join(first)
        model = '_'.join(parts[:dataset_start])
        return model, dataset
    return '_'.join(parts[:dataset_start]), '_'.join(parts[dataset_start:])


def discover_all_strengths(intervention_root):
    """
    扫描 intervention_root 下所有 strength_* 目录名，提取所有干预强度。
    返回强度列表（如 ["0.1", "0.3", "1.0"]）。
    """
    strengths = set()
    for path in sorted(glob.glob(os.path.join(intervention_root, 'strength_*'))):
        if not os.path.isdir(path):
            continue
        name = os.path.basename(path)
        if name.startswith('strength_'):
            strength_str = name.replace('strength_', '', 1)
        else:
            strength_str = name
        strengths.add(strength_str)
    return sorted(list(strengths), key=lambda x: float(x))


def discover_eval_files(intervention_root):
    """
    扫描 intervention_root 下各 strength_* 目录中的所有 baseline_eval.jsonl 文件。
    返回: list of (strength_str, baseline_file, intervention_file, model, dataset)
    """
    out = []
    
    # 扫描所有 strength_* 目录
    for strength_dir in sorted(glob.glob(os.path.join(intervention_root, 'strength_*'))):
        if not os.path.isdir(strength_dir):
            continue
        
        # 提取强度值
        dir_name = os.path.basename(strength_dir)
        if dir_name.startswith('strength_'):
            strength_str = dir_name.replace('strength_', '', 1)
        else:
            strength_str = dir_name
        
        # 扫描该目录下的所有 baseline_eval.jsonl 文件
        baseline_files = glob.glob(os.path.join(strength_dir, '*_baseline_eval.jsonl'))
        
        for baseline_file in baseline_files:
            # 从文件名提取信息：{model}_{dataset}_baseline_eval.jsonl 或 {model}_{dataset}_{dataset}_baseline_eval.jsonl
            basename = os.path.basename(baseline_file)
            if not basename.endswith('_baseline_eval.jsonl'):
                continue
            
            model = None
            dataset = None
            
            # 优先从 summary 文件获取 model_name / dataset_name，再统一去掉 model 中的 dataset 后缀
            summary_file = baseline_file.replace('_baseline_eval.jsonl', '_summary.json')
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        model = summary.get('model_name', '')
                        dataset = summary.get('dataset_name', '')
                except Exception:
                    pass
            
            if model and dataset:
                model, dataset = normalize_model_dataset(model, dataset)
            else:
                # 从文件名解析：支持 model_dataset_dataset 重复命名
                model, dataset = _parse_model_dataset_from_baseline_basename(basename)
            
            # 构建 intervention_eval 文件路径
            intervention_file = baseline_file.replace('_baseline_eval.jsonl', '_intervention_eval.jsonl')
            
            if not os.path.exists(intervention_file):
                print(f"[WARN] 未找到对应的干预评测文件: {intervention_file}")
                continue
            
            out.append((strength_str, baseline_file, intervention_file, model, dataset))
    
    return out


def aggregate_all(intervention_root):
    """
    汇总所有评测文件的统计。
    返回: list of dict，每项含 strength, model, dataset, file_path 及统计字段。
    """
    rows = []
    for strength_str, baseline_file, intervention_file, model, dataset in discover_eval_files(intervention_root):
        stats = extract_stats_from_eval_files(baseline_file, intervention_file)
        if stats is None:
            print(f"跳过（无法提取统计）: {baseline_file}")
            continue
        
        rows.append({
            'strength': strength_str,
            'model': model,
            'dataset': dataset,
            'baseline_eval_file': baseline_file,
            'intervention_eval_file': intervention_file,
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
    """将汇总结果保存为全局 JSON/CSV，并按每个 strength 分别保存独立 JSON/CSV。"""
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

    # 2) 按 strength 分别统计并保存
    by_strength = defaultdict(list)
    for r in rows:
        by_strength[r['strength']].append(r)
    strengths_to_save = sorted(set(all_strengths or []) | set(by_strength.keys()), key=lambda x: float(x))
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
    从汇总行构建准确率表：保留 baseline_correct、intervention_correct(=baseline_correct+corrected_by_intervention-degraded_by_intervention)、
    total_baseline，并计算 baseline_accuracy、intervention_accuracy。
    只保留 strengths 中的强度（默认 0.1, 0.3, 0.5）。
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
        degraded = r.get('degraded_by_intervention', 0) or 0
        intervention_correct = baseline_correct + corrected - degraded
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
    """在终端按强度分别打印汇总。"""
    print("\n" + "=" * 70)
    print("干预效果汇总（baseline 正确数 | baseline 错误数 | 被干预改正数 | 被干预破坏数）")
    print("=" * 70)
    by_strength = defaultdict(list)
    for r in rows:
        by_strength[r['strength']].append(r)
    strengths_to_print = sorted(set(all_strengths or []) | set(by_strength.keys()), key=lambda x: float(x))
    for strength in strengths_to_print:
        sub = by_strength.get(strength, [])
        print(f"\n【干预强度 strength_{strength}】")
        if not sub:
            print("  该强度下暂无评测结果文件。")
            continue
        for r in sorted(sub, key=lambda x: (x['model'], x['dataset'])):
            name = f"{r['model']}_{r['dataset']}"
            deg = r.get('degraded_by_intervention')
            deg_str = str(deg) if deg is not None else 'N/A'
            print(f"  {name}: baseline 正确={r['baseline_correct']}, baseline 错误={r['baseline_wrong']}, "
                  f"改正={r['corrected_by_intervention']}, 破坏={deg_str}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description='汇总各干预强度下评测结果的干预效果统计')
    parser.add_argument('--intervention_root', type=str, default=INTERVENTION_RESULTS_DEFAULT,
                        help='results/intervention 根目录（将扫描各 strength_* 目录下的评测结果）')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT,
                        help='汇总结果输出目录')
    parser.add_argument('--skip_existing', action='store_true',
                        help='若输出文件已存在则跳过写入')
    parser.add_argument('--no_print', action='store_true',
                        help='不打印终端汇总')
    args = parser.parse_args()

    print("正在扫描各 strength_* 目录下的评测结果文件...")
    all_strengths = discover_all_strengths(args.intervention_root)
    if all_strengths:
        print(f"发现 {len(all_strengths)} 个干预强度: {', '.join(sorted(all_strengths, key=lambda x: float(x)))}")
    rows = aggregate_all(args.intervention_root)
    if not rows:
        print("未找到任何评测结果文件；请确认文件路径并已运行 eval_intervention.py 生成评测结果。")
        return
    save_results(rows, args.output_dir, all_strengths=all_strengths, skip_existing=args.skip_existing)
    # 合并 strength 0.1, 0.3, 0.5 为一张准确率表
    accuracy_table = build_accuracy_table(rows)
    save_accuracy_table(accuracy_table, args.output_dir, skip_existing=args.skip_existing)
    if not args.no_print:
        print_summary(rows, all_strengths=all_strengths)
    print("Done.")


if __name__ == '__main__':
    main()
