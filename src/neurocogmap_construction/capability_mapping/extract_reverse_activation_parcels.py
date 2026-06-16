#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract Reverse-Activation Parcels and Aggregate by Capability

需求：
- 从正向（strength=0.5）与负向（strength=-1.0）干预目录读取每数据集 parcel 分数（默认字段 logprob_diff_avg）。
- 反向激活定义：
  * 正向负相关：正向分数 < 0 的 parcel（保留其分数值）
  * 负向正相关：负向分数 > 0 的 parcel（保留其分数值）
- 产物：
  * 每数据集 JSON：保存两类字典与二者并集的 parcel 名称列表（并集不需要保存分数）。
  * 综合 JSON：所有数据集的并集 set 列表。
  * 按能力聚合：依据 capability→datasets 映射，将各能力涉及的数据集的并集 parcel 记为该能力的反向激活 parcel 集合；同时保留能力内分数据集的列表。

输入目录（默认）：
- 正向：results/intervention/strength_0.5
- 负向：results/intervention/strength_-1.0
能力数据：capability_data_v2/data_stastic/final_merged_capability_dataset_stats.json
输出目录：results/aggrate_final
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List, Set


def _discover_files(result_dir: str) -> Tuple[Optional[str], Dict[str, str]]:
    """查找干预结果文件。返回 (all_results_file, per_dataset_files)。"""
    all_candidates = [
        os.path.join(result_dir, 'all_intervention_results_optimized.json'),
        os.path.join(result_dir, 'all_intervention_results.json'),
    ]
    all_file = None
    for fp in all_candidates:
        if os.path.exists(fp):
            all_file = fp
            break

    per_dataset: Dict[str, str] = {}
    try:
        for fname in os.listdir(result_dir):
            if fname.endswith('_intervention_results_optimized.json'):
                ds = fname.replace('_intervention_results_optimized.json', '')
                per_dataset[ds] = os.path.join(result_dir, fname)
            elif fname.endswith('_intervention_results.json') and 'optimized' not in fname:
                ds = fname.replace('_intervention_results.json', '')
                per_dataset[ds] = os.path.join(result_dir, fname)
    except FileNotFoundError:
        pass

    return all_file, per_dataset


def _load_results(result_dir: str) -> Dict[str, Any]:
    all_file, per_dataset = _discover_files(result_dir)
    results: Dict[str, Any] = {}

    if per_dataset:
        for ds, fp in per_dataset.items():
            with open(fp, 'r', encoding='utf-8') as f:
                results[ds] = json.load(f)
        return results

    if all_file and os.path.exists(all_file):
        with open(all_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'dataset_name' in data and 'intervention_results' in data:
            return {data['dataset_name']: data}
        if isinstance(data, dict):
            return data
        raise ValueError('无法解析汇总文件结构')

    # 回退：加载目录中唯一的 .json 作为一个数据集
    try:
        json_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith('.json')]
    except FileNotFoundError:
        json_files = []
    if len(json_files) == 1:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'dataset_name' in data and 'intervention_results' in data:
            return {data['dataset_name']: data}
        ds_name = os.path.basename(os.path.normpath(result_dir))
        return {ds_name: data}

    raise FileNotFoundError(f"未在目录中找到可用结果文件: {result_dir}")


def _extract_scores(dataset_result: Dict[str, Any], strength: float, score_field: str) -> Dict[str, float]:
    """提取给定强度下的 parcel 分数字典。兼容：
    result['intervention_results'][parcel][str(strength)][score_field]
    若无强度维度，则尝试直接取 score_field 或 'log_diff_avg' 作为回退。
    """
    root = dataset_result.get('intervention_results') if isinstance(dataset_result, dict) else None
    if root is None and isinstance(dataset_result, dict):
        root = dataset_result
    if not isinstance(root, dict):
        return {}

    scores: Dict[str, float] = {}
    for parcel, entry in root.items():
        if not isinstance(entry, dict):
            continue
        # 情况 A：含强度维度
        key = str(float(strength))
        if key in entry and isinstance(entry[key], dict):
            val = entry[key]
            if score_field in val:
                try:
                    scores[str(parcel)] = float(val[score_field])
                    continue
                except Exception:
                    pass
            if 'log_diff_avg' in val:  # 容错字段名
                try:
                    scores[str(parcel)] = float(val['log_diff_avg'])
                    continue
                except Exception:
                    pass
        # 情况 B：无强度维度，直接取字段
        if score_field in entry:
            try:
                scores[str(parcel)] = float(entry[score_field])
                continue
            except Exception:
                pass
        if 'log_diff_avg' in entry:
            try:
                scores[str(parcel)] = float(entry['log_diff_avg'])
                continue
            except Exception:
                pass
    return scores


def _load_capability_stats(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Extract reverse-activation parcels and aggregate by capability.')
    parser.add_argument('--pos_dir', type=str, default='/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_0.5', help='正向目录')
    parser.add_argument('--neg_dir', type=str, default='/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_-1.0', help='负向目录')
    parser.add_argument('--pos_strength', type=float, default=0.5, help='正向强度')
    parser.add_argument('--neg_strength', type=float, default=-1.0, help='负向强度')
    parser.add_argument('--score_field', type=str, default='logprob_diff_avg', help='分数字段名，容错回退到 log_diff_avg')
    parser.add_argument('--capability_stats', type=str, default='/path/to/project_root/neural_area/capability_data_v2/data_stastic/final_merged_capability_dataset_stats.json', help='能力-数据集映射 JSON')
    parser.add_argument('--output_dir', type=str, default='/path/to/project_root/neural_area/connect_cap_parcel/results/aggrate_final', help='输出目录')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir,"by_dataset"), exist_ok=True)

    pos_all = _load_results(args.pos_dir)
    neg_all = _load_results(args.neg_dir)

    datasets = sorted(set(list(pos_all.keys()) + list(neg_all.keys())))

    # 每数据集输出与综合集合
    union_all: Set[str] = set()
    ds_to_combined: Dict[str, List[str]] = {}

    for ds in datasets:
        pos_scores = _extract_scores(pos_all.get(ds, {}), args.pos_strength, args.score_field) if ds in pos_all else {}
        neg_scores = _extract_scores(neg_all.get(ds, {}), args.neg_strength, args.score_field) if ds in neg_all else {}

        # 反向条件过滤
        # 正向负相关：pos < 0
        pos_negcorr = {p: v for p, v in pos_scores.items() if v < 0.0}
        # 负向正相关：neg > 0
        neg_poscorr = {p: v for p, v in neg_scores.items() if v > 0.0}

        combined_set = set(pos_negcorr.keys()) | set(neg_poscorr.keys())
        union_all |= combined_set
        ds_to_combined[ds] = sorted(combined_set)

        # 写入每数据集 JSON
        out_ds = {
            'dataset': ds,
            'pos_strength': args.pos_strength,
            'neg_strength': args.neg_strength,
            'score_field': args.score_field,
            'positive_negative_correlation': {k: float(v) for k, v in pos_negcorr.items()},
            'negative_positive_correlation': {k: float(v) for k, v in neg_poscorr.items()},
            'combined_reverse_parcels': sorted(combined_set)
        }
        ds_out_path = os.path.join(args.output_dir,"by_dataset", f"{ds}.reverse_activation_parcels.json")
        with open(ds_out_path, 'w', encoding='utf-8') as f:
            json.dump(out_ds, f, ensure_ascii=False, indent=2)

    # 综合 set 列表
    union_path = os.path.join(args.output_dir, 'reverse_activation_parcels_all_datasets.json')
    with open(union_path, 'w', encoding='utf-8') as f:
        json.dump({'combined_reverse_parcels_all_datasets': sorted(list(union_all))}, f, ensure_ascii=False, indent=2)

    # 按能力聚合：将各能力涉及的数据集的正向负相关和负向正相关parcels分别合并
    cap_stats = _load_capability_stats(args.capability_stats)
    capability_out: Dict[str, Any] = {}
    
    for capability, ds_map in cap_stats.items():
        if not isinstance(ds_map, dict):
            continue
        
        cap_ds = [ds for ds in ds_map.keys()]
        pos_neg_parcels: Set[str] = set()  # 正向负相关parcels
        neg_pos_parcels: Set[str] = set()  # 负向正相关parcels
        
        for ds in cap_ds:
            # 获取该数据集的正向和负向分数
            pos_scores = _extract_scores(pos_all.get(ds, {}), args.pos_strength, args.score_field) if ds in pos_all else {}
            neg_scores = _extract_scores(neg_all.get(ds, {}), args.neg_strength, args.score_field) if ds in neg_all else {}
            
            # 如果直接找不到，尝试小写匹配
            if not pos_scores and not neg_scores:
                lower_map = {k.lower(): k for k in pos_all.keys()}
                alt_pos = lower_map.get(ds.lower())
                if alt_pos is not None:
                    pos_scores = _extract_scores(pos_all.get(alt_pos, {}), args.pos_strength, args.score_field)
                
                lower_map = {k.lower(): k for k in neg_all.keys()}
                alt_neg = lower_map.get(ds.lower())
                if alt_neg is not None:
                    neg_scores = _extract_scores(neg_all.get(alt_neg, {}), args.neg_strength, args.score_field)
            
            # 提取正向负相关parcels (pos < 0)
            for parcel, score in pos_scores.items():
                if score < 0.0:
                    pos_neg_parcels.add(parcel)
            
            # 提取负向正相关parcels (neg > 0)
            for parcel, score in neg_scores.items():
                if score > 0.0:
                    neg_pos_parcels.add(parcel)

        capability_out[capability] = {
            'positive_negative_parcels': sorted(list(pos_neg_parcels)),
            'negative_positive_parcels': sorted(list(neg_pos_parcels))
        }

    cap_path = os.path.join(args.output_dir, 'capability_reverse_activation_parcels_by_type.json')
    with open(cap_path, 'w', encoding='utf-8') as f:
        json.dump(capability_out, f, ensure_ascii=False, indent=2)

    # 索引
    index_path = os.path.join(args.output_dir, 'reverse_activation_index.json')
    
    # 统计能力级别的parcels数量
    capability_stats = {}
    for capability, data in capability_out.items():
        capability_stats[capability] = {
            'positive_negative_count': len(data['positive_negative_parcels']),
            'negative_positive_count': len(data['negative_positive_parcels']),
            'total_count': len(data['positive_negative_parcels']) + len(data['negative_positive_parcels'])
        }
    
    index = {
        'datasets': datasets,
        'num_datasets': len(datasets),
        'union_count': len(union_all),
        'union_preview': sorted(list(union_all))[:50],
        'capabilities': list(capability_out.keys()),
        'num_capabilities': len(capability_out),
        'capability_stats': capability_stats
    }
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print('✅ 反向激活 Parcel 提取与聚合完成。')
    print(f'📁 输出目录: {args.output_dir}')
    print(f'📊 处理了 {len(datasets)} 个数据集')
    print(f'🧠 聚合了 {len(capability_out)} 个能力')
    print(f'📄 主要输出文件:')
    print(f'   - capability_reverse_activation_parcels_by_type.json (按能力分组的反向激活parcels)')
    print(f'   - reverse_activation_parcels_all_datasets.json (所有数据集的并集)')
    print(f'   - reverse_activation_index.json (统计索引)')


if __name__ == '__main__':
    main()


