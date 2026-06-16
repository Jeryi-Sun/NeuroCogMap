#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge Positive/Negative Intervention Scores (per dataset and parcel)

功能：
1) 读取正向与负向干预结果（目录中可能存在按数据集拆分的 *_intervention_results*.json 或 all_intervention_results*.json）
2) 从结果中提取指定强度（例如正向 0.5、负向 -1.0）下每个 parcel 的分数字段（默认 logprob_diff_avg）
3) 规则：
   - 正向：将 < 0 的置为 0
   - 负向：将 > 0 的置为 0，然后取绝对值（变为正数）
4) 在每个数据集内分别对正向分数与负向分数进行归一化（默认 minmax）
5) 以 0.5 权重对正负归一化分数做平均，得到最终分数并排序
6) 保存到输出目录：
   - {dataset}.merged_intervention_scores.json（含细节）
   - merged_intervention_scores_index.json（索引）
   - merged_intervention_scores_all_datasets.csv（汇总表）

备注：为兼容不同结果结构，本脚本支持两类结构：
- 结构 A（按本项目常见）：{dataset: { 'intervention_results': { parcel: { str(strength): {score_field: value, ...} } } } }
- 结构 B（汇总文件扁平化）：{ 'dataset_name': ..., 'intervention_results': { parcel: { ... } } } 或直接 {dataset: {...}}
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd


def _minmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    denom = vmax - vmin
    if denom == 0.0:
        return {k: 0.0 for k in d.keys()}
    return {k: (float(v) - vmin) / denom for k, v in d.items()}


def _zscore(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if std == 0.0:
        return {k: 0.0 for k in d.keys()}
    return {k: (float(v) - mean) / std for k, v in d.items()}


def _discover_files(result_dir: str) -> Tuple[Optional[str], Dict[str, str]]:
    """查找干预结果文件。
    返回 (all_results_file, per_dataset_files)
    - 优先 all_intervention_results_optimized.json → all_intervention_results.json
    - 其次 {dataset}_intervention_results_optimized.json / {dataset}_intervention_results.json
    """
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
    """加载结果为 {dataset: dataset_result_dict} 的结构。"""
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
        # 兼容结构 B 的两种写法
        if isinstance(data, dict) and 'dataset_name' in data and 'intervention_results' in data:
            results = {data['dataset_name']: data}
        elif isinstance(data, dict):
            results = data
        else:
            raise ValueError('无法解析汇总文件结构')
        return results

    # 若目录下存在非标准命名文件，尝试加载单一大 JSON 为一个数据集
    # 回退：寻找任意 .json 文件
    try:
        json_files = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith('.json')]
    except FileNotFoundError:
        json_files = []
    if len(json_files) == 1:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'dataset_name' in data and 'intervention_results' in data:
            results = {data['dataset_name']: data}
        else:
            # 无 dataset 名称，使用目录名作为 dataset
            ds_name = os.path.basename(os.path.normpath(result_dir))
            results = {ds_name: data}
        return results

    raise FileNotFoundError(f"未在目录中找到可用结果文件: {result_dir}")


def _extract_scores_from_dataset_result(dataset_result: Dict[str, Any],
                                        strength: float,
                                        score_field: str) -> Dict[str, float]:
    """从单个数据集结果字典中抽取 parcel 分数（在给定强度下）。
    兼容两类结构：
    - A: dataset_result['intervention_results'][parcel][str(strength)][score_field]
    - B: dataset_result 直接是 {parcel: {...}} 或 { 'intervention_results': { parcel: {...} } }
    """
    # 优先使用 'intervention_results'
    root = dataset_result.get('intervention_results') if isinstance(dataset_result, dict) else None
    if root is None and isinstance(dataset_result, dict):
        root = dataset_result

    if not isinstance(root, dict):
        raise ValueError('数据集结果结构不符合预期，缺少 intervention_results')

    scores: Dict[str, float] = {}
    for parcel, entry in root.items():
        if isinstance(entry, dict):
            # 情况 A：存在强度维度
            if any(isinstance(v, dict) and score_field in v for v in entry.values()):
                key = str(float(strength))
                if key in entry and isinstance(entry[key], dict) and score_field in entry[key]:
                    try:
                        scores[str(parcel)] = float(entry[key][score_field])
                    except Exception:
                        pass
                else:
                    # 若没有精确 key，选最接近的强度键
                    candidates: List[Tuple[float, Dict[str, Any]]] = []
                    for k, v in entry.items():
                        try:
                            s = float(k)
                        except Exception:
                            continue
                        if isinstance(v, dict) and score_field in v:
                            candidates.append((s, v))
                    if candidates:
                        s_best, v_best = min(candidates, key=lambda t: abs(t[0] - strength))
                        try:
                            scores[str(parcel)] = float(v_best[score_field])
                        except Exception:
                            pass
            else:
                # 情况 B：无强度维度，直接取 score_field
                if score_field in entry:
                    try:
                        scores[str(parcel)] = float(entry[score_field])
                    except Exception:
                        pass
    return scores


def _clip_and_flip(pos_scores: Dict[str, float], neg_scores: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    # 正向：小于 0 的置零
    pos = {p: (v if v > 0.0 else 0.0) for p, v in pos_scores.items()}
    # 负向：大于 0 的置零，然后取绝对值
    neg = {}
    for p, v in neg_scores.items():
        if v > 0.0:
            neg[p] = 0.0
        else:
            neg[p] = abs(v)
    return pos, neg


def _normalize(d: Dict[str, float], mode: str) -> Dict[str, float]:
    if mode == 'zscore':
        return _zscore(d)
    return _minmax(d)


def main():
    parser = argparse.ArgumentParser(description='Merge positive/negative intervention scores per dataset and parcel')
    parser.add_argument('--pos_dir', type=str,
                        default='/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_0.5',
                        help='正向干预结果目录（强度 0.5）')
    parser.add_argument('--neg_dir', type=str,
                        default='/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/strength_-1.0',
                        help='负向干预结果目录（强度 -1.0）')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate_intervention',
                        help='输出目录')
    parser.add_argument('--score_field', type=str, default='logprob_diff_avg', help='使用的分数字段')
    parser.add_argument('--pos_strength', type=float, default=0.5, help='正向强度')
    parser.add_argument('--neg_strength', type=float, default=-1.0, help='负向强度')
    parser.add_argument('--norm', type=str, default='minmax', choices=['minmax', 'zscore'], help='归一化方法')
    parser.add_argument('--dataset_filter', type=str, default='', help='仅处理指定数据集（精确匹配）')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载正负向结果
    pos_all = _load_results(args.pos_dir)
    neg_all = _load_results(args.neg_dir)

    # 统一数据集集合（以正向结果为主，补齐负向存在但正向没有的情况）
    datasets = sorted(set(list(pos_all.keys()) + list(neg_all.keys())))
    if args.dataset_filter:
        datasets = [d for d in datasets if d == args.dataset_filter]
        if not datasets:
            raise SystemExit(f"未找到指定数据集: {args.dataset_filter}")

    index: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for ds in datasets:
        ds_pos_raw = _extract_scores_from_dataset_result(pos_all.get(ds, {}), args.pos_strength, args.score_field) if ds in pos_all else {}
        ds_neg_raw = _extract_scores_from_dataset_result(neg_all.get(ds, {}), args.neg_strength, args.score_field) if ds in neg_all else {}

        # 规则裁剪与取绝对值
        ds_pos_clip, ds_neg_clip = _clip_and_flip(ds_pos_raw, ds_neg_raw)

        # 分别在数据集内部归一化
        ds_pos_norm = _normalize(ds_pos_clip, args.norm)
        ds_neg_norm = _normalize(ds_neg_clip, args.norm)

        # 合并（0.5 加权平均）；缺失按 0 处理
        parcels = set(ds_pos_norm.keys()) | set(ds_neg_norm.keys())
        merged: Dict[str, float] = {}
        for p in parcels:
            merged[p] = 0.5 * float(ds_pos_norm.get(p, 0.0)) + 0.5 * float(ds_neg_norm.get(p, 0.0))

        merged_sorted = sorted(merged.items(), key=lambda t: t[1], reverse=True)

        # 输出单数据集 JSON
        out_path = os.path.join(args.output_dir, f"{ds}.merged_intervention_scores.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': ds,
                'score_field': args.score_field,
                'pos_strength': args.pos_strength,
                'neg_strength': args.neg_strength,
                'norm': args.norm,
                'pos_scores_raw': ds_pos_raw,
                'neg_scores_raw': ds_neg_raw,
                'pos_scores_clipped': ds_pos_clip,
                'neg_scores_clipped_abs': ds_neg_clip,
                'pos_scores_norm': ds_pos_norm,
                'neg_scores_norm': ds_neg_norm,
                'merged_scores': merged_sorted
            }, f, ensure_ascii=False, indent=2)

        # 更新索引与 CSV 行
        index[ds] = {
            'count_parcels': len(merged_sorted),
            'top10': merged_sorted[:10]
        }
        for parcel, score in merged_sorted:
            rows.append({
                'dataset': ds,
                'parcel': parcel,
                'merged_score': score,
                'pos_norm': ds_pos_norm.get(parcel, 0.0),
                'neg_norm': ds_neg_norm.get(parcel, 0.0)
            })

    # 输出索引与 CSV
    index_path = os.path.join(args.output_dir, 'merged_intervention_scores_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, 'merged_intervention_scores_all_datasets.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')

    print('✅ 干预分数合并完成。输出目录:', args.output_dir)


if __name__ == '__main__':
    main()


