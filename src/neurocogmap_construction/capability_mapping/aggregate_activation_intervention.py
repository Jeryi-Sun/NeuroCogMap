#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate Activation Rankings and Parcel Intervention Results

根据给定算法，将激活强度排序文件与干预结果(top-K)进行融合，得到每个数据集的最终 Parcel 排名与分数。

算法（每个数据集独立执行）：
1) Stage 1: Activation Ranking
   π_A ← sort_indices(A, descending)
2) Stage 2: Intervention Re-Ranking
   TopK ← π_A[1..K]
   HasL ← {p ∈ TopK | p ∈ keys(L)}
   π_L ← sort_indices({L[p] | p ∈ HasL}, descending)
3) Stage 3: Score Normalization
   A'_i = (A_i - min(A)) / (max(A) - min(A))，若分母为0则置0
   L'_i = ((L_i - min(L)) / (max(L) - min(L))) if p_i∈keys(L) else 0，若分母为0则置0
4) Stage 4: Final Score and Merge
   S_i = α*A'_i + (1-α)*L'_i
   π ← sort_indices(S, descending)

输入：
- 激活强度JSON（由 rank_activation_connect_capability_parcel.py 保存的 parcel_activation_strengths.json）
- 干预结果目录（由 parcel_intervention.py 生成的每数据集 *intervention_results*.json，或汇总 all_intervention_results*.json）

输出：
- 每数据集最终结果JSON：{dataset}.final_parcel_scores.json
- 汇总CSV：final_parcel_scores_all_datasets.csv
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

def _zscore_series(series: Dict[str, float]) -> Dict[str, float]:
    if not series:
        raise ValueError("归一化失败：series 为空")
    values = np.array(list(series.values()), dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))  # population std (ddof=0)
    if std == 0.0:
        # 全部相等时，z-score 置为 0
        return {k: 0.0 for k in series.keys()}
    return {k: (float(v) - mean) / std for k, v in series.items()}


def _minmax_series(series: Dict[str, float]) -> Dict[str, float]:
    if not series:
        raise ValueError("归一化失败：series 为空")
    values = np.array(list(series.values()), dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    denom = vmax - vmin
    if denom == 0.0:
        # 全部相等时，归一化置为 0
        return {k: 0.0 for k in series.keys()}
    return {k: (float(v) - vmin) / denom for k, v in series.items()}


def _summarize_series(series: Dict[str, float]) -> Dict[str, Any]:
    if not series:
        return {
            'count': 0,
            'min': None,
            'p1': None,
            'p5': None,
            'p50': None,
            'p95': None,
            'p99': None,
            'max': None,
            'mean': None,
            'std': None
        }
    values = np.array(list(series.values()), dtype=float)
    values_sorted = np.sort(values)
    def pct(p: float) -> float:
        return float(np.percentile(values_sorted, p))
    return {
        'count': int(values.size),
        'min': float(values_sorted[0]),
        'p1': pct(1),
        'p5': pct(5),
        'p50': pct(50),
        'p95': pct(95),
        'p99': pct(99),
        'max': float(values_sorted[-1]),
        'mean': float(np.mean(values)),
        'std': float(np.std(values))
    }


def _load_activation_strengths(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 支持两种格式：
    # 1) { dataset: { parcel: activation, ... } }
    # 2) { dataset: [ [parcel, activation], ... ] }（rank文件）
    normalized: Dict[str, Dict[str, float]] = {}
    for ds, val in data.items():
        if isinstance(val, dict):
            try:
                normalized[ds] = {str(k): float(v) for k, v in val.items()}
            except Exception as e:
                raise ValueError(f"激活文件解析失败：数据集 {ds} 内数值无法转为 float: {e}")
        elif isinstance(val, list):
            # list of [parcel, score]
            d: Dict[str, float] = {}
            for item in val:
                if not (isinstance(item, list) and len(item) == 2):
                    raise ValueError(f"激活文件解析失败：数据集 {ds} 中存在非法列表项 {item}")
                p, s = item
                try:
                    d[str(p)] = float(s)
                except Exception as e:
                    raise ValueError(f"激活文件解析失败：数据集 {ds} 的 {p} 分数无法转为 float: {e}")
            if not d:
                raise ValueError(f"激活文件解析失败：数据集 {ds} 激活列表为空")
            normalized[ds] = d
        else:
            raise ValueError(f"激活文件解析失败：数据集 {ds} 的值类型不支持: {type(val)}")
    return normalized


def _discover_intervention_files(intervention_dir: str, prefer_optimized: bool) -> Tuple[Optional[str], Dict[str, str]]:
    """返回 (all_results_file, per_dataset_files)
    - 优先使用 all_intervention_results_optimized.json 或 all_intervention_results.json
    - 其次使用每数据集的 {dataset}_intervention_results(_optimized).json
    """
    all_file_candidates = []
    if prefer_optimized:
        all_file_candidates.append(os.path.join(intervention_dir, 'all_intervention_results_optimized.json'))
    all_file_candidates.append(os.path.join(intervention_dir, 'all_intervention_results.json'))

    all_results_file = None
    for fp in all_file_candidates:
        if os.path.exists(fp):
            all_results_file = fp
            break

    # per-dataset 文件
    per_dataset_files: Dict[str, str] = {}
    for fname in os.listdir(intervention_dir):
        if prefer_optimized and fname.endswith('_intervention_results_optimized.json'):
            dataset = fname.replace('_intervention_results_optimized.json', '')
            per_dataset_files[dataset] = os.path.join(intervention_dir, fname)
        elif (not prefer_optimized) and fname.endswith('_intervention_results.json') and 'optimized' not in fname:
            dataset = fname.replace('_intervention_results.json', '')
            per_dataset_files[dataset] = os.path.join(intervention_dir, fname)

    return all_results_file, per_dataset_files


def _load_intervention_results(intervention_dir: str, prefer_optimized: bool) -> Dict[str, Any]:
    """返回结构：{dataset: { 'intervention_results': { parcel: {strength: result_dict} } } }"""
    all_file, per_dataset_files = _discover_intervention_files(intervention_dir, prefer_optimized)

    results: Dict[str, Any] = {}

    # 优先使用 per-dataset 文件（如果存在任何一个）
    if per_dataset_files:
        for dataset, fp in per_dataset_files.items():
            with open(fp, 'r', encoding='utf-8') as f:
                results[dataset] = json.load(f)
        return results

    # 否则回退到 all_* 汇总文件
    if all_file and os.path.exists(all_file):
        with open(all_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        # 兼容两种汇总结构
        if 'dataset_name' in results and 'intervention_results' in results:
            results = {results['dataset_name']: results}
        if not results:
            raise ValueError("干预汇总文件解析为空")
        return results

    raise FileNotFoundError("未能在干预目录中找到任何可用结果文件（per-dataset 或 all_*）")


def _load_merged_intervention_scores(merged_dir: str) -> Dict[str, Dict[str, float]]:
    """读取由 merge_intervention_scores.py 生成的合并后结果
    期望读取 {dataset}.merged_intervention_scores.json 文件，结构示例：
    {
      'dataset': ds,
      'merged_scores': [[parcel, score], ...],
      ...
    }
    返回：{ dataset: { parcel: score, ... } }
    若找不到逐数据集文件，尝试读取 merged_intervention_scores_index.json 仅用于枚举数据集，再逐个读取。
    """
    results: Dict[str, Dict[str, float]] = {}
    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(f"合并后干预目录不存在：{merged_dir}")

    # 优先直接扫描逐数据集文件
    found_any = False
    for fname in os.listdir(merged_dir):
        if fname.endswith('.merged_intervention_scores.json'):
            found_any = True
            ds = fname.replace('.merged_intervention_scores.json', '')
            fp = os.path.join(merged_dir, fname)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                merged_list = data.get('merged_scores', [])
                scores: Dict[str, float] = {}
                for item in merged_list:
                    if isinstance(item, list) and len(item) == 2:
                        p, s = item
                        try:
                            scores[str(p)] = float(s)
                        except Exception:
                            pass
                results[ds] = scores
            except Exception:
                continue

    if found_any:
        if not results:
            raise ValueError('无法从合并后结果文件中解析任何分数')
        return results

    # 退化：尝试通过索引文件枚举数据集，再按约定读取
    index_path = os.path.join(merged_dir, 'merged_intervention_scores_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            idx = json.load(f)
        if isinstance(idx, dict):
            for ds in idx.keys():
                fp = os.path.join(merged_dir, f"{ds}.merged_intervention_scores.json")
                if os.path.exists(fp):
                    with open(fp, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    merged_list = data.get('merged_scores', [])
                    scores: Dict[str, float] = {}
                    for item in merged_list:
                        if isinstance(item, list) and len(item) == 2:
                            p, s = item
                            try:
                                scores[str(p)] = float(s)
                            except Exception:
                                pass
                    results[ds] = scores
        if results:
            return results

    raise FileNotFoundError('未能在合并后干预目录中找到任何 {dataset}.merged_intervention_scores.json 文件')


def _reduce_intervention_scores(dataset_result: Dict[str, Any],
                                score_field: str,
                                strength_mode: str,
                                at_strength: float) -> Dict[str, float]:
    """
    将单数据集的干预结果转为 {parcel: score}
    - score_field: 选择干预结果中的哪个字段作为 L 值（默认 'logprob_diff'）
    - strength_mode:
        * 'at' 固定取某个强度（由 at_strength 指定）
        * 'mean' 对所有强度取均值
        * 'max_abs' 取绝对值最大的强度对应的分数（保留原符号）
    """
    l_scores: Dict[str, float] = {}
    inter = dataset_result.get('intervention_results', {})
    for parcel, by_strength in inter.items():
        # by_strength: { str(strength): intervention_result_dict }
        if not isinstance(by_strength, dict) or not by_strength:
            raise ValueError(f"干预结果解析失败：{parcel} 缺少强度维度数据")

        # 解析可用强度
        items: List[Tuple[float, float]] = []  # (strength, score)
        for k, v in by_strength.items():
            try:
                s = float(k)
            except Exception:
                raise ValueError(f"干预结果解析失败：强度键 {k} 无法转为浮点数")
            if not isinstance(v, dict):
                raise ValueError(f"干预结果解析失败：强度 {k} 的值不是对象：{type(v)}")
            if score_field not in v:
                raise KeyError(f"干预结果缺少字段 {score_field} 于强度 {k}")
            try:
                score = float(v[score_field])
            except Exception as e:
                raise ValueError(f"干预结果解析失败：{score_field} 无法转为 float（强度 {k}）: {e}")
            items.append((s, score))

        if not items:
            raise ValueError(f"干预结果解析失败：{parcel} 未提取到任何有效分数项")

        if strength_mode == 'at':
            # 选取最接近 at_strength 的条目
            s_best, sc_best = min(items, key=lambda t: abs(t[0] - at_strength))
            l_scores[parcel] = sc_best
        elif strength_mode == 'mean':
            l_scores[parcel] = float(np.mean([sc for _, sc in items]))
        elif strength_mode == 'max_abs':
            s_best, sc_best = max(items, key=lambda t: abs(t[1]))
            l_scores[parcel] = sc_best
        else:
            # 默认 mean
            l_scores[parcel] = float(np.mean([sc for _, sc in items]))

    return l_scores


def _make_pair_key(dataset: str, parcel: str) -> str:
    return f"{dataset}||{parcel}"


def aggregate_one_dataset(dataset: str,
                          activation_strengths: Dict[str, float],
                          dataset_l_scores: Dict[str, float],
                          K: int,
                          alpha: float,
                          a_z_global: Dict[str, float],
                          l_z_global: Dict[str, float],
                          score_field: str,
                          strength_mode: str,
                          at_strength: float,
                          l_use_abs: bool,
                          per_dataset_norm: bool,
                          norm: str) -> Dict[str, Any]:
    """
    返回：{
        'dataset': dataset,
        'final_ranking': [ [parcel, S], ... ],
        'detail': { parcel: { 'A': v, 'A_norm': v', 'L': v, 'L_norm': v', 'S': s } }
    }
    """
    # A 全量分数
    A: Dict[str, float] = dict(activation_strengths)
    parcels = list(A.keys())
    if not parcels:
        raise ValueError(f"数据集 {dataset} 激活分数为空")

    # L（可能只覆盖 TopK 或其子集），此处直接使用预计算的分数
    L: Dict[str, float] = dict(dataset_l_scores)
    if not L:
        raise FileNotFoundError(f"数据集 {dataset} 缺少干预结果，无法继续聚合")

    # 根据需求：L 使用 |logprob_diff| 表示重要性（可配置）
    if l_use_abs and L:
        L = {p: abs(v) for p, v in L.items()}

    # Stage 1 排序
    pi_A = sorted(parcels, key=lambda p: A.get(p, 0.0), reverse=True)
    # Stage 2 TopK & HasL（仅用于理解流程；最终融合在 Stage 4 完成）
    topK_list = pi_A[:K]
    topK = set(topK_list)
    hasL = [p for p in topK_list if p in L]
    # π_L 仅作参考（结果仍以融合分数为准）。若无交集，则回退为在 L 的全体上取前 K
    if hasL:
        pi_L = sorted(hasL, key=lambda p: L[p], reverse=True)
    else:
        if not L:
            raise ValueError(f"数据集 {dataset} 干预分数 L 为空")
        pi_L = [p for p, _ in sorted(L.items(), key=lambda t: t[1], reverse=True)[:K]]

    # 将候选集合限定为 TopK 与 π_L 的并集
    candidate_set = set(topK_list) | set(pi_L)
    if not candidate_set:
        raise ValueError(f"数据集 {dataset} 候选集合为空")

    # Stage 3 归一化：
    # - 若 per_dataset_norm=True，则对 A 与（裁剪过的）L 在该数据集内做归一化
    # - 否则使用 main 中计算的全局归一化分数
    if not A:
        raise ValueError(f"数据集 {dataset} 激活分数为空")

    if per_dataset_norm:
        # 局部归一化（A 与 L 分别独立在数据集内归一化）
        if norm == 'zscore':
            a_norm_map = _zscore_series({p: A.get(p, 0.0) for p in candidate_set})
            l_norm_map = _zscore_series({p: dataset_l_scores.get(p, 0.0) for p in candidate_set})
        else:
            a_norm_map = _minmax_series({p: A.get(p, 0.0) for p in candidate_set})
            l_norm_map = _minmax_series({p: dataset_l_scores.get(p, 0.0) for p in candidate_set})
    else:
        # 使用全局归一化查表
        a_norm_map = {p: float(a_z_global.get(_make_pair_key(dataset, p), 0.0)) for p in candidate_set}
        l_norm_map = {p: float(l_z_global.get(_make_pair_key(dataset, p), 0.0)) for p in candidate_set}

    # Stage 4 融合
    S: Dict[str, float] = {}
    for p in candidate_set:
        a_norm = a_norm_map.get(p, 0.0)
        l_component = l_norm_map.get(p, 0.0)
        S[p] = float(alpha * a_norm + (1.0 - alpha) * l_component)

    final_ranking = sorted(((p, S[p]) for p in candidate_set if p in S), key=lambda t: t[1], reverse=True)
    
    # 组装详细信息
    detail = {}
    for p in candidate_set:
        detail[p] = {
            'A': float(A.get(p, 0.0)),
            'A_norm': float(a_norm_map.get(p, 0.0)),
            'L': float(L.get(p, 0.0)) if p in L else 0.0,
            'L_norm': float(l_norm_map.get(p, 0.0)),
            'S': float(S.get(p, 0.0))
        }

    return {
        'dataset': dataset,
        'K': K,
        'alpha': alpha,
        'score_field': score_field,
        'strength_mode': strength_mode,
        'at_strength': at_strength,
        'l_use_abs': l_use_abs,
        'topK_activation_parcels': list(topK),
        'topK_intervened_parcels': pi_L,
        'final_ranking': final_ranking,
        'detail': detail
    }


def main():
    parser = argparse.ArgumentParser(description='Aggregate activation and intervention results (per algorithm).')
    parser.add_argument('--activation_file', type=str,
                        default='/path/to/project_root/neural_area/connect_cap_parcel/results/rank_activation/parcel_activation_rankings.json',
                        help='rank_activation 的 JSON，可为 {dataset:{parcel:val}} 或 {dataset:[[parcel,val],...]}')
    parser.add_argument('--intervention_dir', type=str,
                        default='/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate_intervention',
                        help='干预结果目录：可为原始 parcel_intervention 结果目录或合并后目录')
    parser.add_argument('--output_dir', type=str,
                        default='/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate',
                        help='聚合输出目录')
    parser.add_argument('--K', type=int, default=50, help='Top-K 参数（与干预覆盖一致或更大）')
    parser.add_argument('--alpha', type=float, default=0.5, help='融合权重 α ∈ [0,1]')
    parser.add_argument('--prefer_optimized', action='store_true', help='优先读取 *_optimized.json（仅 raw 模式有效）')
    parser.add_argument('--dataset_filter', type=str, default='', help='仅处理该数据集（精确名称匹配）')
    parser.add_argument('--score_field', type=str, default='logprob_diff', help='从干预结果中选作 L 的字段（raw 模式有效）')
    parser.add_argument('--strength_mode', type=str, default='mean', choices=['mean', 'at', 'max_abs'],
                        help='干预强度聚合方式：mean/at/max_abs（raw 模式有效）')
    parser.add_argument('--at_strength', type=float, default=-1.0, help='strength_mode=at 时的目标强度（raw 模式有效）')
    parser.add_argument('--l_abs', action='store_true', default=True,
                        help='将 L 分数取绝对值用于重要性（raw 模式默认开启）')
    parser.add_argument('--norm', type=str, default='zscore', choices=['zscore', 'minmax'],
                        help='归一化方法：zscore 或 minmax（支持全局或按数据集）')
    parser.add_argument('--intervention_mode', type=str, default='merged', choices=['raw', 'merged'],
                        help='干预输入模式：raw=原始干预输出，merged=使用 merge_intervention_scores.py 的输出')
    parser.add_argument('--per_dataset_norm', action='store_true', help='在每个数据集内部对 A 与 L 分数各自归一化后再加权')
    parser.add_argument('--report_stats', action='store_true', help='在归一化之前输出全局激活与干预分数的统计摘要')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 加载激活强度
    activation_all = _load_activation_strengths(args.activation_file)
    
    # 2) 加载干预结果
    if args.intervention_mode == 'merged':
        # 读取合并后的干预结果：返回 {dataset: {parcel: score}}
        merged_l = _load_merged_intervention_scores(args.intervention_dir)
        intervention_all = {ds: {'intervention_results': {p: {'merged': v} for p, v in merged_l.get(ds, {}).items()}} for ds in merged_l.keys()}
        # 在 merged 模式下，后续 reduce 阶段将直接读取这些 merged 分数
        score_field_effective = 'merged'
    else:
        intervention_all = _load_intervention_results(args.intervention_dir, args.prefer_optimized)
        score_field_effective = args.score_field

    # 3) 逐数据集聚合
    aggregated: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []  # 用于汇总CSV
    
    activation_datasets = list(activation_all.keys())
    intervention_dataset = list(intervention_all.keys())
    datasets = list(set(activation_datasets) & set(intervention_dataset))
    print("length of datasets: ", len(datasets), "intervention_dataset: ", len(intervention_dataset), "activation_datasets: ", len(activation_datasets))
    if not datasets:
        raise SystemExit(f"未在激活强度文件中找到指定数据集: {args.dataset_filter}")
    if args.dataset_filter:
        datasets = [d for d in datasets if d == args.dataset_filter]
        if not datasets:
            raise SystemExit(f"未在激活强度文件中找到指定数据集: {args.dataset_filter}")

    # 先构建全局 (dataset, parcel) 级别激活与干预分数字典
    a_global: Dict[str, float] = {}
    l_global: Dict[str, float] = {}
    # 同时缓存每数据集的 L 分数，避免重复 reduce
    l_per_dataset: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        A_ds = activation_all.get(ds, {})
        if not A_ds:
            raise ValueError(f"数据集 {ds} 的激活分数为空")
        for p, v in A_ds.items():
            a_global[_make_pair_key(ds, str(p))] = float(v)

        ds_inter = intervention_all.get(ds) if isinstance(intervention_all, dict) else None
        if ds_inter is None:
            raise FileNotFoundError(f"数据集 {ds} 缺少干预结果，无法继续聚合")
        if args.intervention_mode == 'merged':
            # 直接读取合并后的分数
            root = ds_inter.get('intervention_results', {})
            l_scores = {}
            for parcel, entry in root.items():
                if isinstance(entry, dict) and 'merged' in entry:
                    try:
                        l_scores[str(parcel)] = float(entry['merged'])
                    except Exception:
                        pass
        else:
            l_scores = _reduce_intervention_scores(ds_inter, score_field_effective, args.strength_mode, args.at_strength)
            if args.l_abs and l_scores:
                l_scores = {k: abs(v) for k, v in l_scores.items()}
        l_per_dataset[ds] = l_scores
        for p, v in l_scores.items():
            l_global[_make_pair_key(ds, str(p))] = float(v)

    # 统计报告（可选）
    if args.report_stats:
        a_stats = _summarize_series(a_global)
        l_stats = _summarize_series(l_global)
        report = {
            'activation_stats': a_stats,
            'intervention_stats': l_stats
        }
        stats_path = os.path.join(args.output_dir, 'pre_normalization_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print('ℹ️ 已输出全局分数统计摘要:', stats_path)

    # 计算全局归一化（当未启用 per_dataset_norm 时用于查表）
    if args.norm == 'zscore':
        a_norm_global = _zscore_series(a_global) if a_global else {}
        l_norm_global = _zscore_series(l_global) if l_global else {}
    else:
        a_norm_global = _minmax_series(a_global) if a_global else {}
        l_norm_global = _minmax_series(l_global) if l_global else {}

    # 聚合每个数据集（使用全局 z-score）
    for ds in datasets:
        A_ds = activation_all.get(ds, {})
        dataset_l_scores = l_per_dataset.get(ds, {})
        result = aggregate_one_dataset(
            dataset=ds,
            activation_strengths=A_ds,
            dataset_l_scores=dataset_l_scores,
            K=args.K,
            alpha=args.alpha,
            a_z_global=a_norm_global,
            l_z_global=l_norm_global,
            score_field=args.score_field,
            strength_mode=args.strength_mode,
            at_strength=args.at_strength,
            l_use_abs=args.l_abs,
            per_dataset_norm=args.per_dataset_norm,
            norm=args.norm,
        )

        aggregated[ds] = {
            'final_ranking': result['final_ranking'],
            'K': result['K'],
            'alpha': result['alpha'],
            'topK_activation_parcels': result['topK_activation_parcels'],
            'topK_intervened_parcels': result['topK_intervened_parcels']
        }

        # 输出该数据集详细JSON
        out_path = os.path.join(args.output_dir, f"{ds}.final_parcel_scores.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 收集CSV行（仅取前K或全部）
        for parcel, score in result['final_ranking']:
            rows.append({
                'dataset': ds,
                'parcel': parcel,
                'score': score,
                'A': result['detail'][parcel]['A'],
                'A_norm': result['detail'][parcel]['A_norm'],
                'L': result['detail'][parcel]['L'],
                'L_norm': result['detail'][parcel]['L_norm']
            })

    # 4) 输出总体索引文件与CSV
    index_path = os.path.join(args.output_dir, 'final_parcel_scores_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(args.output_dir, 'final_parcel_scores_all_datasets.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')

    print('✅ 聚合完成。输出目录:', args.output_dir)


if __name__ == '__main__':
    main()


