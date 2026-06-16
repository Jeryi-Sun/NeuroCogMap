#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Capability → Parcel Weighted Rankings

依据 merged_capability_dataset_stats.json 中各能力(capability)下数据集的 "relevance_score" 作为权重，
结合聚合脚本（aggregate_activation_intervention.py / run_aggregate_pipeline.py）产出的每数据集
Parcel 最终分数（S），计算每个能力到 Parcel 的加权相关性分数与排序。

加权策略（默认）：
- 对于某能力 C，设其关联数据集集合 D(C)，每个数据集 d∈D(C) 有权重 w_d（取自 relevance_score）；
- 对每个 Parcel p，综合分数为：Score_C(p) = Σ_{d∈D(C)} (ŵ_d * S_d(p))，其中 ŵ_d = w_d / Σ w；
- 若某数据集结果中不存在该 Parcel，则该项视为 0；
- 最终按 Score_C(p) 降序排序，输出排名与详细分解。

输入：
- --capability_stats：能力-数据集映射与 relevance_score（merged_capability_dataset_stats.json）
- --aggregate_dir：聚合结果目录，包含 "{dataset}.final_parcel_scores.json" 和/或索引文件
  * 兼容 run_aggregate_pipeline.sh 默认输出目录：results/aggregate_data_parcel
  * 兼容 aggregate_activation_intervention.py 默认输出目录：results/aggregate
- --output_dir：能力→Parcel 排名的输出目录（默认 results/capability_parcel）

输出：
- capability_{capability}.parcel_ranking.json：
  {
    "capability": str,
    "datasets_used": [dataset,...],
    "weight_sum": float,
    "ranking": [[parcel, score], ...],
    "detail": { parcel: { "score": float, "by_dataset": { ds: { "weight": ŵ_d, "parcel_score": S_d(p), "contrib": ŵ_d*S_d(p) } } } }
  }
- capability_parcel_ranking_index.json：各能力的摘要索引（前若干名）
- capability_parcel_ranking_all.csv：展平后的汇总 CSV（capability, parcel, score, dataset, weight, parcel_score, contrib）

注意：
- 数据集名称可能存在变体（例如是否带有 "_qa"）。本脚本内置名称匹配与映射策略：
  1) 直接精确匹配；
  2) 去除或添加后缀 "_qa" 再匹配；
  3) 小写再匹配；
  若仍未匹配上，则跳过该数据集并给出告警。
"""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import csv


DEFAULT_CAPABILITY_STATS = \
    '/path/to/project_root/neural_area/capability_data_v2/data_stastic/merged_capability_dataset_stats.json'

# 可能的默认聚合结果目录（二选一都支持）
DEFAULT_AGGREGATE_DIR_CANDIDATES = [
    '/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate_data_parcel',
    '/path/to/project_root/neural_area/connect_cap_parcel/results/aggregate',
]

DEFAULT_OUTPUT_DIR = \
    '/path/to/project_root/neural_area/connect_cap_parcel/results/capability_parcel'


def _load_capability_stats(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 结构：{ capability: { dataset: { relevance_score: float, ... }, ... }, ... }
    return data


def _list_dataset_result_files(aggregate_dir: str) -> Dict[str, str]:
    """扫描目录下 {dataset}.final_parcel_scores.json 文件，返回 {dataset_name: filepath}。
    dataset_name 使用文件名前缀。
    """
    mapping: Dict[str, str] = {}
    if not os.path.isdir(aggregate_dir):
        return mapping
    for fname in os.listdir(aggregate_dir):
        if not fname.endswith('.final_parcel_scores.json'):
            continue
        ds = fname[:-len('.final_parcel_scores.json')]
        mapping[ds] = os.path.join(aggregate_dir, fname)
    return mapping


def _load_one_dataset_result(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: float(max(0.0, float(v)) / total) for k, v in weights.items()}


def _candidate_keys(name: str) -> List[str]:
    name_l = name.lower()
    variants = {name, name_l}
    # 去除或添加 _qa 变体
    if name_l.endswith('_qa'):
        variants.add(name_l[:-3])
        variants.add(name[:-3])
    else:
        variants.add(name_l + '_qa')
        variants.add(name + '_qa')
    return list(variants)


def _match_dataset_to_file(dataset_name: str, available: Dict[str, str]) -> Optional[str]:
    # 精确命中
    if dataset_name in available:
        return available[dataset_name]
    # 小写命中
    lower_map = {k.lower(): v for k, v in available.items()}
    if dataset_name.lower() in lower_map:
        return lower_map[dataset_name.lower()]
    # 变体尝试
    for cand in _candidate_keys(dataset_name):
        if cand in available:
            return available[cand]
        if cand in lower_map:
            return lower_map[cand]
    return None


def _collect_all_parcel_scores(dataset_files: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """返回 {dataset: {parcel: S}}，仅使用 result['final_ranking'] 或 result['detail'][p]['S']。
    优先从 final_ranking 读取 (parcel, score)。
    """
    ds_to_scores: Dict[str, Dict[str, float]] = {}
    for ds, fp in dataset_files.items():
        try:
            result = _load_one_dataset_result(fp)
        except Exception:
            continue
        scores: Dict[str, float] = {}
        ranking = result.get('final_ranking')
        if isinstance(ranking, list):
            for item in ranking:
                if isinstance(item, list) and len(item) == 2:
                    p, s = item
                    try:
                        scores[str(p)] = float(s)
                    except Exception:
                        continue
        if not scores:
            detail = result.get('detail', {})
            for p, info in detail.items():
                try:
                    scores[str(p)] = float(info.get('S', 0.0))
                except Exception:
                    continue
        if scores:
            ds_to_scores[ds] = scores
    return ds_to_scores


def compute_capability_parcel_rankings(
    capability_stats_path: str,
    aggregate_dir: str,
    output_dir: str,
    topn: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    capability_stats = _load_capability_stats(capability_stats_path)

    # 扫描聚合结果
    dataset_files = _list_dataset_result_files(aggregate_dir)
    if not dataset_files:
        raise FileNotFoundError(f"聚合结果目录为空或不存在：{aggregate_dir}")

    # 预加载全部数据集的 parcel 分数，使用文件名前缀作为数据集名键
    ds_to_scores = _collect_all_parcel_scores(dataset_files)

    # 为名称匹配准备 available 映射
    available: Dict[str, str] = {ds: ds for ds in ds_to_scores.keys()}

    # 汇总 CSV 收集
    csv_rows: List[Dict[str, Any]] = []

    # 索引摘要
    index_summary: Dict[str, Any] = {}

    for capability, ds_info in capability_stats.items():
        # 收集该能力涉及的数据集原始权重
        raw_weights: Dict[str, float] = {}
        for ds_name, meta in ds_info.items():
            if not isinstance(meta, dict):
                continue
            try:
                w = float(meta.get('relevance_score', 0.0))
            except Exception:
                w = 0.0
            if w <= 0:
                continue
            raw_weights[ds_name] = w

        if not raw_weights:
            # 无有效数据集权重，跳过
            continue

        # 名称匹配：找到每个数据集对应的实际结果键
        matched_ds_to_key: Dict[str, str] = {}
        for ds_name in raw_weights.keys():
            key = _match_dataset_to_file(ds_name, available)
            if key is None:
                # 尝试再用去尾/加尾策略匹配（已在 _match 内部完成），告警留给汇总
                continue
            matched_ds_to_key[ds_name] = key

        if not matched_ds_to_key:
            # 无匹配结果，跳过
            index_summary[capability] = {
                'datasets_used': [],
                'note': 'no matched datasets in aggregate results'
            }
            continue

        # 仅对匹配成功的数据集进行归一化权重
        used_weights = {ds: raw_weights[ds] for ds in matched_ds_to_key.keys()}
        norm_weights = _normalize_weights(used_weights)

        # 聚合到 Parcel：Score_C(p) = Σ ŵ_d * S_d(p)
        parcel_score_sum: Dict[str, float] = defaultdict(float)
        parcel_detail: Dict[str, Dict[str, Any]] = {}

        for orig_ds, key in matched_ds_to_key.items():
            w_norm = norm_weights.get(orig_ds, 0.0)
            parcel_scores = ds_to_scores.get(key, {})
            for p, s in parcel_scores.items():
                contrib = w_norm * s
                parcel_score_sum[p] += contrib
                if p not in parcel_detail:
                    parcel_detail[p] = {
                        'score': 0.0,
                        'by_dataset': {}
                    }
                parcel_detail[p]['by_dataset'][orig_ds] = {
                    'weight': w_norm,
                    'parcel_score': s,
                    'contrib': contrib
                }

        # 填充分数
        for p in parcel_detail.keys():
            parcel_detail[p]['score'] = float(parcel_score_sum.get(p, 0.0))

        # 排序与裁剪
        ranking = sorted(parcel_score_sum.items(), key=lambda t: t[1], reverse=True)
        if topn > 0:
            ranking = ranking[:topn]

        # 写入能力 JSON
        cap_out = {
            'capability': capability,
            'datasets_used': list(matched_ds_to_key.keys()),
            'weight_sum': float(sum(used_weights.values())),
            'ranking': [[p, float(s)] for p, s in ranking],
            'detail': {p: parcel_detail[p] for p, _ in ranking},
        }
        cap_json_path = os.path.join(output_dir, f"capability_{capability}.parcel_ranking.json")
        with open(cap_json_path, 'w', encoding='utf-8') as f:
            json.dump(cap_out, f, ensure_ascii=False, indent=2)

        # 索引摘要
        index_summary[capability] = {
            'datasets_used': list(matched_ds_to_key.keys()),
            'topK_preview': [[p, float(s)] for p, s in ranking[: min(10, len(ranking))]],
        }

        # CSV 行展开：仅对 topN 的 parcel 展开 by_dataset 明细
        for p, total_s in ranking:
            ds_map = parcel_detail[p]['by_dataset']
            for ds, info in ds_map.items():
                csv_rows.append({
                    'capability': capability,
                    'parcel': p,
                    'score': float(total_s),
                    'dataset': ds,
                    'weight': float(info.get('weight', 0.0)),
                    'parcel_score': float(info.get('parcel_score', 0.0)),
                    'contrib': float(info.get('contrib', 0.0)),
                })

    # 写入索引摘要
    index_path = os.path.join(output_dir, 'capability_parcel_ranking_index.json')
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)

    # 写入 CSV
    if csv_rows:
        csv_path = os.path.join(output_dir, 'capability_parcel_ranking_all.csv')
        fieldnames = ['capability', 'parcel', 'score', 'dataset', 'weight', 'parcel_score', 'contrib']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

    print('✅ 能力→Parcel 加权排名计算完成。输出目录:', output_dir)


def _resolve_aggregate_dir(user_dir: Optional[str]) -> str:
    if user_dir:
        return user_dir
    for cand in DEFAULT_AGGREGATE_DIR_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    # 默认返回首个候选（即使不存在，后续会报错更清晰）
    return DEFAULT_AGGREGATE_DIR_CANDIDATES[0]


def main():
    parser = argparse.ArgumentParser(description='Compute capability→parcel weighted rankings.')
    parser.add_argument('--capability_stats', type=str, default=DEFAULT_CAPABILITY_STATS,
                        help='merged_capability_dataset_stats.json 路径')
    parser.add_argument('--aggregate_dir', type=str, default='',
                        help='聚合结果目录（包含 {dataset}.final_parcel_scores.json）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='能力→Parcel 排名输出目录')
    parser.add_argument('--topn', type=int, default=0, help='每个能力保留前N个 Parcel；0=保留全部')

    args = parser.parse_args()

    aggregate_dir = _resolve_aggregate_dir(args.aggregate_dir)

    compute_capability_parcel_rankings(
        capability_stats_path=args.capability_stats,
        aggregate_dir=aggregate_dir,
        output_dir=args.output_dir,
        topn=args.topn,
    )


if __name__ == '__main__':
    main()
