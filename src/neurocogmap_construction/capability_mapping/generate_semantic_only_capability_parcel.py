#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate semantic-only capability→parcel relationships for capabilities missing
from the existing final aggregation output.

用途：
- 对于未出现在 aggrate_final/final_capability_parcel_all.json 的能力，
  直接依据相似度矩阵 CSV（或 detailed CSV 作为回退）生成该能力与 Parcel 的关系，
  并写入 aggrate_final/final_capability_parcel_all_semantic_only.json。

输出结构（每个能力一个条目）：
{
  capability: {
    "capability": str,
    "source": "semantic_only",
    "ranking": [[parcel, sim_score], ...],
    "detail": {
      parcel: { "sim_score": float, "sim_norm": float }
    }
  }, ...
}

说明：
- ranking 以相似度分数从高到低排序；sim_norm 为该能力内部的 Z-score 标准化。
- 优先从相似度矩阵 CSV 读取（行=capability，列=parcel），若不可用则回退到 detailed CSV。
"""

import os
import json
import argparse
import csv
from typing import Dict, List, Any, Tuple
import math

# 默认路径与 aggregate_final_capability_parcel.py 保持一致
DEFAULT_OUTPUT_DIR = \
    '/path/to/project_root/neural_area/connect_cap_parcel/results/aggrate_final'

DEFAULT_FINAL_ALL_JSON = os.path.join(DEFAULT_OUTPUT_DIR, 'final_capability_parcel_all.json')

DEFAULT_SIMILARITY_MATRIX_CSV = \
    '/path/to/project_root/neural_area/connect_cap_parcel/results/cap_parcel_similarity/capability_parcel_similarity_matrix_qwen.csv'

DEFAULT_SIMILARITY_DETAILED_CSV = \
    '/path/to/project_root/neural_area/connect_cap_parcel/code/capability_parcel_similarity_matrix_detailed.csv'


def _zscore(values: List[float]) -> List[float]:
    if not values:
        return []
    n = float(len(values))
    mean = sum(values) / n
    var = sum((v - mean) * (v - mean) for v in values) / n
    std = math.sqrt(var)
    if std == 0.0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _read_existing_final_caps(path: str) -> List[str]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return list(data.keys()) if isinstance(data, dict) else []
    except Exception:
        return []


def _read_similarity_matrix(path: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, float]]]:
    """返回 (capabilities, parcels, cap->parcel->score)"""
    if not os.path.isfile(path):
        return [], [], {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], [], {}
    header = rows[0]
    parcels = header[1:]
    caps: List[str] = []
    scores: Dict[str, Dict[str, float]] = {}
    for r in rows[1:]:
        if not r:
            continue
        cap = str(r[0]).strip()
        if not cap:
            continue
        caps.append(cap)
        mp: Dict[str, float] = {}
        for p, v in zip(parcels, r[1:]):
            try:
                mp[p] = float(v)
            except Exception:
                mp[p] = 0.0
        scores[cap] = mp
    return caps, parcels, scores


def _read_similarity_detailed(path: str) -> Tuple[List[str], List[str], Dict[str, Dict[str, float]]]:
    """从 detailed CSV 构建 cap->parcel->max(similarity) 简易矩阵。"""
    if not os.path.isfile(path):
        return [], [], {}
    caps_set = set()
    parcels_set = set()
    scores: Dict[str, Dict[str, float]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cap = str(row.get('capability_key', '')).strip()
                pk = row.get('parcel_key')
                pid = row.get('parcel_id')
                parcel = str(pk if pk not in (None, '') else pid).strip()
                if not cap or not parcel:
                    continue
                s = float(row.get('similarity_score', 0.0))
            except Exception:
                continue
            caps_set.add(cap)
            parcels_set.add(parcel)
            mp = scores.setdefault(cap, {})
            if parcel in mp:
                if s > mp[parcel]:
                    mp[parcel] = s
            else:
                mp[parcel] = s
    return sorted(caps_set), sorted(parcels_set), scores


def generate_semantic_only(
    final_all_path: str,
    similarity_matrix_csv: str,
    similarity_detailed_csv: str,
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    existing_caps = set(_read_existing_final_caps(final_all_path))

    # 优先使用矩阵 CSV；若不可用则回退 detailed CSV
    caps, parcels, sim_scores = _read_similarity_matrix(similarity_matrix_csv)
    if not sim_scores:
        caps, parcels, sim_scores = _read_similarity_detailed(similarity_detailed_csv)

    if not sim_scores:
        raise SystemExit('未能读取到任何相似度数据（matrix/detailed 均不可用）')

    missing_caps = [c for c in caps if c not in existing_caps]

    result: Dict[str, Any] = {}
    for cap in missing_caps:
        p_to_s = sim_scores.get(cap, {})
        if not p_to_s:
            continue
        # 构建能力内部的 z-score 标准化
        vals = [float(v) for v in p_to_s.values()]
        zs = _zscore(vals)
        parcels_list = list(p_to_s.keys())
        sim_norm_map = {p: z for p, z in zip(parcels_list, zs)}

        # 排序（按原始 sim_score 从高到低）
        ranking = sorted(p_to_s.items(), key=lambda t: t[1], reverse=True)

        detail = {p: {"sim_score": float(s), "sim_norm": float(sim_norm_map.get(p, 0.0))}
                  for p, s in ranking}

        result[cap] = {
            "capability": cap,
            "source": "semantic_only",
            "ranking": [[p, float(s)] for p, s in ranking],
            "detail": detail,
        }

    out_path = os.path.join(output_dir, 'final_capability_parcel_all_semantic_only.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'✅ 已生成语义专用能力-Parcel 关系，共 {len(result)} 个能力：{out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Generate semantic-only capability→parcel relationships for capabilities missing in final output.')
    parser.add_argument('--final_all_path', type=str, default=DEFAULT_FINAL_ALL_JSON,
                        help='已存在的最终融合结果 final_capability_parcel_all.json 路径')
    parser.add_argument('--similarity_matrix_csv', type=str, default=DEFAULT_SIMILARITY_MATRIX_CSV,
                        help='相似度矩阵 CSV（行=capability，列=parcel）')
    parser.add_argument('--similarity_detailed_csv', type=str, default=DEFAULT_SIMILARITY_DETAILED_CSV,
                        help='detailed 相似度 CSV（回退用）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='输出目录（将写入 final_capability_parcel_all_semantic_only.json）')

    args = parser.parse_args()
    generate_semantic_only(
        final_all_path=args.final_all_path,
        similarity_matrix_csv=args.similarity_matrix_csv,
        similarity_detailed_csv=args.similarity_detailed_csv,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()


