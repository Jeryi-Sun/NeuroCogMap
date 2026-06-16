#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate Final Capability → Parcel Rankings

融合两部分结果以得到最终排序：
1) compute_capability_parcel_ranking.py 生成的能力→Parcel 加权分数（基于数据集 relevance_score 的聚合）
2) compute_semantic_similarity.py 生成的 capability-parcel 语义相似性（使用其 detailed CSV）

融合方式：
- 采用全局 Z-score 归一化（跨“全部能力 × 全部 Parcel”的值域）：
  * R_norm(p) = zscore(加权分数)
  * S_norm(p) = zscore(语义相似度)
- 加权融合：Final(p) = (1 - beta) * R_norm(p) + beta * S_norm(p)
- 支持仅一侧可用时的回退（例如没有语义相似度，则 Final = R_norm；反之亦然）

匹配/对齐：
- 能力维度使用 detailed CSV 中的 capability_key 与加权结果目录中的文件名 token（capability_{capability}.parcel_ranking.json）对应
- Parcel 维度：尝试按如下顺序匹配
  1) 直接以 parcel_key 精确匹配
  2) 将两侧转为字符串后与 parcel_id 精确匹配
  3) 大小写、前缀（去/加 "parcel_"、"Parcel_"）的变体匹配

输入：
- --cap_parcel_dir：compute_capability_parcel_ranking.py 的输出目录（包含 capability_*.parcel_ranking.json）
- --similarity_detailed_csv：compute_semantic_similarity.py 产出的 *_detailed.csv 文件
- --output_dir：最终聚合结果输出目录（默认 results/aggrate_final）
- --beta：融合权重 beta ∈ [0,1]（默认 0.5）
- --topn：每个能力保留前 N 个 Parcel（0=全部）

输出：
- final_capability_parcel_all.json：按 capability 汇总的完整结果字典，形如：
  {
    capability: {
      "capability": str,
      "beta": float,
      "topn": int,
      "ranking": [[parcel, final_score], ...],
      "detail": {
        parcel: {
          "final": float,
          "rank_score": float,
          "rank_norm": float,
          "sim_score": float,
          "sim_norm": float
        }, ...
      }
    }, ...
  }
- final_capability_parcel_index.json：各能力的前若干名预览
- final_capability_parcel_all.csv：展平的明细（capability, parcel, final, rank_score, rank_norm, sim_score, sim_norm）
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional
import math
import csv
import sys
from pathlib import Path

try:
    from neurocogmap_release.paths import env_path_str, output_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from neurocogmap_release.paths import env_path_str, output_path


DEFAULT_CAP_PARCEL_DIR = env_path_str("NEUROCOGMAP_CAP_PARCEL_DIR")
DEFAULT_SIMILARITY_DETAILED_CSV = env_path_str("NEUROCOGMAP_SIMILARITY_DETAILED_CSV")
DEFAULT_OUTPUT_DIR = str(output_path("neurocogmap_construction", "capability_mapping", "aggrate_final"))
DEFAULT_SIMILARITY_MATRIX_CSV = env_path_str("NEUROCOGMAP_SIMILARITY_MATRIX_CSV")


def _minmax(series: List[float]) -> Tuple[float, float]:
    vmin = min(series) if series else 0.0
    vmax = max(series) if series else 0.0
    return float(vmin), float(vmax)


def _minmax_normalize_map(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    vmin, vmax = _minmax(list(values.values()))
    denom = vmax - vmin
    if denom == 0:
        # 全相同，统一置为 0
        return {k: 0.0 for k in values}
    return {k: (float(v) - vmin) / denom for k, v in values.items()}


def _zscore_map(values: Dict[str, float]) -> Dict[str, float]:
    """对一组值做 Z-score：返回与输入 keys 对应的 z 值；若样本为空或方差为0，返回全0。"""
    if not values:
        return {}
    xs = [float(v) for v in values.values()]
    n = float(len(xs))
    mean = sum(xs) / n
    var = sum((x - mean) * (x - mean) for x in xs) / n
    std = math.sqrt(var)
    if std == 0.0:
        return {k: 0.0 for k in values}
    return {k: (float(v) - mean) / std for k, v in values.items()}


def _normalize_capability_token(name: str) -> str:
    """规范化能力名称用于跨来源匹配。"""
    s = str(name).lower().strip()
    s = s.replace('capability', '')
    s = re.sub(r'[\s_\-]+', '', s)
    return s


def _read_capability_rank_jsons(cap_parcel_dir: str) -> Dict[str, Dict[str, float]]:
    """读取 capability_{capability}.parcel_ranking.json，返回 {capability: {parcel: rank_score}}"""
    result: Dict[str, Dict[str, float]] = {}
    if not os.path.isdir(cap_parcel_dir):
        return result
    for fname in os.listdir(cap_parcel_dir):
        if not fname.startswith('capability_') or not fname.endswith('.parcel_ranking.json'):
            continue
        cap = fname[len('capability_'):-len('.parcel_ranking.json')]
        fpath = os.path.join(cap_parcel_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        ranking = data.get('ranking', [])
        scores: Dict[str, float] = {}
        for item in ranking:
            if isinstance(item, list) and len(item) == 2:
                p, s = item
                try:
                    scores[str(p)] = float(s)
                except Exception:
                    continue
        result[cap] = scores
    return result


def _read_similarity_detailed(detailed_csv: str) -> List[Dict[str, Any]]:
    """读取 detailed CSV，返回记录列表。预期列：capability_key, parcel_key, parcel_id, similarity_score ..."""
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(detailed_csv):
        return rows
    with open(detailed_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_similarity_matrix_csv(path: str) -> Dict[str, Dict[str, float]]:
    """读取矩阵 CSV：首行是 Parcel 列名，首列是 capability 名称。
    返回 {normalized_capability_token: {normalized_parcel_token: score}}。
    """
    if not os.path.isfile(path):
        return {}
    cap_to_sim: Dict[str, Dict[str, float]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}
    header = rows[0]
    parcel_headers = [h for h in header[1:]]  # 跳过第一列
    norm_parcel_headers = [_normalize_parcel_token(str(h)) for h in parcel_headers]
    for r in rows[1:]:
        if not r:
            continue
        cap_raw = str(r[0]).strip()
        if not cap_raw:
            continue
        cap_norm = _normalize_capability_token(cap_raw)
        vals = r[1:]
        mp: Dict[str, float] = {}
        for nt, v in zip(norm_parcel_headers, vals):
            try:
                mp[nt] = float(v)
            except Exception:
                continue
        cap_to_sim[cap_norm] = mp
    return cap_to_sim


def _normalize_parcel_token(token: str) -> str:
    t = str(token)
    t_low = t.lower().strip()
    # 去常见前缀
    t_low = re.sub(r'^(parcel_|parcel-)', '', t_low)
    t_low = re.sub(r'^(p_|p-)', '', t_low)
    return t_low


def _fuse_one_capability(
    capability: str,
    rank_scores: Dict[str, float],
    sim_rows: List[Dict[str, Any]],
    beta: float,
    topn: int,
) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, float]]]:
    """返回 (ranking_list, detail_map)"""
    # 1) 准备两侧的可用集合与映射
    # rank 侧：parcel_token -> raw_score
    rank_map: Dict[str, float] = {str(k): float(v) for k, v in (rank_scores or {}).items()}

    # sim 侧：把属于该 capability 的记录聚合为 parcel -> max(sim_score)
    sim_map: Dict[str, float] = {}
    for r in sim_rows:
        try:
            cap_key = str(r.get('capability_key', '')).strip()
            if cap_key != capability:
                continue
            sim_s = float(r.get('similarity_score', 0.0))
            # parcel key/id 两种尝试
            pk = r.get('parcel_key')
            pid = r.get('parcel_id')
            cand_keys = []
            if pk is not None:
                cand_keys.append(str(pk))
            if pid is not None:
                cand_keys.append(str(pid))
            # 归一化成多个候选 token
            norm_tokens = set()
            for ck in cand_keys:
                norm_tokens.add(_normalize_parcel_token(ck))
                norm_tokens.add(_normalize_parcel_token('parcel_' + ck))
            for nt in norm_tokens:
                # 以最大相似度为该 parcel 的 sim 值
                if nt in sim_map:
                    if sim_s > sim_map[nt]:
                        sim_map[nt] = sim_s
                else:
                    sim_map[nt] = sim_s
        except Exception:
            continue

    # 构造 rank 侧的归一化 token 映射
    rank_norm_token_to_key: Dict[str, str] = {}
    for k in rank_map.keys():
        nk = _normalize_parcel_token(k)
        rank_norm_token_to_key[nk] = k

    # 将 sim_map 的 token 映射回 rank 的原 key（若可能），否则保留其自身 token 作为 key
    # 同时收集所有候选 parcel 的并集
    all_parcels: List[str] = []
    parcel_to_rank_raw: Dict[str, float] = {}
    parcel_to_sim_raw: Dict[str, float] = {}

    # 先加入 rank 侧的所有 parcel
    for rk, rv in rank_map.items():
        all_parcels.append(rk)
        parcel_to_rank_raw[rk] = rv
        # 若 sim 有匹配，则写入
        nk = _normalize_parcel_token(rk)
        if nk in sim_map:
            parcel_to_sim_raw[rk] = sim_map[nk]

    # 再加入那些只出现在 sim 侧的 parcel
    for nt, sv in sim_map.items():
        if nt in rank_norm_token_to_key:
            # 已在上一步加入
            continue
        # 使用标准化 token 作为 key 存入
        all_parcels.append(nt)
        parcel_to_sim_raw[nt] = sv
        # rank 分数缺失视为 0
        if nt not in parcel_to_rank_raw:
            parcel_to_rank_raw[nt] = 0.0

    # 2) 归一化将由 main 中的“全局 Z-score”提供；此处仅先占位，稍后由调用方覆盖。
    rank_norm = parcel_to_rank_raw
    sim_norm = parcel_to_sim_raw

    # 3) 融合（占位，最终在 aggregate_final 中以全局 Z 分数替换 rank_norm/sim_norm 后再计算）
    final_scores: Dict[str, float] = {}
    for p in all_parcels:
        r_n = float(rank_norm.get(p, 0.0))
        s_n = float(sim_norm.get(p, 0.0))
        # 若某侧完全缺失（该能力所有 parcel 上都无值），其归一化表可能为空，get 默认 0
        final_scores[p] = (1.0 - beta) * r_n + beta * s_n

    ranking = sorted(final_scores.items(), key=lambda t: t[1], reverse=True)
    if topn > 0:
        ranking = ranking[:topn]

    # 4) 细节
    detail: Dict[str, Dict[str, float]] = {}
    for p, fval in ranking:
        detail[p] = {
            'final': float(fval),
            'rank_score': float(parcel_to_rank_raw.get(p, 0.0)),
            'rank_norm': float(rank_norm.get(p, 0.0)),
            'sim_score': float(parcel_to_sim_raw.get(p, 0.0)),
            'sim_norm': float(sim_norm.get(p, 0.0)),
        }

    return ranking, detail


def aggregate_final(
    cap_parcel_dir: str,
    similarity_detailed_csv: str,
    similarity_matrix_csv: Optional[str],
    output_dir: str,
    beta: float,
    topn: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 读取两侧数据
    cap_to_rank_scores = _read_capability_rank_jsons(cap_parcel_dir)
    sim_rows = _read_similarity_detailed(similarity_detailed_csv)
    # 构建“全局（受限于 rank 的 TopN）”原始分数字典：以 (capability||normalized_parcel_token) 为 key
    def _pair_key(cap: str, parcel_token: str) -> str:
        return f"{cap}||{parcel_token}"

    # 1) 先确定每个能力从 rank 侧选取的 Parcel（前 topn；topn<=0 则取全部）
    cap_to_selected_original_keys: Dict[str, List[str]] = {}
    cap_to_selected_norm_tokens: Dict[str, List[str]] = {}
    for cap, rmap in cap_to_rank_scores.items():
        if not rmap:
            continue
        # 按原始分数从大到小排序
        pairs = sorted(((k, float(v)) for k, v in rmap.items()), key=lambda t: t[1], reverse=True)
        if topn > 0:
            pairs = pairs[:topn]
        selected_keys = [str(k) for k, _ in pairs]
        cap_to_selected_original_keys[cap] = selected_keys
        cap_to_selected_norm_tokens[cap] = [_normalize_parcel_token(k) for k in selected_keys]

    # 2) 预聚合相似度，但仅保留入选的 Parcel（基于规范化 token 匹配）
    cap_to_sim_map: Dict[str, Dict[str, float]] = {}
    use_matrix = bool(similarity_matrix_csv and os.path.isfile(similarity_matrix_csv))
    if use_matrix:
        cap_to_sim_matrix = _read_similarity_matrix_csv(similarity_matrix_csv)
        # 构建规范化名称映射：normalized_name -> original capability key
        norm_to_cap: Dict[str, str] = {
            _normalize_capability_token(k): k for k in cap_to_selected_norm_tokens.keys()
        }
        for norm_name, smap_full in cap_to_sim_matrix.items():
            mapped_cap = norm_to_cap.get(norm_name)
            if not mapped_cap:
                continue
            allowed_nt = set(cap_to_selected_norm_tokens[mapped_cap])
            if not smap_full:
                continue
            cap_to_sim_map[mapped_cap] = {nt: float(smap_full.get(nt, 0.0)) for nt in allowed_nt}
    else:
        for r in sim_rows:
            try:
                cap_key_raw = str(r.get('capability_key', '')).strip()
                if not cap_key_raw:
                    continue
                norm_to_cap: Dict[str, str] = {
                    _normalize_capability_token(k): k for k in cap_to_selected_norm_tokens.keys()
                }
                mapped_cap = norm_to_cap.get(_normalize_capability_token(cap_key_raw))
                if not mapped_cap:
                    continue
                allowed_nt = set(cap_to_selected_norm_tokens[mapped_cap])
                sim_s = float(r.get('similarity_score', 0.0))
                pk = r.get('parcel_key')
                pid = r.get('parcel_id')
                cand_keys = []
                if pk is not None:
                    cand_keys.append(str(pk))
                if pid is not None:
                    cand_keys.append(str(pid))
                norm_tokens = set()
                for ck in cand_keys:
                    norm_tokens.add(_normalize_parcel_token(ck))
                    norm_tokens.add(_normalize_parcel_token('parcel_' + ck))
                norm_tokens = {nt for nt in norm_tokens if nt in allowed_nt}
                if not norm_tokens:
                    continue
                mp = cap_to_sim_map.setdefault(mapped_cap, {})
                for nt in norm_tokens:
                    if nt in mp:
                        if sim_s > mp[nt]:
                            mp[nt] = sim_s
                    else:
                        mp[nt] = sim_s
            except Exception:
                continue

    # 3) 汇总“受限集合”的全局 rank/sim 原始值
    rank_global_raw: Dict[str, float] = {}
    sim_global_raw: Dict[str, float] = {}
    for cap, selected_keys in cap_to_selected_original_keys.items():
        rmap = cap_to_rank_scores.get(cap, {})
        for p_raw in selected_keys:
            nt = _normalize_parcel_token(str(p_raw))
            rank_global_raw[_pair_key(cap, nt)] = float(rmap.get(p_raw, 0.0))

    for cap, selected_nts in cap_to_selected_norm_tokens.items():
        smap = cap_to_sim_map.get(cap, {})
        for nt in selected_nts:
            if nt in smap:
                sim_global_raw[_pair_key(cap, nt)] = float(smap[nt])
    # 计算全局 Z-score
    rank_z_global: Dict[str, float] = _zscore_map(rank_global_raw)
    sim_z_global: Dict[str, float] = _zscore_map(sim_global_raw)

    # 融合
    index_preview: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    combined_json_by_capability: Dict[str, Any] = {}

    for capability, selected_keys in cap_to_selected_original_keys.items():
        # 组装 detail 与最终分数，仅针对入选的 parcel
        rmap = cap_to_rank_scores.get(capability, {})
        smap = cap_to_sim_map.get(capability, {})
        detail: Dict[str, Dict[str, float]] = {}
        final_scores: Dict[str, float] = {}
        for p_raw in selected_keys:
            nt = _normalize_parcel_token(str(p_raw))
            pair_k = _pair_key(capability, nt)
            r_raw = float(rmap.get(p_raw, 0.0))
            s_raw = float(smap.get(nt, 0.0)) if nt in smap else 0.0
            r_n = float(rank_z_global.get(pair_k, 0.0))
            s_n = float(sim_z_global.get(pair_k, 0.0))
            final_scores[p_raw] = (1.0 - beta) * r_n + beta * s_n
            detail[p_raw] = {
                'final': float(final_scores[p_raw]),
                'rank_score': r_raw,
                'rank_norm': r_n,
                'sim_score': s_raw,
                'sim_norm': s_n,
            }

        ranking = sorted(final_scores.items(), key=lambda t: t[1], reverse=True)
        # 已限定为 selected_keys，无需再截断；若 topn<=0 代表“全部”，selected_keys 已包含全部

        # 组织单能力 JSON 内容（不再单独写文件）
        out_json = {
            'capability': capability,
            'beta': float(beta),
            'topn': int(topn if topn > 0 else len(selected_keys)),
            'ranking': [[p, float(s)] for p, s in ranking],
            'detail': detail,
        }
        # 汇总到总文件结构中
        combined_json_by_capability[capability] = out_json

        # 预览索引
        index_preview[capability] = {
            'topK_preview': [[p, float(s)] for p, s in ranking[: min(10, len(ranking))]],
        }

        # CSV 展开
        for p, s in ranking:
            info = detail.get(p, {})
            csv_rows.append({
                'capability': capability,
                'parcel': p,
                'final': float(s),
                'rank_score': float(info.get('rank_score', 0.0)),
                'rank_norm': float(info.get('rank_norm', 0.0)),
                'sim_score': float(info.get('sim_score', 0.0)),
                'sim_norm': float(info.get('sim_norm', 0.0)),
            })

    # 写索引与 CSV
    idx_path = os.path.join(output_dir, 'final_capability_parcel_index.json')
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(index_preview, f, ensure_ascii=False, indent=2)

    if csv_rows:
        csv_path = os.path.join(output_dir, 'final_capability_parcel_all.csv')
        fieldnames = ['capability', 'parcel', 'final', 'rank_score', 'rank_norm', 'sim_score', 'sim_norm']
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

    # 写合并后的总 JSON（capability -> 单能力 JSON 内容）
    merged_json_path = os.path.join(output_dir, 'final_capability_parcel_all.json')
    with open(merged_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_json_by_capability, f, ensure_ascii=False, indent=2)

    print('✅ 最终融合完成。输出目录:', output_dir)


def main():
    parser = argparse.ArgumentParser(description='Aggregate final capability→parcel rankings by fusing weighted scores and semantic similarity.')
    parser.add_argument('--cap_parcel_dir', type=str, default=DEFAULT_CAP_PARCEL_DIR,
                        help='compute_capability_parcel_ranking.py 的输出目录')
    parser.add_argument('--similarity_detailed_csv', type=str, default=DEFAULT_SIMILARITY_DETAILED_CSV,
                        help='compute_semantic_similarity.py 生成的 *_detailed.csv 路径')
    parser.add_argument('--similarity_matrix_csv', type=str, default=DEFAULT_SIMILARITY_MATRIX_CSV,
                        help='可选：从相似度矩阵CSV读取（行=capability, 列=parcel）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='最终结果输出目录')
    parser.add_argument('--beta', type=float, default=0.5, help='融合权重 beta ∈ [0,1]')
    parser.add_argument('--topn', type=int, default=0, help='每个能力保留前 N 个 Parcel；0=全部')

    args = parser.parse_args()

    if args.beta < 0.0 or args.beta > 1.0:
        raise SystemExit('beta 必须位于 [0,1] 区间内')
    if not args.cap_parcel_dir:
        raise SystemExit(
            '缺少 capability parcel ranking 输入：请传入 --cap_parcel_dir '
            '或设置 NEUROCOGMAP_CAP_PARCEL_DIR。'
        )
    if not args.similarity_detailed_csv and not args.similarity_matrix_csv:
        raise SystemExit(
            '缺少 capability-parcel 语义相似度输入：请传入 '
            '--similarity_detailed_csv / --similarity_matrix_csv，'
            '或设置 NEUROCOGMAP_SIMILARITY_DETAILED_CSV / NEUROCOGMAP_SIMILARITY_MATRIX_CSV。'
        )

    aggregate_final(
        cap_parcel_dir=args.cap_parcel_dir,
        similarity_detailed_csv=args.similarity_detailed_csv,
        similarity_matrix_csv=args.similarity_matrix_csv,
        output_dir=args.output_dir,
        beta=args.beta,
        topn=args.topn,
    )


if __name__ == '__main__':
    main()
