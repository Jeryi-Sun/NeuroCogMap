#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从分析输出中自动构建指示器配置。

输入（分析输出根目录，如 results/analysis_output/<MODEL_DATA>）：
- parcel_level/top_anomalous_parcels.json（含 activation_diff）
- parcel_level/anomalous_connections.json（含 pos_connections/neg_connections）
- capability_level/top_anomalous_capabilities.json（含 activation_diff 与名称）
- capability_level/anomalous_capability_connections.json（含 pos_connections/neg_connections）

输出：包含以下字段的配置：
- pos_parcels / neg_parcels
- pos_capabilities / neg_capabilities（名称）
- pos_parcel_pairs / neg_parcel_pairs（(i,j) 对）
- pos_capability_pairs / neg_capability_pairs（(name_i, name_j) 对）
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class AutoIndicator:
    pos_parcels: List[int]
    neg_parcels: List[int]
    pos_capabilities: List[str]
    neg_capabilities: List[str]
    pos_parcel_pairs: List[Tuple[int, int]]
    neg_parcel_pairs: List[Tuple[int, int]]
    pos_capability_pairs: List[Tuple[str, str]]
    neg_capability_pairs: List[Tuple[str, str]]


def _load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_auto_indicator(analysis_root: str,
                         top_k_parcel: int = 10,
                         top_k_capability: int = 10,
                         top_k_pairs: int = 25) -> AutoIndicator:
    root = Path(analysis_root)
    p_parcel_top = root / 'parcel_level' / 'top_anomalous_parcels.json'
    p_parcel_conn = root / 'parcel_level' / 'anomalous_connections.json'
    p_cap_top = root / 'capability_level' / 'top_anomalous_capabilities.json'
    p_cap_conn = root / 'capability_level' / 'anomalous_capability_connections.json'

    if not p_parcel_top.exists():
        raise FileNotFoundError(f"缺少文件: {p_parcel_top}")
    if not p_parcel_conn.exists():
        raise FileNotFoundError(f"缺少文件: {p_parcel_conn}")
    if not p_cap_top.exists():
        raise FileNotFoundError(f"缺少文件: {p_cap_top}")
    if not p_cap_conn.exists():
        raise FileNotFoundError(f"缺少文件: {p_cap_conn}")

    # 1) parcel 正负列表
    parcel_top_list = _load_json(p_parcel_top)
    if not isinstance(parcel_top_list, list):
        raise ValueError("top_anomalous_parcels.json 应为列表")
    pos_parcels: List[int] = []
    neg_parcels: List[int] = []
    for item in parcel_top_list:
        pid = int(item.get('parcel_id'))
        diff = float(item.get('activation_diff', 0.0))
        if diff > 0:
            pos_parcels.append(pid)
        elif diff < 0:
            neg_parcels.append(pid)
    pos_parcels = pos_parcels[:top_k_parcel]
    neg_parcels = neg_parcels[:top_k_parcel]

    # 2) capability 正负列表
    cap_top_list = _load_json(p_cap_top)
    if not isinstance(cap_top_list, list):
        raise ValueError("top_anomalous_capabilities.json 应为列表")
    pos_caps: List[str] = []
    neg_caps: List[str] = []
    for item in cap_top_list:
        name = item.get('capability_name')
        diff = float(item.get('activation_diff', 0.0))
        if not isinstance(name, str):
            continue
        if diff > 0:
            pos_caps.append(name)
        elif diff < 0:
            neg_caps.append(name)
    pos_caps = pos_caps[:top_k_capability]
    neg_caps = neg_caps[:top_k_capability]

    # 3) parcel 连接正负
    parcel_conn = _load_json(p_parcel_conn)
    if not isinstance(parcel_conn, dict) or 'anomalous_connections' not in parcel_conn:
        raise ValueError("anomalous_connections.json 格式错误")
    ac = parcel_conn['anomalous_connections']
    pos_pairs: List[Tuple[int, int]] = []
    neg_pairs: List[Tuple[int, int]] = []
    for key, target in [('pos_connections', pos_pairs), ('neg_connections', neg_pairs)]:
        conns = ac.get(key, [])
        for item in conns[:top_k_pairs]:
            pi = item.get('parcel_i', {}).get('id')
            pj = item.get('parcel_j', {}).get('id')
            if pi is None or pj is None:
                continue
            target.append((int(pi), int(pj)))

    # 4) capability 连接正负（名称对）
    cap_conn = _load_json(p_cap_conn)
    if not isinstance(cap_conn, dict) or 'anomalous_connections' not in cap_conn:
        raise ValueError("anomalous_capability_connections.json 格式错误")
    ac2 = cap_conn['anomalous_connections']
    pos_cap_pairs: List[Tuple[str, str]] = []
    neg_cap_pairs: List[Tuple[str, str]] = []
    for key, target in [('pos_connections', pos_cap_pairs), ('neg_connections', neg_cap_pairs)]:
        conns = ac2.get(key, [])
        for item in conns[:top_k_pairs]:
            ni = item.get('capability_i', {}).get('name')
            nj = item.get('capability_j', {}).get('name')
            if isinstance(ni, str) and isinstance(nj, str):
                target.append((ni, nj))

    return AutoIndicator(
        pos_parcels=pos_parcels,
        neg_parcels=neg_parcels,
        pos_capabilities=pos_caps,
        neg_capabilities=neg_caps,
        pos_parcel_pairs=pos_pairs,
        neg_parcel_pairs=neg_pairs,
        pos_capability_pairs=pos_cap_pairs,
        neg_capability_pairs=neg_cap_pairs,
    )


