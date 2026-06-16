#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 parcel-level 分析结果中提取 top 异常 parcel 的工具脚本。

参考 jailbreak 的 extract_parcel_ids.py：
- 选取 top_k 个 activation_diff < 0 的 parcel（需要增强）
- 选取 top_k 个 activation_diff > 0 的 parcel（需要抑制）
- 若数量不足或无 activation_diff 则回退到按 rank/顺序取前 top_k*2，再不足则用默认列表

默认用于读取 sycophancy 的
  results/analysis_output/*_gemma-2-9b-it/parcel_level/top_anomalous_parcels.json
"""

import json
import sys
from typing import Any, List, Optional


DEFAULT_PARCELS = [233, 89, 156]


def _normalize_parcels(data: Any) -> List[dict]:
    """兼容 dict(top_parcels=...) 或直接 list。"""
    if isinstance(data, dict) and "top_parcels" in data:
        arr = data["top_parcels"]
    elif isinstance(data, list):
        arr = data
    else:
        return []
    return arr if isinstance(arr, list) else []


def _to_parcel_id(item: Any) -> Optional[int]:
    if isinstance(item, dict) and item.get("parcel_id") is not None:
        return int(item["parcel_id"])
    if isinstance(item, int):
        return item
    return None


def extract_parcel_ids(parcel_json_path: str, top_k: int = 3) -> List[int]:
    try:
        with open(parcel_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 parcel 文件 {parcel_json_path}: {e}", file=sys.stderr)
        return DEFAULT_PARCELS

    raw = _normalize_parcels(data)
    if not raw:
        print("[WARNING] 未找到 parcel 列表，使用默认", file=sys.stderr)
        return DEFAULT_PARCELS

    # 有 activation_diff 时与 jailbreak 一致：负/正各取 top_k
    has_diff = isinstance(raw[0], dict) and "activation_diff" in raw[0]
    if has_diff:
        negative = [p for p in raw if p.get("activation_diff", 0.0) < 0]
        positive = [p for p in raw if p.get("activation_diff", 0.0) > 0]
        negative = sorted(negative, key=lambda x: x.get("activation_diff", 0.0))[:top_k]
        positive = sorted(
            positive, key=lambda x: x.get("activation_diff", 0.0), reverse=True
        )[:top_k]
        parcel_ids = [
            p.get("parcel_id")
            for p in negative + positive
            if p.get("parcel_id") is not None
        ]
    else:
        # 无 activation_diff：按顺序取前 top_k*2 个有效 parcel_id
        parcel_ids = []
        for item in raw:
            pid = _to_parcel_id(item)
            if pid is not None:
                parcel_ids.append(pid)
            if len(parcel_ids) >= top_k * 2:
                break
        parcel_ids = parcel_ids[: top_k * 2]

    if not parcel_ids:
        return DEFAULT_PARCELS
    return parcel_ids


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "[ERROR] 用法: extract_parcel_ids.py <top_anomalous_parcels.json> [--top_k N]",
            file=sys.stderr,
        )
        sys.exit(1)

    parcel_json_path = sys.argv[1]
    top_k = 5
    if len(sys.argv) >= 4 and sys.argv[2] == "--top_k":
        try:
            top_k = int(sys.argv[3])
        except ValueError:
            print(f"[WARNING] 无效 --top_k，使用默认 3", file=sys.stderr)

    parcel_ids = extract_parcel_ids(parcel_json_path, top_k=top_k)
    print(" ".join(str(pid) for pid in parcel_ids))


if __name__ == "__main__":
    main()
