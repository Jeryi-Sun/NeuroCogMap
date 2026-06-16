#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 parcel-level 分析结果中提取 top 异常 parcel 的工具脚本。

保持与幻觉干预版本一致：
- 选取 top 3 activation_diff < 0 的 parcel（需要增强）
- 选取 top 3 activation_diff > 0 的 parcel（需要抑制）
- 若数量不足则回退到默认 parcel 列表 (233, 89, 156)
"""

import json
import sys
from typing import List


DEFAULT_PARCELS = [233, 89, 156]


def extract_parcel_ids(parcel_json_path: str, top_k: int = 3) -> List[int]:
    try:
        with open(parcel_json_path, 'r', encoding='utf-8') as f:
            parcels = json.load(f)
    except Exception as e:
        print(f"[ERROR] 无法读取 parcel 文件 {parcel_json_path}: {e}", file=sys.stderr)
        return DEFAULT_PARCELS

    negative = [p for p in parcels if p.get("activation_diff", 0.0) < 0]
    positive = [p for p in parcels if p.get("activation_diff", 0.0) > 0]

    negative = sorted(negative, key=lambda x: x.get("activation_diff", 0.0))[:top_k]
    positive = sorted(positive, key=lambda x: x.get("activation_diff", 0.0), reverse=True)[:top_k]

    parcel_ids = [p.get("parcel_id") for p in negative + positive if p.get("parcel_id") is not None]

    if not parcel_ids:
        return DEFAULT_PARCELS

    return parcel_ids


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_parcel_ids.py <top_anomalous_parcels.json>", file=sys.stderr)
        sys.exit(1)

    parcel_json_path = sys.argv[1]
    parcel_ids = extract_parcel_ids(parcel_json_path)

    print(" ".join(str(pid) for pid in parcel_ids))


if __name__ == "__main__":
    main()


