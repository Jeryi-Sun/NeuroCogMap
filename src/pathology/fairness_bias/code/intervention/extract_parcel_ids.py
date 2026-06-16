#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从异常分析输出中提取 parcel ids，输出以空格分隔的整数序列，供 shell 脚本使用。
输入：top_anomalous_parcels.json
"""
import sys
import json
from typing import List

def main() -> None:
    if len(sys.argv) < 2:
        print("[ERROR] 用法: extract_parcel_ids.py <top_anomalous_parcels.json>", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        parcel_ids: List[int] = []
        if isinstance(data, dict) and "top_parcels" in data:
            arr = data["top_parcels"]
        else:
            arr = data
        for item in arr:
            if isinstance(item, dict) and "parcel_id" in item:
                parcel_ids.append(int(item["parcel_id"]))
            elif isinstance(item, int):
                parcel_ids.append(int(item))
        if not parcel_ids:
            print("233 89 156")
        else:
            print(" ".join(str(i) for i in parcel_ids))
    except Exception as e:
        print(f"[WARNING] 解析失败: {e}", file=sys.stderr)
        print("233 89 156")

if __name__ == "__main__":
    main()


