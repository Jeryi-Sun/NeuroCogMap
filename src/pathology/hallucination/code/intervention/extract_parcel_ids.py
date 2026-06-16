#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel ID提取工具

从异常分析文件中提取top3负向激活和top3正向激活的parcel ID
"""

import json
import sys
import os

def extract_top_parcels(json_file):
    """
    从异常分析文件中提取top3负向和top3正向激活的parcel ID
    
    Args:
        json_file: top_anomalous_parcels.json文件路径
        
    Returns:
        str: 空格分隔的parcel ID字符串
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            parcels = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}", file=sys.stderr)
        return ""
    
    # 按activation_diff排序
    negative_parcels = [p for p in parcels if p.get('activation_diff', 0) < 0 and p.get('is_significant', False)]
    positive_parcels = [p for p in parcels if p.get('activation_diff', 0) > 0 and p.get('is_significant', False)]
    
    # 排序：负向按activation_diff升序（最负的在前），正向按activation_diff降序（最正的在前）
    negative_parcels.sort(key=lambda x: x.get('activation_diff', 0))
    positive_parcels.sort(key=lambda x: x.get('activation_diff', 0), reverse=True)
    
    # 取top3
    top_negative = negative_parcels[:3]
    top_positive = positive_parcels[:3]
    
    # 合并并提取ID
    all_parcels = top_negative + top_positive
    parcel_ids = [str(p['parcel_id']) for p in all_parcels]
    
    # 打印详细信息到stderr
    print('Top3负向激活parcels:', file=sys.stderr)
    for p in top_negative:
        print(f'  Parcel {p["parcel_id"]}: activation_diff={p.get("activation_diff", 0):.3f}, function={p.get("function_name", "Unknown")}', file=sys.stderr)
    
    print('Top3正向激活parcels:', file=sys.stderr)
    for p in top_positive:
        print(f'  Parcel {p["parcel_id"]}: activation_diff={p.get("activation_diff", 0):.3f}, function={p.get("function_name", "Unknown")}', file=sys.stderr)
    
    return ' '.join(parcel_ids)

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python extract_parcel_ids.py <json_file>", file=sys.stderr)
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: 文件不存在 {json_file}", file=sys.stderr)
        sys.exit(1)
    
    parcel_ids = extract_top_parcels(json_file)
    
    if parcel_ids:
        print(parcel_ids)
    else:
        print("Error: 无法提取parcel IDs", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
