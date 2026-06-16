#!/usr/bin/env python3
"""
对 lebel 数据集的 uts02 和 uts03 进行平均计算

功能：
1. 从 summary_results.csv 读取 lebel_uts02 和 lebel_uts03 的数据
2. 对每个 method，计算两个 uts 的平均值
3. 对 parcel 相关的值也进行平均
4. 输出平均后的结果到新的 CSV 文件
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def parse_semicolon_list(value: str) -> List[float]:
    """解析分号分隔的字符串列表为浮点数列表"""
    if not value or value == "":
        return []
    try:
        return [float(x.strip()) for x in value.split(";") if x.strip()]
    except Exception:
        return []


def format_semicolon_list(values: List[float], format_str: str = "{:.6f}") -> str:
    """将浮点数列表格式化为分号分隔的字符串"""
    if not values:
        return ""
    return ";".join(format_str.format(v) for v in values)


def average_two_results(result1: Dict[str, Any], result2: Dict[str, Any]) -> Dict[str, Any]:
    """对两个结果进行平均"""
    averaged = {
        "data_source": "lebel_dataset",
        "method": result1.get("method", ""),
        "status": "success" if result1.get("status") == "success" and result2.get("status") == "success" else "partial",
    }
    
    # 数值字段：直接平均
    numeric_fields = [
        "n_parcels", "corr_mean", "corr_median", "corr_max", "corr_min", "corr_std",
        "p_mean", "p_significant", "p_significant_percent",
        "cp_significant", "cp_significant_percent"
    ]
    
    for field in numeric_fields:
        val1 = result1.get(field, 0)
        val2 = result2.get(field, 0)
        try:
            val1 = float(val1) if val1 != "" else 0
            val2 = float(val2) if val2 != "" else 0
            averaged[field] = (val1 + val2) / 2.0
        except (ValueError, TypeError):
            averaged[field] = ""
    
    # Parcel indices: 应该相同，取第一个
    parcel_indices1 = result1.get("parcel_indices", "")
    parcel_indices2 = result2.get("parcel_indices", "")
    if parcel_indices1 == parcel_indices2:
        averaged["parcel_indices"] = parcel_indices1
    else:
        # 如果不相同，取交集或第一个
        averaged["parcel_indices"] = parcel_indices1
    
    # Parcel correlations: 逐元素平均
    corr1 = parse_semicolon_list(result1.get("parcel_correlations", ""))
    corr2 = parse_semicolon_list(result2.get("parcel_correlations", ""))
    if len(corr1) == len(corr2) and len(corr1) > 0:
        avg_corr = [(c1 + c2) / 2.0 for c1, c2 in zip(corr1, corr2)]
        averaged["parcel_correlations"] = format_semicolon_list(avg_corr)
    else:
        averaged["parcel_correlations"] = result1.get("parcel_correlations", "")
    
    # Parcel p_values: 逐元素平均（或取几何平均？这里用算术平均）
    p1 = parse_semicolon_list(result1.get("parcel_p_values", ""))
    p2 = parse_semicolon_list(result2.get("parcel_p_values", ""))
    if len(p1) == len(p2) and len(p1) > 0:
        # 对于 p 值，可以使用几何平均（更合理）或算术平均
        # 这里使用算术平均
        avg_p = [(p1_val + p2_val) / 2.0 for p1_val, p2_val in zip(p1, p2)]
        averaged["parcel_p_values"] = format_semicolon_list(avg_p, "{:.6e}" if any(p < 0.001 for p in avg_p) else "{:.6f}")
    else:
        averaged["parcel_p_values"] = result1.get("parcel_p_values", "")
    
    # Parcel corrected_p_values: 逐元素平均
    cp1 = parse_semicolon_list(result1.get("parcel_corrected_p_values", ""))
    cp2 = parse_semicolon_list(result2.get("parcel_corrected_p_values", ""))
    if len(cp1) == len(cp2) and len(cp1) > 0:
        avg_cp = [(cp1_val + cp2_val) / 2.0 for cp1_val, cp2_val in zip(cp1, cp2)]
        averaged["parcel_corrected_p_values"] = format_semicolon_list(avg_cp, "{:.6e}" if any(cp < 0.001 for cp in avg_cp) else "{:.6f}")
    else:
        averaged["parcel_corrected_p_values"] = result1.get("parcel_corrected_p_values", "")
    
    # 其他字段：保留第一个或合并
    other_fields = ["file_size", "reason", "metrics_path", "output_dir", "output_file"]
    for field in other_fields:
        if field in result1:
            averaged[field] = result1.get(field, "")
        elif field in result2:
            averaged[field] = result2.get(field, "")
    
    return averaged


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    input_csv = data_dir / "summary_results.csv"
    output_csv = data_dir / "summary_results_lebel_avg.csv"
    
    if not input_csv.exists():
        print(f"[ERROR] 输入文件不存在: {input_csv}")
        return
    
    # 读取 CSV 文件
    results = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    
    print(f"[INFO] 从 {input_csv} 读取了 {len(results)} 条记录")
    
    # 筛选出 lebel_uts02 和 lebel_uts03 的数据
    lebel_uts02 = {r["method"]: r for r in results if r.get("data_source") == "lebel_uts02"}
    lebel_uts03 = {r["method"]: r for r in results if r.get("data_source") == "lebel_uts03"}
    
    print(f"[INFO] lebel_uts02: {len(lebel_uts02)} 个方法")
    print(f"[INFO] lebel_uts03: {len(lebel_uts03)} 个方法")
    
    if lebel_uts02:
        print(f"  - lebel_uts02 方法: {sorted(lebel_uts02.keys())}")
    if lebel_uts03:
        print(f"  - lebel_uts03 方法: {sorted(lebel_uts03.keys())}")
    
    # 找出共同的方法
    common_methods = set(lebel_uts02.keys()) & set(lebel_uts03.keys())
    print(f"\n[INFO] 共同方法: {sorted(common_methods)}")
    
    if not common_methods:
        print("[WARN] 没有找到共同的方法")
        return
    
    # 对每个方法进行平均
    averaged_results = []
    for method in sorted(common_methods):
        result1 = lebel_uts02[method]
        result2 = lebel_uts03[method]
        averaged = average_two_results(result1, result2)
        averaged_results.append(averaged)
        print(f"[INFO] 已平均: {method}")
    
    # 确定字段顺序
    if averaged_results:
        fieldnames = list(averaged_results[0].keys())
        # 重新排序，将重要字段放在前面
        priority_fields = [
            "data_source", "method", "n_parcels",
            "corr_mean", "corr_median", "corr_max", "corr_min", "corr_std",
            "p_mean", "p_significant", "p_significant_percent",
            "cp_significant", "cp_significant_percent",
            "parcel_indices", "parcel_correlations", "parcel_p_values", "parcel_corrected_p_values",
            "file_size", "status", "reason"
        ]
        
        ordered_fieldnames = []
        for field in priority_fields:
            if field in fieldnames:
                ordered_fieldnames.append(field)
        
        # 添加其他字段
        for field in fieldnames:
            if field not in ordered_fieldnames:
                ordered_fieldnames.append(field)
        
        fieldnames = ordered_fieldnames
    
    # 保存到 CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(averaged_results)
    
    print(f"\n[INFO] 平均结果已保存到: {output_csv}")
    print(f"[INFO] 共 {len(averaged_results)} 个方法的结果")
    
    # 显示摘要
    print("\n" + "="*100)
    print("平均结果摘要")
    print("="*100)
    print(f"{'方法':<20} {'Parcels':<10} {'Corr均值':<12} {'Corr中位':<12} {'显著数':<10} {'显著%':<10}")
    print("-"*100)
    for r in averaged_results:
        method = r.get("method", "")
        n_parcels = r.get("n_parcels", 0)
        corr_mean = r.get("corr_mean", 0)
        corr_median = r.get("corr_median", 0)
        p_sig = r.get("p_significant", 0)
        p_sig_pct = r.get("p_significant_percent", 0)
        print(f"{method:<20} {n_parcels:<10.0f} {corr_mean:<12.4f} {corr_median:<12.4f} {p_sig:<10.0f} {p_sig_pct:<10.1f}")
    print("="*100)


if __name__ == "__main__":
    main()
