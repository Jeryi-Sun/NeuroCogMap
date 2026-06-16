#!/usr/bin/env python3
"""
从 metrics.pkl 文件中筛选出与 language parcels 相关的数据。

功能：
1. 读取 language parcels 索引（从 language_parcel_overlap_and_accuracy.json）
2. 从指定的 metrics.pkl 文件中筛选出这些 parcels 的数据
3. 保存筛选后的结果到新的 pickle 文件

支持单文件处理和批量处理模式。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


PROJECT_ROOT = Path(
    "/path/to/project_root/Human_LLM_align/litcoder_core"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 metrics.pkl 中筛选 language parcels 相关数据"
    )
    parser.add_argument(
        "--focus_parcels_json",
        type=Path,
        required=True,
        help="包含筛选 parcel 索引的 JSON 文件（如 language_parcel_overlap_and_accuracy.json）",
    )
    parser.add_argument(
        "--focus_parcels_key",
        type=str,
        default="selected_parcel_indices",
        help="focus_parcels_json 中用于读取 parcel 索引列表的键名（默认 selected_parcel_indices）",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=None,
        help="单个 metrics.pkl 文件路径（单文件模式）",
    )
    parser.add_argument(
        "--batch_dir",
        type=Path,
        default=None,
        help="批量处理目录，扫描所有 run_* 子目录中的 metrics.pkl（批量模式）",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="输出文件路径（仅单文件模式使用，批量模式自动生成）",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="若输出文件已存在则跳过",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在输出文件",
    )
    return parser.parse_args()


def load_focus_parcels(focus_json: Path, key: str) -> List[int]:
    """从 JSON 文件中加载要筛选的 parcel 索引"""
    if not focus_json.exists():
        raise FileNotFoundError(f"[ERROR] 找不到 focus parcels JSON: {focus_json}")
    
    with open(focus_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if key not in data:
        raise KeyError(f"[ERROR] JSON 中缺少键 '{key}'，可用键: {list(data.keys())}")
    
    parcel_indices = data[key]
    if not isinstance(parcel_indices, list):
        raise TypeError(f"[ERROR] '{key}' 应为列表，实际类型: {type(parcel_indices)}")
    
    parcel_set = set(int(idx) for idx in parcel_indices)
    return sorted(parcel_set)


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """加载 metrics.pkl 文件"""
    if not metrics_path.exists():
        raise FileNotFoundError(f"[ERROR] 找不到 metrics.pkl: {metrics_path}")
    
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)
    
    if not isinstance(metrics, dict):
        raise TypeError(f"[ERROR] metrics.pkl 期望为 dict，实际类型: {type(metrics)}")
    
    return metrics


def filter_metrics_by_parcels(
    metrics: Dict[str, Any],
    parcel_indices: List[int],
) -> Dict[str, Any]:
    """
    从 metrics 字典中筛选出指定 parcel 索引的数据。
    
    假设 metrics 中的数组字段（如 correlations, p_values 等）按 human_parcel_idx 顺序排列。
    """
    parcel_set = set(parcel_indices)
    n_total = None
    
    # 确定总 parcel 数量（从第一个数组字段推断）
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim == 1:
                if n_total is None:
                    n_total = len(arr)
                elif len(arr) != n_total:
                    print(f"[WARN] 字段 '{key}' 长度 ({len(arr)}) 与预期 ({n_total}) 不一致，跳过")
                    continue
    
    if n_total is None:
        raise ValueError("[ERROR] metrics.pkl 中未找到任何数组字段")
    
    # 检查 parcel 索引是否有效
    max_idx = max(parcel_indices) if parcel_indices else -1
    if max_idx >= n_total:
        raise ValueError(
            f"[ERROR] parcel 索引 {max_idx} 超出范围 [0, {n_total-1}]"
        )
    
    # 筛选数据
    filtered_metrics = {}
    # 初始化 meta 信息
    filtered_metrics["meta"] = {
        "original_n_parcels": n_total,
        "filtered_n_parcels": len(parcel_indices),
        "selected_parcel_indices": parcel_indices,
    }
    
    # 如果原始 metrics 中有 meta 信息，保留它
    if "meta" in metrics and isinstance(metrics["meta"], dict):
        filtered_metrics["meta"]["original_meta"] = metrics["meta"]
    
    for key, value in metrics.items():
        if key == "meta":
            # meta 已经在上面处理过了
            continue
        
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim == 1 and len(arr) == n_total:
                # 一维数组，按索引筛选
                filtered_metrics[key] = arr[parcel_indices].tolist()
            elif arr.ndim == 2 and arr.shape[0] == n_total:
                # 二维数组，第一维是 parcel
                filtered_metrics[key] = arr[parcel_indices, :].tolist()
            elif arr.ndim == 2 and arr.shape[1] == n_total:
                # 二维数组，第二维是 parcel
                filtered_metrics[key] = arr[:, parcel_indices].tolist()
            else:
                # 其他形状，保留原样
                filtered_metrics[key] = value
        else:
            # 非数组字段，保留原样
            filtered_metrics[key] = value
    
    return filtered_metrics


def convert_to_json_serializable(obj: Any) -> Any:
    """
    将包含 numpy 数组的对象转换为 JSON 可序列化的格式。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def process_single_metrics(
    metrics_path: Path,
    parcel_indices: List[int],
    output_path: Path,
    skip_existing: bool = False,
    overwrite: bool = False,
) -> bool:
    """
    处理单个 metrics.pkl 文件。
    
    Returns:
        True 如果成功处理，False 如果跳过
    """
    print(f"[DEBUG] 正在处理: {metrics_path}")
    print(f"[DEBUG] 输出路径: {output_path}")
    print(f"[DEBUG] skip_existing={skip_existing}, overwrite={overwrite}")
    
    # 根据输出路径的扩展名确定 pickle 和 JSON 文件路径
    if output_path.suffix == ".json":
        # 如果输出路径是 .json，直接使用它作为 JSON 路径，pickle 保存为 .pkl
        json_output_path = output_path
        pkl_output_path = output_path.with_suffix(".pkl")
    elif output_path.suffix == ".pkl":
        # 如果输出路径是 .pkl，直接使用它作为 pickle 路径，JSON 保存为 .json
        pkl_output_path = output_path
        json_output_path = output_path.with_suffix(".json")
    else:
        # 如果扩展名不是 .pkl 或 .json，默认保存为 .pkl，JSON 为 .json
        pkl_output_path = output_path.with_suffix(".pkl")
        json_output_path = output_path.with_suffix(".json")
    
    # 检查是否跳过（检查 pickle 和 JSON 文件）
    if skip_existing and pkl_output_path.exists() and json_output_path.exists() and not overwrite:
        print(f"[INFO] 检测到已存在结果 {pkl_output_path} 和 {json_output_path}，跳过。")
        print(f"[WARN] 如果结果不同，请使用 --overwrite 或删除已存在的输出文件")
        return False
    
    try:
        metrics = load_metrics(metrics_path)
        print(f"[DEBUG] 成功加载 metrics.pkl，包含字段: {list(metrics.keys())}")
        
        filtered_metrics = filter_metrics_by_parcels(metrics, parcel_indices)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存 pickle 格式
        with open(pkl_output_path, "wb") as f:
            pickle.dump(filtered_metrics, f)
        print(f"[INFO] 筛选后的 metrics (pickle) 已保存至 {pkl_output_path}")
        
        # 保存 JSON 格式
        json_serializable_metrics = convert_to_json_serializable(filtered_metrics)
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(json_serializable_metrics, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 筛选后的 metrics (JSON) 已保存至 {json_output_path}")
        
        print(
            f"[INFO] 从 {filtered_metrics['meta']['original_n_parcels']} 个 parcels 筛选到 {len(parcel_indices)} 个。"
        )
        
        # 显示一些关键信息用于验证
        if "correlations" in filtered_metrics:
            corr = np.asarray(filtered_metrics["correlations"])
            print(f"[DEBUG] 筛选后的 correlations: 数量={len(corr)}, 均值={np.mean(corr):.6f}, 范围=[{np.min(corr):.6f}, {np.max(corr):.6f}]")
        
        return True
    except Exception as e:
        print(f"[ERROR] 处理 {metrics_path} 时出错: {e}", file=sys.stderr)
        raise


def main() -> None:
    args = parse_args()
    
    # 加载要筛选的 parcel 索引
    parcel_indices = load_focus_parcels(args.focus_parcels_json, args.focus_parcels_key)
    print(f"[INFO] 从 {args.focus_parcels_json} 加载了 {len(parcel_indices)} 个 parcel 索引")
    
    # 批量处理模式
    if args.batch_dir is not None:
        if not args.batch_dir.exists():
            raise FileNotFoundError(f"[ERROR] 批量处理目录不存在: {args.batch_dir}")
        if not args.batch_dir.is_dir():
            raise ValueError(f"[ERROR] 指定的路径不是目录: {args.batch_dir}")
        
        # 扫描所有 run_* 子目录中的 metrics.pkl
        metrics_files = list(args.batch_dir.glob("run_*/metrics.pkl"))
        if not metrics_files:
            print(f"[WARN] 在 {args.batch_dir} 下未找到任何 metrics.pkl 文件")
            return
        
        print(f"[INFO] 找到 {len(metrics_files)} 个 metrics.pkl 文件，开始批量处理...")
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for metrics_path in sorted(metrics_files):
            # 批量模式下，输出文件名保持为 metrics_language_parcels.pkl
            # process_single_metrics 会自动生成对应的 .json 文件
            output_path = metrics_path.parent / "metrics_language_parcels.pkl"
            try:
                result = process_single_metrics(
                    metrics_path,
                    parcel_indices,
                    output_path,
                    args.skip_existing,
                    args.overwrite,
                )
                if result:
                    success_count += 1
                else:
                    skip_count += 1
            except Exception as e:
                error_count += 1
                print(f"[ERROR] 处理 {metrics_path} 失败: {e}", file=sys.stderr)
        
        print(f"\n[INFO] 批量处理完成: 成功 {success_count}，跳过 {skip_count}，失败 {error_count}")
        if error_count > 0:
            sys.exit(1)
        return
    
    # 单文件处理模式
    if args.metrics_path is None:
        raise ValueError("[ERROR] 单文件模式需要指定 --metrics_path")
    
    if args.output_path is None:
        # 默认输出为 .pkl，process_single_metrics 会自动生成对应的 .json
        args.output_path = args.metrics_path.parent / "metrics_language_parcels.pkl"
    
    process_single_metrics(
        args.metrics_path,
        parcel_indices,
        args.output_path,
        args.skip_existing,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
