#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchy-Level（A/B/C/D 四层）能力激活差异聚合脚本

使用说明：
1) 需要先运行 `analysis_capability_level.py`，确保生成 capability-level 的分析结果：
   <model_output_dir>/capability_level/capability_level_analysis_complete.json
2) 本脚本读取其中的 `activation_analysis.activation_diff`（按 capability 顺序的列表），
   并通过 `capability_cog_mapping_flat.json` 将每个 capability 映射到 A/B/C/D 四层，
   输出一个 JSON： {"A":[...], "B":[...], "C":[...], "D":[...]}
   其中列表元素即对应能力的 activation_diff（激活差异）。

注意：
- capability 名称在不同文件中可能大小写不一致，本脚本会做“归一化匹配”；
  如果仍无法匹配，会直接报错并打印缺失项（不会悄悄填 0 或平均值）。
- 可选用 `final_capability_parcel_all.json` 做顺序一致性校验（不通过会报错）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize_capability_name(name: str) -> str:
    """
    将 capability 名称归一化到“只保留字母数字、小写”的形式，便于跨文件匹配。
    例：'Induction And Inference Capability' -> 'inductionandinferencecapability'
    """
    if not isinstance(name, str):
        raise TypeError(f"capability name 必须是 str，但得到: {type(name)}")
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_activation_diff_and_names(analysis_json: dict) -> Tuple[List[str], List[float]]:
    if "activation_analysis" not in analysis_json:
        raise KeyError("capability analysis JSON 缺少字段: activation_analysis")
    activation_analysis = analysis_json["activation_analysis"]

    if "activation_diff" not in activation_analysis:
        raise KeyError("capability analysis JSON 缺少字段: activation_analysis.activation_diff")
    activation_diff = activation_analysis["activation_diff"]
    if not isinstance(activation_diff, list):
        raise TypeError("activation_analysis.activation_diff 不是 list")

    if "capability_names" not in activation_analysis:
        raise KeyError("capability analysis JSON 缺少字段: activation_analysis.capability_names")
    capability_names = activation_analysis["capability_names"]
    if not isinstance(capability_names, list):
        raise TypeError("activation_analysis.capability_names 不是 list")

    if len(capability_names) != len(activation_diff):
        raise ValueError(
            f"capability_names 与 activation_diff 长度不一致: "
            f"{len(capability_names)} vs {len(activation_diff)}"
        )

    # activation_diff 可能是 int/float，统一转 float
    try:
        activation_diff_f = [float(x) for x in activation_diff]
    except Exception as e:
        print(f"[ERROR] activation_diff 无法转换为 float: {e}", file=sys.stderr)
        raise

    capability_names_s = []
    for x in capability_names:
        if not isinstance(x, str):
            raise TypeError(f"capability_names 中存在非 str 元素: {type(x)} -> {x}")
        capability_names_s.append(x)

    return capability_names_s, activation_diff_f


def _build_capability_to_layer_map(mapping_flat_json: dict) -> Dict[str, str]:
    """
    从 capability_cog_mapping_flat.json 构造:
      normalized_capability_name -> layer_id ('A'|'B'|'C'|'D')
    """
    if not isinstance(mapping_flat_json, dict):
        raise TypeError("capability_cog_mapping_flat.json 顶层必须是 dict")

    norm_to_layer: Dict[str, str] = {}
    norm_collisions: Dict[str, List[str]] = {}

    for cap_name, meta in mapping_flat_json.items():
        norm = _normalize_capability_name(cap_name)
        if not isinstance(meta, dict):
            raise TypeError(f"mapping_flat[{cap_name}] 不是 dict")
        if "category" not in meta:
            raise KeyError(f"mapping_flat[{cap_name}] 缺少字段: category")

        category = meta["category"]
        if not isinstance(category, str) or len(category) == 0:
            raise ValueError(f"mapping_flat[{cap_name}].category 非法: {category}")

        layer_id = category[0].upper()
        if layer_id not in {"A", "B", "C", "D"}:
            raise ValueError(f"mapping_flat[{cap_name}].category 首字母不是 A/B/C/D: {category}")

        if norm in norm_to_layer and norm_to_layer[norm] != layer_id:
            norm_collisions.setdefault(norm, []).append(cap_name)
        else:
            norm_to_layer[norm] = layer_id

    if norm_collisions:
        # 这类问题会导致匹配歧义，直接报错最安全
        collision_preview = {k: v[:5] for k, v in list(norm_collisions.items())[:10]}
        raise ValueError(f"capability 名称归一化后发生冲突，无法唯一映射: {collision_preview}")

    return norm_to_layer


def _validate_order_with_mapping_json(
    capability_names: List[str],
    mapping_json_path: Path,
) -> None:
    """
    用 final_capability_parcel_all.json 的 key 顺序，校验 capability_names 顺序一致。
    """
    mapping_json = _load_json(mapping_json_path)
    if not isinstance(mapping_json, dict):
        raise TypeError(f"{mapping_json_path} 顶层必须是 dict（capability->...）")
    mapping_keys = list(mapping_json.keys())
    if len(mapping_keys) != len(capability_names):
        raise ValueError(
            f"顺序校验失败：{mapping_json_path} 的 key 数量与 capability_names 长度不一致: "
            f"{len(mapping_keys)} vs {len(capability_names)}"
        )

    # 使用“精确字符串匹配”做严格校验（你已说明按顺序一一对应）
    mismatches = []
    for i, (a, b) in enumerate(zip(mapping_keys, capability_names)):
        if a != b:
            mismatches.append((i, a, b))
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise ValueError(
            "顺序校验失败：final_capability_parcel_all.json keys 与 capability_names 不一致。"
            f"前若干不一致示例: {mismatches}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 capability-level 的 activation_diff 聚合到 A/B/C/D 四个认知层级，并输出 JSON（含层级均值与排序信息）。"
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default="/path/to/project_root/safety_explanation/hallucination/results/analysis_output/dolly_close_gemma-2-2b",
        help="模型结果输出目录（包含 capability_level 子目录）。例如: .../analysis_output/dolly_close_gemma-2-2b",
    )
    parser.add_argument(
        "--capability-analysis-json",
        type=str,
        default=None,
        help="capability level 完整分析 JSON 路径。默认自动使用 <model-output-dir>/capability_level/capability_level_analysis_complete.json",
    )
    parser.add_argument(
        "--capability-parcel-mapping-json",
        type=str,
        default="/path/to/project_root/neural_area/connect_cap_parcel/results/aggrate_final/final_capability_parcel_all.json",
        help="final_capability_parcel_all.json，用于校验 capability 顺序是否一致（可关闭校验）。",
    )
    parser.add_argument(
        "--disable-order-validation",
        action="store_true",
        help="关闭用 final_capability_parcel_all.json 校验 capability 顺序的一致性。",
    )
    parser.add_argument(
        "--capability-cog-mapping-flat-json",
        type=str,
        default="/path/to/project_root/capability_analysis/data/capability_cog_mapping_flat.json",
        help="capability->A/B/C/D 的映射文件（flat）。",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="输出 JSON 路径。默认: <model-output-dir>/hierarchy_level/hierarchy_level_activation_diff.json",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 output-json 已存在则跳过（不覆盖）。",
    )
    parser.add_argument(
        "--use-abs-for-mean",
        action="store_true",
        help="在计算每个层级的平均值之前先取绝对值（只影响统计与排序，不改变原始列表）。",
    )
    args = parser.parse_args()

    model_output_dir = Path(args.model_output_dir)
    capability_analysis_json_path = (
        Path(args.capability_analysis_json)
        if args.capability_analysis_json is not None
        else model_output_dir / "capability_level" / "capability_level_analysis_complete.json"
    )

    output_json_path = (
        Path(args.output_json)
        if args.output_json is not None
        else model_output_dir / "hierarchy_level" / "hierarchy_level_activation_diff.json"
    )

    if args.skip_existing and output_json_path.exists():
        print(f"[SKIP] 输出已存在，跳过生成: {output_json_path}")
        return

    analysis_json = _load_json(capability_analysis_json_path)
    if not isinstance(analysis_json, dict):
        raise TypeError(f"{capability_analysis_json_path} 顶层必须是 dict")

    capability_names, activation_diff = _extract_activation_diff_and_names(analysis_json)

    if not args.disable_order_validation:
        _validate_order_with_mapping_json(
            capability_names=capability_names,
            mapping_json_path=Path(args.capability_parcel_mapping_json),
        )

    mapping_flat_json = _load_json(Path(args.capability_cog_mapping_flat_json))
    if not isinstance(mapping_flat_json, dict):
        raise TypeError(f"{args.capability_cog_mapping_flat_json} 顶层必须是 dict")

    norm_to_layer = _build_capability_to_layer_map(mapping_flat_json)

    out: Dict[str, List[float]] = {"A": [], "B": [], "C": [], "D": []}
    missing: List[str] = []

    for cap_name, diff in zip(capability_names, activation_diff):
        norm = _normalize_capability_name(cap_name)
        layer_id = norm_to_layer.get(norm)
        if layer_id is None:
            missing.append(cap_name)
            continue
        out[layer_id].append(diff)

    if missing:
        missing_preview = missing[:30]
        print(
            "[ERROR] 以下 capability 无法在 capability_cog_mapping_flat.json 中找到 A/B/C/D 映射（归一化后仍缺失）。",
            file=sys.stderr,
        )
        for x in missing_preview:
            print(f"  - {x}", file=sys.stderr)
        if len(missing) > len(missing_preview):
            print(f"  ... 以及另外 {len(missing) - len(missing_preview)} 个", file=sys.stderr)
        raise ValueError("capability->层级映射缺失，已停止。请补全 mapping_flat 或修正命名。")

    # 统计信息：每层平均值与排序（可选是否先取绝对值）
    layer_means: Dict[str, float] = {}
    for layer_id, values in out.items():
        if len(values) == 0:
            layer_means[layer_id] = float("nan")
            continue
        if args.use_abs_for_mean:
            vals = [abs(v) for v in values]
        else:
            vals = values
        layer_means[layer_id] = sum(vals) / float(len(vals))

    # 按平均值从小到大排序层级（忽略 nan）
    sorted_layers = sorted(
        [k for k, v in layer_means.items() if not (v != v)],
        key=lambda k: layer_means[k],
        reverse=False,
    )

    output_data = {
        "A": out["A"],
        "B": out["B"],
        "C": out["C"],
        "D": out["D"],
        "layer_mean": layer_means,
        "layer_sorted_by_mean": sorted_layers,
        "mean_type": "abs" if args.use_abs_for_mean else "signed",
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    counts = {k: len(v) for k, v in out.items()}
    print(f"[OK] 已写出: {output_json_path}")
    print(f"[INFO] 每层数量: {counts}")
    print(f"[INFO] 层级平均值({output_data['mean_type']}): {layer_means}")
    print(f"[INFO] 排序(从小到大): {sorted_layers}")


if __name__ == "__main__":
    main()

