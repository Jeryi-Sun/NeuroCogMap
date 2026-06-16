#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 step1_train_feature_pcc_ranked_by_aic.csv 中的 Parcel 相关性映射到 Capability 与 Hierarchy(A/B/C/D) 层面。

说明：
1) 输入 CSV 的 feature_idx 被视为 parcel_id（例如 249 -> parcel_249）。
2) 遇到空值行会跳过，并在统计 JSON 与日志中记录。
3) 输出三个文件到实验目录：
   - step1_train_feature_pcc_mapped_capability_by_aic.csv
   - step1_train_feature_pcc_mapped_hierarchy_by_aic.csv
   - step1_train_feature_pcc_mapping_stats_by_aic.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def _recompute_abs_metrics(row: Dict[str, object], metric_fields: List[str]) -> int:
    """
    强制保证 abs_* 指标与对应原始指标一致：abs_xxx = abs(xxx)。
    返回修正次数（用于日志统计）。
    """
    fixes = 0
    for field in metric_fields:
        if not field.startswith("abs_"):
            continue
        raw_field = field[4:]
        if raw_field not in row:
            continue

        raw_val = row.get(raw_field)
        abs_val = row.get(field)
        if not isinstance(raw_val, float) or math.isnan(raw_val):
            continue

        recomputed = abs(raw_val)
        if not isinstance(abs_val, float) or math.isnan(abs_val) or abs(abs_val - recomputed) > 1e-8:
            row[field] = recomputed
            fixes += 1
    return fixes


def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"name 必须是 str，但得到: {type(name)}")
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _load_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_empty(v: str | None) -> bool:
    return v is None or str(v).strip() == ""


def _parse_input_csv(
    input_csv: Path,
) -> Tuple[List[Dict[str, float]], List[str], Dict[str, object]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"输入 CSV 不存在: {input_csv}")

    parsed_rows: List[Dict[str, float]] = []
    metric_fields: List[str] = []
    stats = {
        "total_rows": 0,
        "valid_rows": 0,
        "skipped_rows_empty": 0,
        "skipped_rows_non_numeric": 0,
    }

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 表头为空: {input_csv}")
        if "feature_idx" not in reader.fieldnames:
            raise KeyError(f"CSV 缺少必需字段 feature_idx: {input_csv}")

        metric_fields = [x for x in reader.fieldnames if x != "feature_idx"]
        if not metric_fields:
            raise ValueError(f"CSV 中找不到可聚合的指标列: {input_csv}")

        for row in reader:
            stats["total_rows"] += 1
            feature_idx_raw = row.get("feature_idx")

            if _is_empty(feature_idx_raw):
                stats["skipped_rows_empty"] += 1
                continue

            # 约束：按用户要求，行中存在空值时整行跳过
            has_empty = False
            for field in metric_fields:
                if _is_empty(row.get(field)):
                    has_empty = True
                    break
            if has_empty:
                stats["skipped_rows_empty"] += 1
                continue

            try:
                feature_idx = int(str(feature_idx_raw).strip())
                values = {field: float(str(row[field]).strip()) for field in metric_fields}
            except Exception:
                stats["skipped_rows_non_numeric"] += 1
                continue

            one = {"feature_idx": feature_idx}
            one.update(values)
            parsed_rows.append(one)
            stats["valid_rows"] += 1

    if not parsed_rows:
        raise ValueError(f"CSV 解析后没有有效数据行: {input_csv}")

    return parsed_rows, metric_fields, stats


def _build_capability_weights(mapping_json: dict) -> Dict[str, Dict[int, float]]:
    if not isinstance(mapping_json, dict):
        raise TypeError("final_capability_parcel_all.json 顶层必须是 dict")

    cap_to_parcel_weights: Dict[str, Dict[int, float]] = {}

    for cap_name, cap_data in mapping_json.items():
        if not isinstance(cap_data, dict):
            raise TypeError(f"mapping[{cap_name}] 不是 dict")
        ranking = cap_data.get("ranking")
        if not isinstance(ranking, list):
            raise KeyError(f"mapping[{cap_name}] 缺少或非法 ranking 字段")

        parcel_weights: Dict[int, float] = {}
        for item in ranking:
            if not isinstance(item, list) or len(item) != 2:
                continue
            parcel_name, w = item
            if not isinstance(parcel_name, str) or not parcel_name.startswith("parcel_"):
                continue
            try:
                parcel_idx = int(parcel_name.split("_", 1)[1])
                weight = float(w)
            except Exception:
                continue
            # 与 capability-level 分析保持一致：负权重截断为 0
            weight = max(weight, 0.0)
            if weight == 0.0:
                continue
            parcel_weights[parcel_idx] = parcel_weights.get(parcel_idx, 0.0) + weight

        if parcel_weights:
            total = sum(parcel_weights.values())
            if total <= 0:
                continue
            cap_to_parcel_weights[cap_name] = {k: v / total for k, v in parcel_weights.items()}

    if not cap_to_parcel_weights:
        raise ValueError("无法从 final_capability_parcel_all.json 构建有效 capability 权重")
    return cap_to_parcel_weights


def _build_capability_to_hierarchy(mapping_flat_json: dict) -> Dict[str, str]:
    if not isinstance(mapping_flat_json, dict):
        raise TypeError("capability_cog_mapping_flat.json 顶层必须是 dict")

    norm2layer: Dict[str, str] = {}
    for cap_name, meta in mapping_flat_json.items():
        if not isinstance(meta, dict):
            continue
        category = meta.get("category")
        if not isinstance(category, str) or len(category) == 0:
            continue
        layer = category[0].upper()
        if layer in {"A", "B", "C", "D"}:
            norm2layer[_normalize_name(cap_name)] = layer
    return norm2layer


def _aggregate_capability(
    parsed_rows: List[Dict[str, float]],
    metric_fields: List[str],
    cap_to_parcel_weights: Dict[str, Dict[int, float]],
    top_k_max: int,
    top_k_min: int,
    selection_metric_field: str,
    recompute_abs_metrics: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    parcel_metric: Dict[int, Dict[str, float]] = {}
    for row in parsed_rows:
        idx = int(row["feature_idx"])
        parcel_metric[idx] = {m: float(row[m]) for m in metric_fields}

    capability_rows: List[Dict[str, object]] = []
    covered_parcels = set(parcel_metric.keys())
    used_parcels = set()
    abs_metric_fix_count = 0

    if selection_metric_field not in metric_fields:
        raise KeyError(f"CSV 中缺少选择聚合用的列: {selection_metric_field}")

    for cap_name, pweights in cap_to_parcel_weights.items():
        sum_w_total = 0.0
        selected_sum_w = 0.0
        selected_parcels: List[int] = []

        # 只在该 capability 对应且在 CSV 中存在的 parcel 上进行极值选择
        candidates: List[Tuple[int, float]] = []
        metric_sum = {m: 0.0 for m in metric_fields}

        # pweights 在构建时已经做了归一化，但这里仍保留显式求和，便于统计与健壮性
        for pid, w in pweights.items():
            sum_w_total += w
            if pid not in parcel_metric:
                continue
            used_parcels.add(pid)
            candidates.append((pid, float(parcel_metric[pid][selection_metric_field])))

        # 排序后取极值
        # - max: pcc 最大（降序取前 top_k_max）
        # - min: pcc 最小（升序取前 top_k_min）
        # 特殊规则：当 top_k_max == 0 且 top_k_min == 0 时，使用该 capability 下全部候选 parcel
        candidates_sorted_desc = sorted(candidates, key=lambda x: x[1], reverse=True)
        candidates_sorted_asc = list(reversed(candidates_sorted_desc))

        if top_k_max == 0 and top_k_min == 0:
            top_max = []
            top_min = []
            selected_parcels = [pid for pid, _ in candidates]
        else:
            top_max = candidates_sorted_desc[: max(0, top_k_max)]
            top_min = candidates_sorted_asc[: max(0, top_k_min)]
            selected_set = {pid for pid, _ in top_max} | {pid for pid, _ in top_min}
            selected_parcels = list(selected_set)

        mapped_count = len(selected_parcels)

        for pid in selected_parcels:
            w = pweights.get(pid, 0.0)
            selected_sum_w += w
            for m in metric_fields:
                metric_sum[m] += w * parcel_metric[pid][m]

        top_max_selected_count = len({pid for pid, _ in top_max})
        top_min_selected_count = len({pid for pid, _ in top_min})

        row: Dict[str, object] = {
            "capability_name": cap_name,
            "mapped_parcel_count": mapped_count,  # 仅极值 parcel 数
            "mapping_total_parcel_count": len(pweights),
            "weight_coverage_ratio": (selected_sum_w / sum_w_total) if sum_w_total > 0 else float("nan"),
            "selected_top_k_max": int(top_max_selected_count),
            "selected_top_k_min": int(top_min_selected_count),
        }
        for m in metric_fields:
            row[m] = (metric_sum[m] / selected_sum_w) if selected_sum_w > 0 else float("nan")
        if recompute_abs_metrics:
            abs_metric_fix_count += _recompute_abs_metrics(row=row, metric_fields=metric_fields)
        capability_rows.append(row)

    capability_rows.sort(
        key=lambda x: abs(x.get("pcc_with_train_aic", float("-inf")))
        if not (isinstance(x.get("pcc_with_train_aic"), float) and math.isnan(x["pcc_with_train_aic"]))
        else -1.0,
        reverse=True,
    )

    stats = {
        "input_unique_parcel_count": len(covered_parcels),
        "input_parcel_used_by_any_capability": len(used_parcels),
        "input_parcel_unmapped_count": len(covered_parcels - used_parcels),
        "input_parcel_unmapped_preview": sorted(list(covered_parcels - used_parcels))[:30],
        "selection_metric_field": selection_metric_field,
        "top_k_max": int(top_k_max),
        "top_k_min": int(top_k_min),
        "use_all_candidates_when_topk_zero_zero": bool(top_k_max == 0 and top_k_min == 0),
        "abs_metric_fix_count": int(abs_metric_fix_count),
    }
    return capability_rows, stats


def _aggregate_hierarchy(
    capability_rows: List[Dict[str, object]],
    metric_fields: List[str],
    cap_to_layer: Dict[str, str],
    recompute_abs_metrics: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    acc: Dict[str, Dict[str, float]] = {x: {"count": 0.0} for x in ["A", "B", "C", "D"]}
    for layer in ["A", "B", "C", "D"]:
        for m in metric_fields:
            acc[layer][f"sum_{m}"] = 0.0

    missing_caps: List[str] = []
    abs_metric_fix_count = 0

    for row in capability_rows:
        cap_name = str(row["capability_name"])
        norm = _normalize_name(cap_name)
        layer = cap_to_layer.get(norm)
        if layer is None:
            missing_caps.append(cap_name)
            continue

        acc[layer]["count"] += 1.0
        for m in metric_fields:
            v = row.get(m)
            if isinstance(v, float) and not math.isnan(v):
                acc[layer][f"sum_{m}"] += float(v)

    hierarchy_rows: List[Dict[str, object]] = []
    for layer in ["A", "B", "C", "D"]:
        cnt = acc[layer]["count"]
        row: Dict[str, object] = {
            "hierarchy_level": layer,
            "mapped_capability_count": int(cnt),
        }
        for m in metric_fields:
            row[m] = (acc[layer][f"sum_{m}"] / cnt) if cnt > 0 else float("nan")
        if recompute_abs_metrics:
            abs_metric_fix_count += _recompute_abs_metrics(row=row, metric_fields=metric_fields)
        hierarchy_rows.append(row)

    hierarchy_rows.sort(
        key=lambda x: abs(x.get("pcc_with_train_aic", float("-inf")))
        if not (isinstance(x.get("pcc_with_train_aic"), float) and math.isnan(x["pcc_with_train_aic"]))
        else -1.0,
        reverse=True,
    )

    stats = {
        "capability_missing_hierarchy_count": len(missing_caps),
        "capability_missing_hierarchy_preview": missing_caps[:30],
        "abs_metric_fix_count": int(abs_metric_fix_count),
    }
    return hierarchy_rows, stats


def _write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 feature-level(parcel) 相关性聚合到 capability 与 hierarchy(A/B/C/D) 层面。"
    )
    parser.add_argument("--input-csv", type=str, required=True, help="输入 CSV 路径")
    parser.add_argument(
        "--capability-parcel-mapping-json",
        type=str,
        default="/path/to/project_root/neural_area/connect_cap_parcel/results/aggrate_final/final_capability_parcel_all.json",
        help="capability->parcel ranking 映射 JSON",
    )
    parser.add_argument(
        "--capability-cog-mapping-flat-json",
        type=str,
        default="/path/to/project_root/capability_analysis/data/capability_cog_mapping_flat.json",
        help="capability->A/B/C/D 映射 JSON",
    )
    parser.add_argument(
        "--output-capability-csv",
        type=str,
        default=None,
        help="输出 capability CSV 路径，默认写到 input 同目录",
    )
    parser.add_argument(
        "--output-hierarchy-csv",
        type=str,
        default=None,
        help="输出 hierarchy CSV 路径，默认写到 input 同目录",
    )
    parser.add_argument(
        "--output-stats-json",
        type=str,
        default=None,
        help="输出统计 JSON 路径，默认写到 input 同目录",
    )
    parser.add_argument("--skip-existing", action="store_true", help="若输出已存在则跳过")
    parser.add_argument(
        "--top-k-max",
        type=int,
        default=10,
        help="capability 聚合时：选择 pcc_with_train_aic 最大的前 K 个 parcel（仅限该 capability 对应 parcel）",
    )
    parser.add_argument(
        "--top-k-min",
        type=int,
        default=10,
        help="capability 聚合时：选择 pcc_with_train_aic 最小的前 K 个 parcel（仅限该 capability 对应 parcel）",
    )
    parser.add_argument(
        "--disable-abs-recompute",
        action="store_true",
        help="关闭 abs_* 指标重算（默认开启：abs_xxx 将由 xxx 重算，保证一致性）",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = input_csv.parent

    out_capability = (
        Path(args.output_capability_csv)
        if args.output_capability_csv is not None
        else out_dir / "step1_train_feature_pcc_mapped_capability_by_aic.csv"
    )
    out_hierarchy = (
        Path(args.output_hierarchy_csv)
        if args.output_hierarchy_csv is not None
        else out_dir / "step1_train_feature_pcc_mapped_hierarchy_by_aic.csv"
    )
    out_stats = (
        Path(args.output_stats_json)
        if args.output_stats_json is not None
        else out_dir / "step1_train_feature_pcc_mapping_stats_by_aic.json"
    )

    if args.skip_existing and out_capability.exists() and out_hierarchy.exists() and out_stats.exists():
        print(f"[SKIP] 输出已存在，跳过: {input_csv}")
        return

    parsed_rows, metric_fields, csv_stats = _parse_input_csv(input_csv)
    cap_parcel_mapping = _load_json(Path(args.capability_parcel_mapping_json))
    cap_cog_mapping = _load_json(Path(args.capability_cog_mapping_flat_json))

    if not isinstance(cap_parcel_mapping, dict):
        raise TypeError("capability-parcel mapping 顶层必须是 dict")
    if not isinstance(cap_cog_mapping, dict):
        raise TypeError("capability-cog mapping 顶层必须是 dict")

    cap_to_parcel_weights = _build_capability_weights(cap_parcel_mapping)
    cap_to_layer = _build_capability_to_hierarchy(cap_cog_mapping)

    capability_rows, capability_stats = _aggregate_capability(
        parsed_rows=parsed_rows,
        metric_fields=metric_fields,
        cap_to_parcel_weights=cap_to_parcel_weights,
        top_k_max=args.top_k_max,
        top_k_min=args.top_k_min,
        selection_metric_field="pcc_with_train_aic",
        recompute_abs_metrics=(not args.disable_abs_recompute),
    )
    hierarchy_rows, hierarchy_stats = _aggregate_hierarchy(
        capability_rows=capability_rows,
        metric_fields=metric_fields,
        cap_to_layer=cap_to_layer,
        recompute_abs_metrics=(not args.disable_abs_recompute),
    )

    capability_fields = [
        "capability_name",
        "mapped_parcel_count",
        "mapping_total_parcel_count",
        "weight_coverage_ratio",
        "selected_top_k_max",
        "selected_top_k_min",
        *metric_fields,
    ]
    hierarchy_fields = ["hierarchy_level", "mapped_capability_count", *metric_fields]

    _write_csv(out_capability, capability_rows, capability_fields)
    _write_csv(out_hierarchy, hierarchy_rows, hierarchy_fields)
    with out_stats.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_csv": str(input_csv),
                "csv_parsing": csv_stats,
                "capability_aggregation": capability_stats,
                "hierarchy_aggregation": hierarchy_stats,
                "outputs": {
                    "capability_csv": str(out_capability),
                    "hierarchy_csv": str(out_hierarchy),
                    "stats_json": str(out_stats),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] 已写出 capability: {out_capability}")
    print(f"[OK] 已写出 hierarchy: {out_hierarchy}")
    print(f"[OK] 已写出 stats: {out_stats}")
    print(f"[INFO] CSV rows: total={csv_stats['total_rows']} valid={csv_stats['valid_rows']}")
    print(
        f"[INFO] CSV skipped: empty={csv_stats['skipped_rows_empty']} non_numeric={csv_stats['skipped_rows_non_numeric']}"
    )
    print(
        f"[INFO] abs 指标重算修正: capability={capability_stats['abs_metric_fix_count']} "
        f"hierarchy={hierarchy_stats['abs_metric_fix_count']}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise
