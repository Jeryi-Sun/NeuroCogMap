#!/usr/bin/env python3
"""
根据 language ROI 与 Schaefer parcels 的顶点重合度，选择一批“语言相关” parcels，
并在这些 parcels 上统计编码模型的预测准确度（correlation 等）。

依赖：
- encoding.brain_projection.vertix2parcel.VertexToParcelMapper
- dataset/roi_masks/language/language_mask_left.npy, language_mask_right.npy
- data_analysis 已生成的 selected JSON（例如 human_best_llm_selected_best_test_from_20251122_133937.json）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

LITCODER_CORE = Path(__file__).resolve().parents[1] / "litcoder_core"
if str(LITCODER_CORE) not in sys.path:
    sys.path.insert(0, str(LITCODER_CORE))

try:
    from neurocogmap_release.paths import env_path, output_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from neurocogmap_release.paths import env_path, output_path

from encoding.brain_projection.vertix2parcel import VertexToParcelMapper
from encoding.brain_projection.vertex_to_roi import VertexToROIMapper


DEFAULT_LEBEL_DATASET_ROOT = env_path("NEUROCOGMAP_LEBEL_DATASET_DIR") or Path("external_resources/lebel_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="根据 language ROI 选择重合度高的 parcels，并在这些 parcels 上统计预测准确度"
    )
    p.add_argument(
        "--lh_annot_path",
        type=Path,
        default=DEFAULT_LEBEL_DATASET_ROOT
        / "annotation"
        / "lh.Schaefer2018_100Parcels_7Networks_order.annot",
        help="左半球 Schaefer annot 路径",
    )
    p.add_argument(
        "--rh_annot_path",
        type=Path,
        default=DEFAULT_LEBEL_DATASET_ROOT
        / "annotation"
        / "rh.Schaefer2018_100Parcels_7Networks_order.annot",
        help="右半球 Schaefer annot 路径",
    )
    p.add_argument(
        "--language_mask_left",
        type=Path,
        default=DEFAULT_LEBEL_DATASET_ROOT
        / "roi_masks"
        / "language"
        / "language_mask_left.npy",
        help="左半球 language ROI mask (.npy, bool)",
    )
    p.add_argument(
        "--language_mask_right",
        type=Path,
        default=DEFAULT_LEBEL_DATASET_ROOT
        / "roi_masks"
        / "language"
        / "language_mask_right.npy",
        help="右半球 language ROI mask (.npy, bool)",
    )
    p.add_argument(
        "--selected_json",
        type=Path,
        required=True,
        help="编码结果的 selected JSON 路径，如 human_best_llm_selected_best_test_from_20251122_133937.json",
    )
    p.add_argument(
        "--top_k_parcels",
        type=int,
        default=30,
        help="按 language ROI 重合度排序后选取的前 K 个 parcels（默认 30）",
    )
    p.add_argument(
        "--min_overlap_fraction",
        type=float,
        default=0.0,
        help="可选：parcel 内至少有多少比例顶点在 language ROI 中才保留（0~1，默认不设门槛）",
    )
    p.add_argument(
        "--output_json",
        type=Path,
        default=output_path(
            "brain_alignment",
            "fig5_language_parcels",
            "language_parcel_overlap_and_accuracy.json",
        ),
        help="输出 JSON 路径",
    )
    return p.parse_args()


def compute_parcel_language_overlap(
    mapper: VertexToParcelMapper,
    roi_mapper: VertexToROIMapper,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    计算每个 parcel 与 language ROI 的顶点重合度。

    Returns:
        overlaps: 每个 parcel 的信息（名称、索引、重合顶点数、比例等）
        parcel_indices: 与 overlaps 顺序一致的 parcel 索引数组（用于匹配 human_parcel_idx）
    """
    # language mask: left + right -> 全局索引
    mask_left = roi_mapper.mask_left.astype(bool)
    mask_right = roi_mapper.mask_right.astype(bool)
    if mask_left.ndim != 1:
        mask_left = mask_left.reshape(-1)
    if mask_right.ndim != 1:
        mask_right = mask_right.reshape(-1)

    n_left = len(mask_left)
    n_right = len(mask_right)
    roi_mask_full = np.concatenate([mask_left, mask_right], axis=0)
    if roi_mask_full.dtype != bool:
        roi_mask_full = roi_mask_full.astype(bool)

    total_roi_vertices = int(np.sum(roi_mask_full))
    if total_roi_vertices == 0:
        raise ValueError("language ROI 中没有任何顶点（mask 全为 False）")

    # VertexToParcelMapper 的 parcel_names 顺序 = LH parcels + RH parcels
    parcel_names = mapper.parcel_names

    overlaps: List[Dict[str, Any]] = []
    parcel_indices: List[int] = []

    # 构建与 VertexToParcelMapper 一致的 LH/RH lookup
    # 其内部使用 _build_lookup，顺序与 parcel_names 一致，我们通过 parcel_to_vertex 反推索引更稳妥
    # 这里直接利用 parcel_to_vertex 的结构化扩展：对单位向量逐个投影得到每个 parcel 的 vertex mask
    n_parcels = len(parcel_names)
    eye = np.eye(n_parcels, dtype=float)
    parcel_vertex_masks = mapper.parcel_to_vertex(eye, fill_value=0.0) > 0
    # parcel_vertex_masks 形状: (n_parcels, n_vertices)

    for idx in range(n_parcels):
        name = parcel_names[idx]
        vertex_mask = parcel_vertex_masks[idx]  # bool, length = n_vertices
        parcel_vertex_count = int(np.sum(vertex_mask))
        if parcel_vertex_count == 0:
            continue

        # 计算与 language ROI 的重合
        overlap_vertices = vertex_mask & roi_mask_full
        overlap_count = int(np.sum(overlap_vertices))
        frac_in_parcel = overlap_count / parcel_vertex_count
        frac_of_roi = overlap_count / total_roi_vertices

        overlaps.append(
            {
                "parcel_idx": idx,
                "parcel_name": name,
                "parcel_vertex_count": parcel_vertex_count,
                "overlap_vertex_count": overlap_count,
                "frac_in_parcel": float(frac_in_parcel),
                "frac_of_language_roi": float(frac_of_roi),
            }
        )
        parcel_indices.append(idx)

    return overlaps, np.asarray(parcel_indices, dtype=int)


def compute_accuracy_for_parcels(
    selected_json: Path,
    selected_parcel_indices: List[int],
) -> Dict[str, Any]:
    """
    在指定的 parcel 索引集合上，统计 selected JSON 中的预测准确度。

    假设 human_parcel_idx 与 VertexToParcelMapper.parcel_names 的顺序一致。
    """
    if not selected_json.exists():
        raise FileNotFoundError(f"selected_json 不存在: {selected_json}")

    data = json.loads(selected_json.read_text())
    rows = data.get("selected_results")
    if not isinstance(rows, list):
        raise TypeError("selected_json 中缺少 selected_results(list)")

    idx_set = set(int(i) for i in selected_parcel_indices)
    used_rows = [r for r in rows if int(r["human_parcel_idx"]) in idx_set]
    if not used_rows:
        raise ValueError("在 selected_json 中没有找到任何匹配的 human_parcel_idx，请检查索引映射")

    base_corr = np.asarray([r["base_correlation"] for r in used_rows], dtype=np.float64)
    base_p = np.asarray([r["base_p_value"] for r in used_rows], dtype=np.float64)
    base_cp = np.asarray([r["base_corrected_p_value"] for r in used_rows], dtype=np.float64)
    base_sig = np.asarray([1.0 if r["base_significant"] else 0.0 for r in used_rows], dtype=np.float64)

    def topk_mean(arr: np.ndarray, k: int) -> float:
        k = min(k, arr.size)
        if k <= 0:
            return float("nan")
        sorted_arr = np.sort(arr)[::-1]
        return float(np.mean(sorted_arr[:k]))

    q25 = float(np.percentile(base_corr, 25))
    q75 = float(np.percentile(base_corr, 75))

    return {
        "n_parcels": len(idx_set),
        "n_rows": len(used_rows),
        "mean_base_correlation": float(np.mean(base_corr)),
        "median_base_correlation": float(np.median(base_corr)),
        "top20_mean_base_correlation": topk_mean(base_corr, 20),
        "top30_mean_base_correlation": topk_mean(base_corr, 30),
        "q25_base_correlation": q25,
        "q75_base_correlation": q75,
        "std_base_correlation": float(np.std(base_corr)),
        "min_base_correlation": float(np.min(base_corr)),
        "max_base_correlation": float(np.max(base_corr)),
        "mean_base_p_value": float(np.mean(base_p)),
        "mean_base_corrected_p_value": float(np.mean(base_cp)),
        "base_significant_rate": float(np.mean(base_sig)),
        "n_base_significant": int(np.sum(base_sig)),
    }


def main() -> None:
    args = parse_args()

    # 构建 mapper
    parcel_mapper = VertexToParcelMapper(
        lh_annot_path=str(args.lh_annot_path),
        rh_annot_path=str(args.rh_annot_path),
        use_nanmean=True,
    )
    roi_mapper = VertexToROIMapper(
        mask_left_path=str(args.language_mask_left),
        mask_right_path=str(args.language_mask_right),
        combine_hemispheres=True,
        use_nanmean=True,
    )

    overlaps, parcel_indices = compute_parcel_language_overlap(parcel_mapper, roi_mapper)

    # 按重合度排序：这里默认按 overlap_vertex_count 降序，再按 frac_in_parcel 次级排序
    overlaps_sorted = sorted(
        overlaps,
        key=lambda x: (x["overlap_vertex_count"], x["frac_in_parcel"]),
        reverse=True,
    )

    # 过滤：先按 parcel 内重合比例 >= min_overlap_fraction，再截取 top_k
    filtered = [
        o for o in overlaps_sorted if o["frac_in_parcel"] >= args.min_overlap_fraction
    ]
    top_k = filtered[: args.top_k_parcels] if args.top_k_parcels > 0 else filtered

    selected_parcel_idxs = [int(o["parcel_idx"]) for o in top_k]

    # 计算这些 parcel 上的编码准确度
    accuracy_stats = compute_accuracy_for_parcels(
        selected_json=args.selected_json,
        selected_parcel_indices=selected_parcel_idxs,
    )

    payload = {
        "meta": {
            "lh_annot_path": str(args.lh_annot_path),
            "rh_annot_path": str(args.rh_annot_path),
            "language_mask_left": str(args.language_mask_left),
            "language_mask_right": str(args.language_mask_right),
            "selected_json": str(args.selected_json),
            "top_k_parcels": args.top_k_parcels,
            "min_overlap_fraction": args.min_overlap_fraction,
        },
        "overlaps_sorted": overlaps_sorted,
        "selected_parcels": top_k,
        "selected_parcel_indices": selected_parcel_idxs,
        "accuracy_stats_on_selected_parcels": accuracy_stats,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 已保存结果到: {args.output_json}")
    print(
        "[INFO] 选中的 parcel 数量:",
        len(selected_parcel_idxs),
        "mean_base_correlation (selected):",
        accuracy_stats["mean_base_correlation"],
    )


if __name__ == "__main__":
    main()
