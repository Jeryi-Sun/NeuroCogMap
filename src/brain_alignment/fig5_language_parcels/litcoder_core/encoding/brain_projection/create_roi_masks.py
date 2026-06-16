#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建语言网络和视觉网络的 ROI 掩码

使用方法：
    python create_roi_masks.py \
        --lana_path /path/to/lana/probability/map \
        --output_dir /path/to/output \
        --top_percent 10

参考原文：
    Language network (LanA). LanA provides a per-vertex probability of language selectivity.
    After transferring the LanA map to fsaverage5, we define the language ROI as the top 10%
    highest-probability vertices (default setting used throughout).
"""

import argparse
import numpy as np
from nilearn import datasets, surface
from nilearn import plotting
import nibabel as nib
from pathlib import Path
import json
from typing import Tuple, Dict, Optional
from functools import lru_cache
from scipy.spatial import cKDTree

from roi_mapping import (
    transfer_to_fsaverage5,
    create_language_roi_mask,
    create_visual_roi_mask,
    visualize_roi_mask,
    apply_roi_mask
)


MESH_VERTEX_COUNTS = {
    "fsaverage7": 163842,
    "fsaverage6": 40962,
    "fsaverage5": 10242,
    "fsaverage4": 2562,
    "fsaverage3": 642
}


def infer_mesh_from_vertex_count(vertex_count: int) -> str:
    for mesh_name, count in MESH_VERTEX_COUNTS.items():
        if vertex_count == count:
            return mesh_name
    raise ValueError(
        f"无法根据顶点数 {vertex_count} 确定 fsaverage 网格，请确认输入文件是否为标准表面。"
    )


@lru_cache(maxsize=16)
def _get_resample_indices(source_mesh: str, target_mesh: str, hemi: str) -> np.ndarray:
    """
    使用最近邻在球面上计算从 source_mesh -> target_mesh 的索引映射。
    """
    fs_source = datasets.fetch_surf_fsaverage(mesh=source_mesh)
    fs_target = datasets.fetch_surf_fsaverage(mesh=target_mesh)
    hemi_key = "left" if hemi.lower().startswith("l") else "right"
    source_coords = surface.load_surf_mesh(fs_source[f"sphere_{hemi_key}"])[0]
    target_coords = surface.load_surf_mesh(fs_target[f"sphere_{hemi_key}"])[0]
    tree = cKDTree(source_coords)
    _, indices = tree.query(target_coords, k=1)
    return indices


def _load_single_hemi_surface(
    file_path: Path,
    hemi: str,
    target_mesh: str = "fsaverage5"
) -> np.ndarray:
    """
    加载单半球 per-vertex 概率图（可能以 NIfTI/GIFTI 形式存储），必要时下采样到 target_mesh。
    """
    img = nib.load(str(file_path))
    data = np.asarray(img.get_fdata()).squeeze()
    if data.ndim != 1:
        data = data.reshape(data.shape[0], -1)
        if data.shape[1] != 1:
            raise ValueError(
                f"[ERROR] {file_path} 形状为 {img.shape}，无法解释为单半球 per-vertex 数组。"
            )
        data = data[:, 0]
    vertex_count = data.shape[0]
    source_mesh = infer_mesh_from_vertex_count(vertex_count)
    if source_mesh == target_mesh:
        return data
    indices = _get_resample_indices(source_mesh, target_mesh, hemi)
    return data[indices]


def load_surface_probability_map(
    file_path: str,
    right_file_path: Optional[str] = None,
    target_mesh: str = "fsaverage5",
    *,
    use_neuromaps: bool = False,
    neuromaps_density: str = "10k",
    neuromaps_method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载表面概率图（支持多种格式）
    
    Args:
        file_path: 文件路径（可以是 .gii, .mgz, 或体积文件路径）
        
    Returns:
        (left_prob, right_prob): 左右半球的概率图
    """
    file_path = Path(file_path)
    
    if right_file_path is not None:
        right_path = Path(right_file_path)
        if use_neuromaps:
            left_prob, right_prob = transfer_to_fsaverage5(
                surface_inputs={
                    "left": str(file_path),
                    "right": str(right_path)
                },
                target_density=neuromaps_density,
                neuromaps_method=neuromaps_method
            )
            return left_prob, right_prob
        left_prob = _load_single_hemi_surface(file_path, hemi="left", target_mesh=target_mesh)
        right_prob = _load_single_hemi_surface(right_path, hemi="right", target_mesh=target_mesh)
        return left_prob, right_prob
    
    # 如果是 .gii 格式（Gifti）
    if file_path.suffix == '.gii' or 'gii' in str(file_path):
        left_path = str(file_path).replace('right', 'left').replace('rh.', 'lh.')
        right_path = str(file_path).replace('left', 'right').replace('lh.', 'rh.')
        
        if Path(left_path).exists() and Path(right_path).exists():
            left_img = nib.load(left_path)
            right_img = nib.load(right_path)
            left_prob = left_img.darrays[0].data if hasattr(left_img, 'darrays') else left_img.get_fdata()
            right_prob = right_img.darrays[0].data if hasattr(right_img, 'darrays') else right_img.get_fdata()
        else:
            raise ValueError("请提供匹配的左右半球 GIfTI 文件（包含 'lh'/'rh' 或 'left'/'right' 标识）。")
    else:
        volume_img = nib.load(str(file_path))
        volume_data = volume_img.get_fdata()
        # 检测是否为“伪体积”per-vertex 数据
        if volume_data.ndim == 4 and volume_data.shape[1:] == (1, 1, 1):
            raise ValueError(
                "检测到 per-vertex NIfTI（单半球概率图）。请使用 --lana_left_path/--lana_right_path 单独传入左右半球文件。"
            )
        fsaverage = datasets.fetch_surf_fsaverage(mesh=target_mesh)
        left_prob, right_prob = transfer_to_fsaverage5(
            volume_data,
            volume_img.affine,
            fsaverage
        )
    
    return left_prob, right_prob


def save_roi_mask(
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    output_dir: Path,
    roi_name: str
):
    """
    保存 ROI 掩码
    
    Args:
        mask_left: 左半球掩码
        mask_right: 右半球掩码
        output_dir: 输出目录
        roi_name: ROI 名称（如 'language', 'visual'）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import pdb; pdb.set_trace()
    # 保存为 numpy 数组
    np.save(output_dir / f"{roi_name}_mask_left.npy", mask_left)
    np.save(output_dir / f"{roi_name}_mask_right.npy", mask_right)
    
    # 保存为 JSON（包含统计信息）
    stats = {
        "roi_name": roi_name,
        "left_hemisphere": {
            "total_vertices": int(len(mask_left)),
            "roi_vertices": int(np.sum(mask_left)),
            "percentage": float(np.sum(mask_left) / len(mask_left) * 100)
        },
        "right_hemisphere": {
            "total_vertices": int(len(mask_right)),
            "roi_vertices": int(np.sum(mask_right)),
            "percentage": float(np.sum(mask_right) / len(mask_right) * 100)
        }
    }
    
    with open(output_dir / f"{roi_name}_mask_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ 已保存 {roi_name} ROI 掩码到: {output_dir}")
    print(f"   左半球: {stats['left_hemisphere']['roi_vertices']} / {stats['left_hemisphere']['total_vertices']} 顶点 ({stats['left_hemisphere']['percentage']:.2f}%)")
    print(f"   右半球: {stats['right_hemisphere']['roi_vertices']} / {stats['right_hemisphere']['total_vertices']} 顶点 ({stats['right_hemisphere']['percentage']:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="创建语言网络和视觉网络的 ROI 掩码"
    )
    parser.add_argument(
        "--lana_path",
        type=str,
        help="LanA 语言网络概率图路径（可以是 .gii 或体积文件）"
    )
    parser.add_argument(
        "--lana_left_path",
        type=str,
        help="LanA 左半球 per-vertex 概率图路径（可与 lana_right_path 搭配使用）"
    )
    parser.add_argument(
        "--lana_right_path",
        type=str,
        help="LanA 右半球 per-vertex 概率图路径（必须与 lana_left_path 同时提供）"
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        help="视觉网络概率图路径（可选）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--top_percent",
        type=float,
        default=10.0,
        help="选择前百分之多少的顶点作为 ROI（默认 10%）"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="是否生成可视化图像"
    )
    parser.add_argument(
        "--use_neuromaps",
        action="store_true",
        help="若提供左右半球 surface 文件，使用 neuromaps 进行 fsaverage 对齐重采样"
    )
    parser.add_argument(
        "--neuromaps_density",
        type=str,
        default="10k",
        help="调用 neuromaps 时的目标密度，fsaverage5 对应 10k"
    )
    parser.add_argument(
        "--neuromaps_method",
        type=str,
        default="linear",
        help="调用 neuromaps 时的插值方式（默认 linear）"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取 fsaverage5 数据
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    print(f"📊 fsaverage5 表面信息:")
    print(f"   左半球顶点数: {surface.load_surf_mesh(fsaverage['pial_left'])[0].shape[0]}")
    print(f"   右半球顶点数: {surface.load_surf_mesh(fsaverage['pial_right'])[0].shape[0]}")
    
    # 处理语言网络 ROI
    language_input_desc = None
    lana_prob_left = lana_prob_right = None
    if args.lana_left_path or args.lana_right_path:
        if not (args.lana_left_path and args.lana_right_path):
            raise ValueError("必须同时提供 --lana_left_path 与 --lana_right_path。")
        lana_prob_left, lana_prob_right = load_surface_probability_map(
            args.lana_left_path,
            right_file_path=args.lana_right_path,
            target_mesh="fsaverage5",
            use_neuromaps=args.use_neuromaps,
            neuromaps_density=args.neuromaps_density,
            neuromaps_method=args.neuromaps_method
        )
        language_input_desc = [
            f"   左半球文件: {args.lana_left_path}",
            f"   右半球文件: {args.lana_right_path}"
        ]
    elif args.lana_path:
        lana_prob_left, lana_prob_right = load_surface_probability_map(args.lana_path)
        language_input_desc = [f"   输入文件: {args.lana_path}"]
    
    if lana_prob_left is not None and lana_prob_right is not None:
        print(f"\n🔤 处理语言网络 (LanA)...")
        if language_input_desc:
            for line in language_input_desc:
                print(line)
        
        # 创建语言网络 ROI 掩码
        language_mask_left, language_mask_right = create_language_roi_mask(
            lana_prob_left,
            lana_prob_right,
            top_percent=args.top_percent
        )
        
        # 保存掩码
        save_roi_mask(
            language_mask_left,
            language_mask_right,
            output_dir,
            "language"
        )
        
        # 可视化
        if args.visualize:
            visualize_roi_mask(
                language_mask_left,
                language_mask_right,
                fsaverage,
                title=f"Language Network ROI (Top {args.top_percent}%)",
                output_path=str(output_dir / "language_roi.png")
            )
    
    # 处理视觉网络 ROI
    if args.visual_path:
        print(f"\n👁️  处理视觉网络...")
        print(f"   输入文件: {args.visual_path}")
        
        # 加载视觉网络概率图
        visual_prob_left, visual_prob_right = load_surface_probability_map(args.visual_path)
        
        # 创建视觉网络 ROI 掩码
        visual_mask_left, visual_mask_right = create_visual_roi_mask(
            visual_prob_left,
            visual_prob_right,
            top_percent=args.top_percent
        )
        import pdb; pdb.set_trace()
        # 保存掩码
        save_roi_mask(
            visual_mask_left,
            visual_mask_right,
            output_dir,
            "visual"
        )
        
        # 可视化
        if args.visualize:
            visualize_roi_mask(
                visual_mask_left,
                visual_mask_right,
                fsaverage,
                title=f"Visual Network ROI (Top {args.top_percent}%)",
                output_path=str(output_dir / "visual_roi.png")
            )
    
    print(f"\n✅ 完成！所有 ROI 掩码已保存到: {output_dir}")


if __name__ == "__main__":
    main()

