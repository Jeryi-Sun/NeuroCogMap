#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将语言网络 (LanA) 和视觉网络映射到 fsaverage 表面的工具函数

参考原文描述：
Language network (LanA). LanA provides a per-vertex probability of language selectivity [25]. 
After transferring the LanA map to fsaverage5, we define the language ROI as the top 10% 
highest-probability vertices (default setting used throughout). This produces a reproducible, 
thresholded mask for the language network.
"""

import numpy as np
from nilearn import datasets, surface
from nilearn import plotting
import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from nibabel.spatialimages import SpatialImage
from functools import lru_cache
from scipy.spatial import cKDTree

try:
    from neuromaps.transforms import fsaverage_to_fsaverage

    _HAS_NEUROMAPS = True
except ImportError:
    _HAS_NEUROMAPS = False

_HAS_WORKBENCH = shutil.which("wb_command") is not None

MESH_VERTEX_COUNTS = {
    "fsaverage7": 163842,
    "fsaverage6": 40962,
    "fsaverage5": 10242,
    "fsaverage4": 2562,
    "fsaverage3": 642
}

DENSITY_TO_FSAVERAGE = {
    "3k": "fsaverage3",
    "10k": "fsaverage5",
    "41k": "fsaverage6",
    "164k": "fsaverage7"
}

SurfaceImageLike = Union[SpatialImage, GiftiImage]
SurfaceInput = Union[str, Path, SurfaceImageLike, np.ndarray]


def _infer_mesh_from_vertex_count(vertex_count: int) -> str:
    for mesh_name, count in MESH_VERTEX_COUNTS.items():
        if vertex_count == count:
            return mesh_name
    raise ValueError(
        f"[ERROR] 无法根据顶点数 {vertex_count} 推断 fsaverage 网格，请确认输入是否为标准表面。"
    )
@lru_cache(maxsize=32)
def _get_resample_indices(source_mesh: str, target_mesh: str, hemi: str) -> np.ndarray:
    fs_source = datasets.fetch_surf_fsaverage(mesh=source_mesh)
    fs_target = datasets.fetch_surf_fsaverage(mesh=target_mesh)
    hemi_key = "left" if hemi.lower().startswith("l") else "right"
    source_coords = surface.load_surf_mesh(fs_source[f"sphere_{hemi_key}"])[0]
    target_coords = surface.load_surf_mesh(fs_target[f"sphere_{hemi_key}"])[0]
    tree = cKDTree(source_coords)
    _, idx = tree.query(target_coords, k=1)
    return idx


def _resample_surface_via_nearest(
    surface_obj: SurfaceInput,
    hemi_desc: str,
    target_mesh: str = "fsaverage5"
) -> np.ndarray:
    img = _ensure_surface_image(surface_obj, hemi_desc)
    data = _surface_image_to_array(img, hemi_desc)
    vertex_count = data.shape[0]
    source_mesh = _infer_mesh_from_vertex_count(vertex_count)
    if source_mesh == target_mesh:
        return data
    indices = _get_resample_indices(source_mesh, target_mesh, hemi_desc)
    return data[indices]


def _ensure_surface_image(surface_obj: SurfaceInput, hemi_desc: str) -> SurfaceImageLike:
    """
    将 surface 输入统一为 nibabel Image，供内部处理（非 neuromaps 使用）。
    """
    if isinstance(surface_obj, (SpatialImage, GiftiImage)):
        return surface_obj
    if isinstance(surface_obj, (str, Path)):
        path = Path(surface_obj).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] 未找到 {hemi_desc} 半球输入文件: {path}")
        return nib.load(str(path))
    if isinstance(surface_obj, np.ndarray):
        arr = np.asarray(surface_obj).squeeze()
        if arr.ndim != 1:
            raise ValueError(
                f"[ERROR] {hemi_desc} 半球 surface ndarray 需要是一维向量，当前形状为 {arr.shape}。"
            )
        gifti_img = GiftiImage()
        gifti_img.add_gifti_data_array(GiftiDataArray(arr.astype(np.float32)))
        return gifti_img
    raise TypeError(
        f"[ERROR] 无法识别 {hemi_desc} 半球输入类型: {type(surface_obj)}。"
    )


def _prepare_surface_for_neuromaps(surface_obj: SurfaceInput, hemi_desc: str) -> Tuple[str, Optional[str]]:
    """
    将输入转为 neuromaps 接受的文件路径；若需临时文件，返回并在调用方负责清理。
    """
    if isinstance(surface_obj, (str, Path)):
        path = Path(surface_obj).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] 未找到 {hemi_desc} 半球输入文件: {path}")
        return str(path), None
    img = _ensure_surface_image(surface_obj, hemi_desc)
    data = _surface_image_to_array(img, hemi_desc)
    tmp = tempfile.NamedTemporaryFile(suffix=".func.gii", delete=False)
    tmp_path = tmp.name
    tmp.close()
    gifti_img = GiftiImage()
    gifti_img.add_gifti_data_array(GiftiDataArray(data.astype(np.float32)))
    nib.save(gifti_img, tmp_path)
    return tmp_path, tmp_path


def _surface_image_to_array(img: SurfaceImageLike, hemi_desc: str) -> np.ndarray:
    """
    从 Gifti/Nifti surface 对象中提取一维 numpy 数组。
    """
    if hasattr(img, "darrays") and getattr(img, "darrays"):
        data = np.asarray(img.darrays[0].data).squeeze()
    else:
        data = np.asarray(img.get_fdata()).squeeze()
    if data.ndim != 1:
        raise ValueError(
            f"[ERROR] {hemi_desc} 半球 surface 数据维度应为 1，当前形状为 {data.shape}。"
        )
    return data


def load_lana_map(lana_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    加载 LanA (Language Network Atlas) 概率图
    
    Args:
        lana_path: LanA 文件路径。如果为 None，需要从网络下载或使用默认路径
        
    Returns:
        包含左右半球概率图的字典，键为 'left' 和 'right'
    """
    if lana_path is None:
        # 如果 LanA 文件需要下载，可以从以下位置获取：
        # https://github.com/brainhack-school2020/project_template/issues
        # 或者使用 nilearn 的 datasets 模块下载
        raise ValueError("请提供 LanA 文件路径，或实现自动下载功能")
    
    # 加载 LanA 概率图（假设是 .gii 格式或 .mgz 格式）
    # 这里需要根据实际文件格式调整
    if lana_path.endswith('.gii'):
        lana_left = nib.load(lana_path.replace('left', 'left') if 'left' in lana_path else f"{lana_path}_left.gii")
        lana_right = nib.load(lana_path.replace('right', 'right') if 'right' in lana_path else f"{lana_path}_right.gii")
        prob_left = lana_left.darrays[0].data
        prob_right = lana_right.darrays[0].data
    else:
        # 如果是其他格式，需要相应调整
        raise NotImplementedError(f"不支持的文件格式: {lana_path}")
    
    return {
        'left': prob_left,
        'right': prob_right
    }


def transfer_to_fsaverage5(
    volume_map: Optional[np.ndarray] = None,
    affine: Optional[np.ndarray] = None,
    fsaverage: Optional[Dict] = None,
    *,
    surface_inputs: Optional[Dict[str, SurfaceInput]] = None,
    target_density: str = "10k",
    neuromaps_method: str = "linear"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将输入概率图映射到 fsaverage5 (10k) 表面。
    
    支持两种输入模式：
        1. volume_map + affine: 体积概率图，使用 nilearn.vol_to_surf 投影。
        2. surface_inputs: 已在 fsaverageX 表面的 per-vertex 数据（遵循 @fsaverage_transfer
           记录推荐，调用 neuromaps 进行球面对齐重采样）。
    
    Args:
        volume_map: 体积数据 (3D array)。仅在 surface_inputs=None 时必需。
        affine: 仿射矩阵。仅在 surface_inputs=None 时必需。
        fsaverage: fsaverage 数据字典。若为 None 自动下载 fsaverage5。
        surface_inputs: dict，键必须包含 'left' 与 'right'，值可为路径、SpatialImage 或 ndarray。
        target_density: neuromaps 目标密度，fsaverage5 对应 '10k'。
        neuromaps_method: neuromaps 插值方式，默认为 'linear'。
        
    Returns:
        (left_surface_data, right_surface_data): 左右半球的表面数据 (np.ndarray)。
    """
    if surface_inputs is not None:
        missing_keys = {"left", "right"} - set(surface_inputs.keys())
        if missing_keys:
            raise ValueError(
                f"[ERROR] surface_inputs 需要同时包含 'left' 与 'right'，缺少: {missing_keys}"
            )
        target_mesh = DENSITY_TO_FSAVERAGE.get(target_density)
        if target_mesh is None:
            raise ValueError(
                f"[ERROR] 不支持的 target_density='{target_density}'，"
                f"可选值: {list(DENSITY_TO_FSAVERAGE.keys())}"
            )
        missing_deps = []
        if not _HAS_NEUROMAPS:
            missing_deps.append("neuromaps")
        if not _HAS_WORKBENCH:
            missing_deps.append("wb_command")
        if missing_deps:
            print(
                f"[WARN] neuromaps 球面重采样缺少依赖 {missing_deps}，"
                "自动回退到基于球面最近邻的下采样。"
            )
            left_data = _resample_surface_via_nearest(surface_inputs["left"], "left", target_mesh)
            right_data = _resample_surface_via_nearest(surface_inputs["right"], "right", target_mesh)
            return left_data, right_data
        left_ref, left_tmp = _prepare_surface_for_neuromaps(surface_inputs["left"], "左")
        right_ref, right_tmp = _prepare_surface_for_neuromaps(surface_inputs["right"], "右")
        print(
            f"[INFO] 使用 neuromaps.fsaverage_to_fsaverage 将 surface 数据重采样至 {target_density}"
            f" ({target_mesh})。"
        )
        try:
            resampled_left = fsaverage_to_fsaverage(
                data=left_ref,
                hemi="L",
                target_density=target_density,
                method=neuromaps_method
            )[0]
            resampled_right = fsaverage_to_fsaverage(
                data=right_ref,
                hemi="R",
                target_density=target_density,
                method=neuromaps_method
            )[0]
        finally:
            for tmp_path in (left_tmp, right_tmp):
                if tmp_path and Path(tmp_path).exists():
                    try:
                        os.remove(tmp_path)
                    except OSError as err:
                        print(f"[WARN] 无法删除临时文件 {tmp_path}: {err}")
        left_data = _surface_image_to_array(resampled_left, "左")
        right_data = _surface_image_to_array(resampled_right, "右")
        return left_data, right_data
    
    if volume_map is None or affine is None:
        raise ValueError(
            "[ERROR] 当未提供 surface_inputs 时，volume_map 与 affine 均不能为空。"
        )
    if fsaverage is None:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    
    mesh_left = surface.load_surf_mesh(fsaverage["pial_left"])
    mesh_right = surface.load_surf_mesh(fsaverage["pial_right"])
    img = nib.Nifti1Image(volume_map, affine)
    data_left = surface.vol_to_surf(img, mesh_left)
    data_right = surface.vol_to_surf(img, mesh_right)
    
    return data_left, data_right


def _nan_safe_percentile(values: np.ndarray, percentile: float, hemi: str) -> float:
    """
    计算忽略 NaN 的百分位，如果全是 NaN 就报错。
    """
    if np.isnan(values).all():
        raise ValueError(f"[ERROR] {hemi} 半球概率图全为 NaN，无法计算百分位。")
    if np.isnan(values).any():
        nan_count = int(np.isnan(values).sum())
        print(f"[WARN] {hemi} 半球概率图包含 {nan_count} 个 NaN，已忽略再计算百分位。")
    return np.nanpercentile(values, percentile)


def _build_mask(
    probs: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    给定概率和阈值，生成 ROI 掩码。NaN 点默认视为 False。
    """
    mask = np.zeros_like(probs, dtype=bool)
    finite_idx = ~np.isnan(probs)
    mask[finite_idx] = probs[finite_idx] >= threshold
    return mask


def create_language_roi_mask(
    lana_prob_left: np.ndarray,
    lana_prob_right: np.ndarray,
    top_percent: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据 LanA 概率图创建语言网络 ROI 掩码
    
    定义语言 ROI 为概率最高的前 top_percent% 的顶点
    
    Args:
        lana_prob_left: 左半球 LanA 概率图
        lana_prob_right: 右半球 LanA 概率图
        top_percent: 选择前百分之多少的顶点（默认 10%）
        
    Returns:
        (left_mask, right_mask): 左右半球的布尔掩码，True 表示在 ROI 内
    """
    # 计算阈值（前 top_percent%），忽略 NaN
    percentile = 100 - top_percent
    threshold_left = _nan_safe_percentile(lana_prob_left, percentile, "左")
    threshold_right = _nan_safe_percentile(lana_prob_right, percentile, "右")
    
    # 创建掩码，确保 NaN 顶点不进入 ROI
    left_mask = _build_mask(lana_prob_left, threshold_left)
    right_mask = _build_mask(lana_prob_right, threshold_right)
    
    print(f"左半球语言 ROI: {np.sum(left_mask)} / {len(left_mask)} 顶点 ({np.sum(left_mask)/len(left_mask)*100:.2f}%)")
    print(f"右半球语言 ROI: {np.sum(right_mask)} / {len(right_mask)} 顶点 ({np.sum(right_mask)/len(right_mask)*100:.2f}%)")
    
    return left_mask, right_mask


def create_visual_roi_mask(
    visual_prob_left: np.ndarray,
    visual_prob_right: np.ndarray,
    top_percent: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据视觉网络概率图创建视觉 ROI 掩码
    
    Args:
        visual_prob_left: 左半球视觉网络概率图
        visual_prob_right: 右半球视觉网络概率图
        top_percent: 选择前百分之多少的顶点（默认 10%）
        
    Returns:
        (left_mask, right_mask): 左右半球的布尔掩码
    """
    # 计算阈值（前 top_percent%）
    threshold_left = np.percentile(visual_prob_left, 100 - top_percent)
    threshold_right = np.percentile(visual_prob_right, 100 - top_percent)
    
    # 创建掩码
    left_mask = visual_prob_left >= threshold_left
    right_mask = visual_prob_right >= threshold_right
    
    print(f"左半球视觉 ROI: {np.sum(left_mask)} / {len(left_mask)} 顶点 ({np.sum(left_mask)/len(left_mask)*100:.2f}%)")
    print(f"右半球视觉 ROI: {np.sum(right_mask)} / {len(right_mask)} 顶点 ({np.sum(right_mask)/len(right_mask)*100:.2f}%)")
    
    return left_mask, right_mask


def apply_roi_mask(
    surface_data_left: np.ndarray,
    surface_data_right: np.ndarray,
    mask_left: np.ndarray,
    mask_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 ROI 掩码应用到表面数据
    
    Args:
        surface_data_left: 左半球表面数据
        surface_data_right: 右半球表面数据
        mask_left: 左半球掩码
        mask_right: 右半球掩码
        
    Returns:
        (masked_left, masked_right): 应用掩码后的数据
    """
    # 创建掩码后的数据（非 ROI 区域设为 NaN）
    masked_left = surface_data_left.copy()
    masked_right = surface_data_right.copy()
    
    masked_left[~mask_left] = np.nan
    masked_right[~mask_right] = np.nan
    
    return masked_left, masked_right


def visualize_roi_mask(
    mask_left: np.ndarray,
    mask_right: np.ndarray,
    fsaverage: Optional[Dict] = None,
    title: str = "Language Network ROI",
    output_path: Optional[str] = None
):
    """
    可视化 ROI 掩码
    
    Args:
        mask_left: 左半球掩码
        mask_right: 右半球掩码
        fsaverage: fsaverage 数据字典
        title: 图像标题
        output_path: 保存路径，如果为 None 则显示图像
    """
    if fsaverage is None:
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    
    # 将布尔掩码转换为数值（用于可视化）
    mask_left_vis = mask_left.astype(float)
    mask_right_vis = mask_right.astype(float)
    
    # 创建图形
    fig = plotting.plot_surf_stat_map(
        fsaverage["infl_left"],
        mask_left_vis,
        hemi="left",
        view="lateral",
        title=f"{title} - Left Hemisphere",
        colorbar=True,
        cmap="hot"
    )
    
    if output_path:
        fig.savefig(output_path.replace('.png', '_left.png'))
        print(f"已保存左半球可视化: {output_path.replace('.png', '_left.png')}")
    
    fig = plotting.plot_surf_stat_map(
        fsaverage["infl_right"],
        mask_right_vis,
        hemi="right",
        view="lateral",
        title=f"{title} - Right Hemisphere",
        colorbar=True,
        cmap="hot"
    )
    
    if output_path:
        fig.savefig(output_path.replace('.png', '_right.png'))
        print(f"已保存右半球可视化: {output_path.replace('.png', '_right.png')}")


def main_example():
    """
    示例：如何使用这些函数创建语言网络 ROI
    """
    # 1. 获取 fsaverage5 数据
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    print(f"fsaverage5 左半球顶点数: {surface.load_surf_mesh(fsaverage['pial_left'])[0].shape[0]}")
    print(f"fsaverage5 右半球顶点数: {surface.load_surf_mesh(fsaverage['pial_right'])[0].shape[0]}")
    
    # 2. 加载 LanA 概率图（需要提供实际路径）
    # lana_path = "/path/to/lana/probability/map"
    # lana_probs = load_lana_map(lana_path)
    
    # 3. 如果 LanA 是体积数据，需要先转换到表面
    # lana_volume = nib.load("/path/to/lana_volume.nii.gz")
    # lana_prob_left, lana_prob_right = transfer_to_fsaverage5(
    #     lana_volume.get_fdata(),
    #     lana_volume.affine,
    #     fsaverage
    # )
    
    # 4. 创建语言网络 ROI 掩码（前 10%）
    # language_mask_left, language_mask_right = create_language_roi_mask(
    #     lana_prob_left,
    #     lana_prob_right,
    #     top_percent=10.0
    # )
    
    # 5. 可视化 ROI
    # visualize_roi_mask(
    #     language_mask_left,
    #     language_mask_right,
    #     fsaverage,
    #     title="Language Network ROI (Top 10%)",
    #     output_path="language_roi.png"
    # )
    
    print("示例代码已准备就绪，请根据实际数据路径取消注释并运行")


if __name__ == "__main__":
    main_example()

