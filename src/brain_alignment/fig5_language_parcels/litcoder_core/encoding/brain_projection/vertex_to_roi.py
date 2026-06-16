"""
将 fsaverage5 顶点级别的数值映射到语言 ROI，取平均作为 Y 值。
"""

import logging
from typing import List, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VertexToROIMapper:
    """将 fsaverage5 顶点级别的数值映射到语言 ROI，取平均作为 Y 值。
    
    根据语言 ROI mask，将对应的顶点取出来取平均。
    输出可以是：
    - 单个值：左右脑 ROI 的平均值
    - 两个值：左脑 ROI 平均值和右脑 ROI 平均值（按顺序）
    """

    def __init__(
        self,
        mask_left_path: str,
        mask_right_path: str,
        combine_hemispheres: bool = True,
        use_nanmean: bool = True,
    ) -> None:
        """
        Args:
            mask_left_path: 左半球 ROI mask 文件路径（.npy 格式，布尔数组）
            mask_right_path: 右半球 ROI mask 文件路径（.npy 格式，布尔数组）
            combine_hemispheres: 如果 True，返回左右脑的平均值（单个值）；
                                 如果 False，返回 [左脑平均值, 右脑平均值]（两个值）
            use_nanmean: 是否使用 np.nanmean（否则使用 np.mean）
        """
        # 加载 mask
        mask_left_path = Path(mask_left_path)
        mask_right_path = Path(mask_right_path)
        
        if not mask_left_path.exists():
            raise FileNotFoundError(f"左半球 mask 文件不存在: {mask_left_path}")
        if not mask_right_path.exists():
            raise FileNotFoundError(f"右半球 mask 文件不存在: {mask_right_path}")
        
        self.mask_left = np.load(mask_left_path).astype(bool)
        self.mask_right = np.load(mask_right_path).astype(bool)
        
        # 确保是一维数组
        if self.mask_left.ndim != 1:
            self.mask_left = self.mask_left.reshape(-1)
        if self.mask_right.ndim != 1:
            self.mask_right = self.mask_right.reshape(-1)
        
        self.combine_hemispheres = combine_hemispheres
        self._reducer = np.nanmean if use_nanmean else np.mean
        
        # 统计信息
        n_left = np.sum(self.mask_left)
        n_right = np.sum(self.mask_right)
        n_total = n_left + n_right
        
        logger.info(
            "加载语言 ROI mapper: 左脑 %d 个顶点，右脑 %d 个顶点，总计 %d 个顶点",
            n_left,
            n_right,
            n_total,
        )
        logger.info(
            "输出模式: %s",
            "左右脑合并平均（单个值）" if combine_hemispheres else "左右脑分别平均（两个值）"
        )

    @property
    def n_vertices(self) -> int:
        """返回总顶点数（左脑 + 右脑）"""
        return len(self.mask_left) + len(self.mask_right)

    @property
    def roi_names(self) -> List[str]:
        """返回 ROI 名称列表"""
        if self.combine_hemispheres:
            return ["Language_ROI"]
        else:
            return ["Language_ROI_Left", "Language_ROI_Right"]

    def project(self, brain_data: np.ndarray) -> np.ndarray:
        """
        将顶点级别的脑响应映射到语言 ROI 平均值。

        Args:
            brain_data: 形状为 (timepoints, n_vertices) 的数组
                       其中 n_vertices = n_left_vertices + n_right_vertices
                       前 n_left_vertices 列是左脑，后 n_right_vertices 列是右脑

        Returns:
            如果 combine_hemispheres=True:
                形状为 (timepoints, 1) 的数组，包含左右脑 ROI 的平均值
            如果 combine_hemispheres=False:
                形状为 (timepoints, 2) 的数组，第一列是左脑 ROI 平均值，第二列是右脑 ROI 平均值
        """
        if brain_data.ndim != 2:
            raise ValueError(
                f"brain_data 应为 2D 数组，当前维度 {brain_data.ndim}"
            )

        n_left_vertices = len(self.mask_left)
        n_right_vertices = len(self.mask_right)
        expected_vertices = n_left_vertices + n_right_vertices

        if brain_data.shape[1] != expected_vertices:
            raise ValueError(
                "脑数据顶点数与 mask 不匹配："
                f"brain_data.shape[1]={brain_data.shape[1]}, 期望 {expected_vertices} "
                f"(左脑 {n_left_vertices} + 右脑 {n_right_vertices})"
            )

        # 分离左右脑数据
        lh_data = brain_data[:, :n_left_vertices]
        rh_data = brain_data[:, n_left_vertices:]

        # 提取 ROI 内的顶点
        lh_roi_data = lh_data[:, self.mask_left]
        rh_roi_data = rh_data[:, self.mask_right]

        # 计算平均值
        if lh_roi_data.size == 0:
            logger.warning("左脑 ROI 中没有顶点，使用 NaN")
            lh_mean = np.full((brain_data.shape[0], 1), np.nan)
        else:
            lh_mean = self._reducer(lh_roi_data, axis=1, keepdims=True)

        if rh_roi_data.size == 0:
            logger.warning("右脑 ROI 中没有顶点，使用 NaN")
            rh_mean = np.full((brain_data.shape[0], 1), np.nan)
        else:
            rh_mean = self._reducer(rh_roi_data, axis=1, keepdims=True)

        # 根据 combine_hemispheres 决定输出
        if self.combine_hemispheres:
            # 合并左右脑的平均值
            combined_mean = self._reducer(
                np.concatenate([lh_mean, rh_mean], axis=1), axis=1, keepdims=True
            )
            return combined_mean
        else:
            # 分别返回左右脑的平均值
            return np.concatenate([lh_mean, rh_mean], axis=1)

    def get_roi_vertex_indices(self) -> dict:
        """
        获取 ROI 内的顶点索引（用于调试或可视化）。
        
        Returns:
            包含 'left' 和 'right' 键的字典，每个值是顶点索引数组
        """
        left_indices = np.where(self.mask_left)[0]
        right_indices = np.where(self.mask_right)[0]
        
        # 右脑的索引需要加上左脑的顶点数
        right_indices_global = right_indices + len(self.mask_left)
        
        return {
            "left": {
                "local_indices": left_indices.tolist(),
                "global_indices": left_indices.tolist(),
                "count": int(len(left_indices)),
            },
            "right": {
                "local_indices": right_indices.tolist(),
                "global_indices": right_indices_global.tolist(),
                "count": int(len(right_indices)),
            },
        }

