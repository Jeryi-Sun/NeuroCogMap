import logging
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class VertexToParcelMapper:
    """将 fsaverage5 顶点级别的数值映射并平均到 Schaefer parcel 级别。"""

    def __init__(
        self,
        lh_annot_path: str,
        rh_annot_path: str,
        drop_label_names: Sequence[str] = ("???", "unknown"),
        use_nanmean: bool = True,
    ) -> None:
        """
        Args:
            lh_annot_path: 左半球 .annot 文件路径
            rh_annot_path: 右半球 .annot 文件路径
            drop_label_names: 不参与平均的 label 名称
            use_nanmean: 是否使用 np.nanmean（否则使用 np.mean）
        """
        self.lh_labels, _, self.lh_names = nib.freesurfer.read_annot(lh_annot_path)
        self.rh_labels, _, self.rh_names = nib.freesurfer.read_annot(rh_annot_path)
        self.drop_label_names = {name.lower() for name in drop_label_names}
        self._reducer = np.nanmean if use_nanmean else np.mean

        self.lh_lookup = self._build_lookup(self.lh_labels, self.lh_names, "LH")
        self.rh_lookup = self._build_lookup(self.rh_labels, self.rh_names, "RH")

        logger.info(
            "Loaded vertex->parcel mapper. LH parcels=%d, RH parcels=%d",
            len(self.lh_lookup),
            len(self.rh_lookup),
        )

    @property
    def parcel_names(self) -> List[str]:
        """返回 LH+RH 的 parcel 名称列表。"""
        return [name for name, _ in self.lh_lookup] + [name for name, _ in self.rh_lookup]

    def project(self, brain_data: np.ndarray) -> np.ndarray:
        """
        将顶点级别的脑响应映射到 parcel 级别。

        Args:
            brain_data: 形状为 (timepoints, n_vertices) 的数组

        Returns:
            形状为 (timepoints, n_parcels) 的数组，顺序为 LH parcels + RH parcels
        """
        if brain_data.ndim != 2:
            raise ValueError(
                f"brain_data 应为 2D 数组，当前维度 {brain_data.ndim}"
            )

        n_lh = len(self.lh_labels)
        n_rh = len(self.rh_labels)
        expected_vertices = n_lh + n_rh

        if brain_data.shape[1] != expected_vertices:
            raise ValueError(
                "脑数据顶点数与注释文件不匹配："
                f"Y.shape[1]={brain_data.shape[1]}, 期望 {expected_vertices}"
            )

        lh_vertices = brain_data[:, :n_lh]
        rh_vertices = brain_data[:, n_lh:]

        lh_projected = self._average_vertices(lh_vertices, self.lh_lookup, "LH")
        rh_projected = self._average_vertices(rh_vertices, self.rh_lookup, "RH")

        return np.concatenate([lh_projected, rh_projected], axis=1)

    @property
    def n_vertices(self) -> int:
        return len(self.lh_labels) + len(self.rh_labels)

    def parcel_to_vertex(
        self,
        parcel_values: np.ndarray,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        将 parcel 级别的数值扩展回顶点 (fsaverage5) 空间，便于可视化。

        Args:
            parcel_values: 形状为 (n_parcels,) 或 (batch, n_parcels) 的数组
            fill_value: 默认填充值（mask 时可设为 False）

        Returns:
            与输入 batch 对应的顶点数组
        """
        values = np.asarray(parcel_values)
        squeeze = False
        if values.ndim == 1:
            values = values[None, :]
            squeeze = True

        total_parcels = len(self.parcel_names)
        if values.shape[1] != total_parcels:
            raise ValueError(
                f"parcel_values 的第二维应为 {total_parcels}，当前 {values.shape[1]}"
            )

        lh_vertices = np.full(
            (values.shape[0], len(self.lh_labels)),
            fill_value,
            dtype=values.dtype,
        )
        rh_vertices = np.full(
            (values.shape[0], len(self.rh_labels)),
            fill_value,
            dtype=values.dtype,
        )

        idx = 0
        for _, vertex_idx in self.lh_lookup:
            lh_vertices[:, vertex_idx] = values[:, idx][:, None]
            idx += 1
        for _, vertex_idx in self.rh_lookup:
            rh_vertices[:, vertex_idx] = values[:, idx][:, None]
            idx += 1

        expanded = np.concatenate([lh_vertices, rh_vertices], axis=1)
        return expanded[0] if squeeze else expanded

    def _build_lookup(
        self,
        labels: np.ndarray,
        names: Sequence[bytes],
        hemisphere_prefix: str,
    ) -> List[Tuple[str, np.ndarray]]:
        lookup: List[Tuple[str, np.ndarray]] = []
        for label_idx, raw_name in enumerate(names):
            name = raw_name.decode("utf-8")
            if name.lower() in self.drop_label_names:
                continue
            mask = np.where(labels == label_idx)[0]
            if mask.size == 0:
                logger.warning(
                    "%s parcel %s (id=%d) 没有对应顶点，已跳过",
                    hemisphere_prefix,
                    name,
                    label_idx,
                )
                continue
            lookup.append((f"{hemisphere_prefix}_{name}", mask))
        if not lookup:
            raise ValueError(f"{hemisphere_prefix} 没有可用的 parcel，检查 annot 文件是否正确")
        return lookup

    def _average_vertices(
        self,
        vertex_data: np.ndarray,
        lookup: List[Tuple[str, np.ndarray]],
        hemisphere: str,
    ) -> np.ndarray:
        parcel_responses = []
        for parcel_name, indices in lookup:
            parcel_values = vertex_data[:, indices]
            if parcel_values.size == 0:
                raise ValueError(f"{hemisphere} parcel {parcel_name} 没有对应的顶点")
            parcel_responses.append(self._reducer(parcel_values, axis=1, keepdims=True))

        return np.hstack(parcel_responses)