import logging
from collections import OrderedDict
from typing import Dict, List, Sequence, Tuple

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


class VertexToSevenNetworkMapper:
    """将 fsaverage5 顶点级别的数据聚合到 Yeo 7Networks 级别。"""

    def __init__(
        self,
        lh_annot_path: str,
        rh_annot_path: str,
        drop_label_names: Sequence[str] = (
            "???",
            "unknown",
            "background+freesurfer_defined_medial",
        ),
        use_nanmean: bool = True,
    ) -> None:
        """
        Args:
            lh_annot_path: 左半球 .annot 文件路径
            rh_annot_path: 右半球 .annot 文件路径
            drop_label_names: 不参与计算的 label 名称
            use_nanmean: 是否在聚合时使用 np.nanmean
        """
        self.lh_labels, _, self.lh_names = nib.freesurfer.read_annot(lh_annot_path)
        self.rh_labels, _, self.rh_names = nib.freesurfer.read_annot(rh_annot_path)
        self.drop_label_names = {name.lower() for name in drop_label_names}
        self._reducer = np.nanmean if use_nanmean else np.mean

        self.lh_lookup = self._build_network_lookup(self.lh_labels, self.lh_names, "LH")
        self.rh_lookup = self._build_network_lookup(self.rh_labels, self.rh_names, "RH")

        logger.info(
            "Loaded vertex->7Networks mapper. LH nets=%d, RH nets=%d",
            len(self.lh_lookup),
            len(self.rh_lookup),
        )

    @property
    def network_names(self) -> List[str]:
        """返回按 LH, RH 顺序排列的 7Networks 名称。"""
        return [name for name, _ in self.lh_lookup] + [name for name, _ in self.rh_lookup]

    def project(self, brain_data: np.ndarray) -> np.ndarray:
        """
        将顶点级别的数据聚合到 7Networks 级别。

        Args:
            brain_data: 形状为 (timepoints, n_vertices) 的数组

        Returns:
            形状为 (timepoints, n_networks) 的数组
        """
        if brain_data.ndim != 2:
            raise ValueError(f"brain_data 应为 2D 数组，当前维度 {brain_data.ndim}")

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

        lh_projected = self._aggregate_networks(lh_vertices, self.lh_lookup, "LH")
        rh_projected = self._aggregate_networks(rh_vertices, self.rh_lookup, "RH")

        return np.concatenate([lh_projected, rh_projected], axis=1)

    @property
    def n_vertices(self) -> int:
        return len(self.lh_labels) + len(self.rh_labels)

    def network_to_vertex(
        self,
        network_values: np.ndarray,
        fill_value: float = np.nan,
    ) -> np.ndarray:
        """
        将 7Networks 级别的数值扩展回 fsaverage5 顶点空间，用于可视化。

        Args:
            network_values: 形状为 (n_networks,) 或 (batch, n_networks) 的数组
            fill_value: 默认填充值（mask 时可设为 False）

        Returns:
            与输入 batch 对应的顶点数组
        """
        values = np.asarray(network_values)
        squeeze = False
        if values.ndim == 1:
            values = values[None, :]
            squeeze = True

        total_networks = len(self.network_names)
        if values.shape[1] != total_networks:
            raise ValueError(
                f"network_values 的第二维应为 {total_networks}，当前 {values.shape[1]}"
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

    def _build_network_lookup(
        self,
        labels: np.ndarray,
        names: Sequence[bytes],
        hemisphere_prefix: str,
    ) -> List[Tuple[str, np.ndarray]]:
        grouped_vertices: "OrderedDict[str, List[np.ndarray]]" = OrderedDict()
        for label_idx, raw_name in enumerate(names):
            name = raw_name.decode("utf-8")
            if name.lower() in self.drop_label_names:
                continue

            base_name = self._extract_network_name(name)
            mask = np.where(labels == label_idx)[0]

            if mask.size == 0:
                logger.warning(
                    "%s network %s (id=%d) 没有对应顶点，已跳过",
                    hemisphere_prefix,
                    base_name,
                    label_idx,
                )
                continue

            key = f"{hemisphere_prefix}_{base_name}"
            grouped_vertices.setdefault(key, []).append(mask)

        if not grouped_vertices:
            raise ValueError(f"{hemisphere_prefix} 没有可用的 7Networks，检查 annot 文件是否正确")

        lookup: List[Tuple[str, np.ndarray]] = []
        for net_name, vertex_lists in grouped_vertices.items():
            lookup.append((net_name, np.concatenate(vertex_lists)))
        return lookup

    def _aggregate_networks(
        self,
        vertex_data: np.ndarray,
        lookup: List[Tuple[str, np.ndarray]],
        hemisphere: str,
    ) -> np.ndarray:
        network_responses = []
        for network_name, indices in lookup:
            network_values = vertex_data[:, indices]
            if network_values.size == 0:
                raise ValueError(f"{hemisphere} network {network_name} 没有对应的顶点")
            network_responses.append(self._reducer(network_values, axis=1, keepdims=True))

        return np.hstack(network_responses)

    @staticmethod
    def _extract_network_name(full_name: str) -> str:
        """
        去掉末尾的 parcel 序号，例如 7Networks_LH_Default_PFC_5 -> 7Networks_LH_Default
        """
        parts = full_name.split("_")
        if len(parts) >= 3 and parts[0].lower() == "7networks":
            return "_".join(parts[:3])
        if parts and parts[-1].isdigit():
            return "_".join(parts[:-1])
        return full_name

