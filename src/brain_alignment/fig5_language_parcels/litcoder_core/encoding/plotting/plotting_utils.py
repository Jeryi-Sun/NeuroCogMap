from abc import ABC, abstractmethod
import logging
import numpy as np
from nilearn import plotting, datasets
from nilearn.plotting.cm import cold_hot
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any
import io
from PIL import Image

from encoding.brain_projection.vertix2parcel import VertexToParcelMapper
from encoding.brain_projection.vertex_to_roi import VertexToROIMapper
from encoding.brain_projection.vertex_to_7net import VertexToSevenNetworkMapper

logger = logging.getLogger(__name__)


class Logger(ABC):
    """Abstract base class for different logging backends."""

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        """Log a matplotlib figure as an image."""
        pass

    @abstractmethod
    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        """Log a histogram of values."""
        pass


class WandBLogger(Logger):
    """Weights & Biases logger implementation."""

    def __init__(self):
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        log_dict = {name: value}
        if step is not None:
            log_dict["step"] = step
        self.wandb.log(log_dict)

    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        log_dict = {name: self.wandb.Image(figure)}
        if step is not None:
            log_dict["step"] = step
        self.wandb.log(log_dict)

    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        log_dict = {name: self.wandb.Histogram(values)}
        if step is not None:
            log_dict["step"] = step
        self.wandb.log(log_dict)


class TensorBoardLogger(Logger):
    """TensorBoard logger implementation."""

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError(
                "tensorboard not installed. Install with: pip install tensorboard torch"
            )

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)

    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        # Convert matplotlib figure to PIL Image, then to tensor
        buf = io.BytesIO()
        figure.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        pil_image = Image.open(buf)

        # Convert PIL to numpy array (H, W, C)
        img_array = np.array(pil_image)
        if len(img_array.shape) == 3:
            # Convert from HWC to CHW format for tensorboard
            img_array = img_array.transpose(2, 0, 1)

        self.writer.add_image(name, img_array, step, dataformats="CHW")
        buf.close()

    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        self.writer.add_histogram(name, values, step)

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class NullLogger(Logger):
    """Null logger that does nothing - used when logging is disabled."""

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Do nothing."""
        pass

    def log_image(
        self, name: str, figure: plt.Figure, step: Optional[int] = None
    ) -> None:
        """Do nothing."""
        pass

    def log_histogram(
        self, name: str, values: np.ndarray, step: Optional[int] = None
    ) -> None:
        """Do nothing."""
        pass


class BrainPlotter:
    """A class to handle brain surface visualization and correlation plots."""

    FSAVERAGE5_VERTEX_COUNT = 2 * 10242

    def __init__(
        self,
        logger: Logger,
        vertex_to_parcel_mapper: Optional[VertexToParcelMapper] = None,
        vertex_to_seven_network_mapper: Optional[VertexToSevenNetworkMapper] = None,
        vertex_to_roi_mapper: Optional[VertexToROIMapper] = None,
    ):
        """Initialize with a specific logger backend.

        Args:
            logger: Logger instance (WandBLogger, TensorBoardLogger, etc.)
            vertex_to_parcel_mapper: Optional mapper for parcel-level projections
            vertex_to_roi_mapper: Optional mapper for ROI-level projections
        """
        self.logger = logger
        self.vertex_to_parcel_mapper = vertex_to_parcel_mapper
        self.vertex_to_seven_network_mapper = vertex_to_seven_network_mapper
        self.vertex_to_roi_mapper = vertex_to_roi_mapper

    @staticmethod
    def plot_surface_correlations(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Significant Prediction Correlations",
        only_significant: bool = True,
        is_volume: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Plot correlations on brain surface with ONE shared colorbar on the right.
        """
        if is_volume:
            print("Skipping surface plotting for volume data")
            return None

        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        N = 10242

        # Apply mask if requested
        masked_correlations = correlations.astype(float).copy()
        if only_significant:
            masked_correlations[~significant_mask.astype(bool)] = np.nan

        # Split hemispheres
        left_correlations = masked_correlations[:N]
        right_correlations = masked_correlations[N : 2 * N]

        # Symmetric color scale across panes
        vmax = np.nanmax(np.abs(masked_correlations))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        norm = Normalize(vmin=-vmax, vmax=vmax)
        cmap = cold_hot

        fig = plt.figure(figsize=(15, 10))

        # Left Lateral
        ax1 = fig.add_subplot(231, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            left_correlations,
            hemi="left",
            view="lateral",
            colorbar=False,
            axes=ax1,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Left Lateral",
        )

        # Left Medial
        ax2 = fig.add_subplot(232, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_left"],
            left_correlations,
            hemi="left",
            view="medial",
            colorbar=False,
            axes=ax2,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Left Medial",
        )

        # Right Lateral
        ax3 = fig.add_subplot(234, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            right_correlations,
            hemi="right",
            view="lateral",
            colorbar=False,
            axes=ax3,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Right Lateral",
        )

        # Right Medial
        ax4 = fig.add_subplot(235, projection="3d")
        plotting.plot_surf_stat_map(
            fsaverage["infl_right"],
            right_correlations,
            hemi="right",
            view="medial",
            colorbar=False,
            axes=ax4,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            title="Right Medial",
        )

        # One shared colorbar on the right margin
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(sm, cax=cax)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.97])
        return fig

    @staticmethod
    def plot_all_correlations_histogram(
        correlations: np.ndarray, title: str = "All Correlations Distribution"
    ) -> plt.Figure:
        """Plot histogram of all correlations."""
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        valid_correlations = correlations[~np.isnan(correlations)]
        sns.histplot(
            valid_correlations,
            bins=100,
            color="blue",
            label="All",
            kde=True,
            stat="density",
        )
        plt.legend()
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        return fig

    @staticmethod
    def plot_significant_correlations_histogram(
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        title: str = "Significant Correlations Distribution",
    ) -> plt.Figure:
        """Plot histogram of significant correlations."""
        fig = plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        sig_correlations = correlations[significant_mask]
        valid_sig_correlations = sig_correlations[~np.isnan(sig_correlations)]

        sns.histplot(
            valid_sig_correlations,
            bins=100,
            color="green",
            label="Significant",
            kde=True,
            stat="density",
        )
        plt.legend()
        plt.xlabel("Correlation")
        plt.ylabel("Density")
        plt.title(title)
        return fig

    def log_plots(
        self,
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        prefix: str = "",
        step: Optional[int] = None,
        is_volume: bool = False,
        language_mask: Optional[np.ndarray] = None,
        roi_masks: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        支持输入 fsaverage5 顶点长度 (20484)、Schaefer parcel 长度或 ROI 长度（需初始化映射器）。
        Log brain surface plots and correlation histograms using the configured logger.

        Args:
            correlations: correlations array - can be:
                - (20484,) for fsaverage5 surface (L then R)
                - (n_parcels,) for parcel-level (will be expanded if mapper available)
                - (1,) or (2,) for ROI-level (will be logged as scalars)
            significant_mask: boolean mask of same length as correlations
            prefix: optional namespace for log keys
            step: optional step number for logging
            is_volume: skip surface plotting if True
            language_mask: optional boolean mask for language network
            roi_masks: optional dict[str, np.ndarray] of additional ROI masks
        """

        def _sanitize(name: str) -> str:
            return "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name.strip()
            ).lower()

        N = 10242
        full_len = 2 * N

        # 检查是否为 ROI 级别数据
        is_roi_data = self._is_roi_data(correlations)
        
        if is_roi_data:
            # ROI 级别数据：直接记录标量值
            self._log_roi_plots(correlations, significant_mask, prefix, step)
            return

        # Sanity checks for vertex/parcel data
        correlations = self._ensure_vertex_length(
            correlations, is_mask=False, context="correlations", is_volume=is_volume
        )
        significant_mask = self._ensure_vertex_length(
            significant_mask,
            is_mask=True,
            context="significant_mask",
            is_volume=is_volume,
        ).astype(bool)

        # All correlations histogram
        fig_all = self.plot_all_correlations_histogram(
            correlations, title="All Correlations Distribution"
        )
        self.logger.log_image(f"{prefix}correlation_histogram_all", fig_all, step)
        plt.close(fig_all)

        valid_correlations = correlations[~np.isnan(correlations)]
        self.logger.log_histogram(
            f"{prefix}correlation_histogram_data_all", valid_correlations, step
        )

        # Surface plots (if not volume)
        if not is_volume:
            fig_significant = self.plot_surface_correlations(
                correlations,
                significant_mask,
                title="Significant Prediction Correlations",
                only_significant=True,
                is_volume=is_volume,
            )
            if fig_significant is not None:
                self.logger.log_image(
                    f"{prefix}brain_surface_significant", fig_significant, step
                )
                plt.close(fig_significant)

            fig_all_surface = self.plot_surface_correlations(
                correlations,
                significant_mask,
                title="All Prediction Correlations",
                only_significant=False,
                is_volume=is_volume,
            )
            if fig_all_surface is not None:
                self.logger.log_image(
                    f"{prefix}brain_surface_all", fig_all_surface, step
                )
                plt.close(fig_all_surface)

        # Significant correlations histogram
        fig_sig = self.plot_significant_correlations_histogram(
            correlations,
            significant_mask,
            title="Significant Correlations Distribution",
        )
        self.logger.log_image(
            f"{prefix}correlation_histogram_significant", fig_sig, step
        )
        plt.close(fig_sig)

        sig_correlations = correlations[significant_mask]
        valid_sig_correlations = sig_correlations[~np.isnan(sig_correlations)]
        self.logger.log_histogram(
            f"{prefix}correlation_histogram_data_significant",
            valid_sig_correlations,
            step,
        )

        # Language network analysis
        if language_mask is not None:
            language_mask = self._ensure_vertex_length(
                language_mask,
                is_mask=True,
                context="language_mask",
                is_volume=is_volume,
            ).astype(bool)

            lang_vals = correlations[language_mask]
            mean_v = float(np.nanmean(lang_vals)) if lang_vals.size else np.nan
            median_v = float(np.nanmedian(lang_vals)) if lang_vals.size else np.nan

            self.logger.log_scalar(f"{prefix}lanA_mean", mean_v, step)
            self.logger.log_scalar(f"{prefix}lanA_median", median_v, step)

            clean = lang_vals[~np.isnan(lang_vals)]
            if clean.size:
                self.logger.log_histogram(f"{prefix}lanA_hist", clean, step)

            if not is_volume:
                fig_lang = self.plot_surface_correlations(
                    correlations=correlations,
                    significant_mask=language_mask,
                    title="Language Network — Masked",
                    only_significant=True,
                    is_volume=is_volume,
                )
                if fig_lang is not None:
                    self.logger.log_image(f"{prefix}lanA_surface", fig_lang, step)
                    plt.close(fig_lang)

        # ROI analysis
        if roi_masks:
            if not isinstance(roi_masks, dict):
                raise TypeError(
                    "`roi_masks` must be a dict like {'V1': mask, 'AC1': mask, ...}"
                )

            for name, mask in roi_masks.items():
                arr = self._ensure_vertex_length(
                    mask,
                    is_mask=True,
                    context=f"roi_mask[{name}]",
                    is_volume=is_volume,
                ).astype(bool)

                key = _sanitize(name)
                roi_vals = correlations[arr]
                mean_v = float(np.nanmean(roi_vals)) if roi_vals.size else np.nan
                median_v = float(np.nanmedian(roi_vals)) if roi_vals.size else np.nan

                self.logger.log_scalar(f"{prefix}{key}_mean", mean_v, step)
                self.logger.log_scalar(f"{prefix}{key}_median", median_v, step)

                clean = roi_vals[~np.isnan(roi_vals)]
                if clean.size:
                    self.logger.log_histogram(f"{prefix}{key}_hist", clean, step)

                if not is_volume:
                    fig_roi = self.plot_surface_correlations(
                        correlations=correlations,
                        significant_mask=arr,
                        title=f"{name} — Masked",
                        only_significant=True,
                        is_volume=is_volume,
                    )
                    if fig_roi is not None:
                        self.logger.log_image(f"{prefix}{key}_surface", fig_roi, step)
                        plt.close(fig_roi)

    def _ensure_vertex_length(
        self,
        values: np.ndarray,
        *,
        is_mask: bool,
        context: str,
        is_volume: bool,
    ) -> np.ndarray:
        """确保输入为 fsaverage5 顶点长度，如是 parcel 则自动展开。"""
        arr = np.asarray(values)
        if is_volume:
            return arr

        if self.vertex_to_parcel_mapper is not None:
            expected_vertices = self.vertex_to_parcel_mapper.n_vertices
        elif self.vertex_to_seven_network_mapper is not None:
            expected_vertices = self.vertex_to_seven_network_mapper.n_vertices
        else:
            expected_vertices = self.FSAVERAGE5_VERTEX_COUNT

        parcel_count = (
            len(self.vertex_to_parcel_mapper.parcel_names)
            if self.vertex_to_parcel_mapper is not None
            else None
        )
        seven_net_count = (
            len(self.vertex_to_seven_network_mapper.network_names)
            if self.vertex_to_seven_network_mapper is not None
            else None
        )

        if arr.ndim != 1:
            raise ValueError(f"{context} 必须是一维数组，当前维度 {arr.ndim}")

        if arr.shape[0] == expected_vertices:
            return arr

        if (
            self.vertex_to_parcel_mapper is not None
            and arr.shape[0] == len(self.vertex_to_parcel_mapper.parcel_names)
        ):
            fill_value = False if is_mask else np.nan
            expanded = self.vertex_to_parcel_mapper.parcel_to_vertex(
                arr, fill_value=fill_value
            )
            return expanded.astype(arr.dtype, copy=False)

        if (
            self.vertex_to_seven_network_mapper is not None
            and arr.shape[0] == len(self.vertex_to_seven_network_mapper.network_names)
        ):
            fill_value = False if is_mask else np.nan
            expanded = self.vertex_to_seven_network_mapper.network_to_vertex(
                arr, fill_value=fill_value
            )
            return expanded.astype(arr.dtype, copy=False)

        raise ValueError(
            f"{context} 长度为 {arr.shape[0]}，既非顶点数 {expected_vertices}"
            + (f" 也非 parcel 数 {parcel_count}" if parcel_count is not None else "")
            + (f" 也非 7Networks 数 {seven_net_count}" if seven_net_count is not None else "")
        )

    def _is_roi_data(self, values: np.ndarray) -> bool:
        """检查是否为 ROI 级别数据（1 或 2 个值）"""
        arr = np.asarray(values)
        if arr.ndim != 1:
            return False
        
        # 如果启用了 ROI mapper，且长度匹配，则认为是 ROI 数据
        if self.vertex_to_roi_mapper is not None:
            expected_roi_len = len(self.vertex_to_roi_mapper.roi_names)
            if arr.shape[0] == expected_roi_len:
                return True
        
        # 如果没有 ROI mapper，但长度是 1 或 2，且不是顶点/parcel 长度，也可能是 ROI 数据
        # 但为了避免误判，只在有 ROI mapper 时才认为是 ROI 数据
        return False

    def _log_roi_plots(
        self,
        correlations: np.ndarray,
        significant_mask: np.ndarray,
        prefix: str = "",
        step: Optional[int] = None,
    ):
        """记录 ROI 级别的绘图（标量值和简单图表）"""
        arr = np.asarray(correlations)
        sig_mask = np.asarray(significant_mask).astype(bool)
        
        if self.vertex_to_roi_mapper is None:
            logger.warning("ROI mapper 未初始化，无法记录 ROI 级别的绘图")
            return
        
        roi_names = self.vertex_to_roi_mapper.roi_names
        
        # 记录每个 ROI 的标量值
        for i, roi_name in enumerate(roi_names):
            if i < arr.shape[0]:
                corr_val = float(arr[i]) if not np.isnan(arr[i]) else np.nan
                sig_val = bool(sig_mask[i]) if i < sig_mask.shape[0] else False
                
                self.logger.log_scalar(f"{prefix}{roi_name}_correlation", corr_val, step)
                self.logger.log_scalar(f"{prefix}{roi_name}_significant", float(sig_val), step)
        
        # 如果只有一个 ROI，记录为单个值
        if len(roi_names) == 1:
            self.logger.log_scalar(f"{prefix}roi_correlation", float(arr[0]) if not np.isnan(arr[0]) else np.nan, step)
        
        # 绘制简单的柱状图
        if arr.shape[0] <= 2:
            fig = plt.figure(figsize=(8, 6))
            x_pos = np.arange(len(roi_names))
            colors = ['green' if (i < sig_mask.shape[0] and sig_mask[i]) else 'blue' 
                     for i in range(len(roi_names))]
            
            bars = plt.bar(x_pos, arr, color=colors, alpha=0.7)
            plt.xlabel('ROI')
            plt.ylabel('Correlation')
            plt.title('ROI Correlations')
            plt.xticks(x_pos, roi_names, rotation=45, ha='right')
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            self.logger.log_image(f"{prefix}roi_correlations_bar", fig, step)
            plt.close(fig)
        
        # 记录所有 ROI 值的直方图（如果有多个值）
        if arr.shape[0] > 1:
            valid_vals = arr[~np.isnan(arr)]
            if valid_vals.size > 0:
                self.logger.log_histogram(f"{prefix}roi_correlations_hist", valid_vals, step)
