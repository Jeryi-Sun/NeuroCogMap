
"""
Abstract trainer that accepts components as dependencies.
"""

import logging
from typing import Dict, List, Union, Any, Optional
import numpy as np
from datetime import datetime
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import fdrcorrection

from encoding.utils import ModelSaver, zs
from encoding.brain_projection.vertix2parcel import VertexToParcelMapper
from encoding.brain_projection.vertex_to_7net import VertexToSevenNetworkMapper
from encoding.brain_projection.vertex_to_roi import VertexToROIMapper
from encoding.features.FIR_expander import FIR
from encoding.plotting.plotting_utils import BrainPlotter, TensorBoardLogger, WandBLogger, NullLogger

logger = logging.getLogger(__name__)


class AbstractTrainer:
    """
    A completely abstract trainer that accepts all components as dependencies.
    
    This trainer doesn't know about datasets, assemblies, or specific feature types.
    It just orchestrates the pipeline: extract → downsample → FIR → trim → train.
    """
    
    def __init__(
        self,
        assembly: Any,                    # Data source
        feature_extractors: List[Any],    # List of feature extractors
        downsampler: Any,                 # Downsampling component
        model: Any,                       # Training model
        fir_delays: List[int],            # FIR delay parameters
        trimming_config: Dict,            # Trimming configuration
        use_train_test_split: bool = False,  # Data structuring mode
        # Feature extraction parameters
        layer_idx: int = 9,
        lookback: int = 256,
        dataset_type: str = "unknown",
        # Logging parameters
        logger_backend: str = "wandb",
        wandb_project_name: str = "abstract-trainer",
        results_dir: str = "results",
        run_name: Optional[str] = None,
        # Processing parameters
        downsample_config: Optional[Dict] = None,
        story_selection: Optional[List[str]] = None,
        reverse_order: bool = False,
        forward_fill_zeros: bool = False,
        parcel_projection_config: Optional[Dict[str, Any]] = None,
        seven_network_projection_config: Optional[Dict[str, Any]] = None,
        roi_projection_config: Optional[Dict[str, Any]] = None,
        test_story_from_end: int = 1,
        eval_story_from_end: Optional[int] = None,
        two_fold_last_story_eval_test: bool = False,
        # K-fold cross-validation parameters (for Narratives dataset)
        n_outer_folds: int = 5,
        n_inner_folds: int = 5,
        folding_type: str = "kfold",
        chunk_length: int = 20,
    ):
        """
        Initialize with all components as dependencies.
        
        Args:
            assembly: Data assembly (has .stories, .get_brain_data(), etc.)
            feature_extractors: List of feature extraction components
            downsampler: Downsampling component
            model: Model with fit_predict() method
            fir_delays: List of FIR delays to apply
            trimming_config: Dict specifying how to trim data
            use_train_test_split: Whether to use train/test split vs concatenation
            layer_idx: Layer index for feature extraction
            lookback: Context lookback for feature extraction
            dataset_type: Dataset type for caching
            logger_backend: "wandb", "tensorboard", or "none"
            wandb_project_name: Project name for wandb
            results_dir: Directory for results
            run_name: Custom run name
            downsample_config: Downsampling parameters
            story_selection: Specific stories to process (None = all)
            forward_fill_zeros: If True, forward fill zero rows with previous row values
            n_outer_folds: Number of outer CV folds (for k-fold cross-validation)
            n_inner_folds: Number of inner CV folds (for nested CV)
            folding_type: Type of CV folding: "kfold", "chunked", "chunked_contiguous", "timeseries", "group"
            chunk_length: Length of chunks for respecting fMRI autocorrelation
        """
        self.assembly = assembly
        self.feature_extractors = feature_extractors
        self.downsampler = downsampler
        self.model = model
        self.fir_delays = fir_delays
        self.trimming_config = trimming_config
        self.use_train_test_split = use_train_test_split
        self.downsample_config = downsample_config or {}
        self.reverse_order = reverse_order
        self.forward_fill_zeros = forward_fill_zeros
        self.parcel_projection_config = parcel_projection_config
        self.seven_network_projection_config = seven_network_projection_config
        self.roi_projection_config = roi_projection_config

        projection_count = sum(
            config is not None
            for config in (
                self.parcel_projection_config,
                self.seven_network_projection_config,
                self.roi_projection_config,
            )
        )
        if projection_count > 1:
            raise ValueError(
                "parcel_projection、seven_network_projection、roi_projection 目前互斥，请只启用一种。"
            )

        self.vertex_to_parcel_mapper: Optional[VertexToParcelMapper] = None
        if parcel_projection_config is not None:
            try:
                self.vertex_to_parcel_mapper = VertexToParcelMapper(**parcel_projection_config)
                logger.info(
                    "Parcel projection已启用，共 %d 个 parcels",
                    len(self.vertex_to_parcel_mapper.parcel_names),
                )
            except Exception as exc:
                logger.exception("初始化 VertexToParcelMapper 失败")
                raise exc

        self.vertex_to_seven_network_mapper: Optional[VertexToSevenNetworkMapper] = None
        if seven_network_projection_config is not None:
            try:
                self.vertex_to_seven_network_mapper = VertexToSevenNetworkMapper(
                    **seven_network_projection_config
                )
                logger.info(
                    "7Networks projection已启用，共 %d 个网络",
                    len(self.vertex_to_seven_network_mapper.network_names),
                )
            except Exception as exc:
                logger.exception("初始化 VertexToSevenNetworkMapper 失败")
                raise exc
        
        self.vertex_to_roi_mapper: Optional[VertexToROIMapper] = None
        if roi_projection_config is not None:
            try:
                self.vertex_to_roi_mapper = VertexToROIMapper(**roi_projection_config)
                logger.info(
                    "ROI projection已启用，输出维度: %s",
                    self.vertex_to_roi_mapper.roi_names,
                )
            except Exception as exc:
                logger.exception("初始化 VertexToROIMapper 失败")
                raise exc
        
        # K-fold cross-validation parameters
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.folding_type = folding_type
        self.chunk_length = chunk_length
        
        # Feature extraction parameters
        self.layer_idx = layer_idx
        self.lookback = lookback
        self.dataset_type = dataset_type
        if test_story_from_end < 1:
            raise ValueError("test_story_from_end 必须>=1")
        self.test_story_from_end = test_story_from_end
        if eval_story_from_end is not None and eval_story_from_end < 1:
            raise ValueError("eval_story_from_end 必须>=1")
        self.eval_story_from_end = eval_story_from_end
        self.two_fold_last_story_eval_test = bool(two_fold_last_story_eval_test)
        if self.two_fold_last_story_eval_test and not self.use_train_test_split:
            raise ValueError("two_fold_last_story_eval_test=True 需要 use_train_test_split=True")
        
        # Story selection
        if story_selection is None:
            self.stories_to_process = self.assembly.stories
        elif isinstance(story_selection, int):
            # Single story index (1-based)
            self.stories_to_process = [self.assembly.stories[story_selection - 1]]
        else:
            # List of story names
            self.stories_to_process = story_selection
        
        # Setup logging
        self.setup_logger(logger_backend, wandb_project_name, results_dir, run_name)
        self.model_saver = ModelSaver(base_dir=results_dir)
        self.brain_plotter = BrainPlotter(
            self.experiment_logger,
            vertex_to_parcel_mapper=self.vertex_to_parcel_mapper,
            vertex_to_seven_network_mapper=self.vertex_to_seven_network_mapper,
            vertex_to_roi_mapper=self.vertex_to_roi_mapper,
        )
        
        logger.info(f"Abstract trainer initialized")
        logger.info(f"Feature extractors: {len(self.feature_extractors)}")
        logger.info(f"Stories to process: {len(self.stories_to_process)}")
        logger.info(f"Layer idx: {self.layer_idx}")
        logger.info(f"Lookback: {self.lookback}")
        logger.info(f"Dataset type: {self.dataset_type}")
        logger.info(f"FIR delays: {self.fir_delays}")
        logger.info(f"Use train/test split: {self.use_train_test_split}")
        if not self.use_train_test_split:
            logger.info(f"K-fold CV: outer_folds={self.n_outer_folds}, inner_folds={self.n_inner_folds}, folding_type={self.folding_type}")
    
    def setup_logger(self, backend: str, project_name: str, results_dir: str, run_name: Optional[str]):
        """Setup experiment logger."""
        if run_name is None:
            run_name = f"abstract-trainer-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        if backend == "wandb":
            try:
                import wandb
                wandb.init(project=project_name, name=run_name)
                self.experiment_logger = WandBLogger()
            except ImportError as e:
                raise ImportError("wandb not installed. Install with: pip install wandb") from e
        elif backend == "tensorboard":
            run_dir = f"{results_dir}/runs/{run_name}"
            self.experiment_logger = TensorBoardLogger(log_dir=run_dir)
        elif backend == "none":
            self.experiment_logger = NullLogger()
        else:
            raise ValueError(f"Unsupported logger_backend '{backend}'. Use 'wandb', 'tensorboard', or 'none'.")
    
    def extract_and_downsample_features(self, reverse_order: bool = False) -> Dict[str, np.ndarray]:
        """Extract and downsample features for all stories."""
        all_features = {}
        
        for story in self.stories_to_process[::-1] if reverse_order else self.stories_to_process:
            idx = self.assembly.stories.index(story)
            story_features = []
            # Extract features from each extractor
            for extractor in self.feature_extractors:
                features = self._extract_single_features(extractor, story, idx)
                
                # Apply forward fill if enabled (before downsampling)
                if self.forward_fill_zeros:
                    # Handle tuple return (speech features with times)
                    if isinstance(features, tuple):
                        features_data, times = features
                        features_data = self._forward_fill_zeros(features_data)
                        features = (features_data, times)
                    else:
                        features = self._forward_fill_zeros(features)
                
                # Downsample if needed
                if self._should_downsample(extractor):
                    downsampled = self._downsample_features(features, idx)
                else:
                    downsampled = features
                
                story_features.append(downsampled)
            
            # Concatenate features from multiple extractors
            if len(story_features) > 1:
                # Align timepoints
                min_length = min(feat.shape[0] for feat in story_features)
                story_features = [feat[:min_length] for feat in story_features]
                combined = np.concatenate(story_features, axis=1)
            else:
                combined = story_features[0]
            
            all_features[story] = combined
            logger.info(f"Story {story}: feature shape {combined.shape}")
        
        return all_features
    
    def _extract_single_features(self, extractor, story: str, idx: int):
        """Extract features from a single extractor."""
        from encoding.features.factory import FeatureExtractorFactory
        
        # Use the factory's caching method
        return FeatureExtractorFactory.extract_features_with_caching(
            extractor, self.assembly, story, idx, self.layer_idx, self.lookback, self.dataset_type
        )
    
    def _should_downsample(self, extractor) -> bool:
        """Determine if this extractor needs downsampling."""
        # Simple heuristic: wordrate doesn't need downsampling
        extractor_name = extractor.__class__.__name__.lower()
        return 'wordrate' not in extractor_name
    
    def _downsample_features(self, features, story_idx: int):
        """Downsample features for a story."""
        if isinstance(features, tuple):
            # Speech features
            features, times = features
            tr_times = self.assembly.get_tr_times()[story_idx]
            split_indices = self.assembly.get_split_indices()[story_idx]
            
            return self.downsampler.downsample(
                data=features,
                data_times=times,
                tr_times=tr_times,
                split_indices=split_indices,
                **self.downsample_config
            )
        else:
            # Text-based features
            split_indices = self.assembly.get_split_indices()[story_idx]
            data_times = self.assembly.get_data_times()[story_idx]
            tr_times = self.assembly.get_tr_times()[story_idx]
            
            return self.downsampler.downsample(
                data=features,
                data_times=data_times,
                tr_times=tr_times,
                split_indices=split_indices,
                **self.downsample_config
            )
    
    def _forward_fill_zeros(self, features: np.ndarray) -> np.ndarray:
        """Forward fill zero rows with previous row values.
        
        从前到后扫描，如果某一行为全0，则用前面一行的值补上。
        如果前面一行也是全0，则继续向前查找第一个非0行。
        
        Args:
            features: Input feature array of shape (n_timepoints, n_features)
            
        Returns:
            Feature array with zero rows forward-filled
        """
        if features.size == 0:
            return features
        
        # Create a copy to avoid modifying the original
        filled_features = features.copy()
        
        # Forward fill: for each row, if it's all zeros, use the previous row
        for i in range(len(filled_features)):
            if np.all(filled_features[i] == 0):
                if i == 0:
                    # First row is all zeros, keep it as zeros (no previous row)
                    logger.warning("First row is all zeros, keeping it as zeros (no previous row to fill from)")
                else:
                    # Use the previous row (which may have been filled already)
                    filled_features[i] = filled_features[i - 1]
        
        return filled_features
    
    def apply_fir_delays(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply FIR delays to features."""
        delayed_features = {}
        for story, feat in features.items():
            delayed_features[story] = FIR.make_delayed(feat, self.fir_delays)
            logger.info(f"Story {story}: delayed feature shape {delayed_features[story].shape}")
        return delayed_features
    
    def structure_data(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Structure data according to training paradigm."""
        vertex_brain_data = {}
        for story in self.stories_to_process:
            idx = self.assembly.stories.index(story)
            vertex_brain_data[story] = self.assembly.get_brain_data()[idx]

        if self.vertex_to_parcel_mapper is not None:
            brain_data = self._maybe_project_brain_data(vertex_brain_data)
        elif self.vertex_to_seven_network_mapper is not None:
            brain_data = self._maybe_project_brain_data_to_7net(vertex_brain_data)
        elif self.vertex_to_roi_mapper is not None:
            brain_data = self._maybe_project_brain_data_to_roi(vertex_brain_data)
        else:
            brain_data = vertex_brain_data
        
        if self.use_train_test_split:
            return self._create_train_test_split(features, brain_data)
        else:
            return self._create_concatenated_data(features, brain_data)
    
    def _create_train_test_split(self, features: Dict, brain_data: Dict) -> Dict[str, np.ndarray]:
        """Create train/test split (Lebel style)."""
        stories = list(features.keys())
        if self.test_story_from_end > len(stories):
            raise ValueError(
                f"test_story_from_end={self.test_story_from_end} 超过可用故事数量 {len(stories)}"
            )
        print(f"train/test split stories: {stories}")
        split_idx = len(stories) - self.test_story_from_end
        test_stories = [stories[split_idx]]
        train_stories_full = stories[:split_idx] + stories[split_idx + 1 :]
        if self.two_fold_last_story_eval_test:
            if self.eval_story_from_end is not None:
                raise ValueError(
                    "two_fold_last_story_eval_test 模式下不应同时设置 eval_story_from_end"
                )
            if len(train_stories_full) == 0:
                raise ValueError(
                    "two_fold_last_story_eval_test 需要至少 2 个 story（其余 story 作为训练集）"
                )
            holdout_story = test_stories[0]
            print(f"train stories: {train_stories_full}")
            print(f"two-fold holdout story: {holdout_story}")

            train_feat_start = self.trimming_config.get("train_features_start", 0)
            train_feat_end = self.trimming_config.get("train_features_end", None)
            train_targ_start = self.trimming_config.get("train_targets_start", 0)
            train_targ_end = self.trimming_config.get("train_targets_end", None)

            X_train = np.nan_to_num(np.vstack([
                zs(features[story][train_feat_start:train_feat_end])
                for story in train_stories_full
            ]))
            Y_train = np.vstack([
                zs(brain_data[story][train_targ_start:train_targ_end])
                for story in train_stories_full
            ])

            test_feat_start = self.trimming_config.get("test_features_start", 0)
            test_feat_end = self.trimming_config.get("test_features_end", None)
            test_targ_start = self.trimming_config.get("test_targets_start", 0)
            test_targ_end = self.trimming_config.get("test_targets_end", None)

            holdout_x = np.nan_to_num(zs(features[holdout_story][test_feat_start:test_feat_end]))
            holdout_y = zs(brain_data[holdout_story][test_targ_start:test_targ_end])
            if holdout_x.shape[0] != holdout_y.shape[0]:
                raise ValueError(
                    f"holdout story 时间维不一致: X={holdout_x.shape[0]}, Y={holdout_y.shape[0]}"
                )
            n_tp = holdout_x.shape[0]
            if n_tp < 2:
                raise ValueError(
                    f"holdout story 时间点不足以做 2 折切分: n_timepoints={n_tp}"
                )
            mid = n_tp // 2
            if mid == 0 or mid == n_tp:
                raise ValueError(
                    f"holdout story 对半切分失败: n_timepoints={n_tp}, mid={mid}"
                )

            logger.info(
                "Two-fold holdout split enabled. holdout_story=%s, n_timepoints=%d, split=%d|%d",
                holdout_story,
                n_tp,
                mid,
                n_tp - mid,
            )
            logger.info(f"Train: X{X_train.shape}, Y{Y_train.shape}")

            return {
                "Rstim": X_train,
                "Rresp": Y_train,
                "two_fold_eval_test": True,
                "Pstim_fold1": holdout_x[mid:],
                "Presp_fold1": holdout_y[mid:],
                "Estim_fold1": holdout_x[:mid],
                "Eresp_fold1": holdout_y[:mid],
                "Pstim_fold2": holdout_x[:mid],
                "Presp_fold2": holdout_y[:mid],
                "Estim_fold2": holdout_x[mid:],
                "Eresp_fold2": holdout_y[mid:],
            }
        eval_stories: List[str] = []
        train_stories = train_stories_full
        if self.eval_story_from_end is not None:
            if self.eval_story_from_end > len(train_stories_full):
                raise ValueError(
                    f"eval_story_from_end={self.eval_story_from_end} 超过可用训练故事数量 {len(train_stories_full)}"
                )
            eval_idx = len(train_stories_full) - self.eval_story_from_end
            eval_stories = [train_stories_full[eval_idx]]
            train_stories = train_stories_full[:eval_idx] + train_stories_full[eval_idx + 1 :]
        print(f"train stories: {train_stories}")
        print(f"test stories: {test_stories}")
        if eval_stories:
            print(f"evaluation stories: {eval_stories}")
        # Training data
        train_feat_start = self.trimming_config.get("train_features_start", 0)
        train_feat_end = self.trimming_config.get("train_features_end", None)
        train_targ_start = self.trimming_config.get("train_targets_start", 0)
        train_targ_end = self.trimming_config.get("train_targets_end", None)
        
        X_train = np.nan_to_num(np.vstack([
            zs(features[story][train_feat_start:train_feat_end])
            for story in train_stories
        ]))
        Y_train = np.vstack([
            zs(brain_data[story][train_targ_start:train_targ_end])
            for story in train_stories
        ])
        
        # Test data
        test_feat_start = self.trimming_config.get("test_features_start", 0)
        test_feat_end = self.trimming_config.get("test_features_end", None)
        test_targ_start = self.trimming_config.get("test_targets_start", 0)
        test_targ_end = self.trimming_config.get("test_targets_end", None)
        
        X_test = np.nan_to_num(np.vstack([
            zs(features[story][test_feat_start:test_feat_end])
            for story in test_stories
        ]))
        Y_test = np.vstack([
            zs(brain_data[story][test_targ_start:test_targ_end])
            for story in test_stories
        ])
        
        # 确保 X_test 和 Y_test 的样本数一致，如果不一致则自动对齐
        if X_test.shape[0] != Y_test.shape[0]:
            min_samples = min(X_test.shape[0], Y_test.shape[0])
            if X_test.shape[0] != min_samples:
                logger.warning(
                    f"X_test 样本数 ({X_test.shape[0]}) 与 Y_test ({Y_test.shape[0]}) 不一致，"
                    f"将 X_test 截断到 {min_samples} 个样本"
                )
                X_test = X_test[:min_samples]
            if Y_test.shape[0] != min_samples:
                logger.warning(
                    f"Y_test 样本数 ({Y_test.shape[0]}) 与 X_test ({X_test.shape[0]}) 不一致，"
                    f"将 Y_test 截断到 {min_samples} 个样本"
                )
                Y_test = Y_test[:min_samples]
        
        logger.info(f"Train: X{X_train.shape}, Y{Y_train.shape}")
        logger.info(f"Test: X{X_test.shape}, Y{Y_test.shape}")
        data_dict = {"Rstim": X_train, "Rresp": Y_train, "Pstim": X_test, "Presp": Y_test}
        if eval_stories:
            X_eval = np.nan_to_num(np.vstack([
                zs(features[story][test_feat_start:test_feat_end])
                for story in eval_stories
            ]))
            Y_eval = np.vstack([
                zs(brain_data[story][test_targ_start:test_targ_end])
                for story in eval_stories
            ])
            
            # 确保 X_eval 和 Y_eval 的样本数一致，如果不一致则自动对齐
            if X_eval.shape[0] != Y_eval.shape[0]:
                min_samples = min(X_eval.shape[0], Y_eval.shape[0])
                if X_eval.shape[0] != min_samples:
                    logger.warning(
                        f"X_eval 样本数 ({X_eval.shape[0]}) 与 Y_eval ({Y_eval.shape[0]}) 不一致，"
                        f"将 X_eval 截断到 {min_samples} 个样本"
                    )
                    X_eval = X_eval[:min_samples]
                if Y_eval.shape[0] != min_samples:
                    logger.warning(
                        f"Y_eval 样本数 ({Y_eval.shape[0]}) 与 X_eval ({X_eval.shape[0]}) 不一致，"
                        f"将 Y_eval 截断到 {min_samples} 个样本"
                    )
                    Y_eval = Y_eval[:min_samples]
            logger.info(f"Eval: X{X_eval.shape}, Y{Y_eval.shape}")
            data_dict["Estim"] = X_eval
            data_dict["Eresp"] = Y_eval
        
        return data_dict
    
    def _create_concatenated_data(self, features: Dict, brain_data: Dict) -> Dict[str, np.ndarray]:
        """Create concatenated data (LPP/Narratives style)."""
        story_order = self.stories_to_process
        
        X = np.concatenate([features[story] for story in story_order], axis=0)
        Y = np.concatenate([brain_data[story] for story in story_order], axis=0)
        
        # Apply trimming
        feat_start = self.trimming_config.get("features_start", 0)
        feat_end = self.trimming_config.get("features_end", None)
        targ_start = self.trimming_config.get("targets_start", 0)
        targ_end = self.trimming_config.get("targets_end", None)
        
        X = X[feat_start:feat_end]
        Y = Y[targ_start:targ_end]
        
        logger.info(f"Final: X{X.shape}, Y{Y.shape}")
        
        return {"X": X, "Y": Y}

    def _maybe_project_brain_data(self, brain_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """在需要时将脑数据从顶点级别投影到 parcel 级别。
        
        验证映射后的 parcel 顺序：
        1. 数组列索引从 0 开始（0, 1, 2, ..., n_parcels-1）
        2. 顺序为先左脑（LH）parcels，后右脑（RH）parcels
        3. parcel_names 的顺序与返回数组的列索引一一对应
        """
        if self.vertex_to_parcel_mapper is None:
            return brain_data

        # 验证 parcel 顺序和索引
        parcel_names = self.vertex_to_parcel_mapper.parcel_names
        n_lh = len(self.vertex_to_parcel_mapper.lh_lookup)
        n_rh = len(self.vertex_to_parcel_mapper.rh_lookup)
        n_total = len(parcel_names)
        
        # 验证：总 parcel 数 = LH + RH
        if n_total != n_lh + n_rh:
            raise ValueError(
                f"Parcel 数量不匹配：总数为 {n_total}，但 LH={n_lh}, RH={n_rh}, 总和={n_lh+n_rh}"
            )
        
        # 验证：前 n_lh 个应该是 LH，后 n_rh 个应该是 RH
        lh_names = parcel_names[:n_lh]
        rh_names = parcel_names[n_lh:]
        
        if not all(name.startswith("LH_") for name in lh_names):
            wrong_names = [name for name in lh_names if not name.startswith("LH_")]
            raise ValueError(
                f"前 {n_lh} 个 parcel 中发现了非 LH parcel：{wrong_names[:5]}"
            )
        
        if not all(name.startswith("RH_") for name in rh_names):
            wrong_names = [name for name in rh_names if not name.startswith("RH_")]
            raise ValueError(
                f"后 {n_rh} 个 parcel 中发现了非 RH parcel：{wrong_names[:5]}"
            )
        
        logger.info(
            "✅ Parcel 映射验证通过："
            f"总 parcel 数={n_total} (LH={n_lh}, RH={n_rh}), "
            f"列索引范围=[0, {n_total-1}], "
            f"顺序=先左脑后右脑"
        )
        logger.info(
            f"前3个 LH parcels: {lh_names[:3]}, "
            f"后3个 RH parcels: {rh_names[-3:] if len(rh_names) >= 3 else rh_names}"
        )

        projected = {}
        for story, data in brain_data.items():
            projected_data = self.vertex_to_parcel_mapper.project(data)
            
            # 验证返回数组的列数
            if projected_data.shape[1] != n_total:
                raise ValueError(
                    f"故事 {story}: 投影后的列数 {projected_data.shape[1]} 与 parcel 总数 {n_total} 不匹配"
                )
            
            projected[story] = projected_data
            logger.info(
                "故事 %s: 顶点->parcel 映射完成，形状 %s -> %s",
                story,
                data.shape,
                projected_data.shape,
            )
        
        return projected

    def _maybe_project_brain_data_to_7net(
        self, brain_data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if self.vertex_to_seven_network_mapper is None:
            return brain_data

        network_names = self.vertex_to_seven_network_mapper.network_names
        n_lh = len(self.vertex_to_seven_network_mapper.lh_lookup)
        n_rh = len(self.vertex_to_seven_network_mapper.rh_lookup)
        n_total = len(network_names)

        if n_total != n_lh + n_rh:
            raise ValueError(
                f"7Networks 数量不匹配：总数为 {n_total}，但 LH={n_lh}, RH={n_rh}, 总和={n_lh+n_rh}"
            )

        lh_names = network_names[:n_lh]
        rh_names = network_names[n_lh:]

        if not all(name.startswith("LH_") for name in lh_names):
            wrong_names = [name for name in lh_names if not name.startswith("LH_")]
            raise ValueError(f"LH 网络命名不符合约定：{wrong_names[:5]}")

        if not all(name.startswith("RH_") for name in rh_names):
            wrong_names = [name for name in rh_names if not name.startswith("RH_")]
            raise ValueError(f"RH 网络命名不符合约定：{wrong_names[:5]}")

        projected = {}
        for story, data in brain_data.items():
            projected_data = self.vertex_to_seven_network_mapper.project(data)
            if projected_data.shape[1] != n_total:
                raise ValueError(
                    f"故事 {story}: 7Networks 投影后的列数 {projected_data.shape[1]} 与网络总数 {n_total} 不匹配"
                )
            projected[story] = projected_data
            logger.info(
                "故事 %s: 顶点->7Networks 映射完成，形状 %s -> %s",
                story,
                data.shape,
                projected_data.shape,
            )

        return projected
    
    def _maybe_project_brain_data_to_roi(self, brain_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """在需要时将脑数据从顶点级别投影到语言 ROI 平均值。
        
        根据语言 ROI mask，将对应的顶点取出来取平均作为 Y 值。
        
        Returns:
            投影后的脑数据字典，每个故事的形状为 (timepoints, n_roi)
            其中 n_roi = 1（如果 combine_hemispheres=True）或 2（如果 combine_hemispheres=False）
        """
        if self.vertex_to_roi_mapper is None:
            return brain_data
        
        roi_names = self.vertex_to_roi_mapper.roi_names
        logger.info(
            "✅ ROI 映射验证通过："
            f"ROI 名称={roi_names}, "
            f"输出维度={len(roi_names)}"
        )
        
        projected = {}
        for story, data in brain_data.items():
            projected_data = self.vertex_to_roi_mapper.project(data)
            
            # 验证返回数组的列数
            expected_cols = len(roi_names)
            if projected_data.shape[1] != expected_cols:
                raise ValueError(
                    f"故事 {story}: 投影后的列数 {projected_data.shape[1]} 与 ROI 数量 {expected_cols} 不匹配"
                )
            
            projected[story] = projected_data
            logger.info(
                "故事 %s: 顶点->ROI 映射完成，形状 %s -> %s",
                story,
                data.shape,
                projected_data.shape,
            )
        
        return projected
    
    def train(self, **model_kwargs) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        # Step 1: Extract and downsample features
        features = self.extract_and_downsample_features(self.reverse_order)
        # Step 2: Apply FIR delays
        delayed_features = self.apply_fir_delays(features)
        # Step 3: Structure data for training
        data = self.structure_data(delayed_features)
        # Step 4: Train model
        logger.info("Starting model training...")
        if "Rstim" in data:
            # Train/test split
            if data.get("two_fold_eval_test", False):
                fold_metrics: List[Dict[str, Any]] = []
                fold_weights: List[np.ndarray] = []
                fold_best_alphas: List[np.ndarray] = []
                for fold_idx in (1, 2):
                    logger.info("Running holdout two-fold evaluation: fold %d/2", fold_idx)
                    fold_metric, fold_weight, fold_alpha = self.model.fit_predict(
                        features=data["Rstim"],
                        targets=data["Rresp"],
                        X_test=data[f"Pstim_fold{fold_idx}"],
                        y_test=data[f"Presp_fold{fold_idx}"],
                        X_eval=data[f"Estim_fold{fold_idx}"],
                        y_eval=data[f"Eresp_fold{fold_idx}"],
                        **model_kwargs
                    )
                    fold_metrics.append(fold_metric)
                    fold_weights.append(fold_weight)
                    fold_best_alphas.append(fold_alpha)
                metrics = self._aggregate_two_fold_train_test_metrics(fold_metrics)
                weights = np.mean(np.stack(fold_weights, axis=0), axis=0)
                best_alphas = np.mean(np.stack(fold_best_alphas, axis=0), axis=0)
            else:
                metrics, weights, best_alphas = self.model.fit_predict(
                    features=data["Rstim"],
                    targets=data["Rresp"],
                    X_test=data["Pstim"],
                    y_test=data["Presp"],
                    X_eval=data.get("Estim"),
                    y_eval=data.get("Eresp"),
                    **model_kwargs
                )
        else:
            # Cross-validation (k-fold for Narratives dataset)
            # Merge k-fold parameters with model_kwargs, allowing model_kwargs to override defaults
            cv_kwargs = {
                "n_outer_folds": self.n_outer_folds,
                "n_inner_folds": self.n_inner_folds,
                "folding_type": self.folding_type,
                "chunk_length": self.chunk_length,
            }
            # model_kwargs can override the default CV parameters
            cv_kwargs.update(model_kwargs)
            metrics, weights, best_alphas = self.model.fit_predict(
                features=data["X"],
                targets=data["Y"],
                **cv_kwargs
            )
        
        # Step 5: Log and save results
        self.log_metrics(metrics)
        evaluation_metrics = metrics.get("evaluation_metrics")
        # 说明：
        # - train-test 模式下，如果存在单独的 evaluation 集（valid），metrics 中会带有 "evaluation_metrics"。
        # - full CV（如 narratives 单故事）本身没有单独的 valid 集，这里的 metrics 就是 cross-validated 的性能。
        #   为了兼容之前读取 evaluation_metrics.pkl 的下游代码，我们在 full CV 场景下也把同一份指标存到 evaluation_metrics.pkl。
        # - two_fold_last_story_eval_test 模式下，仅保存两个结果文件：
        #   metrics.pkl = 前一半作为 test 的结果；evaluation_metrics.pkl = 后一半作为 test 的结果。
        metrics_to_save = dict(metrics)
        if "two_fold_details" in metrics:
            fold_details = metrics["two_fold_details"]
            if not isinstance(fold_details, dict) or "fold1" not in fold_details or "fold2" not in fold_details:
                raise ValueError("two_fold_details 格式错误，期望包含 fold1 和 fold2")
            # fold2: 前一半作为 test；fold1: 后一半作为 test
            metrics_to_save = dict(fold_details["fold2"])
            metrics_to_save["half_name"] = "first_half_as_test"
            evaluation_metrics = dict(fold_details["fold1"])
            evaluation_metrics["half_name"] = "second_half_as_test"
        elif evaluation_metrics is None and not self.use_train_test_split:
            evaluation_metrics = dict(metrics)
        if evaluation_metrics is not None:
            logger.info(
                "Evaluation metrics available. median=%.4f, mean=%.4f",
                evaluation_metrics["median_score"],
                evaluation_metrics["mean_score"],
            )
        metrics_to_save.pop("evaluation_metrics", None)
        metrics_to_save.pop("two_fold_details", None)

        # 将 model_kwargs 中可能出现的 numpy 类型转换为 JSON 友好的类型
        safe_model_kwargs = {}
        for k, v in model_kwargs.items():
            try:
                import numpy as _np  # 局部导入以避免循环依赖
            except Exception:
                _np = None

            if _np is not None and isinstance(v, _np.ndarray):
                safe_model_kwargs[k] = v.tolist()
            else:
                safe_model_kwargs[k] = v

        self.save_model(weights, best_alphas, metrics_to_save, safe_model_kwargs, evaluation_metrics)
        
        logger.info(f"Training complete. Median correlation: {metrics['median_score']:.4f}")
        
        return metrics

    def _aggregate_two_fold_train_test_metrics(
        self, fold_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if len(fold_metrics) != 2:
            raise ValueError(f"期望 2 个 fold 的 metrics，实际得到 {len(fold_metrics)}")

        def _aggregate_metric_block(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
            corrs = np.stack([np.asarray(m["correlations"], dtype=float) for m in blocks], axis=0)
            best_alphas = np.stack([np.asarray(m["best_alphas"], dtype=float) for m in blocks], axis=0)
            pvals = np.stack([np.asarray(m["p_values"], dtype=float) for m in blocks], axis=0)
            if corrs.shape[1] == 0:
                raise ValueError("无法聚合空的相关系数数组")

            mean_corrs = np.mean(corrs, axis=0)
            combined_pvals = []
            for voxel_p in pvals.T:
                if np.all(voxel_p == 1.0):
                    combined_pvals.append(1.0)
                else:
                    _, p = combine_pvalues(voxel_p.tolist(), method="fisher")
                    combined_pvals.append(float(p))
            combined_pvals = np.asarray(combined_pvals, dtype=float)
            significant_mask, corrected_pvals = fdrcorrection(combined_pvals, alpha=0.05)
            n_significant = int(np.sum(significant_mask))

            out = {
                "median_score": float(np.median(mean_corrs)),
                "mean_score": float(np.mean(mean_corrs)),
                "std_score": float(np.std(mean_corrs)),
                "min_score": float(np.min(mean_corrs)),
                "max_score": float(np.max(mean_corrs)),
                "best_alphas": np.mean(best_alphas, axis=0).tolist(),
                "correlations": mean_corrs.tolist(),
                "p_values": combined_pvals.tolist(),
                "corrected_p_values": corrected_pvals.tolist(),
                "significant_mask": significant_mask.tolist(),
                "n_significant": n_significant,
                "percent_significant": float(n_significant / mean_corrs.shape[0] * 100),
            }
            if n_significant > 0:
                sig_corr = mean_corrs[significant_mask]
                out.update(
                    {
                        "median_significant_score": float(np.median(sig_corr)),
                        "mean_significant_score": float(np.mean(sig_corr)),
                        "min_significant_score": float(np.min(sig_corr)),
                        "max_significant_score": float(np.max(sig_corr)),
                    }
                )
            return out

        aggregated = _aggregate_metric_block(fold_metrics)
        eval_blocks = [m["evaluation_metrics"] for m in fold_metrics if "evaluation_metrics" in m]
        if len(eval_blocks) == 2:
            aggregated["evaluation_metrics"] = _aggregate_metric_block(eval_blocks)
        aggregated["two_fold_details"] = {
            "fold1": fold_metrics[0],
            "fold2": fold_metrics[1],
        }
        return aggregated
    
    def log_metrics(self, metrics: Dict):
        """Log metrics to configured backend."""
        self.experiment_logger.log_scalar("median_correlation", float(metrics["median_score"]))
        self.experiment_logger.log_scalar("mean_correlation", float(metrics["mean_score"]))
        self.experiment_logger.log_scalar("std_correlation", float(metrics["std_score"]))
        
        if "correlations" in metrics and "significant_mask" in metrics:
            correlations = np.array(metrics["correlations"])
            significant_mask = np.array(metrics["significant_mask"], dtype=bool)
            self.brain_plotter.log_plots(correlations, significant_mask, "", False)
        
        if "best_alpha" in metrics:
            self.experiment_logger.log_scalar("best_alpha", float(metrics["best_alpha"]))
        if "n_significant" in metrics:
            self.experiment_logger.log_scalar("n_significant_voxels", float(metrics["n_significant"]))
    
    def save_model(self, weights, best_alphas, metrics, model_kwargs, evaluation_metrics=None):
        """Save model results."""
        hyperparams = {
            "fir_delays": self.fir_delays,
            "trimming_config": self.trimming_config,
            "use_train_test_split": self.use_train_test_split,
            "downsample_config": self.downsample_config,
            "layer_idx": self.layer_idx,
            "lookback": self.lookback,
            "dataset_type": self.dataset_type,
            "stories_processed": len(self.stories_to_process),
            **model_kwargs
        }
        
        self.model_saver.save_encoding_model(
            weights=weights,
            best_alphas=best_alphas,
            hyperparams=hyperparams,
            metrics=metrics,
            evaluation_metrics=evaluation_metrics,
        )

