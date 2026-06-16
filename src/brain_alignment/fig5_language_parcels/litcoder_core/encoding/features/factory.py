from typing import Dict, Any, Union, Optional, Tuple
import numpy as np
from datetime import datetime
from .base import BaseFeatureExtractor
from .language_model import LanguageModelFeatureExtractor
from .speech_model import SpeechFeatureExtractor
from .simple_features import WordRateFeatureExtractor
from .embeddings import StaticEmbeddingFeatureExtractor
from .sae_model import SAEModelFeatureExtractor
from .saeact_model import SAEActModelFeatureExtractor
from .bert_model import BertModelFeatureExtractor
from ..utils import ActivationCache, SpeechActivationCache


class FeatureExtractorFactory:
    """Factory class for creating feature extractors with caching support."""

    _extractors = {
        "language_model": LanguageModelFeatureExtractor,
        "speech": SpeechFeatureExtractor,
        "wordrate": WordRateFeatureExtractor,
        "embeddings": StaticEmbeddingFeatureExtractor,
        "sae_model": SAEModelFeatureExtractor,
        "saeact_model": SAEActModelFeatureExtractor,
        "bert_model": BertModelFeatureExtractor,
    }

    @classmethod
    def create_extractor(
        cls,
        modality: str,
        model_name: str,
        config: Dict[str, Any],
        cache_dir: str = "cache",
    ) -> BaseFeatureExtractor:
        """Create a feature extractor based on modality and model name.

        Args:
            modality: The type of feature extractor ('language_model', 'speech', 'wordrate', 'embeddings', 'sae_model')
            model_name: The specific model name (e.g., 'gpt2-small', 'word2vec', 'openai/whisper-tiny')
            config: Configuration dictionary for the extractor
            cache_dir: Directory for caching

        Returns:
            BaseFeatureExtractor: The appropriate feature extractor instance

        Raises:
            ValueError: If modality is not supported
        """
        if modality not in cls._extractors:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Supported modalities: {list(cls._extractors.keys())}"
            )

        extractor_class = cls._extractors[modality]

        # Add model_name to config if not present
        if "model_name" not in config:
            config["model_name"] = model_name

        # TODO: Change later to use **config for all extractors. But for now, only speech will use **config
        # ideally, they should all use a config, and that config should be a class.
        if modality == "language_model":
            extractor = extractor_class(config)
        elif modality == "speech":
            extractor = extractor_class(**config)
        elif modality == "sae_model":
            extractor = extractor_class(config)
        elif modality == "saeact_model":
            extractor = extractor_class(config)
        elif modality == "bert_model":
            extractor = extractor_class(config)
        else:
            extractor = extractor_class(config)

        print(f"this is the config: {config}")

        # Add caching capability
        if modality in ["language_model", "speech", "sae_model", "saeact_model", "bert_model"]:
            extractor.cache_dir = cache_dir
            if modality == "speech":
                extractor.speech_cache = SpeechActivationCache(cache_dir=cache_dir)
            else:
                extractor.activation_cache = ActivationCache(cache_dir=cache_dir)

        return extractor

    @classmethod
    def extract_features_with_caching(
        cls,
        extractor: BaseFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int = 9,
        lookback: int = 256,
        dataset_type: str = "narratives",
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Extract features with caching support.

        Args:
            extractor: The feature extractor instance
            assembly: The assembly containing data
            story: Story name
            idx: Story index
            layer_idx: Layer index for multi-layer extractors
            lookback: Number of tokens to look back (for language models)
            dataset_type: Type of dataset (e.g., 'narratives', 'lebel', etc.)

        Returns:
            Features array, or (features, times) tuple for speech
        """
        modality = cls._get_modality_from_extractor(extractor)

        if modality == "language_model":
            return cls._extract_language_model_features(
                extractor, assembly, story, idx, layer_idx, lookback, dataset_type
            )
        elif modality == "speech":
            return cls._extract_speech_features(
                extractor, assembly, story, idx, layer_idx, dataset_type
            )
        elif modality == "wordrate":
            word_rates = assembly.get_word_rates()[idx]
            return extractor.extract_features(word_rates)
        elif modality == "embeddings":
            words = assembly.get_words()[idx]
            return extractor.extract_features(words)
        elif modality == "sae_model":
            return cls._extract_sae_model_features(
                extractor, assembly, story, idx, layer_idx, lookback, dataset_type
            )
        elif modality == "saeact_model":
            return cls._extract_sae_model_features(
                extractor, assembly, story, idx, layer_idx, lookback, dataset_type
            )
        elif modality == "bert_model":
            return cls._extract_bert_model_features(
                extractor, assembly, story, idx, layer_idx, lookback, dataset_type
            )
        else:
            raise ValueError(f"Unknown modality: {modality}")

    @classmethod
    def _get_modality_from_extractor(cls, extractor: BaseFeatureExtractor) -> str:
        """Get modality from extractor instance."""
        if isinstance(extractor, LanguageModelFeatureExtractor):
            return "language_model"
        elif isinstance(extractor, SpeechFeatureExtractor):
            return "speech"
        elif isinstance(extractor, WordRateFeatureExtractor):
            return "wordrate"
        elif isinstance(extractor, StaticEmbeddingFeatureExtractor):
            return "embeddings"
        elif isinstance(extractor, SAEModelFeatureExtractor):
            return "sae_model"
        elif isinstance(extractor, SAEActModelFeatureExtractor):
            return "saeact_model"
        elif isinstance(extractor, BertModelFeatureExtractor):
            return "bert_model"
        else:
            raise ValueError(f"Unknown extractor type: {type(extractor)}")

    @classmethod
    def _extract_language_model_features(
        cls,
        extractor: LanguageModelFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int,
        lookback: int = 256,
        dataset_type: str = "narratives",
    ) -> np.ndarray:
        """Extract language model features with caching."""
        texts = assembly.get_stimuli()[idx]

        # Try to load cached activations
        cache_key = extractor.activation_cache._get_cache_key(
            story=story,
            lookback=lookback,  # You can make this configurable
            model_name=extractor.model_name,
            context_type=getattr(extractor, "context_type", "fullcontext"),
            last_token=getattr(extractor, "last_token", False),
            dataset_type=dataset_type,
            raw=True,
        )
        print(f"this is the last token: {getattr(extractor, 'last_token', False)}")
        print(f"this is the lookback: {lookback}")
        print(f'this is the layer: {layer_idx}')

        lazy_cache = extractor.activation_cache.load_multi_layer_activations(cache_key)

        if lazy_cache is not None:
            return lazy_cache.get_layer(layer_idx)
        else:
            # Compute and cache features
            all_features = extractor.extract_all_layers(texts)

            # Create metadata for caching
            metadata = {
                "model_name": extractor.model_name,
                "story": story,
                "lookback": lookback,
                "context_type": getattr(extractor, "context_type", "fullcontext"),
                "hook_type": extractor.hook_type,
                "last_token": getattr(extractor, "last_token", False),
                "dataset_type": dataset_type,
                "available_layers": list(all_features.keys()),
                "created_at": datetime.now().isoformat(),
            }

            # Save to cache
            extractor.activation_cache.save_multi_layer_activations(
                cache_key, all_features, metadata
            )

            return all_features[layer_idx]

    @classmethod
    def _extract_bert_model_features(
        cls,
        extractor: BertModelFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int,
        lookback: int = 512,
        dataset_type: str = "narratives",
    ) -> np.ndarray:
        """Extract BERT model features with caching."""
        texts = assembly.get_stimuli()[idx]

        # Try to load cached activations
        cache_key = extractor.activation_cache._get_cache_key(
            story=story,
            lookback=lookback,
            model_name=extractor.model_name,
            context_type=getattr(extractor, "context_type", "fullcontext"),
            last_token=getattr(extractor, "last_token", False),
            dataset_type=dataset_type,
            raw=True,
        )
        print(f"this is the last token: {getattr(extractor, 'last_token', False)}")
        print(f"this is the lookback: {lookback}")
        print(f'this is the layer: {layer_idx}')

        lazy_cache = extractor.activation_cache.load_multi_layer_activations(cache_key)

        if lazy_cache is not None:
            return lazy_cache.get_layer(layer_idx)
        else:
            # Compute and cache features
            all_features = extractor.extract_all_layers(texts)

            # Create metadata for caching
            metadata = {
                "model_name": extractor.model_name,
                "story": story,
                "lookback": lookback,
                "context_type": getattr(extractor, "context_type", "fullcontext"),
                "last_token": getattr(extractor, "last_token", False),
                "dataset_type": dataset_type,
                "available_layers": list(all_features.keys()),
                "created_at": datetime.now().isoformat(),
            }

            # Save to cache
            extractor.activation_cache.save_multi_layer_activations(
                cache_key, all_features, metadata
            )

            return all_features[layer_idx]

    @classmethod
    def _extract_speech_features(
        cls,
        extractor: SpeechFeatureExtractor,
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int,
        dataset_type: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract speech features with caching."""
        wav_path = assembly.get_audio_path()[idx]

        # Try to load from cache
        cache_key = extractor.speech_cache.get_cache_key(
            audio_id=wav_path,
            model_name=extractor.model_name,
            chunk_size=extractor.chunk_size,
            context_size=extractor.context_size,
            pool=extractor.pool,
            target_sample_rate=extractor.target_sample_rate,
            dataset_type=dataset_type,
            extra={"layer_mode": "all"},
        )

        lazy = extractor.speech_cache.load_multi_layer_activations(cache_key)

        if lazy is not None:
            # Validate cached data
            lazy.validate_params(
                expected={
                    "model_name": extractor.model_name,
                    "chunk_size": extractor.chunk_size,
                    "context_size": extractor.context_size,
                    "pool": extractor.pool,
                    "target_sample_rate": extractor.target_sample_rate,
                    "dataset_type": dataset_type,
                }
            )
            features = lazy.get_layer(layer_idx)
            times = lazy.get_times()
        else:
            # Compute and cache features
            layer_to_feats, times = extractor.extract_all_layers(wav_path)
            if len(layer_to_feats) == 0:
                raise RuntimeError(
                    "extract_all_layers returned no layers (audio too short?)."
                )

            # Save to cache
            metadata = {
                "modality": "speech",
                "audio_id": wav_path,
                "model_name": extractor.model_name,
                "chunk_size": extractor.chunk_size,
                "context_size": extractor.context_size,
                "pool": extractor.pool,
                "target_sample_rate": extractor.target_sample_rate,
                "dataset_type": dataset_type,
                "available_layers": sorted(layer_to_feats.keys()),
            }

            extractor.speech_cache.save_multi_layer_activations(
                cache_key,
                all_layer_activations=layer_to_feats,
                metadata=metadata,
                times=times,
            )

            features = layer_to_feats[layer_idx]

        return features, times

    @classmethod
    def _extract_sae_model_features(
        cls,
        extractor: Union[SAEModelFeatureExtractor, SAEActModelFeatureExtractor],
        assembly: Any,
        story: str,
        idx: int,
        layer_idx: int,
        lookback: int = 256,
        dataset_type: str = "narratives",
    ) -> np.ndarray:
        """Extract SAE model features with caching support.
        
        Note: For sae_model, layer_idx is interpreted as parcel_id.
        If layer_idx is provided, it will be used to select a specific parcel.
        Otherwise, uses the parcel_id(s) from initialization.
        If multiple parcel_ids are provided in initialization, features will be concatenated.
        """
        import hashlib
        import json
        
        texts = assembly.get_stimuli()[idx]
        
        # 检查是否有多个 parcel（通过 extractor.parcel_names）
        has_multiple_parcels = len(extractor.parcel_names) > 1
        
        # Determine which parcel to extract
        # If layer_idx is provided and is a valid parcel ID, use it
        # Otherwise, use the parcel_id from initialization
        parcel_id = None
        if layer_idx is not None and layer_idx >= 0:
            # Try to interpret layer_idx as parcel_id
            parcel_id = layer_idx
        
        # Determine modality based on extractor type
        if isinstance(extractor, SAEActModelFeatureExtractor):
            modality_str = "saeact_model"
        else:
            modality_str = "sae_model"

        # 统一 cache key：同一 story 下按配置参数缓存“所有 parcel”
        cache_params = {
            "modality": modality_str,
            "story": story,
            "model_name": extractor.model_name,
            "parcel_mapping_path": extractor.parcel_mapping_path,
            "last_token": extractor.last_token,
            "dataset_type": dataset_type,
            "sae_release": extractor.sae_release,
        }
        cache_key = hashlib.md5(
            json.dumps(cache_params, sort_keys=True).encode()
        ).hexdigest()

        # 多 parcel：优先从缓存逐个加载后拼接；缺失时回退实时提取并写缓存
        if has_multiple_parcels:
            requested_parcel_ids = []
            for parcel_name in extractor.parcel_names:
                try:
                    requested_parcel_ids.append(int(str(parcel_name).split("_")[-1]))
                except (TypeError, ValueError, IndexError) as ex:
                    raise ValueError(f"无法解析 parcel 名称为 ID: {parcel_name}") from ex

            lazy_cache = extractor.activation_cache.load_multi_layer_activations(cache_key)
            if lazy_cache is not None:
                try:
                    available_parcels = set(int(p) for p in lazy_cache.get_available_layers())
                    missing = [pid for pid in requested_parcel_ids if pid not in available_parcels]
                    if not missing:
                        cached_features = [lazy_cache.get_layer(pid) for pid in requested_parcel_ids]
                        print(
                            f"[INFO] 多 Parcel 模式命中缓存，加载并拼接 {len(requested_parcel_ids)} 个 Parcel 特征"
                        )
                        return np.concatenate(cached_features, axis=1)
                    print(
                        f"[INFO] 多 Parcel 缓存不完整，缺失 {len(missing)} 个 Parcel（示例: {missing[:10]}），将重新计算并更新缓存"
                    )
                except Exception as ex:
                    print(f"[WARN] 读取多 Parcel 缓存失败，将回退到实时提取: {ex}")

            print(
                f"[INFO] 多 Parcel 模式缓存未命中，实时提取全部 Parcel 后按请求顺序拼接 {len(requested_parcel_ids)} 个 Parcel 特征"
            )
            all_parcel_features = extractor.extract_all_parcels(texts)
            concatenated_source = []
            for pid in requested_parcel_ids:
                if pid in all_parcel_features:
                    concatenated_source.append(all_parcel_features[pid])
                else:
                    raise RuntimeError(f"实时提取结果缺少请求的 parcel_id={pid}，无法拼接")
            concatenated_features = np.concatenate(concatenated_source, axis=1)
            metadata = {
                "modality": modality_str,
                "model_name": extractor.model_name,
                "story": story,
                "parcel_mapping_path": extractor.parcel_mapping_path,
                "last_token": extractor.last_token,
                "dataset_type": dataset_type,
                "sae_release": extractor.sae_release,
                "available_parcels": sorted([str(k) for k in all_parcel_features.keys()]),
                "created_at": datetime.now().isoformat(),
            }
            all_parcel_features_int_keys = {}
            for k, v in all_parcel_features.items():
                if isinstance(k, int):
                    all_parcel_features_int_keys[k] = v
                else:
                    try:
                        parcel_id_int = int(str(k).split("_")[-1])
                        all_parcel_features_int_keys[parcel_id_int] = v
                    except (ValueError, IndexError) as ex:
                        print(f"[WARN] 无法将 Parcel 键 '{k}' 解析为整数，使用原始键: {ex}")
                        all_parcel_features_int_keys[k] = v
            extractor.activation_cache.save_multi_layer_activations(
                cache_key,
                all_parcel_features_int_keys,
                metadata,
                shard_layers=True,
            )
            return concatenated_features

        # 单个 parcel：使用缓存机制
        # Try to load cached activations
        lazy_cache = extractor.activation_cache.load_multi_layer_activations(cache_key)
        
        if lazy_cache is not None:
            # Cache exists, get the requested parcel
            # Use parcel_id if provided, otherwise use the first parcel from initialization
            if parcel_id is not None:
                try:
                    return lazy_cache.get_layer(parcel_id)
                except KeyError as ex:
                    import traceback
                    print(f"[WARN] Parcel {parcel_id} not found in cache, using first parcel")
                    print(f"[WARN] 异常类型: {type(ex).__name__}")
                    print(f"[WARN] 完整 traceback:")
                    traceback.print_exc()
                    available_parcels = lazy_cache.get_available_layers()
                    if available_parcels:
                        return lazy_cache.get_layer(available_parcels[0])
                    else:
                        raise RuntimeError(f"缓存中没有可用的 Parcel，无法继续") from ex
            else:
                # Use the parcel_id from initialization
                if extractor.parcel_names:
                    # Try to get the first parcel ID
                    first_parcel_name = extractor.parcel_names[0]
                    try:
                        parcel_id_from_name = int(first_parcel_name.split('_')[-1])
                        return lazy_cache.get_layer(parcel_id_from_name)
                    except (ValueError, IndexError, KeyError) as ex:
                        import traceback
                        print(f"[WARN] 无法从 Parcel 名称 '{first_parcel_name}' 获取 ID 或缓存中不存在，使用第一个可用 Parcel")
                        print(f"[WARN] 异常类型: {type(ex).__name__}")
                        print(f"[WARN] 完整 traceback:")
                        traceback.print_exc()
                        # Fallback to first available parcel
                        available_parcels = lazy_cache.get_available_layers()
                        if available_parcels:
                            return lazy_cache.get_layer(available_parcels[0])
                        else:
                            raise RuntimeError(f"缓存中没有可用的 Parcel，无法继续") from ex
        else:
            # Cache doesn't exist, compute and cache all parcels
            print(f"[INFO] Computing and caching all parcels for story: {story}")
            all_parcel_features = extractor.extract_all_parcels(texts)
            
            # Create metadata for caching
            # Determine modality based on extractor type
            if isinstance(extractor, SAEActModelFeatureExtractor):
                modality_str = "saeact_model"
            else:
                modality_str = "sae_model"
            
            metadata = {
                "modality": modality_str,
                "model_name": extractor.model_name,
                "story": story,
                "parcel_mapping_path": extractor.parcel_mapping_path,
                "last_token": extractor.last_token,
                "dataset_type": dataset_type,
                "sae_release": extractor.sae_release,
                "available_parcels": sorted([str(k) for k in all_parcel_features.keys()]),
                "created_at": datetime.now().isoformat(),
            }
            
            # Save to cache
            # Convert parcel IDs to int keys for consistency with layer-based caching
            all_parcel_features_int_keys = {}
            for k, v in all_parcel_features.items():
                if isinstance(k, int):
                    all_parcel_features_int_keys[k] = v
                else:
                    # Try to extract int from string like "parcel_61"
                    try:
                        parcel_id_int = int(str(k).split('_')[-1])
                        all_parcel_features_int_keys[parcel_id_int] = v
                    except (ValueError, IndexError) as ex:
                        # Keep as string key if can't parse (non-critical error)
                        print(f"[WARN] 无法将 Parcel 键 '{k}' 解析为整数，使用原始键: {ex}")
                        all_parcel_features_int_keys[k] = v
            
            extractor.activation_cache.save_multi_layer_activations(
                cache_key,
                all_parcel_features_int_keys,
                metadata,
                shard_layers=True,
            )
            
            # Return the requested parcel
            if parcel_id is not None:
                if parcel_id in all_parcel_features_int_keys:
                    return all_parcel_features_int_keys[parcel_id]
                else:
                    # Fallback to first available parcel
                    if all_parcel_features_int_keys:
                        first_key = sorted(all_parcel_features_int_keys.keys())[0]
                        return all_parcel_features_int_keys[first_key]
            else:
                # Use the parcel_id from initialization
                if extractor.parcel_names:
                    first_parcel_name = extractor.parcel_names[0]
                    try:
                        parcel_id_from_name = int(first_parcel_name.split('_')[-1])
                        if parcel_id_from_name in all_parcel_features_int_keys:
                            return all_parcel_features_int_keys[parcel_id_from_name]
                    except (ValueError, IndexError) as ex:
                        # Non-critical error, just log and continue
                        print(f"[WARN] 无法从 Parcel 名称 '{first_parcel_name}' 解析 ID: {ex}")
                        pass
                
                # Fallback to first available parcel
                if all_parcel_features_int_keys:
                    first_key = sorted(all_parcel_features_int_keys.keys())[0]
                    return all_parcel_features_int_keys[first_key]
        
        # If we get here, something went wrong
        raise RuntimeError("Failed to extract SAE model features")

    @classmethod
    def get_supported_modalities(cls) -> list:
        """Get list of supported modalities."""
        return list(cls._extractors.keys())

    @classmethod
    def register_extractor(cls, modality: str, extractor_class: type):
        """Register a new feature extractor class.

        Args:
            modality: The modality name
            extractor_class: The extractor class to register
        """
        cls._extractors[modality] = extractor_class
