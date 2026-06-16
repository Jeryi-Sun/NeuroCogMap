from typing import Any, Dict, List, Union, Optional
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from .base import BaseFeatureExtractor


class BertModelFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor that uses BERT model to extract embeddings from text.

    This extractor supports BERT models and can extract features
    from either the [CLS] token or average across all tokens. It supports
    multi-layer extraction with lazy loading.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the BERT model feature extractor.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model_name (str): Name of the BERT model to use (e.g., 'bert-base-uncased')
                - layer_idx (int): Index of the layer to extract features from (for backward compatibility)
                - last_token (bool): Whether to use only the [CLS] token's features (default: True)
                - device (str): Device to run the model on ('cuda' or 'cpu')
                - lookback (int): Maximum number of tokens to process (default: 512, BERT's max)
                - context_type (str): Type of context to use (fullcontext, nocontext, halfcontext)
        """
        super().__init__(config)
        self.model_name = config["model_name"]
        self.layer_idx = config.get("layer_idx", -1)  # For backward compatibility
        self.last_token = config.get("last_token", True)
        self.lookback = config.get("lookback", 512)  # BERT's max sequence length
        self.context_type = config.get("context_type", "fullcontext")

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # 延迟加载模型：只有在真正需要计算、且未命中 ActivationCache 时才加载
        self.model: Optional[BertModel] = None
        self.tokenizer: Optional[BertTokenizer] = None

    def _ensure_model_loaded(self) -> None:
        """确保 BERT 模型和 tokenizer 已经加载（延迟加载）。"""
        if self.model is not None and self.tokenizer is not None:
            return
        print(f"Loading BERT model: {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def extract_features(
        self, stimuli: Union[str, List[str]], layer_idx: Optional[int] = None, **kwargs
    ) -> np.ndarray:
        """Extract features from the input stimuli using a for loop.

        Args:
            stimuli (Union[str, List[str]]): Input text or list of texts
            layer_idx (Optional[int]): Specific layer to extract from. If None, uses self.layer_idx
            **kwargs: Additional arguments for feature extraction

        Returns:
            np.ndarray: Extracted features
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        if isinstance(stimuli, str):
            stimuli = [stimuli]

        # Process each stimulus individually
        all_features = []
        print(f"Processing {len(stimuli)} texts one at a time...")

        for i, text in enumerate(stimuli):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(stimuli)}")

            # Extract features for the current text
            features = self._extract_single_features(text, layer_idx)
            all_features.append(features)

        # Stack all features
        return np.vstack(all_features)

    def extract_all_layers(
        self, stimuli: Union[str, List[str]], **kwargs
    ) -> Dict[int, np.ndarray]:
        """Extract features from all layers for the input stimuli.

        Args:
            stimuli (Union[str, List[str]]): Input text or list of texts
            **kwargs: Additional arguments for feature extraction

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping layer indices to features
        """
        if isinstance(stimuli, str):
            stimuli = [stimuli]

        # Process each stimulus individually
        all_layer_features = {}
        print(f"Processing {len(stimuli)} texts for all layers...")

        for i, text in enumerate(stimuli):
            if i % 10 == 0:
                print(f"Processing text {i+1}/{len(stimuli)}")

            # Extract all layers for the current text
            layer_features = self._extract_single_text_all_layers(text)

            # Accumulate features across texts
            for layer_idx, features in layer_features.items():
                if layer_idx not in all_layer_features:
                    all_layer_features[layer_idx] = []
                all_layer_features[layer_idx].append(features)

        # Stack features for each layer
        for layer_idx in all_layer_features:
            all_layer_features[layer_idx] = np.vstack(all_layer_features[layer_idx])

        return all_layer_features

    def _extract_single_features(self, text: str, layer_idx: int) -> np.ndarray:
        """Extract features from a single text for a specific layer.

        Args:
            text (str): Input text
            layer_idx (int): Layer index to extract from (0-11 for BERT-base, -1 for pooler output)

        Returns:
            np.ndarray: Extracted features for the text
        """
        # 确保模型已加载（只有在真正需要计算时才会触发）
        self._ensure_model_loaded()

        # if the text is '' then return zeros
        if text == "":
            hidden_size = self.model.config.hidden_size
            return np.zeros((1, hidden_size), dtype=np.float32)

        with torch.no_grad():
            # Tokenize text
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.lookback,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Get outputs from all layers
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Extract features from the specified layer
            if layer_idx == -1:
                # Use pooler output (CLS token after final layer)
                features = outputs.pooler_output
            elif 0 <= layer_idx < len(outputs.hidden_states):
                # Use hidden states from the specified layer
                # hidden_states[0] is embedding layer, hidden_states[1:] are transformer layers
                if layer_idx == 0:
                    # Embedding layer
                    features = outputs.hidden_states[0]
                else:
                    # Transformer layer (layer_idx 1-12 corresponds to layers 0-11)
                    features = outputs.hidden_states[layer_idx]
            else:
                raise ValueError(
                    f"Invalid layer_idx {layer_idx}. BERT-base has 12 transformer layers (1-12), "
                    f"layer 0 is embedding layer, or use -1 for pooler output."
                )

            # Handle last_token (CLS token) or average across tokens
            if self.last_token:
                # Get the [CLS] token's features (first token)
                token_features = features[0, 0].unsqueeze(0)  # [1, hidden_size]
            else:
                # Average across all tokens (excluding padding)
                # Use attention_mask to exclude padding tokens
                mask = attention_mask[0].unsqueeze(-1).float()  # [seq_len, 1]
                masked_features = features[0] * mask  # [seq_len, hidden_size]
                sum_features = masked_features.sum(dim=0)  # [hidden_size]
                count = mask.sum()  # scalar
                token_features = (sum_features / count).unsqueeze(0)  # [1, hidden_size]

            # Convert to numpy array
            return token_features.cpu().numpy().astype(np.float32)

    def _extract_single_text_all_layers(self, text: str) -> Dict[int, np.ndarray]:
        """Extract features from all layers for a single text.

        Args:
            text (str): Input text

        Returns:
            Dict[int, np.ndarray]: Dictionary mapping layer indices to features
        """
        # 确保模型已加载（只有在真正需要计算时才会触发）
        self._ensure_model_loaded()

        # if the text is '' then return zeros for all layers
        if text == "":
            hidden_size = self.model.config.hidden_size
            empty_features = np.zeros((1, hidden_size), dtype=np.float32)
            # BERT-base has 12 transformer layers (0-11) plus embedding layer (0) and pooler (-1)
            all_layer_features = {}
            all_layer_features[0] = empty_features  # Embedding layer
            for i in range(1, 13):  # Transformer layers 1-12
                all_layer_features[i] = empty_features
            all_layer_features[-1] = empty_features  # Pooler output
            return all_layer_features

        with torch.no_grad():
            # Tokenize text
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.lookback,
                padding=True,
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Get outputs from all layers
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Extract features from all layers
            all_layer_features = {}
            # Layer 0: embedding layer
            features_0 = outputs.hidden_states[0]
            if self.last_token:
                token_features_0 = features_0[0, 0].unsqueeze(0)
            else:
                mask = attention_mask[0].unsqueeze(-1).float()
                masked_features = features_0[0] * mask
                sum_features = masked_features.sum(dim=0)
                count = mask.sum()
                token_features_0 = (sum_features / count).unsqueeze(0)
            all_layer_features[0] = token_features_0.cpu().numpy().astype(np.float32)

            # Layers 1-12: transformer layers
            for layer_idx in range(1, 13):
                features = outputs.hidden_states[layer_idx]
                if self.last_token:
                    token_features = features[0, 0].unsqueeze(0)
                else:
                    mask = attention_mask[0].unsqueeze(-1).float()
                    masked_features = features[0] * mask
                    sum_features = masked_features.sum(dim=0)
                    count = mask.sum()
                    token_features = (sum_features / count).unsqueeze(0)
                all_layer_features[layer_idx] = (
                    token_features.cpu().numpy().astype(np.float32)
                )

            # Layer -1: pooler output
            if outputs.pooler_output is not None:
                pooler_features = outputs.pooler_output[0].unsqueeze(0)
                all_layer_features[-1] = (
                    pooler_features.cpu().numpy().astype(np.float32)
                )

            return all_layer_features

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_params = ["model_name"]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")

        if "layer_idx" in self.config:
            if not isinstance(self.config["layer_idx"], int):
                raise ValueError("layer_idx must be an integer")

        if "device" in self.config:
            if self.config["device"] not in ["cuda", "cpu", "mps"]:
                raise ValueError("device must be either 'cuda', 'cpu', or 'mps'")

        if "context_type" in self.config:
            valid_context_types = ["fullcontext", "nocontext", "halfcontext"]
            if self.config["context_type"] not in valid_context_types:
                raise ValueError(f"context_type must be one of {valid_context_types}")
