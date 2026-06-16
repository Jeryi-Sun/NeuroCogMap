#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态词向量（word2vec/glove）提取器。
基于 litcoder_core 的 StaticEmbeddingFeatureExtractor，实现 neural 提取脚本所需接口。
"""

import os
import sys
from typing import List
import re

import torch
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
LITCODER_CORE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../litcoder_core"))
if LITCODER_CORE_ROOT not in sys.path:
    sys.path.insert(0, LITCODER_CORE_ROOT)

from encoding.features.embeddings import StaticEmbeddingFeatureExtractor


class EmbeddingTokenLimitedExtractor:
    """静态词向量提取器，返回每个实验的一条 pooled 特征。"""

    def __init__(
        self,
        vector_path: str,
        lowercase: bool = True,
        oov_handling: str = "copy_prev",
        pooling: str = "mean",
        activation_token_pattern: str = r"(?:\s<<)|(?:\b0\b)|(?:\b1\b)",
    ):
        if pooling not in {"mean", "max"}:
            raise ValueError(f"pooling 必须是 mean 或 max，当前: {pooling}")
        self.pooling = pooling
        self.activation_token_re = re.compile(activation_token_pattern)
        self.extractor = StaticEmbeddingFeatureExtractor(
            {
                "vector_path": vector_path,
                "lowercase": lowercase,
                "oov_handling": oov_handling,
                "use_tqdm": False,
            }
        )
        self.dim = int(self.extractor.dim)

    def extract_from_experiment(
        self,
        instruction: str,
        experiments: List[str],
        current_experiment_idx: int,
        **kwargs
    ) -> torch.Tensor:
        """
        提取单个实验的 embedding 特征，输出 shape 为 [1, dim]。
        """
        if current_experiment_idx < 0 or current_experiment_idx >= len(experiments):
            raise IndexError(
                f"current_experiment_idx 越界: {current_experiment_idx}, 实验总数: {len(experiments)}"
            )

        text = experiments[current_experiment_idx]
        token_vecs = self.extractor.extract_features(text)  # [N, D]
        activation_count = len(self.activation_token_re.findall(text))

        if token_vecs.size == 0:
            pooled = np.zeros((self.dim,), dtype=np.float32)
        else:
            if self.pooling == "mean":
                pooled = token_vecs.mean(axis=0).astype(np.float32)
            else:
                pooled = token_vecs.max(axis=0).astype(np.float32)

        if activation_count <= 0:
            return torch.empty(0, self.dim, dtype=torch.float16)

        repeated = np.repeat(pooled.reshape(1, -1), activation_count, axis=0)
        return torch.from_numpy(repeated).half()

