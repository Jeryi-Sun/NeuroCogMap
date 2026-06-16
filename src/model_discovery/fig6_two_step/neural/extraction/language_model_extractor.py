#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Language Model 的 token 限制特征提取器
参考 language_model.py 的实现
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Optional, Dict, Any
from .base_extractor import BaseTokenLimitedExtractor


class LanguageModelTokenLimitedExtractor(BaseTokenLimitedExtractor):
    """基于 HookedTransformer 的 token 限制特征提取器"""
    
    def __init__(
        self,
        model_name: str,
        tokenizer=None,
        layer_idx: int = 0,
        hook_type: str = "hook_resid_pre",
        last_token: bool = True,
        max_tokens: int = 1024,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            tokenizer: tokenizer 对象（如果为 None，则从模型获取）
            layer_idx: 要提取的层索引
            hook_type: hook 类型（默认: "hook_resid_pre"）
            last_token: 是否只使用最后一个 token 的特征（默认: True）
            max_tokens: 最大 token 数限制（默认: 1024）
            device: 设备类型，如果为 None 则自动检测
        """
        # 设备选择
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # 加载模型
        print(f"加载 HookedTransformer 模型: {model_name}")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            n_devices=2 if torch.cuda.device_count() > 1 else 1,
        )
        self.model.eval()
        
        # 获取 tokenizer
        if tokenizer is None:
            tokenizer = self.model.tokenizer
        
        super().__init__(tokenizer, max_tokens)
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.hook_type = hook_type
        self.last_token = last_token
    
    def extract_from_experiment(
        self,
        instruction: str,
        experiments: List[str],
        current_experiment_idx: int,
        **kwargs
    ) -> torch.Tensor:
        """
        从单个实验提取特征
        
        Args:
            instruction: instruction 部分
            experiments: 累积的实验列表
            current_experiment_idx: 当前实验的索引
            **kwargs: 其他参数
            
        Returns:
            提取的特征 tensor [num_activations, feature_dim]
        """
        # 截断到 token 限制
        truncated_text = self.truncate_to_token_limit(
            instruction, experiments, current_experiment_idx
        )
        
        # Tokenize
        tokenized_prompt = self.tokenizer(truncated_text, return_tensors='pt')
        tokenized_prompt.input_ids = tokenized_prompt.input_ids[:, :self.max_tokens]
        input_ids = tokenized_prompt.input_ids.to(self.device)
        
        # 找到激活 token
        relevant_tokens = self.find_activation_tokens(tokenized_prompt.input_ids)
        
        # 找到当前实验在 tokenized 文本中的位置，只保留当前实验部分的激活 token
        current_experiment = experiments[current_experiment_idx]
        start_idx, end_idx = self.get_experiment_token_range(
            truncated_text, current_experiment, tokenized_prompt.input_ids
        )
        
        # 只考虑当前实验部分的激活 token
        experiment_relevant_tokens = torch.zeros(tokenized_prompt.input_ids.shape[1], dtype=torch.bool)
        if start_idx < relevant_tokens.shape[0]:
            experiment_relevant_tokens[start_idx:end_idx] = relevant_tokens[start_idx:end_idx]
        relevant_tokens = experiment_relevant_tokens
        
        if not relevant_tokens.any():
            # 如果没有找到激活 token，返回空 tensor
            return torch.empty(0, self.model.cfg.d_model, dtype=torch.float16)
        
        # 前向传播
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                input_ids, prepend_bos=True, return_type=None
            )
            
            # 获取指定层的特征
            hook_name = f"blocks.{self.layer_idx}.{self.hook_type}"
            features = cache[hook_name]  # [1, seq_len, d_model]
            
            # 提取激活 token 的特征
            activation_features = features[0, relevant_tokens, :]  # [num_activations, d_model]
            
            # 转换为 half precision
            activation_features = activation_features.half()
        
        return activation_features.cpu()
    
    def extract_all_layers_from_experiment(
        self,
        instruction: str,
        experiments: List[str],
        current_experiment_idx: int,
        layers: List[int],
        **kwargs
    ) -> Dict[int, torch.Tensor]:
        """
        从单个实验提取所有指定层的特征
        
        Args:
            instruction: instruction 部分
            experiments: 累积的实验列表
            current_experiment_idx: 当前实验的索引
            layers: 要提取的层列表
            **kwargs: 其他参数
            
        Returns:
            字典，键为层索引，值为特征 tensor [num_activations, feature_dim]
        """
        # 截断到 token 限制
        truncated_text = self.truncate_to_token_limit(
            instruction, experiments, current_experiment_idx
        )
        
        # Tokenize
        tokenized_prompt = self.tokenizer(truncated_text, return_tensors='pt')
        tokenized_prompt.input_ids = tokenized_prompt.input_ids[:, :self.max_tokens]
        input_ids = tokenized_prompt.input_ids.to(self.device)
        
        # 找到激活 token
        relevant_tokens = self.find_activation_tokens(tokenized_prompt.input_ids)
        
        # 找到当前实验在 tokenized 文本中的位置，只保留当前实验部分的激活 token
        current_experiment = experiments[current_experiment_idx]
        start_idx, end_idx = self.get_experiment_token_range(
            truncated_text, current_experiment, tokenized_prompt.input_ids
        )
        
        # 只考虑当前实验部分的激活 token
        experiment_relevant_tokens = torch.zeros(tokenized_prompt.input_ids.shape[1], dtype=torch.bool)
        if start_idx < relevant_tokens.shape[0]:
            experiment_relevant_tokens[start_idx:end_idx] = relevant_tokens[start_idx:end_idx]
        relevant_tokens = experiment_relevant_tokens
        
        if not relevant_tokens.any():
            # 如果没有找到激活 token，返回空 tensor
            d_model = self.model.cfg.d_model
            return {layer: torch.empty(0, d_model, dtype=torch.float16) for layer in layers}
        
        # 前向传播
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                input_ids, prepend_bos=True, return_type=None
            )
            
            # 提取所有指定层的特征
            layer_features = {}
            for layer_idx in layers:
                hook_name = f"blocks.{layer_idx}.{self.hook_type}"
                features = cache[hook_name]  # [1, seq_len, d_model]
                
                # 提取激活 token 的特征
                activation_features = features[0, relevant_tokens, :]  # [num_activations, d_model]
                
                # 转换为 half precision
                layer_features[layer_idx] = activation_features.half().cpu()
        
        return layer_features

