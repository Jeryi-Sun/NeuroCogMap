#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础提取器，提供 token 限制和上下文累积的通用功能
"""

import torch
import re
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseTokenLimitedExtractor(ABC):
    """基础 token 限制提取器，提供通用功能"""
    
    def __init__(self, tokenizer, max_tokens: int = 1024):
        """
        Args:
            tokenizer: tokenizer 对象
            max_tokens: 最大 token 数限制
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
    
    def truncate_to_token_limit(
        self, 
        instruction: str, 
        experiments: List[str], 
        current_experiment_idx: int = 0
    ) -> str:
        """
        根据 token 限制截断 prompt，保留 instruction 和尽可能多的最近实验
        
        Args:
            instruction: instruction 部分
            experiments: 实验列表（累积的）
            current_experiment_idx: 当前要处理的实验索引（在 experiments 中的索引）
            
        Returns:
            截断后的完整 prompt 文本
        """
        # 首先尝试包含所有到当前实验的内容
        context_experiments = experiments[:current_experiment_idx + 1]
        full_text = instruction + '\n' + '\n'.join(context_experiments)
        
        # 计算 token 数量
        tokenized = self.tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        num_tokens = tokenized.input_ids.shape[1]
        
        if num_tokens <= self.max_tokens:
            return full_text
        
        # 如果超过限制，从 instruction 后的早期实验开始删除
        # 保留 instruction 和当前实验，尽可能多地保留最近的实验
        current_experiment = experiments[current_experiment_idx]
        
        # 尝试只保留 instruction + 当前实验
        minimal_text = instruction + '\n' + current_experiment
        tokenized_minimal = self.tokenizer(minimal_text, return_tensors='pt', add_special_tokens=False)
        num_tokens_minimal = tokenized_minimal.input_ids.shape[1]
        
        if num_tokens_minimal > self.max_tokens:
            # 如果连 instruction + 当前实验都超过限制，需要截断 instruction
            print(f"警告: instruction + 当前实验 ({num_tokens_minimal} tokens) 超过限制 ({self.max_tokens} tokens)")
            # 截断 instruction，保留最后部分
            instruction_tokens = self.tokenizer(instruction, return_tensors='pt', add_special_tokens=False)
            current_experiment_tokens = self.tokenizer(current_experiment, return_tensors='pt', add_special_tokens=False)
            current_experiment_token_count = current_experiment_tokens.input_ids.shape[1]
            
            available_tokens = self.max_tokens - current_experiment_token_count - 10  # 留一些余量
            if available_tokens > 0:
                # 截断 instruction
                instruction_ids = instruction_tokens.input_ids[0][-available_tokens:]
                truncated_instruction = self.tokenizer.decode(instruction_ids, skip_special_tokens=True)
                return truncated_instruction + '\n' + current_experiment
            else:
                # 如果当前实验本身就超过限制，只返回当前实验的一部分
                available_for_experiment = self.max_tokens - 10
                experiment_ids = current_experiment_tokens.input_ids[0][:available_for_experiment]
                return self.tokenizer.decode(experiment_ids, skip_special_tokens=True)
        
        # 从后往前添加实验，直到达到 token 限制
        result_experiments = [current_experiment]
        result_text = instruction + '\n' + current_experiment
        
        for i in range(current_experiment_idx - 1, -1, -1):
            test_experiments = experiments[i:current_experiment_idx + 1]
            test_text = instruction + '\n' + '\n'.join(test_experiments)
            tokenized_test = self.tokenizer(test_text, return_tensors='pt', add_special_tokens=False)
            num_tokens_test = tokenized_test.input_ids.shape[1]
            
            if num_tokens_test <= self.max_tokens:
                result_experiments.insert(0, experiments[i])
                result_text = test_text
            else:
                break
        
        return result_text
    
    def find_activation_tokens(
        self, 
        tokenized_input_ids: torch.Tensor,
        only_left_token: bool = False
    ) -> torch.Tensor:
        """
        在 tokenized 文本中找到激活 token 的位置
        
        Args:
            tokenized_input_ids: tokenized 后的 input_ids
            
        Returns:
            boolean tensor 指示哪些位置是激活 token
        """
        # 获取激活 token IDs
        left_token = " <<"
        l_id = None
        try:
            left_tokenized = self.tokenizer(left_token, add_special_tokens=False)
            if len(left_tokenized.input_ids) > 0:
                l_id = left_tokenized.input_ids[-1]
        except:
            pass
        
        zero_token = "0"
        zero_id = None
        try:
            zero_tokenized = self.tokenizer(zero_token, add_special_tokens=False)
            if len(zero_tokenized.input_ids) > 0:
                zero_id = zero_tokenized.input_ids[-1]
        except:
            pass
        
        one_token = "1"
        one_id = None
        try:
            one_tokenized = self.tokenizer(one_token, add_special_tokens=False)
            if len(one_tokenized.input_ids) > 0:
                one_id = one_tokenized.input_ids[-1]
        except:
            pass
        
        # 查找激活 token
        relevant_tokens = torch.zeros(tokenized_input_ids.shape[1], dtype=torch.bool)
        
        if l_id is not None:
            relevant_tokens_l = (tokenized_input_ids == l_id).squeeze()
            relevant_tokens = relevant_tokens | relevant_tokens_l
        
        if zero_id is not None and not only_left_token:
            relevant_tokens_0 = (tokenized_input_ids == zero_id).squeeze()
            relevant_tokens = relevant_tokens | relevant_tokens_0
        
        if one_id is not None and not only_left_token:
            relevant_tokens_1 = (tokenized_input_ids == one_id).squeeze()
            relevant_tokens = relevant_tokens | relevant_tokens_1
        
        return relevant_tokens
    
    def get_experiment_token_range(
        self,
        truncated_text: str,
        current_experiment: str,
        tokenized_input_ids: torch.Tensor
    ) -> Tuple[int, int]:
        """
        获取当前实验在 tokenized 文本中的 token 范围
        
        Args:
            truncated_text: 截断后的文本
            current_experiment: 当前实验文本
            tokenized_input_ids: tokenized 后的 input_ids
            
        Returns:
            (start_idx, end_idx) token 索引范围
        """
        # 找到当前实验在截断文本中的位置
        experiment_start_in_text = truncated_text.rfind(current_experiment)
        
        if experiment_start_in_text != -1:
            # Tokenize 前缀部分来找到当前实验在 tokenized 文本中的起始位置
            prefix_text = truncated_text[:experiment_start_in_text]
            prefix_tokenized = self.tokenizer(prefix_text, return_tensors='pt', add_special_tokens=False)
            prefix_token_count = prefix_tokenized.input_ids.shape[1]
            
            # Tokenize 当前实验部分
            experiment_tokenized = self.tokenizer(current_experiment, return_tensors='pt', add_special_tokens=False)
            experiment_token_count = experiment_tokenized.input_ids.shape[1]
            
            start_idx = prefix_token_count
            end_idx = prefix_token_count + experiment_token_count
            return (start_idx, end_idx)
        else:
            # 如果找不到，返回整个范围
            return (0, tokenized_input_ids.shape[1])
    
    @abstractmethod
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
            提取的特征 tensor
        """
        pass

