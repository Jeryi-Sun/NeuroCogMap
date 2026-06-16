#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 SAE Model 的 token 限制特征提取器
参考 sae_model.py 的实现
"""

import torch
import numpy as np
import os
import json
from sae_lens import SAE, HookedSAETransformer
from typing import List, Optional, Dict, Any, Tuple
from .base_extractor import BaseTokenLimitedExtractor
import sys
from pathlib import Path
# 添加 litcoder_core 到路径
litcoder_path = Path(__file__).parent.parent.parent.parent / "litcoder_core"
sys.path.insert(0, str(litcoder_path.parent))
from litcoder_core.encoding.features.sae_model import (
    ParcelMapping, 
    extract_layer_numbers_from_sae_paths, 
    build_layer_to_path
)


class SAEModelTokenLimitedExtractor(BaseTokenLimitedExtractor):
    """基于 SAE 和 Parcel 的 token 限制特征提取器"""
    
    DEFAULT_SAE_PATHS_2B_PT = [
        "layer_0/width_16k/average_l0_105",
        "layer_1/width_16k/average_l0_102",
        "layer_2/width_16k/average_l0_141",
        "layer_3/width_16k/average_l0_59",
        "layer_4/width_16k/average_l0_124",
        "layer_5/width_16k/average_l0_68",
        "layer_6/width_16k/average_l0_70",
        "layer_7/width_16k/average_l0_69",
        "layer_8/width_16k/average_l0_71",
        "layer_9/width_16k/average_l0_73",
        "layer_10/width_16k/average_l0_77",
        "layer_11/width_16k/average_l0_80",
        "layer_12/width_16k/average_l0_82",
        "layer_13/width_16k/average_l0_84",
        "layer_14/width_16k/average_l0_84",
        "layer_15/width_16k/average_l0_78",
        "layer_16/width_16k/average_l0_78",
        "layer_17/width_16k/average_l0_77",
        "layer_18/width_16k/average_l0_74",
        "layer_19/width_16k/average_l0_73",
        "layer_20/width_16k/average_l0_71",
        "layer_21/width_16k/average_l0_70",
        "layer_22/width_16k/average_l0_72",
        "layer_23/width_16k/average_l0_75",
        "layer_24/width_16k/average_l0_73",
        "layer_25/width_16k/average_l0_116",
    ]
    
    def __init__(
        self,
        model_name: str,
        tokenizer=None,
        parcel_id: int = None,
        parcel_mapping_path: str = None,
        sae_release: str = None,
        sae_local_base_dir: str = None,
        sae_paths: List[str] = None,
        last_token: bool = True,
        max_tokens: int = 1024,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: 模型名称
            tokenizer: tokenizer 对象（如果为 None，则从模型获取）
            parcel_id: Parcel ID
            parcel_mapping_path: Parcel 映射文件路径
            sae_release: SAE release ID
            sae_local_base_dir: SAE 本地基础目录
            sae_paths: SAE 路径列表
            last_token: 是否只使用最后一个 token 的特征（默认: True）
            max_tokens: 最大 token 数限制（默认: 1024）
            device: 设备类型，如果为 None 则自动检测
        """
        self.model_name = model_name
        self.parcel_id = parcel_id
        self.parcel_name = f"parcel_{parcel_id}" if isinstance(parcel_id, int) else parcel_id
        self.parcel_mapping_path = parcel_mapping_path
        self.sae_release = sae_release
        self.sae_local_base_dir = sae_local_base_dir
        self.sae_paths = sae_paths
        self.last_token = last_token
        
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
        
        # 加载 Parcel mapping
        self._load_parcel_mapping()
        
        # 构建层到 latent 的映射
        self._build_layer_latent_mapping()
        
        # 加载模型和 SAE
        self._load_model_and_saes()
        
        # 获取 tokenizer（从模型获取或使用提供的）
        if tokenizer is None:
            tokenizer = self.model.tokenizer
        
        super().__init__(tokenizer, max_tokens)
    
    def _load_parcel_mapping(self):
        """加载 Parcel 映射文件"""
        with open(self.parcel_mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取可用层号
        available_layers = extract_layer_numbers_from_sae_paths(self.sae_paths)
        self.parcel_mapping = ParcelMapping(data, available_layers=available_layers)
    
    def _build_layer_latent_mapping(self, parcel_names: List[str] = None):
        """构建层到 latent 的映射
        
        Args:
            parcel_names: Parcel 名称列表，如果为 None 则使用 self.parcel_name
        """
        if parcel_names is None:
            parcel_names = [self.parcel_name]
        
        # 获取 Parcel 对应的 (layer_id, latent_in_layer) 映射
        parcel_latent_pairs = self.parcel_mapping.get_parcel_latent_mapping(parcel_names)
        
        # 构建分层映射：layer_id -> [latent_in_layer, ...]
        self.layer_to_latents: Dict[int, List[int]] = {}
        for layer_id, latent_in_layer in parcel_latent_pairs:
            self.layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
        
        # 构建层号到路径的映射
        layer_to_path = build_layer_to_path(self.sae_paths)
        self.layer_to_path = layer_to_path
        self.required_layers = sorted([
            l for l in self.layer_to_latents.keys() 
            if l in layer_to_path
        ])
        
        if not self.required_layers:
            raise RuntimeError("Parcel 对应的层不在提供的 SAE 路径中，无法提取特征。")
    
    def _load_single_sae_with_retry(self, layer_id: int, sae_path: str) -> Tuple[SAE, str]:
        """加载单个 SAE，支持设备重试机制
        
        Args:
            layer_id: 层ID
            sae_path: SAE路径
            
        Returns:
            Tuple[SAE, str]: (SAE对象, hook名称)
            
        Raises:
            RuntimeError: 如果所有重试都失败
        """
        max_retries = 2
        retry = 0
        success = False
        sae = None
        hook_name = None
        
        while retry <= max_retries and not success:
            try:
                force_device = None
                if retry == 1:
                    force_device = "cuda:0"
                elif retry == 2:
                    force_device = "cuda:1"
                
                # 确定目标设备
                target_device = force_device if force_device is not None else self.device
                
                if force_device is not None:
                    print(f"[INFO] 尝试在设备 {force_device} 上加载 SAE (层 {layer_id}, 路径 {sae_path})")
                else:
                    print(f"[INFO] 尝试在默认设备 {self.device} 上加载 SAE (层 {layer_id}, 路径 {sae_path})")
                
                # 加载 SAE，直接使用目标设备
                sae, _, _ = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_path,
                    device=target_device,
                    local_path=os.path.join(self.sae_local_base_dir, sae_path, "params.npz"),
                )
                
                sae.use_error_term = True
                print(f"[INFO] SAE 已加载到设备 {target_device}")
                
                # 获取 hook 名称
                hook_name = (
                    sae.cfg.metadata.hook_name 
                    if hasattr(sae.cfg, 'metadata') and hasattr(sae.cfg.metadata, 'hook_name')
                    else sae.cfg.hook_name
                )
                if hook_name is None:
                    raise RuntimeError(f"无法确定层 {layer_id} 的 hook 名称")
                
                success = True
                print(f"[INFO] 成功加载 SAE (层 {layer_id})")
                
            except RuntimeError as e:
                msg = str(e)
                print(f"[ERROR] RuntimeError 加载 SAE (层 {layer_id}): {msg}")
                if "Expected all tensors to be on the same device" in msg and retry < max_retries:
                    retry += 1
                    print(f"[INFO] 设备不匹配，重试 {retry}/{max_retries}")
                    # 清理可能残留的 SAE 对象
                    if sae is not None:
                        del sae
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"[ERROR] 加载 SAE 失败，已尝试 {retry + 1} 次")
                    import traceback
                    traceback.print_exc()
                    raise
            except Exception as ex:
                import traceback
                print(f"[ERROR] 加载 SAE 失败 (层 {layer_id}, 路径 {sae_path}): {ex}")
                print(f"[ERROR] 异常类型: {type(ex).__name__}")
                print(f"[ERROR] 完整 traceback:")
                traceback.print_exc()
                # 对于非 RuntimeError 的设备相关错误，也尝试重试
                if retry < max_retries and ("device" in str(ex).lower() or "cuda" in str(ex).lower()):
                    retry += 1
                    print(f"[INFO] 检测到设备相关错误，重试 {retry}/{max_retries}")
                    if sae is not None:
                        del sae
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
        
        if not success:
            raise RuntimeError(f"加载 SAE 失败 (层 {layer_id})，已尝试所有设备")
        
        return sae, hook_name

    def _load_model_and_saes(self):
        """加载模型和所需的 SAE，支持双卡自动切换"""
        # 加载模型
        if "9b" in self.model_name:
            self.model = HookedSAETransformer.from_pretrained(
                self.model_name, 
                device=self.device, 
                dtype=torch.bfloat16,
                n_devices=2 if torch.cuda.device_count() > 1 else 1,
            )
        else:
            self.model = HookedSAETransformer.from_pretrained(
                self.model_name, 
                device=self.device, 
                dtype=torch.bfloat16
            )
        self.model.eval()
        
        # 加载所需的 SAE，支持设备重试
        self.layer_to_sae: Dict[int, SAE] = {}
        self.layer_to_hook: Dict[int, str] = {}
        
        for layer_id in self.required_layers:
            sae_path = self.layer_to_path[layer_id]
            try:
                sae, hook_name = self._load_single_sae_with_retry(layer_id, sae_path)
                self.layer_to_sae[layer_id] = sae
                self.layer_to_hook[layer_id] = hook_name
            except Exception as ex:
                import traceback
                print(f"[ERROR] 加载 SAE 失败 (层 {layer_id}, 路径 {sae_path}): {ex}")
                print(f"[ERROR] 异常类型: {type(ex).__name__}")
                print(f"[ERROR] 完整 traceback:")
                traceback.print_exc()
                raise
    
    def extract_from_experiment(
        self,
        instruction: str,
        experiments: List[str],
        current_experiment_idx: int,
        **kwargs
    ) -> torch.Tensor:
        """
        从单个实验提取特征（重构后的 embedding）
        
        Args:
            instruction: instruction 部分
            experiments: 累积的实验列表
            current_experiment_idx: 当前实验的索引
            **kwargs: 其他参数
            
        Returns:
            提取的特征 tensor [num_activations, d_model]
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
            return torch.empty(0, d_model, dtype=torch.float16)
        
        # 运行模型并缓存所需的 hook
        with torch.no_grad():
            hook_names = [self.layer_to_hook[l] for l in self.layer_to_latents.keys()]
            _, cache = self.model.run_with_cache(
                input_ids,
                stop_at_layer=max(self.layer_to_latents.keys()) + 1,
                names_filter=hook_names
            )
            
            # 收集所有层的重构向量
            all_recon_vectors = []
            
            for layer_id in sorted(self.layer_to_latents.keys()):
                if layer_id not in self.layer_to_sae:
                    continue
                
                sae = self.layer_to_sae[layer_id]
                hook_name = self.layer_to_hook[layer_id]
                
                if hook_name not in cache:
                    continue
                
                # 获取该层的 hidden state
                sae_in = cache[hook_name]  # [1, seq_len, d_model]
                
                # 确保 sae_in 和 SAE 在同一个设备上
                sae_device = next(sae.parameters()).device
                if sae_in.device != sae_device:
                    sae_in = sae_in.to(sae_device)
                
                # SAE 编码得到 feature 激活
                feats = sae.encode(sae_in).squeeze(0)  # [seq_len, n_features]
                
                # 选择 Parcel 对应的 latents
                latents = self.layer_to_latents[layer_id]
                sel = feats[:, latents]  # [seq_len, len(latents)]
                
                # 只保留激活 token 的部分（注意要把 mask 放到和 sel 相同的 device 上）
                relevant_tokens_device = relevant_tokens.to(sel.device)
                sel_activations = sel[relevant_tokens_device]  # [num_activations, len(latents)]
                
                if sel_activations.numel() == 0:
                    continue
                
                # 使用 feature 重构 hidden state
                W_dec_sel = sae.W_dec[latents].to(sel_activations.dtype)  # [len(latents), d_model]
                recon = sel_activations @ W_dec_sel  # [num_activations, d_model]

                # 对于在所有选中 latent 上激活全为 0 的 token，直接回退为原始 hidden state（sae_in）
                # 这样可以避免得到全 0 的重构向量
                zero_mask = (sel_activations == 0).all(dim=-1)  # [num_activations]
                if zero_mask.any():
                    token_indices = torch.nonzero(relevant_tokens_device, as_tuple=False).squeeze(-1)  # [num_activations]
                    # 用原始 hidden state 替换这些位置的重构结果
                    recon[zero_mask] = sae_in[0, token_indices[zero_mask]].to(recon.dtype)
                
                all_recon_vectors.append(recon)
            
            if not all_recon_vectors:
                # 如果没有有效的重构向量，返回空 tensor
                d_model = self.model.cfg.d_model
                print(f"No valid reconstruction vectors found, returning empty tensor with shape: (0, {d_model})")
                return torch.empty(0, d_model, dtype=torch.float16)
            
            # 对所有层的重构向量求平均
            stacked = torch.stack(all_recon_vectors)  # [num_layers, num_activations, d_model]
            final_vec = stacked.mean(dim=0)  # [num_activations, d_model]
            
            return final_vec.half().cpu()
    
    def extract_all_parcels_from_experiment(
        self,
        instruction: str,
        experiments: List[str],
        current_experiment_idx: int,
        parcel_ids: List[int] = None,
        **kwargs
    ) -> Dict[int, torch.Tensor]:
        """
        从单个实验一次性提取所有 Parcel 的特征（重构后的 embedding）
        
        Args:
            instruction: instruction 部分
            experiments: 累积的实验列表
            current_experiment_idx: 当前实验的索引
            parcel_ids: Parcel ID 列表，如果为 None 则提取所有 Parcel
            **kwargs: 其他参数
            
        Returns:
            字典，键为 Parcel ID，值为特征 tensor [num_activations, d_model]
        """
        # 如果未指定 parcel_ids，提取所有 Parcel
        if parcel_ids is None:
            parcel_to_latents = self.parcel_mapping.latent_parcel_assignments.get("parcel_to_latents", {})
            all_parcel_names = sorted(parcel_to_latents.keys())
            parcel_ids = [int(name.split('_')[-1]) for name in all_parcel_names]
        
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
            return {parcel_id: torch.empty(0, d_model, dtype=torch.float16) for parcel_id in parcel_ids}
        
        # 预先构建所有 Parcel 的 layer_to_latents 映射
        parcel_layer_to_latents: Dict[int, Dict[int, List[int]]] = {}
        all_required_layers = set()
        
        for parcel_id in parcel_ids:
            parcel_name = f"parcel_{parcel_id}" if isinstance(parcel_id, int) else parcel_id
            parcel_latent_pairs = self.parcel_mapping.get_parcel_latent_mapping([parcel_name])
            layer_to_latents: Dict[int, List[int]] = {}
            for layer_id, latent_in_layer in parcel_latent_pairs:
                layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
                all_required_layers.add(layer_id)
            parcel_layer_to_latents[parcel_id] = layer_to_latents
        
        # 预加载所有需要的 SAE 层
        missing_layers = [lid for lid in all_required_layers if lid not in self.layer_to_sae]
        if missing_layers:
            for layer_id in missing_layers:
                if layer_id in self.layer_to_path:
                    sae_path = self.layer_to_path[layer_id]
                    try:
                        sae, hook_name = self._load_single_sae_with_retry(layer_id, sae_path)
                        self.layer_to_sae[layer_id] = sae
                        self.layer_to_hook[layer_id] = hook_name
                    except Exception as ex:
                        import traceback
                        print(f"[WARN] 无法加载层 {layer_id} 的 SAE: {ex}")
                        print(f"[WARN] 异常类型: {type(ex).__name__}")
                        print(f"[WARN] 完整 traceback:")
                        traceback.print_exc()
        
        # 运行模型并缓存所有需要的 hook（只运行一次）
        with torch.no_grad():
            available_layers = [lid for lid in all_required_layers if lid in self.layer_to_sae]
            if not available_layers:
                d_model = self.model.cfg.d_model
                return {parcel_id: torch.empty(0, d_model, dtype=torch.float16) for parcel_id in parcel_ids}
            
            hook_names = [self.layer_to_hook[l] for l in available_layers]
            _, cache = self.model.run_with_cache(
                input_ids,
                stop_at_layer=max(available_layers) + 1,
                names_filter=hook_names
            )
            
            # 对每个 Parcel 提取特征
            parcel_features: Dict[int, torch.Tensor] = {}
            
            for parcel_id in parcel_ids:
                layer_to_latents = parcel_layer_to_latents[parcel_id]
                
                # 只处理已加载的层
                available_layer_to_latents = {
                    lid: latents for lid, latents in layer_to_latents.items() 
                    if lid in self.layer_to_sae
                }
                
                if not available_layer_to_latents:
                    d_model = self.model.cfg.d_model
                    parcel_features[parcel_id] = torch.empty(0, d_model, dtype=torch.float16)
                    continue
                
                # 收集所有层的重构向量
                all_recon_vectors = []
                
                for layer_id in sorted(available_layer_to_latents.keys()):
                    sae = self.layer_to_sae[layer_id]
                    hook_name = self.layer_to_hook[layer_id]
                    
                    if hook_name not in cache:
                        continue
                    
                    # 获取该层的 hidden state
                    sae_in = cache[hook_name]  # [1, seq_len, d_model]
                    
                    # 确保 sae_in 和 SAE 在同一个设备上
                    sae_device = next(sae.parameters()).device
                    if sae_in.device != sae_device:
                        sae_in = sae_in.to(sae_device)
                    
                    # SAE 编码得到 feature 激活
                    feats = sae.encode(sae_in).squeeze(0)  # [seq_len, n_features]
                    # 选择 Parcel 对应的 latents
                    latents = available_layer_to_latents[layer_id]
                    sel = feats[:, latents]  # [seq_len, len(latents)]
                    
                    # 只保留激活 token 的部分（把 mask 放到和 sel 相同的 device 上）
                    relevant_tokens_device = relevant_tokens.to(sel.device)
                    sel_activations = sel[relevant_tokens_device]  # [num_activations, len(latents)]
                    
                    if sel_activations.numel() == 0:
                        continue
                    
                    # 使用 feature 重构 hidden state
                    W_dec_sel = sae.W_dec[latents].to(sel_activations.dtype)  # [len(latents), d_model]
                    recon = sel_activations @ W_dec_sel  # [num_activations, d_model]

                    # 对于在所有选中 latent 上激活全为 0 的 token，直接回退为原始 hidden state（sae_in）
                    zero_mask = (sel_activations == 0).all(dim=-1)  # [num_activations]
                    if zero_mask.any():
                        token_indices = torch.nonzero(relevant_tokens_device, as_tuple=False).squeeze(-1)  # [num_activations]
                        recon[zero_mask] = sae_in[0, token_indices[zero_mask]].to(recon.dtype)
                    
                    all_recon_vectors.append(recon)
                
                if not all_recon_vectors:
                    # 如果没有有效的重构向量，返回空 tensor
                    d_model = self.model.cfg.d_model
                    parcel_features[parcel_id] = torch.empty(0, d_model, dtype=torch.float16)
                else:
                    # 对所有层的重构向量求平均
                    stacked = torch.stack(all_recon_vectors)  # [num_layers, num_activations, d_model]
                    final_vec = stacked.mean(dim=0)  # [num_activations, d_model]
                    parcel_features[parcel_id] = final_vec.half().cpu()
        
        return parcel_features

