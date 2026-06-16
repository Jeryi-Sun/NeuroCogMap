#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Union, Optional, Tuple
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import json
import re
import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer
from collections import defaultdict
from tqdm import tqdm
from .base import BaseFeatureExtractor


class ParcelMapping:
    """Parcel 映射管理类，用于处理 Parcel 到 Latent 的映射关系"""
    
    def __init__(self, latent_parcel_assignments: Dict, available_layers: Optional[List[int]] = None):
        self.latent_parcel_assignments = latent_parcel_assignments
        self.num_parcels = self._infer_num_parcels()
        self.available_layers = available_layers or []

    def _infer_num_parcels(self) -> int:
        try:
            return len(self.latent_parcel_assignments.get("parcel_to_latents", {}))
        except Exception as ex:
            import traceback
            print(f"[ERROR] 推断 Parcel 数量失败: {ex}")
            print(f"[ERROR] 异常类型: {type(ex).__name__}")
            print(f"[ERROR] 完整 traceback:")
            traceback.print_exc()
            # 无法推断 Parcel 数量会影响程序逻辑，应该终止程序
            raise RuntimeError(f"无法推断 Parcel 数量，程序无法继续: {ex}") from ex

    def set_available_layers(self, available_layers: List[int]) -> None:
        """设置可用的SAE层号列表"""
        self.available_layers = sorted(available_layers)

    def get_parcel_latent_mapping(self, parcel_input: Union[str, List[str]]) -> List[Tuple[int, int]]:
        """获取 Parcel 对应的 (layer_id, latent_in_layer) 映射列表"""
        if isinstance(parcel_input, str):
            parcel_names = [parcel_input]
        elif isinstance(parcel_input, list):
            parcel_names = parcel_input
        else:
            raise TypeError(f"parcel_input 必须是字符串或列表，当前类型: {type(parcel_input)}")

        mapping: List[Tuple[int, int]] = []
        
        # 如果没有设置可用层，使用旧的连续层映射方式
        if not self.available_layers:
            for parcel_name in parcel_names:
                if parcel_name not in self.latent_parcel_assignments['parcel_to_latents']:
                    raise KeyError(f"未找到 parcel: {parcel_name} 于 parcel_to_latents 映射中")
                latent_ids = self.latent_parcel_assignments['parcel_to_latents'][parcel_name]
                for latent_id in latent_ids:
                    layer_id = latent_id // 16384
                    latent_in_layer = latent_id % 16384
                    mapping.append((layer_id, latent_in_layer))
            return mapping
        
        # 使用实际SAE层号进行映射
        for parcel_name in parcel_names:
            if parcel_name not in self.latent_parcel_assignments['parcel_to_latents']:
                raise KeyError(f"未找到 parcel: {parcel_name} 于 parcel_to_latents 映射中")
            latent_ids = self.latent_parcel_assignments['parcel_to_latents'][parcel_name]
            for latent_id in latent_ids:
                # 先通过latent_id计算出原始的层索引
                original_layer_idx = latent_id // 16384
                
                # 检查原始层索引是否在可用层范围内
                if original_layer_idx < len(self.available_layers):
                    actual_layer_id = self.available_layers[original_layer_idx]
                else:
                    # 如果超出范围，使用模运算回退到可用层
                    layer_index = original_layer_idx % len(self.available_layers)
                    actual_layer_id = self.available_layers[layer_index]
                
                latent_in_layer = latent_id % 16384
                mapping.append((actual_layer_id, latent_in_layer))
        return mapping


def extract_layer_numbers_from_sae_paths(sae_paths: List[str]) -> List[int]:
    """从SAE路径中提取层号"""
    layer_numbers = []
    for path in sae_paths:
        m = re.search(r"layer_(\d+)/", path)
        if m:
            layer_numbers.append(int(m.group(1)))
    return sorted(layer_numbers)


def build_layer_to_path(sae_paths: List[str]) -> Dict[int, str]:
    """构建层号到SAE路径的映射"""
    layer_to_path: Dict[int, str] = {}
    for p in sae_paths:
        m = re.search(r"layer_(\d+)/", p)
        if m:
            layer_to_path[int(m.group(1))] = p
    return layer_to_path


class SAEModelFeatureExtractor(BaseFeatureExtractor):
    """基于 SAE 和 Parcel 的特征提取器
    
    该提取器使用 Sparse Autoencoder (SAE) 和 Parcel 映射来提取特征。
    与传统的基于层的提取不同，这里使用 Parcel 作为特征单元。
    """

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

    def __init__(self, config: Dict[str, Any]):
        """初始化 SAE 模型特征提取器

        Args:
            config (Dict[str, Any]): 配置字典，包含：
                - parcel_id (Union[int, str]): 单个 Parcel ID，如 61 或 "parcel_61"
                - parcel_ids (List[Union[int, str]]): 多个 Parcel ID 列表（可选）
                - parcel_mapping_path (str): latent_parcel_assignments.json 文件路径
                - model_name (str): 模型名称，如 "google/gemma-2-2b"
                - sae_release (str): SAE release ID，默认 "gemma-scope-2b-pt-res"
                - sae_local_base_dir (str): SAE 本地基础目录
                - sae_paths (List[str]): SAE 路径列表（可选，默认使用 2B-PT 全层）
                - last_token (bool): 是否只使用最后一个 token 的特征，默认 True
                - device (str): 设备类型，默认自动检测
        """
        super().__init__(config)
        
        # 解析 Parcel ID
        if "parcel_id" in config:
            parcel_id = config["parcel_id"]
            if isinstance(parcel_id, int):
                self.parcel_names = [f"parcel_{parcel_id}"]
            else:
                self.parcel_names = [parcel_id]
        elif "parcel_ids" in config:
            parcel_ids = config["parcel_ids"]
            self.parcel_names = [
                f"parcel_{pid}" if isinstance(pid, int) else pid 
                for pid in parcel_ids
            ]
        else:
            raise ValueError("必须提供 parcel_id 或 parcel_ids 之一")
        
        self.parcel_mapping_path = config["parcel_mapping_path"]
        self.model_name = config["model_name"]
        self.sae_release = config.get("sae_release", "gemma-scope-2b-pt-res")
        self.sae_local_base_dir = config["sae_local_base_dir"]
        self.sae_paths = config.get("sae_paths", self.DEFAULT_SAE_PATHS_2B_PT)
        self.last_token = config.get("last_token", True)
        
        # 设备选择
        if "device" in config:
            self.device = config["device"]
        elif torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # 加载 Parcel mapping
        self._load_parcel_mapping()
        
        # 构建层到 latent 的映射
        self._build_layer_latent_mapping()
        
        # 延迟加载模型与 SAE：仅在真正需要计算时再加载
        # 这样在命中缓存、只从 ActivationCache 读数据的情况下，就不会重复加载大模型和 SAE。
        self.model: Optional[HookedSAETransformer] = None
        self.layer_to_sae: Dict[int, SAE] = {}
        self.layer_to_hook: Dict[int, str] = {}

    def _ensure_model_and_saes_loaded(self) -> None:
        """确保模型和 SAE 已经加载（延迟加载）。"""
        if self.model is not None and self.layer_to_sae:
            return
        # 如果尚未加载，则真正加载一次
        self._load_model_and_saes()

    def _load_parcel_mapping(self):
        """加载 Parcel 映射文件"""
        try:
            with open(self.parcel_mapping_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取可用层号
            available_layers = extract_layer_numbers_from_sae_paths(self.sae_paths)
            self.parcel_mapping = ParcelMapping(data, available_layers=available_layers)
        except Exception as ex:
            import traceback
            print(f"[ERROR] 加载 Parcel 映射失败: {ex}")
            print(f"[ERROR] 异常类型: {type(ex).__name__}")
            print(f"[ERROR] 完整 traceback:")
            traceback.print_exc()
            # 加载 Parcel 映射失败会影响程序逻辑，应该终止程序
            raise

    def _build_layer_latent_mapping(self):
        """构建层到 latent 的映射（为所有 Parcel 建立映射）"""
        # 获取所有 Parcel 名称
        parcel_to_latents = self.parcel_mapping.latent_parcel_assignments.get("parcel_to_latents", {})
        all_parcel_names = list(parcel_to_latents.keys())
        
        # 获取所有 Parcel 对应的 (layer_id, latent_in_layer) 映射
        parcel_latent_pairs = self.parcel_mapping.get_parcel_latent_mapping(all_parcel_names)
        
        # 构建分层映射：layer_id -> [latent_in_layer, ...]
        self.layer_to_latents: Dict[int, List[int]] = {}
        for layer_id, latent_in_layer in parcel_latent_pairs:
            self.layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
        
        # 构建层号到路径的映射（保存所有层的映射，以便后续提取所有 Parcel 时使用）
        layer_to_path = build_layer_to_path(self.sae_paths)
        self.layer_to_path = layer_to_path  # 保存所有层的路径映射
        # 仅保留需要的层（用于初始化时加载）
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
        try:
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
        except Exception as ex:
            import traceback
            print(f"[ERROR] 加载模型失败: {ex}")
            print(f"[ERROR] 异常类型: {type(ex).__name__}")
            print(f"[ERROR] 完整 traceback:")
            traceback.print_exc()
            # 加载模型失败会影响程序逻辑，应该终止程序
            raise
        
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
                # 加载 SAE 失败会影响程序逻辑，应该终止程序
                raise

    def extract_features(
        self, 
        stimuli: Union[str, List[str]], 
        parcel_id: Optional[Union[int, str]] = None,
        **kwargs
    ) -> np.ndarray:
        """提取特征

        Args:
            stimuli (Union[str, List[str]]): 输入文本或文本列表
            parcel_id (Optional[Union[int, str]]): 可选的 Parcel ID，用于动态指定（覆盖初始化时的配置）
            **kwargs: 其他参数

        Returns:
            np.ndarray: 提取的特征数组。如果有多个 parcel，特征会被拼接（concatenate）在一起
        """
        # 如果提供了 parcel_id，使用它；否则使用初始化时的配置
        if parcel_id is not None:
            if isinstance(parcel_id, int):
                parcel_names = [f"parcel_{parcel_id}"]
            elif isinstance(parcel_id, list):
                parcel_names = [f"parcel_{pid}" if isinstance(pid, int) else pid for pid in parcel_id]
            else:
                parcel_names = [parcel_id]
        else:
            parcel_names = self.parcel_names
        
        if isinstance(stimuli, str):
            stimuli = [stimuli]
        
        # 如果只有一个 parcel，使用原来的方法（合并 latents）
        if len(parcel_names) == 1:
            parcel_latent_pairs = self.parcel_mapping.get_parcel_latent_mapping(parcel_names)
            layer_to_latents: Dict[int, List[int]] = {}
            for layer_id, latent_in_layer in parcel_latent_pairs:
                layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
            
            # 处理每个文本
            all_features = []
            print(f"处理 {len(stimuli)} 个文本，使用单个 Parcel: {parcel_names[0]}...")
            
            for i, text in enumerate(stimuli):
                if i % 10 == 0:
                    print(f"处理文本 {i+1}/{len(stimuli)}")
                
                features = self._extract_single_features(text, layer_to_latents)
                all_features.append(features)
            
            return np.vstack(all_features)
        else:
            # 多个 parcel：分别提取每个 parcel 的特征，然后拼接
            print(f"处理 {len(stimuli)} 个文本，使用 {len(parcel_names)} 个 Parcel: {parcel_names}，特征将拼接...")
            
            # 为每个 parcel 构建 layer_to_latents 映射
            parcel_layer_to_latents: Dict[str, Dict[int, List[int]]] = {}
            for parcel_name in parcel_names:
                parcel_latent_pairs = self.parcel_mapping.get_parcel_latent_mapping([parcel_name])
                layer_to_latents: Dict[int, List[int]] = {}
                for layer_id, latent_in_layer in parcel_latent_pairs:
                    layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
                parcel_layer_to_latents[parcel_name] = layer_to_latents
            
            # 处理每个文本
            all_features = []
            for i, text in enumerate(stimuli):
                if i % 10 == 0:
                    print(f"处理文本 {i+1}/{len(stimuli)}")
                
                # 分别提取每个 parcel 的特征
                parcel_features = []
                for parcel_name in parcel_names:
                    layer_to_latents = parcel_layer_to_latents[parcel_name]
                    features = self._extract_single_features(text, layer_to_latents)
                    parcel_features.append(features)
                
                # 拼接所有 parcel 的特征（在特征维度上拼接）
                concatenated_features = np.concatenate(parcel_features, axis=1)  # [1, d_model * num_parcels]
                all_features.append(concatenated_features)
            
            return np.vstack(all_features)

    def _extract_single_features(
        self, 
        text: str, 
        layer_to_latents: Dict[int, List[int]]
    ) -> np.ndarray:
        """从单个文本提取特征

        Args:
            text (str): 输入文本
            layer_to_latents (Dict[int, List[int]]): 层到 latent 列表的映射

        Returns:
            np.ndarray: 提取的特征向量
        """
        # 只有在真正需要计算时，才确保模型和 SAE 已加载
        self._ensure_model_and_saes_loaded()

        if text == "":
            # 返回零向量，维度为模型的 d_model
            d_model = self.model.cfg.d_model
            return np.zeros((1, d_model))
        
        with torch.no_grad():
            # Tokenize
            tokens = self.model.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
                truncation=True,
                max_length=1024,
            )
            input_ids = tokens["input_ids"].to(self.device)
            
            # 运行模型并缓存所需的 hook
            hook_names = [self.layer_to_hook[l] for l in layer_to_latents.keys()]
            _, cache = self.model.run_with_cache(
                input_ids,
                stop_at_layer=max(layer_to_latents.keys()) + 1,
                names_filter=hook_names
            )
            
            # 从 cache 中提取特征
            return self._extract_features_from_cache(cache, layer_to_latents)

    def _extract_features_from_cache(
        self,
        cache: Dict[str, torch.Tensor],
        layer_to_latents: Dict[int, List[int]]
    ) -> np.ndarray:
        """从已缓存的模型输出中提取特征
        
        Args:
            cache (Dict[str, torch.Tensor]): 模型运行后的缓存
            layer_to_latents (Dict[int, List[int]]): 层到 latent 列表的映射
            
        Returns:
            np.ndarray: 提取的特征向量
        """
        with torch.no_grad():
            # 收集所有层的重构向量
            all_recon_vectors = []
            
            for layer_id in sorted(layer_to_latents.keys()):
                if layer_id not in self.layer_to_sae:
                    continue
                
                sae = self.layer_to_sae[layer_id]
                hook_name = self.layer_to_hook[layer_id]
                
                if hook_name not in cache:
                    continue
                
                # 获取该层的 hidden state
                sae_in = cache[hook_name]  # [1, seq_len, d_model]
                seq_len = sae_in.shape[1]
                
                # 确保 sae_in 和 SAE 在同一个设备上
                sae_device = next(sae.parameters()).device
                if sae_in.device != sae_device:
                    sae_in = sae_in.to(sae_device)
                
                # SAE 编码得到 feature 激活
                feats = sae.encode(sae_in).squeeze(0)  # [seq_len, n_features]
                
                # 检查 feats 中是否有 NaN
                if torch.isnan(feats).any():
                    print(f"[WARN] 层 {layer_id} 的 SAE 编码结果包含 NaN，NaN 数量: {torch.isnan(feats).sum().item()}/{feats.numel()}", file=sys.stderr)
                
                # 选择 Parcel 对应的 latents
                latents = layer_to_latents[layer_id]
                sel = feats[:, latents]  # [seq_len, len(latents)]
                
                # 根据 last_token 参数选择处理方式
                if self.last_token:
                    # 只使用最后一个 token（不管是否激活）
                    last_token_sel = sel[-1:]  # [1, len(latents)]
                    if last_token_sel.numel() == 0:
                        continue
                    
                    # 使用最后一个 token 的 feature 重构 hidden state
                    W_dec_sel = sae.W_dec[latents].to(last_token_sel.dtype)  # [len(latents), d_model]
                    recon = last_token_sel @ W_dec_sel  # [1, d_model]
                    
                    # 检查重构结果中是否有 NaN
                    if torch.isnan(recon).any():
                        print(f"[WARN] 层 {layer_id} 的重构结果包含 NaN，NaN 数量: {torch.isnan(recon).sum().item()}/{recon.numel()}", file=sys.stderr)
                    
                    vec = recon.squeeze(0)  # [d_model]
                else:
                    # 仅保留激活为正的 token
                    if sel.numel() == 0:
                        continue
                    
                    fired_mask = sel.sum(dim=-1) > 0
                    if not torch.any(fired_mask):
                        continue
                    
                    sel = sel[fired_mask]  # [num_fired_tokens, len(latents)]
                    
                    # 使用所选 Parcel 的 feature 重构 hidden state
                    W_dec_sel = sae.W_dec[latents].to(sel.dtype)  # [len(latents), d_model]
                    recon = sel @ W_dec_sel  # [num_fired_tokens, d_model]
                    
                    # 检查重构结果中是否有 NaN
                    if torch.isnan(recon).any():
                        print(f"[WARN] 层 {layer_id} 的重构结果包含 NaN，NaN 数量: {torch.isnan(recon).sum().item()}/{recon.numel()}", file=sys.stderr)
                    
                    # 对 token 维度求平均
                    vec = recon.mean(dim=0)  # [d_model]
                
                all_recon_vectors.append(vec)
            
            if not all_recon_vectors:
                # 如果没有有效的重构向量，返回零向量
                d_model = self.model.cfg.d_model
                print(f"没有有效的重构向量，返回零向量，d_model: {d_model}")
                return np.zeros((1, d_model))
            
            # 对所有层的重构向量求平均
            final_vec = torch.stack(all_recon_vectors).mean(dim=0)  # [d_model]
            
            # 检查是否有 NaN 值
            if torch.isnan(final_vec).any():
                import traceback
                print(f"[ERROR] 最终 embedding 包含 NaN 值！", file=sys.stderr)
                print(f"[ERROR] NaN 数量: {torch.isnan(final_vec).sum().item()}/{final_vec.numel()}", file=sys.stderr)
                print(f"[ERROR] 向量形状: {final_vec.shape}", file=sys.stderr)
                print(f"[ERROR] 向量统计: min={final_vec.min().item():.6f}, max={final_vec.max().item():.6f}, mean={final_vec.mean().item():.6f}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError("最终 embedding 包含 NaN 值，无法继续处理")
            
            # 转换为 numpy 并添加 batch 维度
            result = final_vec.cpu().numpy().reshape(1, -1)
            
            # 再次检查 numpy 数组中的 NaN
            if np.isnan(result).any():
                import traceback
                print(f"[ERROR] 转换后的 numpy 数组包含 NaN 值！", file=sys.stderr)
                print(f"[ERROR] NaN 数量: {np.isnan(result).sum()}/{result.size}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise RuntimeError("转换后的 numpy 数组包含 NaN 值，无法继续处理")
            
            return result

    def extract_all_parcels(
        self, 
        stimuli: Union[str, List[str]], 
        **kwargs
    ) -> Dict[Union[int, str], np.ndarray]:
        """提取所有 Parcel 的特征
        
        该方法会提取 Parcel 映射文件中所有 Parcel 的特征，用于缓存。
        
        Args:
            stimuli (Union[str, List[str]]): 输入文本或文本列表
            **kwargs: 其他参数
            
        Returns:
            Dict[Union[int, str], np.ndarray]: 字典，键为 Parcel ID（如 "parcel_61" 或 61），值为特征数组
        """
        # 只有在真正需要为所有 Parcel 计算特征时，才加载模型与 SAE
        self._ensure_model_and_saes_loaded()

        if isinstance(stimuli, str):
            stimuli = [stimuli]
        # 获取所有 Parcel 名称
        parcel_to_latents = self.parcel_mapping.latent_parcel_assignments.get("parcel_to_latents", {})
        all_parcel_names = sorted(parcel_to_latents.keys())
        
        # 为每个 Parcel 提取特征
        all_parcel_features: Dict[str, List[np.ndarray]] = {}
        
        print(f"提取 {len(all_parcel_names)} 个 Parcel 的特征，处理 {len(stimuli)} 个文本...")
        
        for parcel_name in all_parcel_names:
            all_parcel_features[parcel_name] = []
        
        # 预先加载所有可能需要的 SAE 层（基于 self.layer_to_latents 中所有层）
        all_required_layers = set(self.layer_to_latents.keys())
        missing_layers = [lid for lid in all_required_layers if lid not in self.layer_to_sae]
        if missing_layers:
            print(f"预加载 {len(missing_layers)} 个缺失的 SAE 层...")
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
        
        # 预先构建所有 Parcel 的 layer_to_latents 映射
        parcel_layer_to_latents: Dict[str, Dict[int, List[int]]] = {}
        for parcel_name in all_parcel_names:
            parcel_latent_pairs = self.parcel_mapping.get_parcel_latent_mapping([parcel_name])
            layer_to_latents: Dict[int, List[int]] = {}
            for layer_id, latent_in_layer in parcel_latent_pairs:
                layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
            parcel_layer_to_latents[parcel_name] = layer_to_latents

        # 处理每个文本
        for i, text in tqdm(enumerate(stimuli), desc="Processing texts", total=len(stimuli)):
            if i % 10 == 0:
                print(f"处理文本 {i+1}/{len(stimuli)}")
            
            if text == "":
                # 空文本返回零向量
                d_model = self.model.cfg.d_model
                zero_features = np.zeros((1, d_model))
                for parcel_name in all_parcel_names:
                    all_parcel_features[parcel_name].append(zero_features)
                continue
            
            # 对每个文本，只运行一次模型前向传播，缓存所有需要的层
            with torch.no_grad():
                # Tokenize
                # print(f"Tokenizing text: {text}")
                tokens = self.model.tokenizer(
                    text,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=True,
                    max_length=1024,
                )
                input_ids = tokens["input_ids"].to(self.device)
                # print(f"Input IDs: {input_ids}")

                # 获取所有需要的 hook 名称（基于 self.layer_to_latents 中所有已加载的层）
                available_layers = [lid for lid in self.layer_to_latents.keys() if lid in self.layer_to_sae]
                if not available_layers:
                    # 如果没有可用的层，所有 Parcel 都返回零向量
                    d_model = self.model.cfg.d_model
                    zero_features = np.zeros((1, d_model))
                    for parcel_name in all_parcel_names:
                        all_parcel_features[parcel_name].append(zero_features)
                    continue
                
                hook_names = [self.layer_to_hook[l] for l in available_layers]
                _, cache = self.model.run_with_cache(
                    input_ids,
                    stop_at_layer=max(available_layers) + 1,
                    names_filter=hook_names
                )
            
            # 对每个 Parcel，从共享的 cache 中提取特征
            for parcel_name in all_parcel_names:
                layer_to_latents = parcel_layer_to_latents[parcel_name]
                
                # 只处理已加载的层
                available_layer_to_latents = {
                    lid: latents for lid, latents in layer_to_latents.items() 
                    if lid in self.layer_to_sae
                }
                
                if not available_layer_to_latents:
                    # 如果没有可用的层，返回零向量
                    d_model = self.model.cfg.d_model
                    features = np.zeros((1, d_model))
                else:
                    features = self._extract_features_from_cache(cache, available_layer_to_latents)
                
                all_parcel_features[parcel_name].append(features)
        
        # 将每个 Parcel 的特征堆叠起来
        result: Dict[Union[int, str], np.ndarray] = {}
        for parcel_name, features_list in all_parcel_features.items():
            # 尝试解析 Parcel ID（如 "parcel_61" -> 61）
            try:
                parcel_id = int(parcel_name.split('_')[-1])
                result[parcel_id] = np.vstack(features_list)
            except (ValueError, IndexError) as ex:
                # 如果解析失败，使用原始名称（非关键错误，可以继续）
                print(f"[WARN] 无法解析 Parcel 名称 '{parcel_name}' 为整数 ID，使用原始名称: {ex}")
                result[parcel_name] = np.vstack(features_list)
        return result

    def _validate_config(self) -> None:
        """验证配置参数"""
        required_params = ["parcel_mapping_path", "model_name", "sae_local_base_dir"]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"缺少必需参数: {param}")
        
        if "parcel_id" not in self.config and "parcel_ids" not in self.config:
            raise ValueError("必须提供 parcel_id 或 parcel_ids 之一")
        
        if not os.path.exists(self.config["parcel_mapping_path"]):
            raise ValueError(f"Parcel 映射文件不存在: {self.config['parcel_mapping_path']}")

