#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAE权重加载和IO工具函数

兼容两类 SAE 本地布局：
- Gemma 系列：{base_dir}/{path}/params.npz
- Llama-3.1-8B LXR-8x：{base_dir}/Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors
"""

import os
import numpy as np
import yaml
from typing import Dict, Tuple
import logging

from sae_lens import SAE  # 已在其他模块使用，假设可用
import torch

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAEWeightLoader:
    """SAE权重加载器（Gemma npz + Llama LXR-8x safetensors）"""
    
    def __init__(self, config_path: str):
        """
        初始化权重加载器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.sae_weights_base_dir = self.config['sae_weights_base_dir']
        self.latent_per_layer = self.config['latent_per_layer']
        
        # 可选字段：用于区分 gemma / llama_scope_lxr_8x
        self.model_name = self.config.get('model_name', '')
        self.sae_release = self.config.get('sae_release', '')
        
        # 权重缓存
        self._encoder_cache: Dict[int, np.ndarray] = {}
        self._decoder_cache: Dict[int, np.ndarray] = {}
        
        logger.info(f"✅ SAE权重加载器初始化完成，基础目录: {self.sae_weights_base_dir}")
    
    # ------------------------
    # Gemma: 从 params.npz 读取
    # ------------------------
    def _load_npz_weights(self, layer_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从npz文件加载encoder和decoder权重（Gemma 系列）
        
        Args:
            layer_path: SAE层路径
            
        Returns:
            (encoder_weights, decoder_weights)
        """
        params_path = os.path.join(layer_path, "params.npz")
        
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"SAE参数文件不存在: {params_path}")
        
        try:
            # 加载npz文件
            data = np.load(params_path)
            
            # 查找encoder和decoder权重
            encoder_key = None
            decoder_key = None
            
            for key in data.keys():
                if 'encoder' in key.lower() and 'weight' in key.lower():
                    encoder_key = key
                elif 'decoder' in key.lower() and 'weight' in key.lower():
                    decoder_key = key
            
            if encoder_key is None or decoder_key is None:
                # 如果没找到，尝试按形状推断
                available_keys = list(data.keys())
                logger.warning(f"未找到标准的encoder/decoder键，可用键: {available_keys}")
                
                for key in available_keys:
                    shape = data[key].shape
                    if len(shape) == 2:
                        if shape[0] == self.latent_per_layer:  # decoder形状
                            decoder_key = decoder_key or key
                        elif shape[1] == self.latent_per_layer:  # encoder形状
                            encoder_key = encoder_key or key
            
            if encoder_key is None or decoder_key is None:
                raise ValueError(f"无法找到encoder/decoder权重，可用键: {list(data.keys())}")
            
            encoder_weights = data[encoder_key]
            decoder_weights = data[decoder_key]
            
            # 确保形状正确
            if encoder_weights.shape[1] != self.latent_per_layer:
                raise ValueError(f"Encoder权重形状错误: {encoder_weights.shape}, 期望第二维为 {self.latent_per_layer}")
            
            if decoder_weights.shape[0] != self.latent_per_layer:
                raise ValueError(f"Decoder权重形状错误: {decoder_weights.shape}, 期望第一维为 {self.latent_per_layer}")
            
            logger.info(f"✅ 成功加载 Gemma 层权重 - Encoder: {encoder_weights.shape}, Decoder: {decoder_weights.shape}")
            
            return encoder_weights, decoder_weights
            
        except Exception as e:
            logger.error(f"❌ 加载 Gemma SAE 权重失败: {e}")
            raise
    
    # ------------------------
    # Llama LXR-8x: 从 final.safetensors 读取
    # ------------------------
    def _load_llama_sae_weights(self, layer_id: int, layer_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 sae_lens 从 Llama Scope LXR-8x final.safetensors 加载权重，
        并提取 encoder / decoder 矩阵为 numpy 数组。
        """
        # 从路径中解析出层号（Llama3_1-8B-Base-L{layer}R-8x）
        # 这里 layer_id 本身已经是 0-31，对应 LXR-8x 的层索引，可直接用
        real_dir = layer_path
        local_path = os.path.join(real_dir, "checkpoints", "final.safetensors")
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Llama LXR-8x SAE 文件不存在: {local_path}")
        
        # 与 parcel_intervention 等脚本保持一致的设备选择策略
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # sae_id 使用逻辑 ID，遵循规则（例如 l0r_8x）
        sae_id = f"l{layer_id}r_8x"
        sae_release = self.sae_release or "llama_scope_lxr_8x"
        
        try:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
                local_path=local_path,
            )
        except Exception as e:
            logger.error(f"❌ SAE.from_pretrained 加载 Llama SAE 失败: layer_id={layer_id}, sae_id={sae_id}, error={e}")
            raise
        
        # 提取 encoder / decoder 权重
        # sae.W_enc: [d_model, d_sae], sae.W_dec: [d_sae, d_model]
        try:
            # sae_lens 中 Llama LXR-8x 通常用 bfloat16 存储，需要先转成 float32 再导出到 numpy
            w_enc = sae.W_enc.detach().to(dtype=torch.float32, device="cpu").numpy()
            w_dec = sae.W_dec.detach().to(dtype=torch.float32, device="cpu").numpy()
        except Exception as e:
            logger.error(f"❌ 从 SAE 模型中提取权重失败: layer_id={layer_id}, error={e}")
            raise
        
        # 校验 d_sae 一致
        if w_enc.shape[1] != self.latent_per_layer or w_dec.shape[0] != self.latent_per_layer:
            raise ValueError(
                f"Llama SAE 维度不匹配: enc={w_enc.shape}, dec={w_dec.shape}, "
                f"期望 latent_per_layer={self.latent_per_layer}"
            )
        
        logger.info(
            f"✅ 成功加载 Llama LXR-8x 层 {layer_id} 权重 - "
            f"Encoder: {w_enc.shape}, Decoder: {w_dec.shape}"
        )
        
        return w_enc, w_dec
    
    # ------------------------
    # 公共接口
    # ------------------------
    def _is_llama_lxr(self) -> bool:
        ml = str(self.model_name).lower()
        rl = str(self.sae_release).lower()
        return ("llama" in ml) or ("llama" in rl) or ("lxr" in rl)
    
    def _load_layer_weights(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """根据配置自动选择 Gemma npz 或 Llama LXR-8x safetensors。"""
        layer_config = next((lc for lc in self.config['sae_layers'] if lc['layer'] == layer_id), None)
        if layer_config is None:
            raise ValueError(f"未找到层 {layer_id} 的配置")
        
        layer_path = os.path.join(self.sae_weights_base_dir, layer_config['path'])
        
        if self._is_llama_lxr():
            # Llama-3.1-8B + LXR-8x
            return self._load_llama_sae_weights(layer_id, layer_path)
        else:
            # 默认走 Gemma npz 路径
            return self._load_npz_weights(layer_path)
    
    def get_encoder_weights(self, layer_id: int) -> np.ndarray:
        """
        获取指定层的encoder权重
        
        Returns:
            encoder权重矩阵，形状为 [d_model, n_latent]
        """
        if layer_id in self._encoder_cache:
            return self._encoder_cache[layer_id]
        
        encoder_weights, _ = self._load_layer_weights(layer_id)
        
        # 缓存权重
        self._encoder_cache[layer_id] = encoder_weights
        
        return encoder_weights
    
    def get_decoder_weights(self, layer_id: int) -> np.ndarray:
        """
        获取指定层的decoder权重
        
        Returns:
            decoder权重矩阵，形状为 [n_latent, d_model]
        """
        if layer_id in self._decoder_cache:
            return self._decoder_cache[layer_id]
        
        _, decoder_weights = self._load_layer_weights(layer_id)
        
        # 缓存权重
        self._decoder_cache[layer_id] = decoder_weights
        
        return decoder_weights
    
    def get_weight_vectors(self, layer_id: int, latent_idx: int, vector_type: str) -> np.ndarray:
        """
        获取指定层和latent的权重向量
        
        Args:
            layer_id: 层ID
            latent_idx: latent在层内的索引
            vector_type: 'encoder' 或 'decoder'
        """
        if vector_type == 'encoder':
            weights = self.get_encoder_weights(layer_id)
            vector = weights[:, latent_idx]  # 取列向量
        elif vector_type == 'decoder':
            weights = self.get_decoder_weights(layer_id)
            vector = weights[latent_idx, :]  # 取行向量
        else:
            raise ValueError(f"无效的向量类型: {vector_type}")
        
        return vector
    
    def clear_cache(self):
        """清空权重缓存"""
        self._encoder_cache.clear()
        self._decoder_cache.clear()
        logger.info("✅ 权重缓存已清空")
    
    def get_cache_info(self) -> Dict[str, int]:
        """获取缓存信息"""
        return {
            'encoder_cached_layers': len(self._encoder_cache),
            'decoder_cached_layers': len(self._decoder_cache)
        }
    
    
def test_weight_loading():
    """测试权重加载功能"""
    # 默认使用 2b 配置，如需测试 8b 或 9b，请手动修改为:
    #  - paths_8b.yaml （Llama-3.1-8B + LXR-8x）
    #  - paths_9b.yaml （Gemma-2-9b-it + SAE）
    config_path = "/path/to/project_root/neural_area/global_weight/configs/paths.yaml"
    
    try:
        loader = SAEWeightLoader(config_path)
        
        # 测试加载第一层权重
        layer_id = 0
        encoder_weights = loader.get_encoder_weights(layer_id)
        decoder_weights = loader.get_decoder_weights(layer_id)
        
        print(f"✅ 成功加载层 {layer_id} 权重:")
        print(f"   Encoder形状: {encoder_weights.shape}")
        print(f"   Decoder形状: {decoder_weights.shape}")
        
        # 测试获取特定latent的向量
        latent_idx = 0
        encoder_vector = loader.get_weight_vectors(layer_id, latent_idx, 'encoder')
        decoder_vector = loader.get_weight_vectors(layer_id, latent_idx, 'decoder')
        
        print(f"✅ 成功获取latent {latent_idx} 的向量:")
        print(f"   Encoder向量形状: {encoder_vector.shape}")
        print(f"   Decoder向量形状: {decoder_vector.shape}")
        
        # 显示缓存信息
        cache_info = loader.get_cache_info()
        print(f"✅ 缓存信息: {cache_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    
if __name__ == "__main__":
    test_weight_loading()
