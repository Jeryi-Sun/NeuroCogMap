#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel到Latent映射工具函数
"""

import json
import os
from typing import Dict, List, Tuple, Union, Any
from collections import defaultdict
import yaml


class ParcelLatentMapper:
    """Parcel到Latent映射管理器"""
    
    def __init__(self, config_path: str):
        """
        初始化映射器
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载parcel分配文件
        self.parcel_assignments = self._load_parcel_assignments()
        
        # 构建层路径映射
        self.layer_to_sae_path = self._build_layer_path_mapping()
        
        # 获取可用层列表
        self.available_layers = sorted(self.layer_to_sae_path.keys())
        
        print(f"✅ 加载了 {len(self.available_layers)} 个SAE层")
        print(f"✅ 加载了 {len(self.parcel_assignments.get('parcel_to_latents', {}))} 个parcels")
    
    def _load_parcel_assignments(self) -> Dict[str, Any]:
        """加载parcel分配文件"""
        assignments_path = self.config['parcel_assignments_path']
        
        if not os.path.exists(assignments_path):
            raise FileNotFoundError(f"Parcel分配文件不存在: {assignments_path}")
        
        with open(assignments_path, 'r', encoding='utf-8') as f:
            assignments = json.load(f)
        
        # 如果只有latent_to_parcel，则构建parcel_to_latents
        if "parcel_to_latents" not in assignments:
            print("⚠️ 文件中缺少parcel_to_latents，正在构建...")
            parcel_to_latents = defaultdict(list)
            for k, v in assignments["latent_to_parcel"].items():
                latent_id = int(k.replace("latent_", ""))
                parcel_to_latents[f"parcel_{v}"].append(latent_id)
            assignments["parcel_to_latents"] = dict(parcel_to_latents)
            print(f"✅ 构建了 {len(assignments['parcel_to_latents'])} 个parcel的映射")
        
        return assignments
    
    def _build_layer_path_mapping(self) -> Dict[int, str]:
        """构建层ID到SAE路径的映射"""
        layer_to_path = {}
        base_dir = self.config['sae_weights_base_dir']
        
        for layer_config in self.config['sae_layers']:
            layer_id = layer_config['layer']
            relative_path = layer_config['path']
            full_path = os.path.join(base_dir, relative_path)
            
            # 检查路径是否存在
            if os.path.exists(full_path):
                layer_to_path[layer_id] = full_path
            else:
                print(f"⚠️ SAE层 {layer_id} 路径不存在: {full_path}")
        
        return layer_to_path
    
    def get_parcel_latent_mapping(self, parcel_input: Union[str, List[str]]) -> List[Tuple[int, int]]:
        """
        获取parcel对应的(layer_id, latent_id_in_layer)列表
        
        Args:
            parcel_input: parcel名称或列表，如 "parcel_61" 或 ["parcel_61", "parcel_62"]
            
        Returns:
            [(actual_layer_id, latent_id_in_layer), ...]
        """
        if isinstance(parcel_input, str):
            parcel_names = [parcel_input]
        else:
            parcel_names = parcel_input
        
        mapping = []
        latent_per_layer = self.config['latent_per_layer']
        
        for parcel_name in parcel_names:
            if parcel_name not in self.parcel_assignments["parcel_to_latents"]:
                print(f"⚠️ Parcel {parcel_name} 不存在于分配文件中")
                continue
            
            latent_ids = self.parcel_assignments["parcel_to_latents"][parcel_name]
            
            for latent_id in latent_ids:
                # 计算“原始层索引”（存储时用 0/1/2 这样的索引）
                original_layer_idx = latent_id // latent_per_layer
                
                # 计算层内latent索引
                latent_in_layer = latent_id % latent_per_layer
                
                # 将索引映射到实际可用的 SAE 层（如 [9, 20, 31]）
                if not self.available_layers:
                    raise RuntimeError("没有可用的SAE层，请检查配置中的 sae_layers 是否正确")
                
                if original_layer_idx < len(self.available_layers):
                    # 直接按索引取实际层号：0→available_layers[0]，1→available_layers[1]，…
                    actual_layer = self.available_layers[original_layer_idx]
                else:
                    # 如果索引超出范围，用模运算回退到可用层集合中
                    layer_index = original_layer_idx % len(self.available_layers)
                    actual_layer = self.available_layers[layer_index]
                    print(f"⚠️ Latent {latent_id} 的原始层索引 {original_layer_idx} 超出可用层范围，回退到层 {actual_layer}")
                
                mapping.append((actual_layer, latent_in_layer))
        
        return mapping
    
    def get_all_parcel_mappings(self) -> Dict[str, List[Tuple[int, int]]]:
        """获取所有parcel的latent映射"""
        parcel_mappings = {}
        
        for parcel_name in self.parcel_assignments["parcel_to_latents"]:
            parcel_mappings[parcel_name] = self.get_parcel_latent_mapping(parcel_name)
        
        return parcel_mappings
    
    def get_parcel_info(self) -> Dict[str, Any]:
        """获取parcel统计信息"""
        parcel_info = {}
        
        for parcel_name, latent_ids in self.parcel_assignments["parcel_to_latents"].items():
            mapping = self.get_parcel_latent_mapping(parcel_name)
            
            # 统计每个parcel涉及的层
            layers = set(layer_id for layer_id, _ in mapping)
            
            parcel_info[parcel_name] = {
                'n_latents': len(latent_ids),
                'n_mapped_latents': len(mapping),
                'layers': sorted(layers),
                'latent_ids': latent_ids
            }
        
        return parcel_info


def test_parcel_mapping():
    """测试parcel映射功能"""
    config_path = "/path/to/project_root/neural_area/global_weight/configs/paths.yaml"
    
    try:
        mapper = ParcelLatentMapper(config_path)
        
        # 测试单个parcel
        test_parcel = "parcel_129"
        mapping = mapper.get_parcel_latent_mapping(test_parcel)
        print(f"\n测试parcel {test_parcel}:")
        print(f"映射结果: {mapping[:5]}...")  # 只显示前5个
        
        # 获取parcel信息
        parcel_info = mapper.get_parcel_info()
        print(f"\nParcel统计信息:")
        for parcel_name, info in list(parcel_info.items())[:3]:  # 只显示前3个
            print(f"{parcel_name}: {info['n_latents']} latents, 涉及层 {info['layers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


if __name__ == "__main__":
    test_parcel_mapping()
