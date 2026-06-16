#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel Activation Scaling Intervention

基于top_anomalous_parcels.json的分析结果，通过调节特定Parcel的输出强度来缓解幻觉。
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os


class ParcelScaler:
    """
    Parcel激活调节器
    
    根据activation_diff和is_significant信息，计算并应用scaling系数来调节parcel激活强度
    """
    
    def __init__(self, 
                 parcel_json_path: str,
                 lambda_scale: float = 0.3,
                 smooth: float = 80.0,
                 min_scale: float = -1.0,  # 负值用于抑制
                 max_scale: float = 1.0,   # 正值用于增强
                 logger: Optional[logging.Logger] = None):
        """
        初始化Parcel调节器
        
        Args:
            parcel_json_path: top_anomalous_parcels.json文件路径
            lambda_scale: 调节幅度λ，控制整体调节强度
            smooth: 平滑系数s，控制tanh函数的陡峭程度
            min_scale: 最小缩放系数
            max_scale: 最大缩放系数
            logger: 日志记录器
        """
        self.parcel_json_path = parcel_json_path
        self.lambda_scale = lambda_scale
        self.smooth = smooth
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.logger = logger or logging.getLogger(__name__)
        
        # 加载scaling系数表
        self.scaling_table = self._load_scaling_factors()
        
        # 统计信息
        self.stats = self._compute_stats()
        
        self.logger.info(f"ParcelScaler初始化完成，共加载{len(self.scaling_table)}个需要调节的parcels")
        self.logger.info(f"增强parcels: {self.stats['enhance_count']}个，抑制parcels: {self.stats['suppress_count']}个")
    
    def _load_scaling_factors(self) -> Dict[int, float]:
        """加载并计算scaling系数"""
        if not os.path.exists(self.parcel_json_path):
            raise FileNotFoundError(f"未找到parcel文件: {self.parcel_json_path}")
        
        with open(self.parcel_json_path, 'r', encoding='utf-8') as f:
            parcels = json.load(f)
        
        scaling_table = {}
        
        for parcel in parcels:
            if not parcel.get("is_significant", False):
                continue
                
            parcel_id = parcel["parcel_id"]
            activation_diff = parcel["activation_diff"]
            function_name = parcel.get("function_name", "Unknown")
            
            # 计算scaling系数α_i
            # 注意：由于干预是正向加法，scaling系数以0为基准
            if activation_diff < 0:
                # 需要增强：activation_diff < 0，scaling > 0
                alpha = self.lambda_scale * np.tanh(-activation_diff / self.smooth)
                action = "增强"
            elif activation_diff > 0:
                # 需要抑制：activation_diff > 0，scaling < 0
                alpha = -self.lambda_scale * np.tanh(activation_diff / self.smooth)
                action = "抑制"
            else:
                # 保持不变
                alpha = 0.0
                action = "保持"
            
            # 裁剪到指定范围
            alpha = float(np.clip(alpha, self.min_scale, self.max_scale))
            scaling_table[parcel_id] = alpha
            
            self.logger.debug(f"Parcel {parcel_id} ({function_name}): activation_diff={activation_diff:.3f}, "
                            f"action={action}, alpha={alpha:.3f}")
        
        return scaling_table
    
    def _compute_stats(self) -> Dict[str, Any]:
        """计算统计信息"""
        enhance_count = 0
        suppress_count = 0
        keep_count = 0
        
        for parcel_id, alpha in self.scaling_table.items():
            if alpha > 0.0:
                enhance_count += 1
            elif alpha < 0.0:
                suppress_count += 1
            else:
                keep_count += 1
        
        return {
            "total_parcels": len(self.scaling_table),
            "enhance_count": enhance_count,
            "suppress_count": suppress_count,
            "keep_count": keep_count,
            "avg_scale": np.mean(list(self.scaling_table.values())),
            "min_scale": min(self.scaling_table.values()) if self.scaling_table else 1.0,
            "max_scale": max(self.scaling_table.values()) if self.scaling_table else 1.0
        }
    
    def apply_scaling(self, parcel_activations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        对parcel激活应用scaling调节
        
        Args:
            parcel_activations: {parcel_id: activation_tensor} 格式的激活字典
            
        Returns:
            scaled_activations: 调节后的激活字典
        """
        scaled = {}
        
        for parcel_id, activation in parcel_activations.items():
            scale = self.scaling_table.get(parcel_id, 1.0)
            scaled[parcel_id] = activation * scale
            
            if scale != 1.0:
                self.logger.debug(f"Parcel {parcel_id}: 原始激活范围[{activation.min():.3f}, {activation.max():.3f}], "
                                f"缩放系数={scale:.3f}")
        
        return scaled
    
    def get_scaling_factor(self, parcel_id: int) -> float:
        """获取指定parcel的scaling系数"""
        return self.scaling_table.get(parcel_id, 1.0)
    
    def get_scaling_table(self) -> Dict[int, float]:
        """获取完整的scaling系数表"""
        return self.scaling_table.copy()
    
    def visualize_scaling(self) -> str:
        """生成scaling系数的可视化信息"""
        if not self.scaling_table:
            return "没有需要调节的parcels"
        
        lines = ["Parcel Scaling 系数表:"]
        lines.append("-" * 60)
        lines.append(f"{'Parcel ID':<10} {'Scaling':<10} {'Action':<10}")
        lines.append("-" * 60)
        
        for parcel_id, scale in sorted(self.scaling_table.items()):
            if scale > 0.0:
                action = "增强"
            elif scale < 0.0:
                action = "抑制"
            else:
                action = "保持"
            lines.append(f"{parcel_id:<10} {scale:<10.3f} {action:<10}")
        
        lines.append("-" * 60)
        lines.append(f"统计: 总计{self.stats['total_parcels']}个, "
                    f"增强{self.stats['enhance_count']}个, "
                    f"抑制{self.stats['suppress_count']}个, "
                    f"保持{self.stats['keep_count']}个")
        
        return "\n".join(lines)
    
    def save_table(self, output_path: str):
        """保存scaling系数表到CSV文件"""
        import pandas as pd
        
        data = []
        for parcel_id, scale in self.scaling_table.items():
            if scale > 0.0:
                action = "增强"
            elif scale < 0.0:
                action = "抑制"
            else:
                action = "保持"
            
            data.append({
                "parcel_id": parcel_id,
                "scaling_factor": scale,
                "action": action
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"Scaling系数表已保存到: {output_path}")
    
    def summary(self) -> Dict[str, Any]:
        """返回调节器摘要信息"""
        return {
            "config": {
                "lambda_scale": self.lambda_scale,
                "smooth": self.smooth,
                "min_scale": self.min_scale,
                "max_scale": self.max_scale
            },
            "stats": self.stats,
            "parcel_json_path": self.parcel_json_path
        }


def test_parcel_scaler():
    """测试ParcelScaler功能"""
    # 创建测试数据
    test_data = [
        {"parcel_id": 233, "function_name": "Binary Fact-Check", "activation_diff": -47.82, "is_significant": True},
        {"parcel_id": 89, "function_name": "Numerical Reasoning", "activation_diff": 22.94, "is_significant": True},
        {"parcel_id": 156, "function_name": "Memory", "activation_diff": 0.0, "is_significant": True},
        {"parcel_id": 201, "function_name": "Irrelevant", "activation_diff": 5.2, "is_significant": False}
    ]
    
    # 保存测试数据
    test_file = "/tmp/test_parcels.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 测试ParcelScaler
    scaler = ParcelScaler(test_file, lambda_scale=0.3, smooth=50.0, min_scale=-1.0, max_scale=1.0)
    
    print("=== ParcelScaler 测试 ===")
    print(scaler.visualize_scaling())
    print(f"\n摘要信息: {scaler.summary()}")
    
    # 测试scaling应用
    test_activations = {
        233: torch.tensor([1.0, 2.0, 3.0]),
        89: torch.tensor([0.5, 1.5, 2.5]),
        156: torch.tensor([1.0, 1.0, 1.0]),
        201: torch.tensor([2.0, 3.0, 4.0])  # 这个parcel不在scaling表中
    }
    
    scaled_activations = scaler.apply_scaling(test_activations)
    
    print("\n=== Scaling 应用测试 ===")
    for parcel_id, original in test_activations.items():
        scaled = scaled_activations[parcel_id]
        scale = scaler.get_scaling_factor(parcel_id)
        print(f"Parcel {parcel_id}: {original.tolist()} -> {scaled.tolist()} (scale={scale:.3f})")


if __name__ == "__main__":
    test_parcel_scaler()
