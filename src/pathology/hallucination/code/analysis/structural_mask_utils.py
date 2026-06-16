#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构性连接矩阵Mask工具

用于将结构性连接矩阵作为mask应用到功能连接矩阵计算中，
负向的结构性连接被视为0，正向连接保持原值。

作者: Assistant
日期: 2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Tuple

# 设置日志
logger = logging.getLogger(__name__)


class StructuralMaskProcessor:
    """结构性连接矩阵Mask处理器"""
    
    def __init__(self, structural_matrix_path: str, parcel_dim: int):
        """
        初始化Mask处理器
        
        Args:
            structural_matrix_path: 结构性连接矩阵文件路径
            parcel_dim: Parcel维度
        """
        self.structural_matrix_path = structural_matrix_path
        self.parcel_dim = parcel_dim
        self.structural_matrix = None
        self.mask_matrix = None
        
    def load_structural_matrix(self) -> np.ndarray:
        """
        加载结构性连接矩阵
        
        Returns:
            结构性连接矩阵
        """
        try:
            logger.info(f"加载结构性连接矩阵: {self.structural_matrix_path}")
            
            if self.structural_matrix_path.endswith('.csv'):
                # 从CSV文件加载
                df = pd.read_csv(self.structural_matrix_path, index_col=0)
                self.structural_matrix = df.values
            elif self.structural_matrix_path.endswith('.npy'):
                # 从NPY文件加载
                self.structural_matrix = np.load(self.structural_matrix_path)
            else:
                raise ValueError(f"不支持的文件格式: {self.structural_matrix_path}")
            
            # 检查矩阵维度
            if self.structural_matrix.shape != (self.parcel_dim, self.parcel_dim):
                logger.warning(f"结构性连接矩阵维度不匹配: {self.structural_matrix.shape} != ({self.parcel_dim}, {self.parcel_dim})")
                # 尝试调整维度
                if self.structural_matrix.shape[0] == self.parcel_dim:
                    self.structural_matrix = self.structural_matrix[:, :self.parcel_dim]
                elif self.structural_matrix.shape[1] == self.parcel_dim:
                    self.structural_matrix = self.structural_matrix[:self.parcel_dim, :]
                else:
                    raise ValueError(f"无法调整结构性连接矩阵维度: {self.structural_matrix.shape}")
            
            logger.info(f"结构性连接矩阵加载完成，形状: {self.structural_matrix.shape}")
            return self.structural_matrix
            
        except Exception as e:
            logger.error(f"加载结构性连接矩阵失败: {e}")
            raise
    
    def create_mask_matrix(self, threshold: float = 0.0) -> np.ndarray:
        """
        创建mask矩阵
        
        Args:
            threshold: 阈值，小于此值的结构性连接被视为0
            
        Returns:
            mask矩阵 (0或1)
        """
        if self.structural_matrix is None:
            self.load_structural_matrix()
        
        # 创建mask：结构性连接 > threshold 的位置为1，否则为0
        self.mask_matrix = (self.structural_matrix > threshold).astype(np.float32)
        
        # 确保对角线为1（自连接）
        np.fill_diagonal(self.mask_matrix, 1.0)
        
        logger.info(f"Mask矩阵创建完成，形状: {self.mask_matrix.shape}")
        logger.info(f"非零mask元素数量: {np.sum(self.mask_matrix)}")
        logger.info(f"Mask密度: {np.sum(self.mask_matrix) / (self.parcel_dim * self.parcel_dim):.4f}")
        
        return self.mask_matrix
    
    def apply_mask(self, connectivity_matrix: np.ndarray, 
                   threshold: float = 0.0,
                   mask_type: str = 'binary') -> np.ndarray:
        """
        将mask应用到连接矩阵
        
        Args:
            connectivity_matrix: 功能连接矩阵
            threshold: 结构性连接阈值
            mask_type: mask类型 ('binary' 或 'weighted')
            
        Returns:
            应用mask后的连接矩阵
        """
        if self.mask_matrix is None:
            self.create_mask_matrix(threshold)
        
        if connectivity_matrix.shape != (self.parcel_dim, self.parcel_dim):
            raise ValueError(f"连接矩阵维度不匹配: {connectivity_matrix.shape} != ({self.parcel_dim}, {self.parcel_dim})")
        
        if mask_type == 'binary':
            # 二进制mask：直接相乘
            masked_matrix = connectivity_matrix * self.mask_matrix
        elif mask_type == 'weighted':
            # 加权mask：使用结构性连接强度作为权重
            masked_matrix = connectivity_matrix * self.structural_matrix
        else:
            raise ValueError(f"不支持的mask类型: {mask_type}")
        
        logger.info(f"Mask应用完成，类型: {mask_type}")
        logger.info(f"原始连接矩阵非零元素: {np.sum(connectivity_matrix != 0)}")
        logger.info(f"Mask后连接矩阵非零元素: {np.sum(masked_matrix != 0)}")
        
        return masked_matrix
    
    def get_mask_statistics(self) -> dict:
        """
        获取mask统计信息
        
        Returns:
            统计信息字典
        """
        if self.mask_matrix is None:
            self.create_mask_matrix()
        
        stats = {
            'mask_shape': self.mask_matrix.shape,
            'total_elements': self.mask_matrix.size,
            'non_zero_elements': int(np.sum(self.mask_matrix)),
            'zero_elements': int(np.sum(self.mask_matrix == 0)),
            'mask_density': float(np.sum(self.mask_matrix) / self.mask_matrix.size),
            'diagonal_elements': int(np.sum(np.diag(self.mask_matrix))),
            'off_diagonal_non_zero': int(np.sum(self.mask_matrix) - np.sum(np.diag(self.mask_matrix)))
        }
        
        return stats


def load_structural_mask(structural_matrix_path: str, 
                        parcel_dim: int,
                        threshold: float = 0.0) -> StructuralMaskProcessor:
    """
    便捷函数：加载结构性连接mask
    
    Args:
        structural_matrix_path: 结构性连接矩阵文件路径
        parcel_dim: Parcel维度
        threshold: 结构性连接阈值
        
    Returns:
        StructuralMaskProcessor实例
    """
    processor = StructuralMaskProcessor(structural_matrix_path, parcel_dim)
    processor.load_structural_matrix()
    processor.create_mask_matrix(threshold)
    return processor


def apply_structural_mask(connectivity_matrix: np.ndarray,
                         structural_matrix_path: str,
                         parcel_dim: int,
                         threshold: float = 0.0,
                         mask_type: str = 'binary') -> np.ndarray:
    """
    便捷函数：直接应用结构性连接mask
    
    Args:
        connectivity_matrix: 功能连接矩阵
        structural_matrix_path: 结构性连接矩阵文件路径
        parcel_dim: Parcel维度
        threshold: 结构性连接阈值
        mask_type: mask类型
        
    Returns:
        应用mask后的连接矩阵
    """
    processor = StructuralMaskProcessor(structural_matrix_path, parcel_dim)
    return processor.apply_mask(connectivity_matrix, threshold, mask_type)


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.append('.')
    
    # 测试参数
    structural_matrix_path = "/path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv"
    parcel_dim = 270
    
    try:
        # 创建处理器
        processor = StructuralMaskProcessor(structural_matrix_path, parcel_dim)
        
        # 加载结构性连接矩阵
        structural_matrix = processor.load_structural_matrix()
        
        # 创建mask
        mask_matrix = processor.create_mask_matrix(threshold=0.0)
        
        # 获取统计信息
        stats = processor.get_mask_statistics()
        
        print("结构性连接矩阵统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 测试mask应用
        test_matrix = np.random.randn(parcel_dim, parcel_dim)
        masked_matrix = processor.apply_mask(test_matrix, mask_type='binary')
        
        print(f"\n测试mask应用:")
        print(f"  原始矩阵非零元素: {np.sum(test_matrix != 0)}")
        print(f"  Mask后非零元素: {np.sum(masked_matrix != 0)}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
