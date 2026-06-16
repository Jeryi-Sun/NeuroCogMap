#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel-Parcel结构连接计算器
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import yaml

from utils_mapping import ParcelLatentMapper
from utils_io import SAEWeightLoader

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParcelConnectionCalculator:
    """Parcel-Parcel连接计算器"""
    
    def __init__(self, config_path: str, skip_existing: bool = True):
        """
        初始化连接计算器
        
        Args:
            config_path: 配置文件路径
            skip_existing: 是否跳过已存在的结果文件
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.skip_existing = skip_existing
        self.output_dir = self.config['output_dir']
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.mapper = ParcelLatentMapper(config_path)
        self.weight_loader = SAEWeightLoader(config_path)
        
        # 获取所有parcel映射
        self.parcel_mappings = self.mapper.get_all_parcel_mappings()
        self.parcel_names = list(self.parcel_mappings.keys())
        
        logger.info(f"✅ 连接计算器初始化完成")
        logger.info(f"   处理 {len(self.parcel_names)} 个parcels")
        logger.info(f"   输出目录: {self.output_dir}")
    
    def calculate_parcel_connection(self, src_parcel: str, dst_parcel: str) -> Dict[str, float]:
        """
        计算两个parcel之间的结构连接强度
        
        Args:
            src_parcel: 源parcel名称
            dst_parcel: 目标parcel名称
            
        Returns:
            连接信息字典
        """
        src_latents = self.parcel_mappings[src_parcel]
        dst_latents = self.parcel_mappings[dst_parcel]
        
        total_connection = 0.0
        total_pairs = 0
        cross_layer_pairs = 0
        
        # 遍历所有latent对
        for src_layer, src_idx in src_latents:
            for dst_layer, dst_idx in dst_latents:
                # 只计算跨层连接，且必须是前向连接（src_layer < dst_layer）
                if src_layer >= dst_layer:
                    continue
                
                try:
                    # 获取权重向量
                    src_decoder = self.weight_loader.get_weight_vectors(src_layer, src_idx, 'decoder')
                    dst_encoder = self.weight_loader.get_weight_vectors(dst_layer, dst_idx, 'encoder')
                    
                    # L2归一化
                    src_decoder_norm = src_decoder / (np.linalg.norm(src_decoder) + 1e-12)
                    dst_encoder_norm = dst_encoder / (np.linalg.norm(dst_encoder) + 1e-12)
                    
                    # 计算余弦相似度（内积）
                    connection_strength = np.dot(src_decoder_norm, dst_encoder_norm)
                    
                    total_connection += connection_strength
                    total_pairs += 1
                    cross_layer_pairs += 1
                    
                except Exception as e:
                    logger.warning(f"计算 {src_parcel}->{dst_parcel} 连接时出错: {e}")
                    continue
        
        # 计算平均连接强度
        if total_pairs > 0:
            avg_connection = total_connection / total_pairs
        else:
            avg_connection = 0.0
        
        return {
            'src_parcel': src_parcel,
            'dst_parcel': dst_parcel,
            'connection_strength': avg_connection,
            'n_pairs': total_pairs,
            'cross_layer_pairs': cross_layer_pairs
        }
    
    def calculate_all_connections(self, max_connections: Optional[int] = None) -> List[Dict[str, float]]:
        """
        计算所有parcel对之间的连接
        
        Args:
            max_connections: 最大连接数限制（用于测试）
            
        Returns:
            连接结果列表
        """
        results = []
        total_pairs = len(self.parcel_names) * len(self.parcel_names)
        
        if max_connections:
            total_pairs = min(total_pairs, max_connections)
        
        logger.info(f"开始计算 {total_pairs} 个parcel对之间的连接...")
        
        with tqdm(total=total_pairs, desc="计算连接") as pbar:
            for i, src_parcel in enumerate(self.parcel_names):
                for j, dst_parcel in enumerate(self.parcel_names):
                    if max_connections and len(results) >= max_connections:
                        break
                    
                    # 计算连接
                    connection_info = self.calculate_parcel_connection(src_parcel, dst_parcel)
                    results.append(connection_info)
                    
                    pbar.update(1)
                
                if max_connections and len(results) >= max_connections:
                    break
        
        logger.info(f"✅ 完成连接计算，共 {len(results)} 个连接")
        return results
    
    def save_connections(self, connections: List[Dict[str, float]], filename: str = "parcel_connections.csv"):
        """
        保存连接结果到CSV文件
        
        Args:
            connections: 连接结果列表
            filename: 输出文件名
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # 检查文件是否已存在
        if self.skip_existing and os.path.exists(output_path):
            logger.info(f"⚠️ 文件已存在，跳过保存: {output_path}")
            return output_path
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(connections)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ 连接结果已保存到: {output_path}")
        logger.info(f"   共 {len(connections)} 个连接")
        
        # 显示统计信息
        if len(connections) > 0:
            strengths = [c['connection_strength'] for c in connections]
            logger.info(f"   连接强度统计: 均值={np.mean(strengths):.4f}, "
                       f"标准差={np.std(strengths):.4f}, "
                       f"范围=[{np.min(strengths):.4f}, {np.max(strengths):.4f}]")
        
        return output_path
    
    def build_connection_matrix(self, connections: List[Dict[str, float]]) -> np.ndarray:
        """
        构建parcel连接矩阵
        
        Args:
            connections: 连接结果列表
            
        Returns:
            连接矩阵，形状为 [n_parcels, n_parcels]
        """
        n_parcels = self.config['n_parcels']
        matrix = np.zeros((n_parcels, n_parcels), dtype=np.float32)
        
        # 提取parcel ID
        def extract_parcel_id(parcel_name: str) -> int:
            return int(parcel_name.replace("parcel_", ""))
        
        # 填充矩阵
        for conn in connections:
            src_id = extract_parcel_id(conn['src_parcel'])
            dst_id = extract_parcel_id(conn['dst_parcel'])
            
            if 0 <= src_id < n_parcels and 0 <= dst_id < n_parcels:
                matrix[src_id, dst_id] = conn['connection_strength']
        
        logger.info(f"✅ 构建连接矩阵: {matrix.shape}")
        return matrix
    
    def save_connection_matrix(self, matrix: np.ndarray, filename: str = "parcel_connection_matrix.npy"):
        """
        保存连接矩阵
        
        Args:
            matrix: 连接矩阵
            filename: 输出文件名
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # 检查文件是否已存在
        if self.skip_existing and os.path.exists(output_path):
            logger.info(f"⚠️ 矩阵文件已存在，跳过保存: {output_path}")
            return output_path
        
        # 保存numpy数组
        np.save(output_path, matrix)
        
        # 同时保存CSV格式便于查看
        csv_path = output_path.replace('.npy', '.csv')
        df_matrix = pd.DataFrame(matrix, 
                                index=range(matrix.shape[0]), 
                                columns=range(matrix.shape[1]))
        df_matrix.to_csv(csv_path, float_format="%.6f")
        
        logger.info(f"✅ 连接矩阵已保存:")
        logger.info(f"   .npy文件: {output_path}")
        logger.info(f"   .csv文件: {csv_path}")
        
        return output_path
    
    def run_full_calculation(self, max_connections: Optional[int] = None) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """
        运行完整的连接计算流程
        
        Args:
            max_connections: 最大连接数限制（用于测试）
            
        Returns:
            (连接结果列表, 连接矩阵)
        """
        logger.info("🚀 开始完整连接计算流程...")
        
        # 计算所有连接
        connections = self.calculate_all_connections(max_connections)
        
        # 保存连接结果
        self.save_connections(connections)
        
        # 构建连接矩阵
        matrix = self.build_connection_matrix(connections)
        
        # 保存连接矩阵
        self.save_connection_matrix(matrix)
        
        logger.info("✅ 完整计算流程完成!")
        
        return connections, matrix


def test_connection_calculation():
    """测试连接计算功能"""
    config_path = "/path/to/project_root/neural_area/global_weight/configs/paths.yaml"
    
    try:
        calculator = ParcelConnectionCalculator(config_path, skip_existing=False)
        
        # 测试小规模计算
        logger.info("开始测试连接计算...")
        connections, matrix = calculator.run_full_calculation(max_connections=10)
        
        logger.info(f"✅ 测试完成!")
        logger.info(f"   计算了 {len(connections)} 个连接")
        logger.info(f"   矩阵形状: {matrix.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_connection_calculation()
