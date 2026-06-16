#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel连接可视化工具
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import logging
import yaml

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParcelConnectionVisualizer:
    """Parcel连接可视化器"""
    
    def __init__(self, config_path: str, output_dir: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录，如果为None则使用配置文件中的目录
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = output_dir or self.config['output_dir']
        self.n_parcels = self.config['n_parcels']
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"✅ 可视化器初始化完成，输出目录: {self.output_dir}")
    
    def plot_connection_heatmap(self, matrix: np.ndarray, 
                               title: str = "Parcel-Parcel Structural Connection Matrix",
                               figsize: Tuple[int, int] = (12, 10),
                               save_path: Optional[str] = None) -> str:
        """
        绘制连接矩阵热力图
        
        Args:
            matrix: 连接矩阵
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "parcel_connection_heatmap.png")
        
        plt.figure(figsize=figsize)
        
        # 使用seaborn绘制热力图
        sns.heatmap(matrix, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   cbar_kws={'label': 'Connection Strength'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Target Parcel', fontsize=12)
        plt.ylabel('Source Parcel', fontsize=12)
        
        # 设置坐标轴标签
        step = max(1, self.n_parcels // 20)  # 最多显示20个刻度
        plt.xticks(range(0, self.n_parcels, step), 
                  range(0, self.n_parcels, step))
        plt.yticks(range(0, self.n_parcels, step), 
                  range(0, self.n_parcels, step))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 热力图已保存到: {save_path}")
        return save_path
    
    def plot_connection_distribution(self, connections: List[dict],
                                   title: str = "Connection Strength Distribution",
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None) -> str:
        """
        绘制连接强度分布图
        
        Args:
            connections: 连接结果列表
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "connection_distribution.png")
        
        strengths = [conn['connection_strength'] for conn in connections]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 直方图
        ax1.hist(strengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Connection Strength')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Connection Strength Histogram')
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2.boxplot(strengths, patch_artist=True)
        ax2.set_ylabel('Connection Strength')
        ax2.set_title('Connection Strength Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 分布图已保存到: {save_path}")
        return save_path
    
    def plot_top_connections(self, connections: List[dict], 
                           top_k: int = 20,
                           title: str = "Top-K Strongest Connections",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> str:
        """
        绘制Top-K最强连接图
        
        Args:
            connections: 连接结果列表
            top_k: 显示前K个连接
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"top_{top_k}_connections.png")
        
        # 按连接强度排序
        sorted_connections = sorted(connections, 
                                  key=lambda x: x['connection_strength'], 
                                  reverse=True)[:top_k]
        
        # 准备数据
        labels = [f"{conn['src_parcel']}→{conn['dst_parcel']}" 
                 for conn in sorted_connections]
        strengths = [conn['connection_strength'] for conn in sorted_connections]
        
        plt.figure(figsize=figsize)
        
        # 创建水平条形图
        y_pos = np.arange(len(labels))
        bars = plt.barh(y_pos, strengths, color='lightcoral', alpha=0.7)
        
        # 添加数值标签
        for i, (bar, strength) in enumerate(zip(bars, strengths)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{strength:.4f}', 
                    ha='left', va='center', fontsize=8)
        
        plt.yticks(y_pos, labels)
        plt.xlabel('Connection Strength')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Top-{top_k}连接图已保存到: {save_path}")
        return save_path
    
    def plot_parcel_degree_distribution(self, matrix: np.ndarray,
                                      title: str = "Parcel Degree Distribution",
                                      figsize: Tuple[int, int] = (10, 6),
                                      save_path: Optional[str] = None) -> str:
        """
        绘制parcel连接度分布图
        
        Args:
            matrix: 连接矩阵
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "parcel_degree_distribution.png")
        
        # 计算每个parcel的连接度（非零连接数）
        out_degrees = np.sum(matrix != 0, axis=1)  # 出度
        in_degrees = np.sum(matrix != 0, axis=0)   # 入度
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 出度分布
        ax1.hist(out_degrees, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel('Out-degree')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Parcel Out-degree Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 入度分布
        ax2.hist(in_degrees, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('In-degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Parcel In-degree Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ 连接度分布图已保存到: {save_path}")
        return save_path
    
    def generate_summary_report(self, connections: List[dict], 
                              matrix: np.ndarray,
                              save_path: Optional[str] = None) -> str:
        """
        生成连接分析摘要报告
        
        Args:
            connections: 连接结果列表
            matrix: 连接矩阵
            save_path: 保存路径，如果为None则自动生成
            
        Returns:
            保存路径
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "connection_summary_report.txt")
        
        strengths = [conn['connection_strength'] for conn in connections]
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Parcel-Parcel Structural Connection Analysis Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本统计
            f.write("1. Basic Statistics\n")
            f.write(f"   Total parcels: {self.n_parcels}\n")
            f.write(f"   Total connections: {len(connections)}\n")
            f.write(f"   Non-zero connections: {np.sum(matrix != 0)}\n")
            f.write(f"   Connection density: {np.sum(matrix != 0) / (self.n_parcels * self.n_parcels):.4f}\n\n")
            
            # 连接强度统计
            f.write("2. Connection Strength Statistics\n")
            f.write(f"   Mean: {np.mean(strengths):.6f}\n")
            f.write(f"   Std: {np.std(strengths):.6f}\n")
            f.write(f"   Min: {np.min(strengths):.6f}\n")
            f.write(f"   Max: {np.max(strengths):.6f}\n")
            f.write(f"   Median: {np.median(strengths):.6f}\n\n")
            
            # Top-10最强连接
            f.write("3. Top-10 Strongest Connections\n")
            sorted_connections = sorted(connections, 
                                      key=lambda x: x['connection_strength'], 
                                      reverse=True)[:10]
            for i, conn in enumerate(sorted_connections, 1):
                f.write(f"   {i:2d}. {conn['src_parcel']} → {conn['dst_parcel']}: "
                       f"{conn['connection_strength']:.6f} "
                       f"(pairs: {conn['n_pairs']})\n")
            
            f.write("\n")
            
            # 连接度统计
            out_degrees = np.sum(matrix != 0, axis=1)
            in_degrees = np.sum(matrix != 0, axis=0)
            
            f.write("4. Degree Statistics\n")
            f.write(f"   Average out-degree: {np.mean(out_degrees):.2f}\n")
            f.write(f"   Average in-degree: {np.mean(in_degrees):.2f}\n")
            f.write(f"   Max out-degree: {np.max(out_degrees)}\n")
            f.write(f"   Max in-degree: {np.max(in_degrees)}\n")
            
            # 找出连接度最高的parcels
            max_out_parcel = np.argmax(out_degrees)
            max_in_parcel = np.argmax(in_degrees)
            f.write(f"   Highest out-degree parcel: parcel_{max_out_parcel} (out-degree: {np.max(out_degrees)})\n")
            f.write(f"   Highest in-degree parcel: parcel_{max_in_parcel} (in-degree: {np.max(in_degrees)})\n")
        
        logger.info(f"✅ 摘要报告已保存到: {save_path}")
        return save_path
    
    def create_all_visualizations(self, connections: List[dict], 
                                matrix: np.ndarray) -> List[str]:
        """
        创建所有可视化图表
        
        Args:
            connections: 连接结果列表
            matrix: 连接矩阵
            
        Returns:
            生成的图表文件路径列表
        """
        logger.info("🎨 开始创建可视化图表...")
        
        generated_files = []
        
        try:
            # 1. 连接矩阵热力图
            heatmap_path = self.plot_connection_heatmap(matrix)
            generated_files.append(heatmap_path)
            
            # 2. 连接强度分布图
            dist_path = self.plot_connection_distribution(connections)
            generated_files.append(dist_path)
            
            # 3. Top-20最强连接图
            top_conn_path = self.plot_top_connections(connections, top_k=20)
            generated_files.append(top_conn_path)
            
            # 4. Parcel连接度分布图
            degree_path = self.plot_parcel_degree_distribution(matrix)
            generated_files.append(degree_path)
            
            # 5. 生成摘要报告
            report_path = self.generate_summary_report(connections, matrix)
            generated_files.append(report_path)
            
            logger.info(f"✅ 成功创建 {len(generated_files)} 个可视化文件")
            
        except Exception as e:
            logger.error(f"❌ 创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return generated_files


def test_visualization():
    """测试可视化功能"""
    config_path = "/path/to/project_root/neural_area/global_weight/configs/paths.yaml"
    
    try:
        visualizer = ParcelConnectionVisualizer(config_path)
        
        # 创建测试数据
        n_parcels = 10
        test_matrix = np.random.randn(n_parcels, n_parcels) * 0.1
        test_connections = [
            {
                'src_parcel': f'parcel_{i}',
                'dst_parcel': f'parcel_{j}',
                'connection_strength': test_matrix[i, j],
                'n_pairs': 1
            }
            for i in range(n_parcels)
            for j in range(n_parcels)
        ]
        
        # 测试可视化
        generated_files = visualizer.create_all_visualizations(test_connections, test_matrix)
        
        logger.info(f"✅ 可视化测试完成，生成了 {len(generated_files)} 个文件")
        for file_path in generated_files:
            logger.info(f"   - {file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_visualization()
