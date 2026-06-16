#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连接组分析增强版 (Connection Analysis Plus)

基于已保存的连接矩阵，进行脑科学风格的功能连接组分析：
1. 计算整体网络指标（density, efficiency, modularity等）
2. 计算节点级指标（中心性分析）
3. 发现异常子网（伪NBS分析）
4. 社区与模块结构分析
5. 可视化与报告生成

作者: AI Assistant
日期: 2025
"""

import json
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ConnectionAnalysisPlus:
    """连接组分析增强版"""
    
    def __init__(self, correct_matrix_path: str, incorrect_matrix_path: str,
                 node_names_path: str, output_dir: str,
                 z_threshold: float = 2.0, top_k_percent: float = 5.0):
        """
        初始化分析器
        
        Args:
            correct_matrix_path: 正确样本的平均连接矩阵路径
            incorrect_matrix_path: 幻觉样本的平均连接矩阵路径
            node_names_path: 节点名称文件路径
            output_dir: 输出目录
            z_threshold: Z分数阈值，用于提取显著连接
            top_k_percent: 提取top K%的高差异边
        """
        self.correct_matrix_path = correct_matrix_path
        self.incorrect_matrix_path = incorrect_matrix_path
        self.node_names_path = node_names_path
        self.output_dir = Path(output_dir)
        self.z_threshold = z_threshold
        self.top_k_percent = top_k_percent
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.A_h = None  # 幻觉样本连接矩阵
        self.A_t = None  # 正确样本连接矩阵
        self.Delta_A = None  # 差异矩阵
        self.node_names = []  # 节点名称列表
        self.N = None  # 节点数量
        
    def load_data(self) -> None:
        """加载连接矩阵和节点名称"""
        logger.info("加载连接矩阵数据...")
        
        # 加载连接矩阵
        try:
            self.A_t = np.load(self.correct_matrix_path)
            self.A_h = np.load(self.incorrect_matrix_path)
            logger.info(f"正确样本连接矩阵形状: {self.A_t.shape}")
            logger.info(f"幻觉样本连接矩阵形状: {self.A_h.shape}")
        except Exception as e:
            logger.error(f"加载连接矩阵失败: {e}")
            raise
        
        # 检查矩阵维度
        if self.A_t.shape != self.A_h.shape:
            raise ValueError(f"连接矩阵维度不匹配: {self.A_t.shape} != {self.A_h.shape}")
        
        self.N = self.A_t.shape[0]
        logger.info(f"节点数量: {self.N}")
        
        # 加载节点名称
        try:
            with open(self.node_names_path, 'r', encoding='utf-8') as f:
                node_data = json.load(f)
                if isinstance(node_data, list):
                    self.node_names = [item.get('name', f'Node_{item.get("id", i)}') 
                                     for i, item in enumerate(node_data)]
                elif isinstance(node_data, dict):
                    self.node_names = [node_data.get(str(i), f'Node_{i}') 
                                     for i in range(self.N)]
                else:
                    self.node_names = [f'Node_{i}' for i in range(self.N)]
            logger.info(f"加载了 {len(self.node_names)} 个节点名称")
        except Exception as e:
            logger.warning(f"加载节点名称失败: {e}，将使用默认名称")
            self.node_names = [f'Node_{i}' for i in range(self.N)]
        
        logger.info("数据加载完成")
    
    def step1_preprocess_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1: 基础计算与标准化
        
        Returns:
            处理后的正确和幻觉连接矩阵
        """
        logger.info("Step 1: 基础计算与标准化...")
        
        # 对称化
        A_t_sym = (self.A_t + self.A_t.T) / 2
        A_h_sym = (self.A_h + self.A_h.T) / 2
        
        # 对角置零
        np.fill_diagonal(A_t_sym, 0)
        np.fill_diagonal(A_h_sym, 0)
        
        # 计算差分矩阵
        self.Delta_A = A_h_sym - A_t_sym
        
        logger.info(f"差分矩阵统计: min={np.min(self.Delta_A):.4f}, "
                   f"max={np.max(self.Delta_A):.4f}, "
                   f"mean={np.mean(self.Delta_A):.4f}")
        
        return A_t_sym, A_h_sym
    
    def compute_global_metrics(self, A: np.ndarray) -> Dict[str, float]:
        """
        Step 2: 计算整体网络指标
        
        Args:
            A: 连接矩阵
            
        Returns:
            包含各种全局指标的字典
        """
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for i in range(self.N):
            G.add_node(i)
        
        # 添加边（只添加非零边）
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if abs(A[i, j]) > 1e-10:  # 忽略非常小的值
                    G.add_edge(i, j, weight=A[i, j])
        
        metrics = {}
        
        # 1. 平均强度 (mean strength)
        if G.number_of_edges() > 0:
            strengths = [G[u][v].get('weight', 0) for u, v in G.edges()]
            metrics['mean_strength'] = np.mean(np.abs(strengths))
        else:
            metrics['mean_strength'] = 0.0
        
        # 2. 稀疏度 (density)
        max_edges = self.N * (self.N - 1) / 2
        metrics['density'] = G.number_of_edges() / max_edges if max_edges > 0 else 0.0
        
        # 3. 平均聚类系数 (clustering coefficient)
        try:
            clustering = nx.clustering(G, weight='weight')
            metrics['clustering'] = np.mean(list(clustering.values()))
        except:
            metrics['clustering'] = 0.0
        
        # 4. 全局效率 (global efficiency)
        # 使用加权最短路径
        try:
            # 计算效率需要所有节点对之间的最短路径
            # 对于大图，使用近似方法
            if G.number_of_nodes() <= 100:
                # 小图：精确计算
                efficiency_sum = 0.0
                count = 0
                for i in range(self.N):
                    for j in range(i + 1, self.N):
                        try:
                            # 使用绝对权重作为距离
                            path_length = nx.shortest_path_length(
                                G, i, j, weight=lambda u, v, d: 1.0 / (abs(d['weight']) + 1e-10)
                            )
                            efficiency_sum += 1.0 / path_length if path_length > 0 else 0.0
                            count += 1
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            pass
                metrics['global_efficiency'] = efficiency_sum / count if count > 0 else 0.0
            else:
                # 大图：采样计算
                sample_size = min(1000, self.N * (self.N - 1) // 2)
                efficiency_sum = 0.0
                count = 0
                np.random.seed(42)
                for _ in range(sample_size):
                    i, j = np.random.choice(self.N, 2, replace=False)
                    try:
                        path_length = nx.shortest_path_length(
                            G, i, j, weight=lambda u, v, d: 1.0 / (abs(d['weight']) + 1e-10)
                        )
                        efficiency_sum += 1.0 / path_length if path_length > 0 else 0.0
                        count += 1
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass
                metrics['global_efficiency'] = efficiency_sum / count if count > 0 else 0.0
        except Exception as e:
            logger.warning(f"计算全局效率失败: {e}")
            metrics['global_efficiency'] = 0.0
        
        # 5. 模块度 (modularity)
        try:
            # 使用Louvain算法进行社区检测
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
            metrics['modularity'] = nx.community.modularity(G, communities, weight='weight')
            metrics['num_modules'] = len(communities)
        except Exception as e:
            logger.warning(f"计算模块度失败: {e}")
            metrics['modularity'] = 0.0
            metrics['num_modules'] = 1
        
        return metrics
    
    def compute_node_centrality(self, A: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Step 3: 计算节点级指标（中心性分析）
        
        Args:
            A: 连接矩阵
            
        Returns:
            包含各种中心性指标的字典
        """
        logger.info("计算节点中心性指标...")
        
        # 创建网络图
        G = nx.Graph()
        for i in range(self.N):
            G.add_node(i)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if abs(A[i, j]) > 1e-10:
                    G.add_edge(i, j, weight=A[i, j])
        
        centrality = {}
        
        # 1. Degree centrality (度中心性)
        degree_centrality = nx.degree_centrality(G)
        centrality['degree'] = np.array([degree_centrality.get(i, 0.0) for i in range(self.N)])
        
        # 2. Betweenness centrality (介数中心性)
        try:
            betweenness = nx.betweenness_centrality(G, weight='weight')
            centrality['betweenness'] = np.array([betweenness.get(i, 0.0) for i in range(self.N)])
        except:
            centrality['betweenness'] = np.zeros(self.N)
        
        # 3. Eigenvector centrality (特征向量中心性)
        try:
            eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            centrality['eigenvector'] = np.array([eigenvector.get(i, 0.0) for i in range(self.N)])
        except:
            centrality['eigenvector'] = np.zeros(self.N)
        
        # 4. Participation coefficient (参与系数)
        # 需要先进行社区划分
        try:
            communities = nx.community.louvain_communities(G, weight='weight', seed=42)
            # 创建节点到社区的映射
            node_to_community = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    node_to_community[node] = comm_id
            
            participation = np.zeros(self.N)
            for i in range(self.N):
                # 计算节点i的度
                degree_i = G.degree(i, weight='weight')
                if degree_i == 0:
                    participation[i] = 0.0
                    continue
                
                # 计算每个社区内的连接强度
                comm_connections = {}
                for neighbor in G.neighbors(i):
                    comm_id = node_to_community.get(neighbor, -1)
                    if comm_id not in comm_connections:
                        comm_connections[comm_id] = 0.0
                    comm_connections[comm_id] += abs(G[i][neighbor].get('weight', 0))
                
                # 计算参与系数
                if len(comm_connections) > 1:
                    participation_sum = 0.0
                    for comm_id, strength in comm_connections.items():
                        if comm_id >= 0:
                            participation_sum += (strength / degree_i) ** 2
                    participation[i] = 1 - participation_sum
                else:
                    participation[i] = 0.0
            
            centrality['participation'] = participation
        except Exception as e:
            logger.warning(f"计算参与系数失败: {e}")
            centrality['participation'] = np.zeros(self.N)
        
        return centrality
    
    def find_anomalous_subnetworks(self, Delta_A: np.ndarray) -> List[Dict]:
        """
        Step 4: 发现异常子网（伪NBS分析）
        
        Args:
            Delta_A: 连接差异矩阵
            
        Returns:
            异常子网列表
        """
        logger.info("Step 4: 发现异常子网...")
        
        # 计算Z分数
        upper_tri_indices = np.triu_indices_from(Delta_A, k=1)
        upper_tri_values = Delta_A[upper_tri_indices]
        
        mean_diff = np.mean(upper_tri_values)
        std_diff = np.std(upper_tri_values)
        
        if std_diff < 1e-10:
            logger.warning("差异矩阵标准差太小，无法计算Z分数")
            return []
        
        # 标准化
        Z_matrix = np.zeros_like(Delta_A)
        Z_matrix[upper_tri_indices] = (upper_tri_values - mean_diff) / std_diff
        Z_matrix = (Z_matrix + Z_matrix.T) / 2  # 对称化
        
        # 提取显著连接
        # 方法1: 基于Z分数阈值
        significant_mask = np.abs(Z_matrix) > self.z_threshold
        
        # 方法2: 基于top K%
        abs_delta = np.abs(Delta_A)
        threshold_value = np.percentile(abs_delta[upper_tri_indices], 100 - self.top_k_percent)
        top_k_mask = abs_delta > threshold_value
        
        # 合并两种方法
        final_mask = significant_mask | top_k_mask
        np.fill_diagonal(final_mask, False)
        
        # 构建差异网络图
        G = nx.Graph()
        for i in range(self.N):
            G.add_node(i)
        
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if final_mask[i, j]:
                    G.add_edge(i, j, weight=Delta_A[i, j], z_score=Z_matrix[i, j])
        
        # 提取连通成分（子网）
        subnets = []
        components = list(nx.connected_components(G))
        
        for comp_id, component in enumerate(components):
            if len(component) < 2:  # 至少需要2个节点
                continue
            
            # 提取子网的边
            subgraph = G.subgraph(component)
            edges = list(subgraph.edges(data=True))
            
            # 计算子网统计信息
            node_list = list(component)
            avg_diff = np.mean([Delta_A[i, j] for i, j in subgraph.edges()])
            avg_z = np.mean([e[2]['z_score'] for e in edges])
            
            subnets.append({
                'subnet_id': comp_id,
                'nodes': sorted(node_list),
                'num_nodes': len(node_list),
                'num_edges': len(edges),
                'avg_difference': float(avg_diff),
                'avg_z_score': float(avg_z),
                'node_names': [self.node_names[i] for i in node_list]
            })
        
        # 按节点数量排序
        subnets.sort(key=lambda x: x['num_nodes'], reverse=True)
        
        logger.info(f"发现 {len(subnets)} 个异常子网")
        for i, subnet in enumerate(subnets[:5]):  # 显示前5个
            logger.info(f"  子网 {i+1}: {subnet['num_nodes']} 个节点, "
                       f"{subnet['num_edges']} 条边, "
                       f"平均差异={subnet['avg_difference']:.4f}")
        
        return subnets
    
    def analyze_community_structure(self, A_t: np.ndarray, A_h: np.ndarray) -> Dict:
        """
        Step 5: 社区与模块结构分析
        
        Args:
            A_t: 正确样本连接矩阵
            A_h: 幻觉样本连接矩阵
            
        Returns:
            社区分析结果
        """
        logger.info("Step 5: 社区与模块结构分析...")
        
        def get_communities(A):
            """获取网络的社区划分"""
            G = nx.Graph()
            for i in range(self.N):
                G.add_node(i)
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if abs(A[i, j]) > 1e-10:
                        G.add_edge(i, j, weight=A[i, j])
            
            try:
                communities = nx.community.louvain_communities(G, weight='weight', seed=42)
                modularity = nx.community.modularity(G, communities, weight='weight')
                return communities, modularity, G
            except:
                return [set(range(self.N))], 0.0, G
        
        # 正确样本的社区
        comm_t, Q_t, G_t = get_communities(A_t)
        
        # 幻觉样本的社区
        comm_h, Q_h, G_h = get_communities(A_h)
        
        # 计算模块内和跨模块连接
        def compute_module_stats(A, communities):
            """计算模块内和跨模块连接统计"""
            # 创建节点到社区的映射
            node_to_comm = {}
            for comm_id, comm in enumerate(communities):
                for node in comm:
                    node_to_comm[node] = comm_id
            
            within_connections = []
            between_connections = []
            
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    if abs(A[i, j]) > 1e-10:
                        comm_i = node_to_comm.get(i, -1)
                        comm_j = node_to_comm.get(j, -1)
                        if comm_i == comm_j and comm_i >= 0:
                            within_connections.append(abs(A[i, j]))
                        else:
                            between_connections.append(abs(A[i, j]))
            
            return {
                'within_module_mean': np.mean(within_connections) if within_connections else 0.0,
                'between_module_mean': np.mean(between_connections) if between_connections else 0.0,
                'within_module_count': len(within_connections),
                'between_module_count': len(between_connections)
            }
        
        stats_t = compute_module_stats(A_t, comm_t)
        stats_h = compute_module_stats(A_h, comm_h)
        
        results = {
            'correct': {
                'num_modules': len(comm_t),
                'modularity': float(Q_t),
                'module_stats': stats_t,
                'communities': [sorted(list(comm)) for comm in comm_t]
            },
            'hallucination': {
                'num_modules': len(comm_h),
                'modularity': float(Q_h),
                'module_stats': stats_h,
                'communities': [sorted(list(comm)) for comm in comm_h]
            },
            'delta_modularity': float(Q_h - Q_t)
        }
        
        logger.info(f"正确样本模块度: {Q_t:.4f}, 模块数: {len(comm_t)}")
        logger.info(f"幻觉样本模块度: {Q_h:.4f}, 模块数: {len(comm_h)}")
        logger.info(f"模块度变化: {Q_h - Q_t:.4f}")
        
        return results
    
    def visualize_results(self, A_t: np.ndarray, A_h: np.ndarray, Delta_A: np.ndarray,
                         global_metrics_t: Dict, global_metrics_h: Dict,
                         centrality_t: Dict, centrality_h: Dict,
                         subnets: List[Dict], community_results: Dict) -> None:
        """
        Step 6: 可视化
        
        Args:
            A_t: 正确样本连接矩阵
            A_h: 幻觉样本连接矩阵
            Delta_A: 差异矩阵
            global_metrics_t: 正确样本全局指标
            global_metrics_h: 幻觉样本全局指标
            centrality_t: 正确样本中心性
            centrality_h: 幻觉样本中心性
            subnets: 异常子网列表
            community_results: 社区分析结果
        """
        logger.info("Step 6: 生成可视化...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. ΔFC 热图
        plt.figure(figsize=(12, 10))
        sns.heatmap(Delta_A, cmap='RdBu_r', center=0, 
                   vmin=-np.max(np.abs(Delta_A)), vmax=np.max(np.abs(Delta_A)),
                   square=True, cbar_kws={'label': 'ΔFC (Hallucination - Correct)'})
        plt.title('ΔFC Heatmap (Hallucination - Correct)', fontsize=14, pad=20)
        plt.xlabel('Parcel Index', fontsize=12)
        plt.ylabel('Parcel Index', fontsize=12)
        plt.tight_layout()
        plt.savefig(viz_dir / "delta_fc_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 全局指标对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics_names = ['mean_strength', 'density', 'clustering', 
                        'global_efficiency', 'modularity', 'num_modules']
        metrics_labels = ['Mean Strength', 'Density', 'Clustering', 
                         'Global Efficiency', 'Modularity', 'Num Modules']
        
        for idx, (name, label) in enumerate(zip(metrics_names, metrics_labels)):
            ax = axes[idx // 3, idx % 3]
            values = [global_metrics_t.get(name, 0), global_metrics_h.get(name, 0)]
            bars = ax.bar(['Correct', 'Hallucination'], values, 
                         color=['blue', 'red'], alpha=0.7)
            ax.set_ylabel(label)
            ax.set_title(f'{label} Comparison')
            ax.grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "global_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 节点中心性差异图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        centrality_names = ['degree', 'betweenness', 'eigenvector', 'participation']
        centrality_labels = ['Degree Centrality', 'Betweenness Centrality', 
                           'Eigenvector Centrality', 'Participation Coefficient']
        
        for idx, (name, label) in enumerate(zip(centrality_names, centrality_labels)):
            ax = axes[idx // 2, idx % 2]
            delta_centrality = centrality_h[name] - centrality_t[name]
            
            # 选择差异最大的前20个节点
            top_indices = np.argsort(np.abs(delta_centrality))[-20:][::-1]
            
            x_pos = np.arange(len(top_indices))
            colors = ['red' if delta_centrality[i] > 0 else 'blue' for i in top_indices]
            
            ax.barh(x_pos, delta_centrality[top_indices], color=colors, alpha=0.7)
            ax.set_yticks(x_pos)
            ax.set_yticklabels([self.node_names[i] if i < len(self.node_names) 
                               else f'Node_{i}' for i in top_indices], fontsize=8)
            ax.set_xlabel(f'Δ{label}')
            ax.set_title(f'Top 20 Nodes by Δ{label}')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "node_centrality_differences.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 异常子网可视化（显示最大的几个子网）
        if subnets:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_subnets = subnets[:5]  # 显示前5个子网
            
            subnet_ids = [f"Subnet {s['subnet_id']+1}" for s in top_subnets]
            num_nodes = [s['num_nodes'] for s in top_subnets]
            num_edges = [s['num_edges'] for s in top_subnets]
            
            x = np.arange(len(top_subnets))
            width = 0.35
            
            ax.bar(x - width/2, num_nodes, width, label='Nodes', alpha=0.7, color='blue')
            ax.bar(x + width/2, num_edges, width, label='Edges', alpha=0.7, color='red')
            
            ax.set_xlabel('Subnetwork')
            ax.set_ylabel('Count')
            ax.set_title('Top 5 Anomalous Subnetworks')
            ax.set_xticks(x)
            ax.set_xticklabels(subnet_ids)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "anomalous_subnetworks.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"可视化结果已保存到: {viz_dir}")
    
    def save_results(self, A_t: np.ndarray, A_h: np.ndarray, Delta_A: np.ndarray,
                    global_metrics_t: Dict, global_metrics_h: Dict,
                    centrality_t: Dict, centrality_h: Dict,
                    subnets: List[Dict], community_results: Dict) -> None:
        """
        Step 7: 保存分析结果
        
        Args:
            A_t: 正确样本连接矩阵
            A_h: 幻觉样本连接矩阵
            Delta_A: 差异矩阵
            global_metrics_t: 正确样本全局指标
            global_metrics_h: 幻觉样本全局指标
            centrality_t: 正确样本中心性
            centrality_h: 幻觉样本中心性
            subnets: 异常子网列表
            community_results: 社区分析结果
        """
        logger.info("Step 7: 保存分析结果...")
        
        # 保存数值结果
        results = {
            'global_metrics': {
                'correct': global_metrics_t,
                'hallucination': global_metrics_h,
                'delta': {k: global_metrics_h.get(k, 0) - global_metrics_t.get(k, 0) 
                         for k in global_metrics_t.keys()}
            },
            'node_centrality': {
                'correct': {k: v.tolist() for k, v in centrality_t.items()},
                'hallucination': {k: v.tolist() for k, v in centrality_h.items()},
                'delta': {k: (centrality_h[k] - centrality_t[k]).tolist() 
                         for k in centrality_t.keys()}
            },
            'anomalous_subnetworks': subnets,
            'community_analysis': community_results,
            'node_names': self.node_names
        }
        
        # 保存JSON结果
        results_file = self.output_dir / "connection_analysis_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存矩阵
        np.save(self.output_dir / "delta_connectivity_matrix.npy", Delta_A)
        
        # 生成文本报告
        report_file = self.output_dir / "analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("连接组分析报告 (Connection Analysis Plus)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("一、总体网络指标对比\n")
            f.write("-" * 80 + "\n")
            for metric in ['mean_strength', 'density', 'clustering', 'global_efficiency', 'modularity']:
                val_t = global_metrics_t.get(metric, 0)
                val_h = global_metrics_h.get(metric, 0)
                delta = val_h - val_t
                f.write(f"{metric:20s}: Correct={val_t:8.4f}, Hallucination={val_h:8.4f}, "
                       f"Δ={delta:8.4f} ({delta/val_t*100:+.2f}%)\n")
            f.write("\n")
            
            f.write("二、节点中心性分析（Top 10差异最大的节点）\n")
            f.write("-" * 80 + "\n")
            for cent_type in ['degree', 'betweenness', 'eigenvector', 'participation']:
                delta_cent = centrality_h[cent_type] - centrality_t[cent_type]
                top_indices = np.argsort(np.abs(delta_cent))[-10:][::-1]
                f.write(f"\n{cent_type.upper()} Centrality:\n")
                for idx in top_indices:
                    f.write(f"  Node {idx:3d} ({self.node_names[idx]:30s}): "
                           f"Δ={delta_cent[idx]:8.4f}\n")
            f.write("\n")
            
            f.write("三、异常子网分析\n")
            f.write("-" * 80 + "\n")
            f.write(f"共发现 {len(subnets)} 个异常子网\n\n")
            for i, subnet in enumerate(subnets[:10]):  # 显示前10个
                f.write(f"子网 {i+1} (ID={subnet['subnet_id']}):\n")
                f.write(f"  节点数: {subnet['num_nodes']}, 边数: {subnet['num_edges']}\n")
                f.write(f"  平均差异: {subnet['avg_difference']:.4f}, "
                       f"平均Z分数: {subnet['avg_z_score']:.4f}\n")
                f.write(f"  节点列表: {subnet['node_names'][:5]}...\n")
                f.write("\n")
            
            f.write("四、社区结构分析\n")
            f.write("-" * 80 + "\n")
            f.write(f"正确样本: 模块度={community_results['correct']['modularity']:.4f}, "
                   f"模块数={community_results['correct']['num_modules']}\n")
            f.write(f"幻觉样本: 模块度={community_results['hallucination']['modularity']:.4f}, "
                   f"模块数={community_results['hallucination']['num_modules']}\n")
            f.write(f"模块度变化: ΔQ={community_results['delta_modularity']:.4f}\n")
            if community_results['delta_modularity'] > 0:
                f.write("  → 幻觉网络模块度增高，提示信息整合受损，网络呈现更分散的拓扑结构\n")
            else:
                f.write("  → 幻觉网络模块度降低，提示网络更整合\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("分析完成\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"结果已保存到: {self.output_dir}")
        logger.info(f"  - JSON结果: {results_file}")
        logger.info(f"  - 文本报告: {report_file}")
        logger.info(f"  - 差异矩阵: {self.output_dir / 'delta_connectivity_matrix.npy'}")
    
    def run_analysis(self) -> None:
        """运行完整的连接组分析"""
        try:
            logger.info("开始连接组分析...")
            
            # 加载数据
            self.load_data()
            
            # Step 1: 基础计算与标准化
            A_t, A_h = self.step1_preprocess_matrices()
            
            # Step 2: 计算整体网络指标
            logger.info("Step 2: 计算整体网络指标...")
            global_metrics_t = self.compute_global_metrics(A_t)
            global_metrics_h = self.compute_global_metrics(A_h)
            
            # Step 3: 计算节点级指标
            logger.info("Step 3: 计算节点级指标...")
            centrality_t = self.compute_node_centrality(A_t)
            centrality_h = self.compute_node_centrality(A_h)
            
            # Step 4: 发现异常子网
            subnets = self.find_anomalous_subnetworks(self.Delta_A)
            
            # Step 5: 社区与模块结构分析
            community_results = self.analyze_community_structure(A_t, A_h)
            
            # Step 6: 可视化
            self.visualize_results(A_t, A_h, self.Delta_A,
                                 global_metrics_t, global_metrics_h,
                                 centrality_t, centrality_h,
                                 subnets, community_results)
            
            # Step 7: 保存结果
            self.save_results(A_t, A_h, self.Delta_A,
                            global_metrics_t, global_metrics_h,
                            centrality_t, centrality_h,
                            subnets, community_results)
            
            logger.info("连接组分析完成！")
            
        except Exception as e:
            logger.error(f"分析过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='连接组分析增强版')
    parser.add_argument('--correct_matrix', type=str, required=True,
                       help='正确样本的平均连接矩阵路径 (.npy)')
    parser.add_argument('--incorrect_matrix', type=str, required=True,
                       help='幻觉样本的平均连接矩阵路径 (.npy)')
    parser.add_argument('--node_names', type=str, required=True,
                       help='节点名称文件路径 (.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--z_threshold', type=float, default=2.0,
                       help='Z分数阈值，用于提取显著连接 (默认: 2.0)')
    parser.add_argument('--top_k_percent', type=float, default=5.0,
                       help='提取top K%%的高差异边 (默认: 5.0)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.correct_matrix):
        logger.error(f"正确样本连接矩阵不存在: {args.correct_matrix}")
        sys.exit(1)
    
    if not os.path.exists(args.incorrect_matrix):
        logger.error(f"幻觉样本连接矩阵不存在: {args.incorrect_matrix}")
        sys.exit(1)
    
    if not os.path.exists(args.node_names):
        logger.warning(f"节点名称文件不存在: {args.node_names}")
    
    # 创建分析器并运行分析
    analyzer = ConnectionAnalysisPlus(
        correct_matrix_path=args.correct_matrix,
        incorrect_matrix_path=args.incorrect_matrix,
        node_names_path=args.node_names,
        output_dir=args.output_dir,
        z_threshold=args.z_threshold,
        top_k_percent=args.top_k_percent
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

