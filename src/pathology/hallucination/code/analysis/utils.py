#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数

提供幻觉机制分析中使用的通用工具函数，包括数据处理、统计计算、可视化等。

作者: AI Assistant
日期: 2024
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# 设置日志
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """
        加载JSONL格式文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据列表
        """
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{line_num}行JSON解析错误: {e}")
                        continue
        except FileNotFoundError:
            logger.error(f"文件不存在: {file_path}")
            raise
        except Exception as e:
            logger.error(f"加载文件失败: {e}")
            raise
        
        return data
    
    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str) -> None:
        """
        保存数据为JSONL格式
        
        Args:
            data: 数据列表
            file_path: 文件路径
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"保存JSONL文件失败: {e}")
            raise
    
    @staticmethod
    def normalize_activations(activations: np.ndarray, method: str = 'l2', 
                            epsilon: float = 1e-8) -> np.ndarray:
        """
        归一化激活数据
        
        Args:
            activations: 激活矩阵，形状为 (T, P) 或 (P,)
            method: 归一化方法 ('l2', 'zscore', 'minmax')
            epsilon: 小常数，避免除零
            
        Returns:
            归一化后的激活矩阵
        """
        if method == 'l2':
            # L2归一化
            if activations.ndim == 1:
                norm = np.linalg.norm(activations)
                norm = max(norm, epsilon)
                return activations / norm
            else:
                norms = np.linalg.norm(activations, axis=1, keepdims=True)
                norms = np.maximum(norms, epsilon)
                return activations / norms
        
        elif method == 'zscore':
            # Z-score标准化
            if activations.ndim == 1:
                return (activations - np.mean(activations)) / (np.std(activations) + epsilon)
            else:
                mean = np.mean(activations, axis=0, keepdims=True)
                std = np.std(activations, axis=0, keepdims=True)
                return (activations - mean) / (std + epsilon)
        
        elif method == 'minmax':
            # Min-Max归一化
            if activations.ndim == 1:
                min_val = np.min(activations)
                max_val = np.max(activations)
                if max_val - min_val < epsilon:
                    return np.zeros_like(activations)
                return (activations - min_val) / (max_val - min_val)
            else:
                min_val = np.min(activations, axis=0, keepdims=True)
                max_val = np.max(activations, axis=0, keepdims=True)
                range_val = max_val - min_val
                range_val = np.maximum(range_val, epsilon)
                return (activations - min_val) / range_val
        
        else:
            raise ValueError(f"不支持的归一化方法: {method}")


class StatisticalAnalyzer:
    """统计分析器"""
    
    @staticmethod
    def compute_connectivity_matrix(activations: np.ndarray, 
                                  method: str = 'cosine') -> np.ndarray:
        """
        计算功能连接矩阵
        
        Args:
            activations: 激活矩阵，形状为 (T, P)
            method: 计算方法 ('cosine', 'pearson', 'spearman')
            
        Returns:
            连接矩阵，形状为 (P, P)
        """
        if method == 'cosine':
            # 使用cosine similarity
            normalized = DataProcessor.normalize_activations(activations, method='l2')
            T = normalized.shape[0]
            connectivity = (1.0 / T) * np.dot(normalized.T, normalized)
            return connectivity
        
        elif method == 'pearson':
            # Pearson相关系数
            return np.corrcoef(activations.T)
        
        elif method == 'spearman':
            # Spearman相关系数
            corr_matrix = np.zeros((activations.shape[1], activations.shape[1]))
            for i in range(activations.shape[1]):
                for j in range(activations.shape[1]):
                    corr, _ = stats.spearmanr(activations[:, i], activations[:, j])
                    corr_matrix[i, j] = corr
            return corr_matrix
        
        else:
            raise ValueError(f"不支持的计算方法: {method}")
    
    @staticmethod
    def detect_anomalies(correct_data: np.ndarray, incorrect_data: np.ndarray,
                        method: str = 'ttest', threshold: float = 0.05) -> Dict:
        """
        检测异常
        
        Args:
            correct_data: 正确样本数据，形状为 (N, P)
            incorrect_data: 异常样本数据，形状为 (M, P)
            method: 检测方法 ('ttest', 'mannwhitney', 'zscore')
            threshold: 显著性阈值
            
        Returns:
            异常检测结果字典
        """
        if correct_data.shape[1] != incorrect_data.shape[1]:
            raise ValueError("数据维度不匹配")
        
        p_dim = correct_data.shape[1]
        t_stats = []
        p_values = []
        effect_sizes = []
        
        for i in range(p_dim):
            try:
                if method == 'ttest':
                    t_stat, p_val = stats.ttest_ind(incorrect_data[:, i], correct_data[:, i])
                    t_stats.append(t_stat)
                    p_values.append(p_val)
                    
                    # 计算效应量 (Cohen's d)
                    pooled_std = np.sqrt(((len(correct_data) - 1) * np.var(correct_data[:, i]) + 
                                        (len(incorrect_data) - 1) * np.var(incorrect_data[:, i])) / 
                                       (len(correct_data) + len(incorrect_data) - 2))
                    effect_size = (np.mean(incorrect_data[:, i]) - np.mean(correct_data[:, i])) / pooled_std
                    effect_sizes.append(effect_size)
                
                elif method == 'mannwhitney':
                    u_stat, p_val = stats.mannwhitneyu(incorrect_data[:, i], correct_data[:, i], 
                                                      alternative='two-sided')
                    t_stats.append(u_stat)
                    p_values.append(p_val)
                    
                    # 计算效应量 (Mann-Whitney U)
                    effect_size = (u_stat - len(correct_data) * len(incorrect_data) / 2) / \
                                np.sqrt(len(correct_data) * len(incorrect_data) * 
                                       (len(correct_data) + len(incorrect_data) + 1) / 12)
                    effect_sizes.append(effect_size)
                
                elif method == 'zscore':
                    # Z-score方法
                    combined = np.concatenate([correct_data[:, i], incorrect_data[:, i]])
                    mean_combined = np.mean(combined)
                    std_combined = np.std(combined)
                    
                    z_correct = (np.mean(correct_data[:, i]) - mean_combined) / std_combined
                    z_incorrect = (np.mean(incorrect_data[:, i]) - mean_combined) / std_combined
                    
                    t_stats.append(z_incorrect - z_correct)
                    p_values.append(2 * (1 - stats.norm.cdf(abs(z_incorrect - z_correct))))
                    effect_sizes.append(z_incorrect - z_correct)
                
            except Exception as e:
                logger.warning(f"第{i}个特征异常检测失败: {e}")
                t_stats.append(0.0)
                p_values.append(1.0)
                effect_sizes.append(0.0)
        
        t_stats = np.array(t_stats)
        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)
        
        # 找出显著异常的特征
        significant_mask = p_values < threshold
        significant_indices = np.where(significant_mask)[0]
        
        # 按效应量绝对值排序
        anomaly_scores = np.abs(effect_sizes)
        top_anomalous_indices = np.argsort(anomaly_scores)[::-1]
        
        return {
            't_stats': t_stats.tolist(),
            'p_values': p_values.tolist(),
            'effect_sizes': effect_sizes.tolist(),
            'significant_indices': significant_indices.tolist(),
            'top_anomalous_indices': top_anomalous_indices.tolist(),
            'anomaly_scores': anomaly_scores.tolist()
        }
    
    @staticmethod
    def compute_connectivity_anomaly_score(connectivity_diff: np.ndarray) -> Dict:
        """
        计算连接异常分数
        
        Args:
            connectivity_diff: 连接差异矩阵
            
        Returns:
            连接异常统计字典
        """
        # 上三角部分的平均绝对值
        upper_tri_indices = np.triu_indices_from(connectivity_diff, k=1)
        upper_tri_diff = connectivity_diff[upper_tri_indices]
        mean_abs_diff = np.mean(np.abs(upper_tri_diff))
        
        # Frobenius范数
        frobenius_norm = np.linalg.norm(connectivity_diff, 'fro')
        
        # 最大绝对差异
        max_abs_diff = np.max(np.abs(connectivity_diff))
        
        # 异常连接数量 (超过阈值的连接数)
        threshold = np.std(connectivity_diff) * 2  # 2倍标准差作为阈值
        anomaly_connections = np.sum(np.abs(connectivity_diff) > threshold)
        total_connections = connectivity_diff.shape[0] * (connectivity_diff.shape[0] - 1) // 2
        anomaly_ratio = anomaly_connections / total_connections
        
        return {
            'mean_abs_diff': float(mean_abs_diff),
            'frobenius_norm': float(frobenius_norm),
            'max_abs_diff': float(max_abs_diff),
            'anomaly_connections': int(anomaly_connections),
            'anomaly_ratio': float(anomaly_ratio),
            'threshold': float(threshold)
        }


class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_activation_heatmap(activation_data: np.ndarray, 
                              title: str = "Activation Heatmap",
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> None:
        """
        绘制激活热力图
        
        Args:
            activation_data: 激活数据，形状为 (N, P)
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        sns.heatmap(activation_data, cmap='viridis', cbar=True)
        plt.title(title)
        plt.xlabel('Parcel Index')
        plt.ylabel('Sample Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_connectivity_matrix(connectivity_matrix: np.ndarray,
                               title: str = "Connectivity Matrix",
                               figsize: Tuple[int, int] = (10, 10),
                               save_path: Optional[str] = None) -> None:
        """
        绘制连接矩阵热力图
        
        Args:
            connectivity_matrix: 连接矩阵
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        plt.figure(figsize=figsize)
        sns.heatmap(connectivity_matrix, cmap='RdBu_r', center=0, 
                   square=True, cbar=True)
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_anomaly_ranking(anomaly_scores: np.ndarray, 
                           labels: Optional[List[str]] = None,
                           title: str = "Anomaly Ranking",
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None) -> None:
        """
        绘制异常排名图
        
        Args:
            anomaly_scores: 异常分数数组
            labels: 标签列表
            title: 图表标题
            top_n: 显示前N个
            figsize: 图表大小
            save_path: 保存路径
        """
        # 获取前top_n个异常
        top_indices = np.argsort(anomaly_scores)[::-1][:top_n]
        top_scores = anomaly_scores[top_indices]
        
        if labels is None:
            labels = [f"Item {i}" for i in top_indices]
        else:
            labels = [labels[i] for i in top_indices]
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(top_scores)), top_scores)
        plt.title(title)
        plt.xlabel('Ranking')
        plt.ylabel('Anomaly Score')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_connectivity_network(connectivity_matrix: np.ndarray,
                                threshold: float = 0.3,
                                title: str = "Connectivity Network",
                                figsize: Tuple[int, int] = (12, 12),
                                save_path: Optional[str] = None) -> None:
        """
        绘制连接网络图
        
        Args:
            connectivity_matrix: 连接矩阵
            threshold: 连接阈值
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX未安装，无法绘制网络图")
            return
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        n_nodes = connectivity_matrix.shape[0]
        G.add_nodes_from(range(n_nodes))
        
        # 添加边（只添加超过阈值的连接）
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                if abs(connectivity_matrix[i, j]) > threshold:
                    G.add_edge(i, j, weight=abs(connectivity_matrix[i, j]))
        
        # 绘制网络
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=300, alpha=0.7)
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class FileManager:
    """文件管理器"""
    
    @staticmethod
    def ensure_dir(path: str) -> None:
        """确保目录存在"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """获取文件信息"""
        path = Path(file_path)
        if not path.exists():
            return {'exists': False}
        
        stat = path.stat()
        return {
            'exists': True,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'is_file': path.is_file(),
            'is_dir': path.is_dir()
        }
    
    @staticmethod
    def backup_file(file_path: str, backup_suffix: str = '.backup') -> str:
        """备份文件"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        backup_path = path.with_suffix(path.suffix + backup_suffix)
        import shutil
        shutil.copy2(file_path, backup_path)
        return str(backup_path)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def validate_data_format(data: Dict, required_fields: List[str]) -> bool:
    """
    验证数据格式
    
    Args:
        data: 数据字典
        required_fields: 必需字段列表
        
    Returns:
        是否有效
    """
    for field in required_fields:
        if field not in data:
            logger.error(f"缺少必需字段: {field}")
            return False
    return True


def safe_divide(numerator: Union[int, float], denominator: Union[int, float], 
                default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值
        
    Returns:
        除法结果或默认值
    """
    try:
        if abs(denominator) < 1e-10:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
