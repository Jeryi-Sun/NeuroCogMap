#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case Study 分析工具

基于幻觉case分析设计，对单个样本进行深入的异常分析，
包括时间窗定位、模块异常检测、连接异常分析和干预验证。

作者: Jeryi
日期: 2025
"""

import json
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import traceback
from matplotlib import font_manager as fm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 字体配置：自动选择系统可用的 CJK 字体作为首选，避免中文缺字
def _configure_fonts() -> None:
    try:
        candidate_families = [
            'Noto Sans CJK SC', 'Noto Sans CJK', 'Noto Serif CJK SC', 'Noto Serif CJK',
            'Source Han Sans CN', 'Source Han Sans', 'Source Han Serif CN', 'Source Han Serif',
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Sarasa Gothic SC',
            'Microsoft YaHei', 'SimHei', 'PingFang SC', 'Heiti SC',
            'DejaVu Sans'
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in candidate_families if name in available]
        # 统一使用无衬线族，按可用顺序设置回退链
        plt.rcParams['font.family'] = 'sans-serif'
        if chosen:
            plt.rcParams['font.sans-serif'] = chosen
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        # 兜底：至少保证负号正常
        plt.rcParams['axes.unicode_minus'] = False

_configure_fonts()

class CaseStudyAnalyzer:
    """Case Study 分析器"""
    
    def __init__(self, correct_jsonl_path: str, incorrect_jsonl_path: str,
                 mapping_json_path: str, parcel_desc_path: str, cap_desc_path: str,
                 output_dir: str, window_size: int = 5, top_k: int = 5,
                 epsilon: float = 1e-8):
        """
        初始化Case Study分析器
        
        Args:
            correct_jsonl_path: 正确样本数据路径
            incorrect_jsonl_path: 幻觉样本数据路径
            mapping_json_path: Capability-Parcel映射文件路径
            parcel_desc_path: Parcel功能描述文件路径
            cap_desc_path: Capability描述文件路径
            output_dir: 输出目录
            window_size: 滑窗大小
            top_k: 展示的top异常项数量
            epsilon: L2归一化小常数
        """
        self.correct_jsonl_path = correct_jsonl_path
        self.incorrect_jsonl_path = incorrect_jsonl_path
        self.mapping_json_path = mapping_json_path
        self.parcel_desc_path = parcel_desc_path
        self.cap_desc_path = cap_desc_path
        self.output_dir = Path(output_dir)
        self.window_size = window_size
        self.top_k = top_k
        self.epsilon = epsilon
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.correct_data = []
        self.incorrect_data = []
        self.parcel_descriptions = {}
        self.capability_descriptions = {}
        self.capability_parcel_mapping = {}
        self.mapping_matrix = None
        self.baseline_stats = {}
        
    def load_data(self) -> None:
        """加载所有数据"""
        logger.info("加载数据...")
        
        # 加载问答数据
        self._load_jsonl_data(self.correct_jsonl_path, self.correct_data)
        self._load_jsonl_data(self.incorrect_jsonl_path, self.incorrect_data)
        
        # 加载描述文件
        self._load_descriptions()
        
        # 加载映射文件
        self._load_mapping()
        
        logger.info(f"加载完成: {len(self.correct_data)} 正确样本, {len(self.incorrect_data)} 幻觉样本")
    
    def _load_jsonl_data(self, file_path: str, data_list: List) -> None:
        """加载JSONL数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_list.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"加载数据失败 {file_path}: {e}")
            raise
    
    def _load_descriptions(self) -> None:
        """加载描述文件"""
        # 加载Parcel描述
        try:
            with open(self.parcel_desc_path, 'r', encoding='utf-8') as f:
                self.parcel_descriptions = json.load(f)
        except Exception as e:
            logger.warning(f"加载Parcel描述失败: {e}")
            self.parcel_descriptions = {}
        
        # 加载Capability描述
        try:
            with open(self.cap_desc_path, 'r', encoding='utf-8') as f:
                self.capability_descriptions = json.load(f)
        except Exception as e:
            logger.warning(f"加载Capability描述失败: {e}")
            self.capability_descriptions = {}
    
    def _load_mapping(self) -> None:
        """加载Capability-Parcel映射"""
        try:
            with open(self.mapping_json_path, 'r', encoding='utf-8') as f:
                self.capability_parcel_mapping = json.load(f)
        except Exception as e:
            logger.error(f"加载映射文件失败: {e}")
            raise
    
    def build_mapping_matrix(self, parcel_dim: int) -> np.ndarray:
        """构建Capability-Parcel映射矩阵"""
        capability_names = list(self.capability_parcel_mapping.keys())
        capability_dim = len(capability_names)
        
        mapping_matrix = np.zeros((parcel_dim, capability_dim), dtype=np.float32)
        
        for cap_idx, cap_name in enumerate(capability_names):
            if cap_name not in self.capability_parcel_mapping:
                continue
            
            cap_data = self.capability_parcel_mapping[cap_name]
            if 'ranking' not in cap_data:
                continue
            
            ranking = cap_data['ranking']
            if not isinstance(ranking, list):
                continue
            
            # 提取权重并归一化
            weights = []
            parcel_indices = []
            
            for item in ranking:
                if not isinstance(item, list) or len(item) != 2:
                    continue
                
                parcel_name, weight = item
                try:
                    if parcel_name.startswith('parcel_'):
                        parcel_idx = int(parcel_name.split('_')[1])
                        if 0 <= parcel_idx < parcel_dim:
                            weights.append(float(weight))
                            parcel_indices.append(parcel_idx)
                except (ValueError, IndexError):
                    continue
            
            if len(weights) == 0:
                continue
            
            # 归一化权重
            weights = np.array(weights)
            weights = weights / (np.sum(weights) + self.epsilon)
            
            # 填充映射矩阵
            for parcel_idx, weight in zip(parcel_indices, weights):
                mapping_matrix[parcel_idx, cap_idx] = weight
        
        return mapping_matrix
    
    def _resolve_activation_file(self, is_correct: bool) -> str:
        """根据输入 JSONL 路径稳健地解析激活数据文件路径。"""
        # 期望目录结构：<dataset_dir>/parcels_token_acts/{correct|incorrect}/token_parcels.jsonl
        base_dir = os.path.dirname(self.correct_jsonl_path if is_correct else self.incorrect_jsonl_path)
        target_sub = 'correct' if is_correct else 'incorrect'
        candidate = os.path.join(base_dir, 'parcels_token_acts', target_sub, 'token_parcels.jsonl')
        return candidate

    def load_activation_data(self, sample_id: int, is_correct: bool) -> Optional[np.ndarray]:
        """加载指定样本的激活数据"""
        activation_file = self._resolve_activation_file(is_correct)
        try:
            with open(activation_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        data = json.loads(line.strip())
                        if data.get('index') == sample_id:
                            return np.array(data['token_parcel_acts'], dtype=np.float32)
        except Exception as e:
            logger.error(f"加载激活数据失败: {e}")
        return None
    
    def compute_baseline_stats(self) -> None:
        """计算基线统计信息"""
        logger.info("计算基线统计信息...")
        
        # 加载所有正确样本的激活数据
        correct_activations = []
        activation_file = self._resolve_activation_file(True)
        
        try:
            with open(activation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        activations = np.array(data['token_parcel_acts'], dtype=np.float32)
                        correct_activations.append(activations)
        except Exception as e:
            logger.error(f"加载基线激活数据失败: {e}")
            return
        
        if not correct_activations:
            logger.error("没有基线激活数据")
            return
        
        # 计算Parcel基线统计
        all_parcel_acts = np.concatenate(correct_activations, axis=0)
        parcel_mean = np.mean(all_parcel_acts, axis=0)
        parcel_std = np.std(all_parcel_acts, axis=0)
        
        # 计算Capability基线统计
        if self.mapping_matrix is not None:
            all_capability_acts = []
            for acts in correct_activations:
                cap_acts = np.dot(acts, self.mapping_matrix)
                all_capability_acts.append(cap_acts)
            
            all_capability_acts = np.concatenate(all_capability_acts, axis=0)
            capability_mean = np.mean(all_capability_acts, axis=0)
            capability_std = np.std(all_capability_acts, axis=0)
        else:
            capability_mean = None
            capability_std = None
        
        # 计算连接基线
        parcel_connectivities = []
        for acts in correct_activations:
            normalized_acts = self._normalize_activations(acts)
            conn_matrix = np.dot(normalized_acts.T, normalized_acts) / normalized_acts.shape[0]
            parcel_connectivities.append(conn_matrix)
        
        parcel_conn_baseline = np.mean(parcel_connectivities, axis=0)
        
        if capability_mean is not None:
            capability_connectivities = []
            for acts in correct_activations:
                cap_acts = np.dot(acts, self.mapping_matrix)
                normalized_cap_acts = self._normalize_activations(cap_acts)
                conn_matrix = np.dot(normalized_cap_acts.T, normalized_cap_acts) / normalized_cap_acts.shape[0]
                capability_connectivities.append(conn_matrix)
            
            capability_conn_baseline = np.mean(capability_connectivities, axis=0)
        else:
            capability_conn_baseline = None
        
        self.baseline_stats = {
            'parcel_mean': parcel_mean,
            'parcel_std': parcel_std,
            'capability_mean': capability_mean,
            'capability_std': capability_std,
            'parcel_conn_baseline': parcel_conn_baseline,
            'capability_conn_baseline': capability_conn_baseline
        }
        
        logger.info("基线统计计算完成")
    
    def _normalize_activations(self, activations: np.ndarray) -> np.ndarray:
        """L2归一化激活向量"""
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        norms = np.maximum(norms, self.epsilon)
        return activations / norms
    
    def _token_level_analysis(self, activations: np.ndarray, capability_activations: np.ndarray) -> Dict:
        """Token级别细粒度分析"""
        logger.info("进行Token级别细粒度分析...")
        
        T = activations.shape[0]
        
        # 1. Token级别的Parcel激活分析
        token_parcel_analysis = self._analyze_token_parcel_activations(activations)
        
        # 2. Token级别的Capability激活分析
        token_capability_analysis = self._analyze_token_capability_activations(capability_activations)
        
        # 3. Token级别的连接分析
        token_connection_analysis = self._analyze_token_connections(activations, capability_activations)
        
        # 4. 异常Token识别
        anomaly_tokens = self._identify_anomaly_tokens(activations, capability_activations)
        
        # 5. 激活模式分析
        activation_patterns = self._analyze_activation_patterns(activations, capability_activations)
        
        return {
            'token_parcel_analysis': token_parcel_analysis,
            'token_capability_analysis': token_capability_analysis,
            'token_connection_analysis': token_connection_analysis,
            'anomaly_tokens': anomaly_tokens,
            'activation_patterns': activation_patterns,
            'total_tokens': T
        }
    
    def _analyze_token_parcel_activations(self, activations: np.ndarray) -> Dict:
        """分析Token级别的Parcel激活"""
        T, P = activations.shape
        
        # 计算每个token的激活强度
        token_activation_strengths = np.linalg.norm(activations, axis=1)
        
        # 找出每个token最活跃的Parcels
        top_parcels_per_token = []
        for t in range(T):
            top_indices = np.argsort(activations[t])[::-1][:self.top_k]
            top_parcels = []
            for idx in top_indices:
                top_parcels.append({
                    'parcel_id': int(idx),
                    'activation': float(activations[t, idx]),
                    'description': self.parcel_descriptions.get(f'parcel_{idx}', {}).get('functionality_summary', '未知功能')
                })
            top_parcels_per_token.append(top_parcels)
        
        # 计算激活的时序变化
        activation_trends = self._calculate_activation_trends(activations)
        
        # 识别激活峰值
        activation_peaks = self._identify_activation_peaks(token_activation_strengths)
        
        return {
            'token_activation_strengths': token_activation_strengths.tolist(),
            'top_parcels_per_token': top_parcels_per_token,
            'activation_trends': activation_trends,
            'activation_peaks': activation_peaks
        }
    
    def _analyze_token_capability_activations(self, capability_activations: np.ndarray) -> Dict:
        """分析Token级别的Capability激活"""
        T, C = capability_activations.shape
        capability_names = list(self.capability_parcel_mapping.keys())
        
        # 计算每个token的Capability激活强度
        token_capability_strengths = np.linalg.norm(capability_activations, axis=1)
        
        # 找出每个token最活跃的Capabilities
        top_capabilities_per_token = []
        for t in range(T):
            top_indices = np.argsort(capability_activations[t])[::-1][:self.top_k]
            top_capabilities = []
            for idx in top_indices:
                if idx < len(capability_names):
                    top_capabilities.append({
                        'capability_id': int(idx),
                        'capability_name': capability_names[idx],
                        'activation': float(capability_activations[t, idx]),
                        'description': self.capability_descriptions.get(capability_names[idx], {}).get('description', '未知能力')
                    })
            top_capabilities_per_token.append(top_capabilities)
        
        # 计算Capability激活的时序变化
        capability_trends = self._calculate_activation_trends(capability_activations)
        
        # 识别Capability激活峰值
        capability_peaks = self._identify_activation_peaks(token_capability_strengths)
        
        return {
            'token_capability_strengths': token_capability_strengths.tolist(),
            'top_capabilities_per_token': top_capabilities_per_token,
            'capability_trends': capability_trends,
            'capability_peaks': capability_peaks
        }
    
    def _analyze_token_connections(self, activations: np.ndarray, capability_activations: np.ndarray) -> Dict:
        """分析Token级别的连接"""
        T = activations.shape[0]
        
        # 计算每个token与其前后token的连接强度
        token_connections = []
        for t in range(1, T):
            # Parcel连接
            parcel_conn = np.dot(activations[t-1], activations[t]) / (np.linalg.norm(activations[t-1]) * np.linalg.norm(activations[t]) + self.epsilon)
            
            # Capability连接
            cap_conn = np.dot(capability_activations[t-1], capability_activations[t]) / (np.linalg.norm(capability_activations[t-1]) * np.linalg.norm(capability_activations[t]) + self.epsilon)
            
            token_connections.append({
                'token_position': t,
                'parcel_connection': float(parcel_conn),
                'capability_connection': float(cap_conn),
                'combined_connection': float((parcel_conn + cap_conn) / 2)
            })
        
        # 识别连接异常点
        connection_anomalies = self._identify_connection_anomalies(token_connections)
        
        return {
            'token_connections': token_connections,
            'connection_anomalies': connection_anomalies
        }
    
    def _identify_anomaly_tokens(self, activations: np.ndarray, capability_activations: np.ndarray) -> List[Dict]:
        """识别异常Token"""
        T = activations.shape[0]
        anomaly_tokens = []
        
        # 计算每个token的异常分数
        for t in range(T):
            # Parcel异常分数
            parcel_act = activations[t]
            parcel_z_scores = (parcel_act - self.baseline_stats['parcel_mean']) / (self.baseline_stats['parcel_std'] + self.epsilon)
            parcel_anomaly_score = np.max(np.abs(parcel_z_scores))
            
            # Capability异常分数
            if self.baseline_stats['capability_mean'] is not None:
                cap_act = capability_activations[t]
                cap_z_scores = (cap_act - self.baseline_stats['capability_mean']) / (self.baseline_stats['capability_std'] + self.epsilon)
                cap_anomaly_score = np.max(np.abs(cap_z_scores))
            else:
                cap_anomaly_score = 0.0
            
            combined_anomaly_score = parcel_anomaly_score + cap_anomaly_score
            
            if combined_anomaly_score > 2.0:  # 阈值可调整
                anomaly_tokens.append({
                    'token_position': t,
                    'parcel_anomaly_score': float(parcel_anomaly_score),
                    'capability_anomaly_score': float(cap_anomaly_score),
                    'combined_anomaly_score': float(combined_anomaly_score)
                })
        
        # 按异常分数排序
        anomaly_tokens.sort(key=lambda x: x['combined_anomaly_score'], reverse=True)
        
        return anomaly_tokens[:10]  # 返回前10个最异常的token
    
    def _analyze_activation_patterns(self, activations: np.ndarray, capability_activations: np.ndarray) -> Dict:
        """分析激活模式"""
        T = activations.shape[0]
        
        # 计算激活的时序模式
        parcel_activation_patterns = self._extract_activation_patterns(activations)
        capability_activation_patterns = self._extract_activation_patterns(capability_activations)
        
        # 计算激活的聚类模式
        parcel_clusters = self._cluster_activation_patterns(activations)
        capability_clusters = self._cluster_activation_patterns(capability_activations)
        
        return {
            'parcel_patterns': parcel_activation_patterns,
            'capability_patterns': capability_activation_patterns,
            'parcel_clusters': parcel_clusters,
            'capability_clusters': capability_clusters
        }
    
    def _calculate_activation_trends(self, activations: np.ndarray) -> Dict:
        """计算激活趋势"""
        T = activations.shape[0]
        
        # 计算每个维度的趋势
        trends = []
        for i in range(activations.shape[1]):
            values = activations[:, i]
            # 简单的线性趋势计算
            x = np.arange(T)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            trends.append({
                'dimension': i,
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            })
        
        return trends
    
    def _identify_activation_peaks(self, activation_strengths: np.ndarray) -> List[Dict]:
        """识别激活峰值"""
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(activation_strengths, height=np.mean(activation_strengths) + np.std(activation_strengths))
        
        peak_info = []
        for i, peak in enumerate(peaks):
            peak_info.append({
                'token_position': int(peak),
                'activation_strength': float(activation_strengths[peak]),
                'prominence': float(properties['peak_heights'][i]) if 'peak_heights' in properties else 0.0
            })
        
        return peak_info
    
    def _identify_connection_anomalies(self, token_connections: List[Dict]) -> List[Dict]:
        """识别连接异常点"""
        if not token_connections:
            return []
        
        # 计算连接强度的统计信息
        parcel_conns = [conn['parcel_connection'] for conn in token_connections]
        cap_conns = [conn['capability_connection'] for conn in token_connections]
        
        parcel_mean = np.mean(parcel_conns)
        parcel_std = np.std(parcel_conns)
        cap_mean = np.mean(cap_conns)
        cap_std = np.std(cap_conns)
        
        anomalies = []
        for conn in token_connections:
            parcel_z = abs(conn['parcel_connection'] - parcel_mean) / (parcel_std + self.epsilon)
            cap_z = abs(conn['capability_connection'] - cap_mean) / (cap_std + self.epsilon)
            
            if parcel_z > 2.0 or cap_z > 2.0:  # 阈值可调整
                anomalies.append({
                    'token_position': conn['token_position'],
                    'parcel_z_score': float(parcel_z),
                    'capability_z_score': float(cap_z),
                    'anomaly_type': 'parcel' if parcel_z > cap_z else 'capability'
                })
        
        return anomalies
    
    def _extract_activation_patterns(self, activations: np.ndarray) -> Dict:
        """提取激活模式"""
        T = activations.shape[0]
        
        # 计算激活的统计特征
        patterns = {
            'mean_activation': np.mean(activations, axis=0).tolist(),
            'std_activation': np.std(activations, axis=0).tolist(),
            'max_activation': np.max(activations, axis=0).tolist(),
            'min_activation': np.min(activations, axis=0).tolist(),
            'activation_range': (np.max(activations, axis=0) - np.min(activations, axis=0)).tolist()
        }
        
        return patterns
    
    def _cluster_activation_patterns(self, activations: np.ndarray) -> Dict:
        """聚类激活模式"""
        from sklearn.cluster import KMeans
        
        # 使用K-means对token进行聚类
        n_clusters = min(5, activations.shape[0] // 2)  # 动态确定聚类数
        if n_clusters < 2:
            return {'clusters': [], 'cluster_centers': []}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(activations)
        
        clusters = []
        for i in range(n_clusters):
            cluster_tokens = np.where(cluster_labels == i)[0].tolist()
            clusters.append({
                'cluster_id': i,
                'token_positions': cluster_tokens,
                'cluster_center': kmeans.cluster_centers_[i].tolist(),
                'size': len(cluster_tokens)
            })
        
        return {
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def _compare_with_pair_sample(self, sample_id: int, is_correct: bool, pair_sample_id: int,
                                activations: np.ndarray, capability_activations: np.ndarray) -> Dict:
        """与配对样本进行对比分析"""
        logger.info(f"对比分析样本 {sample_id} 与配对样本 {pair_sample_id}")
        
        # 加载配对样本的激活数据
        pair_activations = self.load_activation_data(pair_sample_id, not is_correct)
        if pair_activations is None:
            logger.warning(f"无法加载配对样本 {pair_sample_id} 的激活数据")
            return None
        
        # 计算配对样本的Capability激活
        pair_capability_activations = np.dot(pair_activations, self.mapping_matrix)
        
        # 确保两个样本的token数量一致（取较短的长度）
        min_length = min(activations.shape[0], pair_activations.shape[0])
        activations = activations[:min_length]
        capability_activations = capability_activations[:min_length]
        pair_activations = pair_activations[:min_length]
        pair_capability_activations = pair_capability_activations[:min_length]
        
        # Token级别的差异分析
        token_differences = self._calculate_token_differences(activations, capability_activations,
                                                            pair_activations, pair_capability_activations)
        
        # 激活模式对比
        activation_comparison = self._compare_activation_patterns(activations, capability_activations,
                                                               pair_activations, pair_capability_activations)
        
        # 连接模式对比
        connection_comparison = self._compare_connection_patterns(activations, capability_activations,
                                                               pair_activations, pair_capability_activations)
        
        return {
            'token_differences': token_differences,
            'activation_comparison': activation_comparison,
            'connection_comparison': connection_comparison,
            'pair_sample_id': pair_sample_id
        }
    
    def _calculate_token_differences(self, activations: np.ndarray, capability_activations: np.ndarray,
                                   pair_activations: np.ndarray, pair_capability_activations: np.ndarray) -> List[Dict]:
        """计算Token级别的差异"""
        T = activations.shape[0]
        token_differences = []
        
        for t in range(T):
            # Parcel激活差异
            parcel_diff = activations[t] - pair_activations[t]
            parcel_diff_magnitude = np.linalg.norm(parcel_diff)
            
            # Capability激活差异
            cap_diff = capability_activations[t] - pair_capability_activations[t]
            cap_diff_magnitude = np.linalg.norm(cap_diff)
            
            # 找出差异最大的Parcels和Capabilities
            top_parcel_diffs = np.argsort(np.abs(parcel_diff))[::-1][:5]
            top_cap_diffs = np.argsort(np.abs(cap_diff))[::-1][:5]
            
            token_differences.append({
                'token_position': t,
                'parcel_diff_magnitude': float(parcel_diff_magnitude),
                'capability_diff_magnitude': float(cap_diff_magnitude),
                'combined_diff_magnitude': float(parcel_diff_magnitude + cap_diff_magnitude),
                'top_parcel_differences': [
                    {
                        'parcel_id': int(idx),
                        'difference': float(parcel_diff[idx]),
                        'description': self.parcel_descriptions.get(f'parcel_{idx}', {}).get('functionality_summary', '未知功能')
                    } for idx in top_parcel_diffs
                ],
                'top_capability_differences': [
                    {
                        'capability_id': int(idx),
                        'capability_name': list(self.capability_parcel_mapping.keys())[idx] if idx < len(self.capability_parcel_mapping) else 'Unknown',
                        'difference': float(cap_diff[idx]),
                        'description': self.capability_descriptions.get(list(self.capability_parcel_mapping.keys())[idx], {}).get('description', '未知能力') if idx < len(self.capability_parcel_mapping) else '未知能力'
                    } for idx in top_cap_diffs
                ]
            })
        
        return token_differences
    
    def _compare_activation_patterns(self, activations: np.ndarray, capability_activations: np.ndarray,
                                   pair_activations: np.ndarray, pair_capability_activations: np.ndarray) -> Dict:
        """对比激活模式"""
        # 计算整体激活模式的相似性
        parcel_similarity = np.corrcoef(activations.flatten(), pair_activations.flatten())[0, 1]
        capability_similarity = np.corrcoef(capability_activations.flatten(), pair_capability_activations.flatten())[0, 1]
        
        # 计算激活强度的差异
        parcel_strength_diff = np.mean(np.linalg.norm(activations, axis=1)) - np.mean(np.linalg.norm(pair_activations, axis=1))
        capability_strength_diff = np.mean(np.linalg.norm(capability_activations, axis=1)) - np.mean(np.linalg.norm(pair_capability_activations, axis=1))
        
        return {
            'parcel_similarity': float(parcel_similarity),
            'capability_similarity': float(capability_similarity),
            'parcel_strength_difference': float(parcel_strength_diff),
            'capability_strength_difference': float(capability_strength_diff)
        }
    
    def _compare_connection_patterns(self, activations: np.ndarray, capability_activations: np.ndarray,
                                   pair_activations: np.ndarray, pair_capability_activations: np.ndarray) -> Dict:
        """对比连接模式"""
        # 计算Parcel连接矩阵
        parcel_conn = np.dot(activations.T, activations) / activations.shape[0]
        pair_parcel_conn = np.dot(pair_activations.T, pair_activations) / pair_activations.shape[0]
        
        # 计算Capability连接矩阵
        cap_conn = np.dot(capability_activations.T, capability_activations) / capability_activations.shape[0]
        pair_cap_conn = np.dot(pair_capability_activations.T, pair_capability_activations) / pair_capability_activations.shape[0]
        
        # 计算连接模式的相似性
        parcel_conn_similarity = np.corrcoef(parcel_conn.flatten(), pair_parcel_conn.flatten())[0, 1]
        cap_conn_similarity = np.corrcoef(cap_conn.flatten(), pair_cap_conn.flatten())[0, 1]
        
        # 计算连接强度的差异
        parcel_conn_diff = np.mean(np.abs(parcel_conn - pair_parcel_conn))
        cap_conn_diff = np.mean(np.abs(cap_conn - pair_cap_conn))
        
        return {
            'parcel_connection_similarity': float(parcel_conn_similarity),
            'capability_connection_similarity': float(cap_conn_similarity),
            'parcel_connection_difference': float(parcel_conn_diff),
            'capability_connection_difference': float(cap_conn_diff)
        }
    
    def analyze_sample(self, sample_id: int, is_correct: bool, pair_sample_id: Optional[int] = None) -> Dict:
        """分析单个样本"""
        logger.info(f"分析样本 {sample_id} ({'正确' if is_correct else '幻觉'})")
        
        # 加载激活数据
        activations = self.load_activation_data(sample_id, is_correct)
        if activations is None:
            raise ValueError(f"无法加载样本 {sample_id} 的激活数据")
        
        # 构建映射矩阵
        if self.mapping_matrix is None:
            self.mapping_matrix = self.build_mapping_matrix(activations.shape[1])
        
        # 计算基线统计（如果还没有）
        if not self.baseline_stats:
            self.compute_baseline_stats()
        
        # 计算Capability激活
        capability_activations = np.dot(activations, self.mapping_matrix)
        
        # Token级别细粒度分析
        token_analysis = self._token_level_analysis(activations, capability_activations)
        
        # 滑窗分析
        window_analysis = self._sliding_window_analysis(activations, capability_activations)
        
        # 定位异常时间窗
        anomaly_windows = self._identify_anomaly_windows(window_analysis)
        
        # 分析异常模块
        module_analysis = self._analyze_anomaly_modules(activations, capability_activations, anomaly_windows)
        
        # 分析连接异常
        connection_analysis = self._analyze_connection_anomalies(activations, capability_activations, anomaly_windows)
        
        # 对比分析（如果有配对样本）
        comparison_analysis = None
        if pair_sample_id:
            comparison_analysis = self._compare_with_pair_sample(sample_id, is_correct, pair_sample_id, 
                                                               activations, capability_activations)
        
        # 获取样本信息
        sample_data = self._get_sample_data(sample_id, is_correct)
        pair_data = self._get_sample_data(pair_sample_id, not is_correct) if pair_sample_id else None
        
        # 生成可视化
        self._generate_visualizations(activations, capability_activations, window_analysis, 
                                    anomaly_windows, module_analysis, connection_analysis, 
                                    sample_data, pair_data, token_analysis, comparison_analysis)
        
        # 生成报告
        report = self._generate_case_report(sample_data, pair_data, anomaly_windows, 
                                          module_analysis, connection_analysis, token_analysis, comparison_analysis)
        
        # 保存结果
        self._save_results(sample_id, window_analysis, anomaly_windows, 
                          module_analysis, connection_analysis, report, token_analysis, comparison_analysis)
        
        return {
            'sample_id': sample_id,
            'is_correct': is_correct,
            'pair_sample_id': pair_sample_id,
            'token_analysis': token_analysis,
            'window_analysis': window_analysis,
            'anomaly_windows': anomaly_windows,
            'module_analysis': module_analysis,
            'connection_analysis': connection_analysis,
            'comparison_analysis': comparison_analysis,
            'report': report
        }
    
    def _sliding_window_analysis(self, activations: np.ndarray, capability_activations: np.ndarray) -> Dict:
        """滑窗分析"""
        T = activations.shape[0]
        windows = []
        
        for start in range(T - self.window_size + 1):
            end = start + self.window_size
            
            # 当前窗口的激活
            window_parcel_acts = activations[start:end]
            window_cap_acts = capability_activations[start:end]
            
            # 计算SpikeScore
            normalized_parcel = self._normalize_activations(window_parcel_acts)
            normalized_cap = self._normalize_activations(window_cap_acts)
            
            # Parcel SpikeScore (z-score)
            parcel_means = np.mean(normalized_parcel, axis=0)
            parcel_z_scores = (parcel_means - self.baseline_stats['parcel_mean']) / (self.baseline_stats['parcel_std'] + self.epsilon)
            parcel_spike_score = np.max(np.abs(parcel_z_scores))
            
            # Capability SpikeScore
            if self.baseline_stats['capability_mean'] is not None:
                cap_means = np.mean(normalized_cap, axis=0)
                cap_z_scores = (cap_means - self.baseline_stats['capability_mean']) / (self.baseline_stats['capability_std'] + self.epsilon)
                cap_spike_score = np.max(np.abs(cap_z_scores))
            else:
                cap_spike_score = 0.0
            
            # 计算连接差异
            parcel_conn = np.dot(normalized_parcel.T, normalized_parcel) / self.window_size
            parcel_conn_diff = np.mean(np.abs(parcel_conn - self.baseline_stats['parcel_conn_baseline']))
            
            if self.baseline_stats['capability_conn_baseline'] is not None:
                cap_conn = np.dot(normalized_cap.T, normalized_cap) / self.window_size
                cap_conn_diff = np.mean(np.abs(cap_conn - self.baseline_stats['capability_conn_baseline']))
            else:
                cap_conn_diff = 0.0
            
            windows.append({
                'start': start,
                'end': end,
                'parcel_spike_score': parcel_spike_score,
                'cap_spike_score': cap_spike_score,
                'parcel_conn_diff': parcel_conn_diff,
                'cap_conn_diff': cap_conn_diff,
                'combined_score': parcel_spike_score + cap_spike_score + parcel_conn_diff + cap_conn_diff
            })
        
        return {
            'windows': windows,
            'total_windows': len(windows),
            'window_size': self.window_size
        }
    
    def _identify_anomaly_windows(self, window_analysis: Dict) -> List[Dict]:
        """识别异常时间窗"""
        windows = window_analysis['windows']
        
        # 按综合分数排序
        sorted_windows = sorted(windows, key=lambda x: x['combined_score'], reverse=True)
        
        # 选择前几个异常窗口
        anomaly_windows = sorted_windows[:min(3, len(sorted_windows))]
        
        # 过滤掉分数过低的窗口
        threshold = np.mean([w['combined_score'] for w in windows]) + np.std([w['combined_score'] for w in windows])
        anomaly_windows = [w for w in anomaly_windows if w['combined_score'] > threshold]
        
        return anomaly_windows
    
    def _analyze_anomaly_modules(self, activations: np.ndarray, capability_activations: np.ndarray, 
                                anomaly_windows: List[Dict]) -> Dict:
        """分析异常模块"""
        if not anomaly_windows:
            return {'parcel_anomalies': [], 'capability_anomalies': []}
        
        # 合并所有异常窗口
        all_anomaly_indices = []
        for window in anomaly_windows:
            all_anomaly_indices.extend(range(window['start'], window['end']))
        
        if not all_anomaly_indices:
            return {'parcel_anomalies': [], 'capability_anomalies': []}
        
        # 分析Parcel异常
        anomaly_parcel_acts = activations[all_anomaly_indices]
        parcel_means = np.mean(anomaly_parcel_acts, axis=0)
        parcel_z_scores = (parcel_means - self.baseline_stats['parcel_mean']) / (self.baseline_stats['parcel_std'] + self.epsilon)
        
        # 找出最异常的Parcel
        parcel_anomaly_indices = np.argsort(np.abs(parcel_z_scores))[::-1][:self.top_k]
        parcel_anomalies = []
        
        for idx in parcel_anomaly_indices:
            parcel_anomalies.append({
                'parcel_id': int(idx),
                'z_score': float(parcel_z_scores[idx]),
                'activation_diff': float(parcel_means[idx] - self.baseline_stats['parcel_mean'][idx]),
                'description': self.parcel_descriptions.get(f'parcel_{idx}', {}).get('functionality_summary', '未知功能')
            })
        
        # 分析Capability异常
        if self.baseline_stats['capability_mean'] is not None:
            anomaly_cap_acts = capability_activations[all_anomaly_indices]
            cap_means = np.mean(anomaly_cap_acts, axis=0)
            cap_z_scores = (cap_means - self.baseline_stats['capability_mean']) / (self.baseline_stats['capability_std'] + self.epsilon)
            
            # 找出最异常的Capability
            cap_anomaly_indices = np.argsort(np.abs(cap_z_scores))[::-1][:self.top_k]
            capability_anomalies = []
            
            capability_names = list(self.capability_parcel_mapping.keys())
            for idx in cap_anomaly_indices:
                if idx < len(capability_names):
                    capability_anomalies.append({
                        'capability_id': int(idx),
                        'capability_name': capability_names[idx],
                        'z_score': float(cap_z_scores[idx]),
                        'activation_diff': float(cap_means[idx] - self.baseline_stats['capability_mean'][idx]),
                        'description': self.capability_descriptions.get(capability_names[idx], {}).get('description', '未知能力')
                    })
        else:
            capability_anomalies = []
        
        return {
            'parcel_anomalies': parcel_anomalies,
            'capability_anomalies': capability_anomalies
        }
    
    def _analyze_connection_anomalies(self, activations: np.ndarray, capability_activations: np.ndarray,
                                    anomaly_windows: List[Dict]) -> Dict:
        """分析连接异常"""
        if not anomaly_windows:
            return {'parcel_connections': [], 'capability_connections': []}
        
        # 合并所有异常窗口
        all_anomaly_indices = []
        for window in anomaly_windows:
            all_anomaly_indices.extend(range(window['start'], window['end']))
        
        if not all_anomaly_indices:
            return {'parcel_connections': [], 'capability_connections': []}
        
        # 分析Parcel连接异常
        anomaly_parcel_acts = activations[all_anomaly_indices]
        normalized_parcel = self._normalize_activations(anomaly_parcel_acts)
        parcel_conn = np.dot(normalized_parcel.T, normalized_parcel) / len(all_anomaly_indices)
        parcel_conn_diff = parcel_conn - self.baseline_stats['parcel_conn_baseline']
        
        # 找出最异常的Parcel连接
        upper_tri_indices = np.triu_indices_from(parcel_conn_diff, k=1)
        upper_tri_diff = parcel_conn_diff[upper_tri_indices]
        top_conn_indices = np.argsort(np.abs(upper_tri_diff))[::-1][:self.top_k]
        
        parcel_connections = []
        for idx in top_conn_indices:
            i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            parcel_connections.append({
                'parcel_i': int(i),
                'parcel_j': int(j),
                'connection_diff': float(parcel_conn_diff[i, j]),
                'description_i': self.parcel_descriptions.get(f'parcel_{i}', {}).get('functionality_summary', '未知功能'),
                'description_j': self.parcel_descriptions.get(f'parcel_{j}', {}).get('functionality_summary', '未知功能')
            })
        
        # 分析Capability连接异常
        if self.baseline_stats['capability_conn_baseline'] is not None:
            anomaly_cap_acts = capability_activations[all_anomaly_indices]
            normalized_cap = self._normalize_activations(anomaly_cap_acts)
            cap_conn = np.dot(normalized_cap.T, normalized_cap) / len(all_anomaly_indices)
            cap_conn_diff = cap_conn - self.baseline_stats['capability_conn_baseline']
            
            # 找出最异常的Capability连接
            upper_tri_indices = np.triu_indices_from(cap_conn_diff, k=1)
            upper_tri_diff = cap_conn_diff[upper_tri_indices]
            top_conn_indices = np.argsort(np.abs(upper_tri_diff))[::-1][:self.top_k]
            
            capability_connections = []
            capability_names = list(self.capability_parcel_mapping.keys())
            for idx in top_conn_indices:
                i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
                if i < len(capability_names) and j < len(capability_names):
                    capability_connections.append({
                        'capability_i': capability_names[i],
                        'capability_j': capability_names[j],
                        'connection_diff': float(cap_conn_diff[i, j]),
                        'description_i': self.capability_descriptions.get(capability_names[i], {}).get('description', '未知能力'),
                        'description_j': self.capability_descriptions.get(capability_names[j], {}).get('description', '未知能力')
                    })
        else:
            capability_connections = []
        
        return {
            'parcel_connections': parcel_connections,
            'capability_connections': capability_connections
        }
    
    def _get_sample_data(self, sample_id: int, is_correct: bool) -> Optional[Dict]:
        """获取样本数据"""
        data_list = self.correct_data if is_correct else self.incorrect_data
        
        for data in data_list:
            if data.get('index') == sample_id:
                return data
        
        return None
    
    def _generate_visualizations(self, activations: np.ndarray, capability_activations: np.ndarray,
                               window_analysis: Dict, anomaly_windows: List[Dict],
                               module_analysis: Dict, connection_analysis: Dict,
                               sample_data: Optional[Dict], pair_data: Optional[Dict],
                               token_analysis: Optional[Dict] = None, comparison_analysis: Optional[Dict] = None) -> None:
        """生成可视化图表"""
        logger.info("生成可视化图表...")
        
        # 1. 时间轴分析图
        self._plot_timeline_analysis(window_analysis, anomaly_windows, sample_data, capability_activations)
        
        # 2. 激活热力图
        self._plot_activation_heatmaps(activations, capability_activations, anomaly_windows)
        
        # 3. 连接差异热力图
        self._plot_connection_heatmaps(connection_analysis)
        
        # 4. 异常模块排名图
        self._plot_anomaly_rankings(module_analysis)
        
        # 5. Token级别细粒度分析图
        if token_analysis:
            self._plot_token_level_analysis(token_analysis, sample_data)
        
        # 6. 对比分析图
        if comparison_analysis:
            self._plot_comparison_analysis(comparison_analysis, sample_data, pair_data)
    
    def _plot_timeline_analysis(self, window_analysis: Dict, anomaly_windows: List[Dict], 
                               sample_data: Optional[Dict], capability_activations: Optional[np.ndarray] = None) -> None:
        """绘制时间轴分析图"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        windows = window_analysis['windows']
        x_positions = [w['start'] for w in windows]
        
        # 上：SpikeScore和ConnShift曲线
        ax1 = axes[0]
        spike_scores = [w['parcel_spike_score'] + w['cap_spike_score'] for w in windows]
        conn_scores = [w['parcel_conn_diff'] + w['cap_conn_diff'] for w in windows]
        
        ax1.plot(x_positions, spike_scores, 'b-', label='Spike Score', linewidth=2)
        ax1.plot(x_positions, conn_scores, 'r-', label='Connection Shift', linewidth=2)
        
        # 标注异常窗口
        for window in anomaly_windows:
            ax1.axvspan(window['start'], window['end'], alpha=0.3, color='red', 
                       label='Anomaly Window' if window == anomaly_windows[0] else "")
        
        ax1.set_title('Anomaly Detection Timeline')
        ax1.set_ylabel('Anomaly Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 中：Top Capability激活曲线
        ax2 = axes[1]
        if self.baseline_stats['capability_mean'] is not None and capability_activations is not None:
            capability_names = list(self.capability_parcel_mapping.keys())
            for i in range(min(5, len(capability_names))):
                ax2.plot(range(len(capability_activations)), capability_activations[:, i], 
                        label=f'Cap {i}', alpha=0.7)
            ax2.set_title('Top Capability Activation Time Series')
            ax2.set_ylabel('Activation Strength')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 下：答案token序列
        ax3 = axes[2]
        if sample_data and 'model_answer' in sample_data:
            answer = sample_data['model_answer']
            tokens = answer.split()
            ax3.text(0.5, 0.5, f"Question: {sample_data.get('question', 'N/A')}\n\nAnswer: {answer}", 
                    transform=ax3.transAxes, fontsize=10, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.axis('off')
            ax3.set_title('Q&A Content')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'timeline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_activation_heatmaps(self, activations: np.ndarray, capability_activations: np.ndarray,
                                anomaly_windows: List[Dict]) -> None:
        """绘制激活热力图"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Parcel激活热力图
        ax1 = axes[0]
        im1 = ax1.imshow(activations.T, aspect='auto', cmap='viridis')
        ax1.set_title('Parcel Activation Heatmap')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Parcel Index')
        plt.colorbar(im1, ax=ax1)
        
        # 标注异常窗口
        for window in anomaly_windows:
            ax1.axvspan(window['start'], window['end'], alpha=0.3, color='red')
        
        # Capability激活热力图
        ax2 = axes[1]
        im2 = ax2.imshow(capability_activations.T, aspect='auto', cmap='viridis')
        ax2.set_title('Capability Activation Heatmap')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Capability Index')
        plt.colorbar(im2, ax=ax2)
        
        # 标注异常窗口
        for window in anomaly_windows:
            ax2.axvspan(window['start'], window['end'], alpha=0.3, color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'activation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_connection_heatmaps(self, connection_analysis: Dict) -> None:
        """绘制连接差异热力图"""
        if not connection_analysis['parcel_connections'] and not connection_analysis['capability_connections']:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 这里可以添加更详细的连接差异热力图
        # 由于篇幅限制，这里只显示基本框架
        axes[0].set_title('Parcel Connection Differences')
        axes[1].set_title('Capability Connection Differences')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'connection_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_anomaly_rankings(self, module_analysis: Dict) -> None:
        """绘制异常模块排名图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Parcel异常排名
        if module_analysis['parcel_anomalies']:
            ax1 = axes[0]
            parcel_data = module_analysis['parcel_anomalies']
            parcel_ids = [f"P{item['parcel_id']}" for item in parcel_data]
            z_scores = [item['z_score'] for item in parcel_data]
            
            bars1 = ax1.bar(range(len(parcel_ids)), z_scores, color='skyblue')
            ax1.set_title('Top Parcel Anomalies')
            ax1.set_xlabel('Parcel ID')
            ax1.set_ylabel('Z-Score')
            ax1.set_xticks(range(len(parcel_ids)))
            ax1.set_xticklabels(parcel_ids, rotation=45)
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars1, z_scores)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.2f}', ha='center', va='bottom')
        
        # Capability异常排名
        if module_analysis['capability_anomalies']:
            ax2 = axes[1]
            cap_data = module_analysis['capability_anomalies']
            cap_names = [item['capability_name'][:20] + '...' if len(item['capability_name']) > 20 
                        else item['capability_name'] for item in cap_data]
            z_scores = [item['z_score'] for item in cap_data]
            
            bars2 = ax2.bar(range(len(cap_names)), z_scores, color='lightcoral')
            ax2.set_title('Top Capability Anomalies')
            ax2.set_xlabel('Capability')
            ax2.set_ylabel('Z-Score')
            ax2.set_xticks(range(len(cap_names)))
            ax2.set_xticklabels(cap_names, rotation=45, ha='right')
            
            # 添加数值标签
            for i, (bar, score) in enumerate(zip(bars2, z_scores)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{score:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anomaly_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_token_level_analysis(self, token_analysis: Dict, sample_data: Optional[Dict]) -> None:
        """绘制Token级别细粒度分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Token激活强度时序图
        ax1 = axes[0, 0]
        parcel_strengths = token_analysis['token_parcel_analysis']['token_activation_strengths']
        cap_strengths = token_analysis['token_capability_analysis']['token_capability_strengths']
        
        ax1.plot(parcel_strengths, 'b-', label='Parcel Activation Strength', linewidth=2)
        ax1.plot(cap_strengths, 'r-', label='Capability Activation Strength', linewidth=2)
        ax1.set_title('Token-level Activation Strengths')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Activation Strength')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 异常Token分布
        ax2 = axes[0, 1]
        anomaly_tokens = token_analysis['anomaly_tokens']
        if anomaly_tokens:
            positions = [token['token_position'] for token in anomaly_tokens]
            scores = [token['combined_anomaly_score'] for token in anomaly_tokens]
            ax2.scatter(positions, scores, c='red', s=100, alpha=0.7)
            ax2.set_title('Anomalous Token Distribution')
            ax2.set_xlabel('Token Position')
            ax2.set_ylabel('Anomaly Score')
            ax2.grid(True, alpha=0.3)
        
        # 3. Token连接强度
        ax3 = axes[1, 0]
        token_connections = token_analysis['token_connection_analysis']['token_connections']
        if token_connections:
            positions = [conn['token_position'] for conn in token_connections]
            parcel_conns = [conn['parcel_connection'] for conn in token_connections]
            cap_conns = [conn['capability_connection'] for conn in token_connections]
            
            ax3.plot(positions, parcel_conns, 'b-', label='Parcel Connection', linewidth=2)
            ax3.plot(positions, cap_conns, 'r-', label='Capability Connection', linewidth=2)
            ax3.set_title('Token-level Connection Strength')
            ax3.set_xlabel('Token Position')
            ax3.set_ylabel('Connection Strength')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 激活峰值分布
        ax4 = axes[1, 1]
        parcel_peaks = token_analysis['token_parcel_analysis']['activation_peaks']
        cap_peaks = token_analysis['token_capability_analysis']['capability_peaks']
        
        if parcel_peaks:
            parcel_positions = [peak['token_position'] for peak in parcel_peaks]
            parcel_strengths = [peak['activation_strength'] for peak in parcel_peaks]
            ax4.scatter(parcel_positions, parcel_strengths, c='blue', s=100, alpha=0.7, label='Parcel Peaks')
        
        if cap_peaks:
            cap_positions = [peak['token_position'] for peak in cap_peaks]
            cap_strengths = [peak['activation_strength'] for peak in cap_peaks]
            ax4.scatter(cap_positions, cap_strengths, c='red', s=100, alpha=0.7, label='Capability Peaks')
        
        ax4.set_title('Activation Peaks Distribution')
        ax4.set_xlabel('Token Position')
        ax4.set_ylabel('Activation Strength')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_level_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comparison_analysis(self, comparison_analysis: Dict, sample_data: Optional[Dict], pair_data: Optional[Dict]) -> None:
        """绘制对比分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # 1. Token级别差异时序图
        ax1 = axes[0, 0]
        token_differences = comparison_analysis['token_differences']
        if token_differences:
            positions = [diff['token_position'] for diff in token_differences]
            parcel_diffs = [diff['parcel_diff_magnitude'] for diff in token_differences]
            cap_diffs = [diff['capability_diff_magnitude'] for diff in token_differences]
            
            ax1.plot(positions, parcel_diffs, 'b-', label='Parcel Difference', linewidth=2)
            ax1.plot(positions, cap_diffs, 'r-', label='Capability Difference', linewidth=2)
            ax1.set_title('Token-level Activation Differences')
            ax1.set_xlabel('Token Position')
            ax1.set_ylabel('Difference Magnitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 激活模式相似性
        ax2 = axes[0, 1]
        activation_comparison = comparison_analysis['activation_comparison']
        similarities = [
            activation_comparison['parcel_similarity'],
            activation_comparison['capability_similarity']
        ]
        labels = ['Parcel Similarity', 'Capability Similarity']
        colors = ['blue', 'red']
        
        bars = ax2.bar(labels, similarities, color=colors, alpha=0.7)
        ax2.set_title('Activation Pattern Similarity')
        ax2.set_ylabel('Similarity Score')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, sim in zip(bars, similarities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom')
        
        # 3. 连接模式相似性
        ax3 = axes[1, 0]
        connection_comparison = comparison_analysis['connection_comparison']
        conn_similarities = [
            connection_comparison['parcel_connection_similarity'],
            connection_comparison['capability_connection_similarity']
        ]
        conn_labels = ['Parcel Connection Similarity', 'Capability Connection Similarity']
        conn_colors = ['green', 'orange']
        
        conn_bars = ax3.bar(conn_labels, conn_similarities, color=conn_colors, alpha=0.7)
        ax3.set_title('Connection Pattern Similarity')
        ax3.set_ylabel('Similarity Score')
        ax3.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, sim in zip(conn_bars, conn_similarities):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{sim:.3f}', ha='center', va='bottom')
        
        # 4. 差异强度分布
        ax4 = axes[1, 1]
        if token_differences:
            combined_diffs = [diff['combined_diff_magnitude'] for diff in token_differences]
            ax4.hist(combined_diffs, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_title('Token Difference Magnitude Distribution')
            ax4.set_xlabel('Difference Magnitude')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_case_report(self, sample_data: Optional[Dict], pair_data: Optional[Dict],
                            anomaly_windows: List[Dict], module_analysis: Dict, 
                            connection_analysis: Dict, token_analysis: Optional[Dict] = None,
                            comparison_analysis: Optional[Dict] = None) -> str:
        """生成Case Study报告"""
        report = f"""# Case Study 分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析类型**: 单样本深度异常分析

---

## 样本信息

### 目标样本
- **样本ID**: {sample_data.get('index', 'N/A') if sample_data else 'N/A'}
- **问题**: {sample_data.get('question', 'N/A') if sample_data else 'N/A'}
- **模型答案**: {sample_data.get('model_answer', 'N/A') if sample_data else 'N/A'}
- **正确性**: {'正确' if sample_data and sample_data.get('is_correct', False) else '幻觉'}
- **置信度分数**: {sample_data.get('score', 'N/A') if sample_data else 'N/A'}

### 配对样本
"""
        
        if pair_data:
            report += f"""- **配对样本ID**: {pair_data.get('index', 'N/A')}
- **配对问题**: {pair_data.get('question', 'N/A')}
- **配对答案**: {pair_data.get('model_answer', 'N/A')}
- **配对正确性**: {'正确' if pair_data.get('is_correct', False) else '幻觉'}

"""
        else:
            report += "- **配对样本**: 未提供\n\n"
        
        # 异常时间窗分析
        report += "## 异常时间窗分析\n\n"
        if anomaly_windows:
            report += f"检测到 {len(anomaly_windows)} 个异常时间窗:\n\n"
            for i, window in enumerate(anomaly_windows, 1):
                report += f"### 异常窗 {i}\n"
                report += f"- **时间范围**: tokens {window['start']}-{window['end']}\n"
                report += f"- **Parcel Spike Score**: {window['parcel_spike_score']:.3f}\n"
                report += f"- **Capability Spike Score**: {window['cap_spike_score']:.3f}\n"
                report += f"- **Parcel连接差异**: {window['parcel_conn_diff']:.3f}\n"
                report += f"- **Capability连接差异**: {window['cap_conn_diff']:.3f}\n"
                report += f"- **综合异常分数**: {window['combined_score']:.3f}\n\n"
        else:
            report += "未检测到显著异常时间窗。\n\n"
        
        # 模块异常分析
        report += "## 模块异常分析\n\n"
        
        # Parcel异常
        if module_analysis['parcel_anomalies']:
            report += "### Top Parcel异常\n\n"
            for i, parcel in enumerate(module_analysis['parcel_anomalies'], 1):
                report += f"**{i}. Parcel {parcel['parcel_id']}**\n"
                report += f"- **Z-Score**: {parcel['z_score']:.3f}\n"
                report += f"- **激活差异**: {parcel['activation_diff']:.3f}\n"
                report += f"- **功能描述**: {parcel['description']}\n\n"
        
        # Capability异常
        if module_analysis['capability_anomalies']:
            report += "### Top Capability异常\n\n"
            for i, cap in enumerate(module_analysis['capability_anomalies'], 1):
                report += f"**{i}. {cap['capability_name']}**\n"
                report += f"- **Z-Score**: {cap['z_score']:.3f}\n"
                report += f"- **激活差异**: {cap['activation_diff']:.3f}\n"
                report += f"- **能力描述**: {cap['description']}\n\n"
        
        # 连接异常分析
        report += "## 连接异常分析\n\n"
        
        if connection_analysis['parcel_connections']:
            report += "### Top Parcel连接异常\n\n"
            for i, conn in enumerate(connection_analysis['parcel_connections'], 1):
                report += f"**{i}. Parcel {conn['parcel_i']} ↔ Parcel {conn['parcel_j']}**\n"
                report += f"- **连接差异**: {conn['connection_diff']:.3f}\n"
                report += f"- **Parcel {conn['parcel_i']}功能**: {conn['description_i']}\n"
                report += f"- **Parcel {conn['parcel_j']}功能**: {conn['description_j']}\n\n"
        
        if connection_analysis['capability_connections']:
            report += "### Top Capability连接异常\n\n"
            for i, conn in enumerate(connection_analysis['capability_connections'], 1):
                report += f"**{i}. {conn['capability_i']} ↔ {conn['capability_j']}**\n"
                report += f"- **连接差异**: {conn['connection_diff']:.3f}\n"
                report += f"- **{conn['capability_i']}描述**: {conn['description_i']}\n"
                report += f"- **{conn['capability_j']}描述**: {conn['description_j']}\n\n"
        
        # Token级别细粒度分析
        if token_analysis:
            report += "## Token级别细粒度分析\n\n"
            
            # Token激活分析
            report += "### Token激活模式分析\n\n"
            total_tokens = token_analysis['total_tokens']
            report += f"**总Token数量**: {total_tokens}\n\n"
            
            # 异常Token分析
            anomaly_tokens = token_analysis['anomaly_tokens']
            if anomaly_tokens:
                report += f"**检测到 {len(anomaly_tokens)} 个异常Token**:\n\n"
                for i, token in enumerate(anomaly_tokens[:5], 1):  # 显示前5个
                    report += f"**{i}. Token位置 {token['token_position']}**\n"
                    report += f"- **Parcel异常分数**: {token['parcel_anomaly_score']:.3f}\n"
                    report += f"- **Capability异常分数**: {token['capability_anomaly_score']:.3f}\n"
                    report += f"- **综合异常分数**: {token['combined_anomaly_score']:.3f}\n\n"
            
            # 激活峰值分析
            parcel_peaks = token_analysis['token_parcel_analysis']['activation_peaks']
            cap_peaks = token_analysis['token_capability_analysis']['capability_peaks']
            
            if parcel_peaks or cap_peaks:
                report += "### 激活峰值分析\n\n"
                if parcel_peaks:
                    report += f"**Parcel激活峰值**: {len(parcel_peaks)} 个\n"
                    for peak in parcel_peaks[:3]:  # 显示前3个
                        report += f"- Token {peak['token_position']}: 强度 {peak['activation_strength']:.3f}\n"
                    report += "\n"
                
                if cap_peaks:
                    report += f"**Capability激活峰值**: {len(cap_peaks)} 个\n"
                    for peak in cap_peaks[:3]:  # 显示前3个
                        report += f"- Token {peak['token_position']}: 强度 {peak['activation_strength']:.3f}\n"
                    report += "\n"
            
            # 连接异常分析
            connection_anomalies = token_analysis['token_connection_analysis']['connection_anomalies']
            if connection_anomalies:
                report += "### Token连接异常分析\n\n"
                report += f"**检测到 {len(connection_anomalies)} 个连接异常点**:\n\n"
                for i, anomaly in enumerate(connection_anomalies[:5], 1):  # 显示前5个
                    report += f"**{i}. Token位置 {anomaly['token_position']}**\n"
                    report += f"- **Parcel Z-Score**: {anomaly['parcel_z_score']:.3f}\n"
                    report += f"- **Capability Z-Score**: {anomaly['capability_z_score']:.3f}\n"
                    report += f"- **异常类型**: {anomaly['anomaly_type']}\n\n"
        
        # 对比分析
        if comparison_analysis:
            report += "## 配对样本对比分析\n\n"
            
            # 激活模式对比
            activation_comparison = comparison_analysis['activation_comparison']
            report += "### 激活模式对比\n\n"
            report += f"- **Parcel激活相似性**: {activation_comparison['parcel_similarity']:.3f}\n"
            report += f"- **Capability激活相似性**: {activation_comparison['capability_similarity']:.3f}\n"
            report += f"- **Parcel激活强度差异**: {activation_comparison['parcel_strength_difference']:.3f}\n"
            report += f"- **Capability激活强度差异**: {activation_comparison['capability_strength_difference']:.3f}\n\n"
            
            # 连接模式对比
            connection_comparison = comparison_analysis['connection_comparison']
            report += "### 连接模式对比\n\n"
            report += f"- **Parcel连接相似性**: {connection_comparison['parcel_connection_similarity']:.3f}\n"
            report += f"- **Capability连接相似性**: {connection_comparison['capability_connection_similarity']:.3f}\n"
            report += f"- **Parcel连接差异**: {connection_comparison['parcel_connection_difference']:.3f}\n"
            report += f"- **Capability连接差异**: {connection_comparison['capability_connection_difference']:.3f}\n\n"
            
            # Token级别差异分析
            token_differences = comparison_analysis['token_differences']
            if token_differences:
                report += "### Token级别差异分析\n\n"
                # 找出差异最大的tokens
                max_diff_tokens = sorted(token_differences, key=lambda x: x['combined_diff_magnitude'], reverse=True)[:5]
                report += f"**差异最大的前5个Token**:\n\n"
                for i, diff in enumerate(max_diff_tokens, 1):
                    report += f"**{i}. Token位置 {diff['token_position']}**\n"
                    report += f"- **Parcel差异强度**: {diff['parcel_diff_magnitude']:.3f}\n"
                    report += f"- **Capability差异强度**: {diff['capability_diff_magnitude']:.3f}\n"
                    report += f"- **综合差异强度**: {diff['combined_diff_magnitude']:.3f}\n"
                    
                    # 显示差异最大的Parcels
                    if diff['top_parcel_differences']:
                        report += "- **主要Parcel差异**:\n"
                        for j, p_diff in enumerate(diff['top_parcel_differences'][:3], 1):
                            report += f"  {j}. Parcel {p_diff['parcel_id']}: {p_diff['difference']:.3f} ({p_diff['description'][:50]}...)\n"
                    
                    # 显示差异最大的Capabilities
                    if diff['top_capability_differences']:
                        report += "- **主要Capability差异**:\n"
                        for j, c_diff in enumerate(diff['top_capability_differences'][:3], 1):
                            report += f"  {j}. {c_diff['capability_name']}: {c_diff['difference']:.3f} ({c_diff['description'][:50]}...)\n"
                    report += "\n"
        
        # 机制假设
        report += "## 机制假设\n\n"
        if anomaly_windows and (module_analysis['parcel_anomalies'] or module_analysis['capability_anomalies']):
            report += "基于分析结果，提出以下机制假设:\n\n"
            
            # 基于异常模式生成假设
            if module_analysis['capability_anomalies']:
                top_cap = module_analysis['capability_anomalies'][0]
                report += f"1. **主要异常能力**: {top_cap['capability_name']} 出现显著异常 (Z-Score: {top_cap['z_score']:.2f})\n"
                report += f"   - 这表明模型在{top_cap['capability_name']}相关任务上出现功能障碍\n\n"
            
            if connection_analysis['capability_connections']:
                top_conn = connection_analysis['capability_connections'][0]
                report += f"2. **连接异常**: {top_conn['capability_i']} 与 {top_conn['capability_j']} 之间的连接出现异常\n"
                report += f"   - 连接差异: {top_conn['connection_diff']:.3f}\n"
                report += f"   - 这可能导致能力间的协调失效\n\n"
            
            report += "3. **综合机制**: 异常时间窗内的模块激活异常和连接断裂共同导致了模型的错误输出\n\n"
        else:
            report += "未检测到显著异常，无法提出明确的机制假设。\n\n"
        
        # 结论
        report += "## 结论\n\n"
        if sample_data and sample_data.get('is_correct', True) == False:
            report += "本案例展示了模型在产生幻觉时的内部神经活动异常模式。通过时间窗分析、模块异常检测和连接分析，我们能够定位到具体的异常发生时刻和涉及的认知模块。\n\n"
        else:
            report += "本案例展示了模型在正确回答时的内部神经活动模式。通过对比分析，可以更好地理解正常认知过程与异常过程的差异。\n\n"
        
        report += "**建议**: 基于这些发现，可以考虑针对异常模块进行干预实验，验证机制假设的有效性。\n\n"
        
        report += "---\n\n"
        report += "*本报告由Case Study分析工具自动生成*"
        
        return report
    
    def _save_results(self, sample_id: int, window_analysis: Dict, anomaly_windows: List[Dict],
                     module_analysis: Dict, connection_analysis: Dict, report: str,
                     token_analysis: Optional[Dict] = None, comparison_analysis: Optional[Dict] = None) -> None:
        """保存分析结果"""
        def _to_serializable(obj):
            # 递归将 numpy 类型转换为原生 Python 类型
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.generic, )):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return [_to_serializable(v) for v in obj]
            return obj

        # 保存详细结果
        results = {
            'sample_id': sample_id,
            'window_analysis': window_analysis,
            'anomaly_windows': anomaly_windows,
            'module_analysis': module_analysis,
            'connection_analysis': connection_analysis,
            'token_analysis': token_analysis,
            'comparison_analysis': comparison_analysis,
            'timestamp': datetime.now().isoformat()
        }
        results = _to_serializable(results)
        
        with open(self.output_dir / 'analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存报告
        with open(self.output_dir / 'case_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存Token级别分析结果
        if token_analysis:
            with open(self.output_dir / 'token_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(token_analysis), f, indent=2, ensure_ascii=False)
        
        # 保存对比分析结果
        if comparison_analysis:
            with open(self.output_dir / 'comparison_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(_to_serializable(comparison_analysis), f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析结果已保存到: {self.output_dir}")
    
    
    def run_case_analysis(self, sample_id: int, is_correct: bool, pair_sample_id: Optional[int] = None) -> Dict:
        """运行完整的case分析"""
        try:
            logger.info(f"开始Case Study分析: 样本 {sample_id}")
            
            # 加载数据
            self.load_data()
            
            # 运行分析
            results = self.analyze_sample(sample_id, is_correct, pair_sample_id)
            
            logger.info("Case Study分析完成！")
            return results
            
        except Exception as e:
            logger.error(f"Case Study分析失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Case Study分析工具')
    parser.add_argument('--correct_jsonl', type=str, required=True,
                       help='正确样本数据路径')
    parser.add_argument('--incorrect_jsonl', type=str, required=True,
                       help='幻觉样本数据路径')
    parser.add_argument('--mapping_json', type=str, required=True,
                       help='Capability-Parcel映射文件路径')
    parser.add_argument('--parcel_desc', type=str, required=True,
                       help='Parcel功能描述文件路径')
    parser.add_argument('--cap_desc', type=str, required=True,
                       help='Capability描述文件路径')
    # 单样本模式参数
    parser.add_argument('--sample_id', type=int,
                       help='要分析的样本ID（与 pairs_json 二选一）')
    parser.add_argument('--is_correct', action='store_true',
                       help='样本是否为正确样本')
    parser.add_argument('--pair_sample_id', type=int,
                       help='配对样本ID')
    parser.add_argument('--out_dir', type=str,
                       help='单样本输出目录（单样本模式必填）')
    parser.add_argument('--window', type=int, default=5,
                       help='滑窗大小')
    parser.add_argument('--topk', type=int, default=5,
                       help='展示的top异常项数量')
    # 批量模式参数（从对比选择 JSON 读取 is_good_pair=true 配对）
    parser.add_argument('--pairs_json', type=str,
                       help='question_contrastive_pairs.json 路径（提供则启用批量模式）')
    parser.add_argument('--output_base', type=str,
                       help='批量模式输出根目录（必填，case_<id> 子目录将自动创建）')
    parser.add_argument('--skip_existing', action='store_true',
                       help='若结果已存在则跳过')
    parser.add_argument('--min_overall_score', type=float, default=None,
                       help='仅保留 overall_score>=阈值 的 is_good_pair（可选）')
    
    args = parser.parse_args()
    
    # 批量模式
    if args.pairs_json:
        if not args.output_base:
            raise ValueError('--output_base 在批量模式下必填')
        with open(args.pairs_json, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        pairs = payload.get('llm_refined_pairs') or []
        if not pairs:
            pairs = []  # direct_pairs 不含 is_good_pair，按需求不选
        # 过滤 is_good_pair
        filtered = []
        for item in pairs:
            if bool(item.get('is_good_pair', False)) is True:
                if args.min_overall_score is not None:
                    if float(item.get('overall_score', 0.0) or 0.0) < float(args.min_overall_score):
                        continue
                h = item.get('hallucination_index')
                c = item.get('correct_index')
                if h is not None and c is not None:
                    filtered.append((int(h), int(c)))
        if not filtered:
            raise ValueError('未在 pairs_json 中找到 is_good_pair=true 的配对（可能为空或低于阈值）')

        success = 0
        total = 0
        for h, c in filtered:
            total += 1
            out_dir = os.path.join(args.output_base, f'case_{h}')
            if args.skip_existing and os.path.exists(os.path.join(out_dir, 'analysis_results.json')):
                logger.info(f'结果已存在且 skip_existing=true，跳过样本 {h}')
                success += 1
                continue
            try:
                analyzer = CaseStudyAnalyzer(
                    correct_jsonl_path=args.correct_jsonl,
                    incorrect_jsonl_path=args.incorrect_jsonl,
                    mapping_json_path=args.mapping_json,
                    parcel_desc_path=args.parcel_desc,
                    cap_desc_path=args.cap_desc,
                    output_dir=out_dir,
                    window_size=args.window,
                    top_k=args.topk
                )
                analyzer.run_case_analysis(sample_id=h, is_correct=False, pair_sample_id=c)
                success += 1
            except Exception as e:
                logger.error(f'批量分析失败：h={h} c={c} 错误: {e}')
                traceback.print_exc()
        logger.info(f'批量分析完成：{success}/{total} 成功')
        return

    # 单样本模式
    if args.sample_id is None or not args.out_dir:
        raise ValueError('单样本模式需要提供 --sample_id 与 --out_dir（或使用 --pairs_json 批量模式）')

    analyzer = CaseStudyAnalyzer(
        correct_jsonl_path=args.correct_jsonl,
        incorrect_jsonl_path=args.incorrect_jsonl,
        mapping_json_path=args.mapping_json,
        parcel_desc_path=args.parcel_desc,
        cap_desc_path=args.cap_desc,
        output_dir=args.out_dir,
        window_size=args.window,
        top_k=args.topk
    )
    analyzer.run_case_analysis(args.sample_id, args.is_correct, args.pair_sample_id)


if __name__ == "__main__":
    main()
