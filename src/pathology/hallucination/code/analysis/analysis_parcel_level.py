#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel-Level 异常分析脚本

基于 LLM 的 Parcel 激活特征，比较模型在 幻觉评测数据集 正确回答与幻觉回答
两类样本中的内部神经活动差异。计算 Parcel-Level 的功能连接网络，
检测单模块异常和连接异常。

作者: Jeryi
日期: 2025
"""

import json
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot
from structural_mask_utils import StructuralMaskProcessor
from sklearn.decomposition import PCA

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParcelLevelAnalyzer:
    """Parcel级别异常分析器"""
    
    def __init__(self, correct_jsonl_path: str, incorrect_jsonl_path: str, 
                 output_dir: str, parcel_info_path: str = None,
                 epsilon: float = 1e-8, significance_threshold: float = 0.05,
                 skip_existing: bool = False, top_k_edges: int = 100,
                 anomalous_select_mode: str = 'posneg',
                 use_structural_mask: bool = False,
                 structural_matrix_path: str = None,
                 structural_threshold: float = 0.0,
                 mask_type: str = 'binary',
                 use_pca_connectivity: bool = False,
                 pca_explained_variance: float = 0.8,
                 max_tokens: int = None):
        """
        初始化分析器
        
        Args:
            correct_jsonl_path: 正确样本激活数据路径
            incorrect_jsonl_path: 幻觉样本激活数据路径
            output_dir: 输出目录
            parcel_info_path: Parcel功能描述信息文件路径
            epsilon: L2归一化的小常数
            significance_threshold: 统计显著性阈值
            skip_existing: 是否跳过已存在的结果文件
            top_k_edges: 可视化时显示前k个最强的连接
            use_structural_mask: 是否使用结构性连接矩阵作为mask
            structural_matrix_path: 结构性连接矩阵文件路径
            structural_threshold: 结构性连接阈值
            mask_type: mask类型 ('binary' 或 'weighted')
            use_pca_connectivity: 是否使用PCA方法计算连接性
            pca_explained_variance: PCA保留的可解释方差比例
            max_tokens: 最大token数量，如果指定则只分析前max_tokens个token（默认None表示使用所有token）
        """
        self.correct_jsonl_path = correct_jsonl_path
        self.incorrect_jsonl_path = incorrect_jsonl_path
        self.output_dir = Path(output_dir)
        self.parcel_info_path = parcel_info_path
        self.epsilon = epsilon
        self.significance_threshold = significance_threshold
        self.skip_existing = skip_existing
        self.top_k_edges = top_k_edges
        # 异常连接挑选模式：'posneg' 或 'abs'
        self.anomalous_select_mode = anomalous_select_mode
        # 结构性连接mask相关参数
        self.use_structural_mask = use_structural_mask
        self.structural_matrix_path = structural_matrix_path
        self.structural_threshold = structural_threshold
        self.mask_type = mask_type
        self.structural_mask_processor = None
        self.use_pca_connectivity = use_pca_connectivity
        self.pca_explained_variance = pca_explained_variance
        self.max_tokens = max_tokens
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.correct_activations = []
        self.incorrect_activations = []
        self.parcel_dim = None
        self.parcel_info = {}  # 存储Parcel功能描述信息
        
        # 初始化结构性连接mask处理器
        if self.use_structural_mask and self.structural_matrix_path:
            try:
                logger.info("初始化结构性连接mask处理器...")
                self.structural_mask_processor = StructuralMaskProcessor(
                    self.structural_matrix_path, 
                    self.parcel_dim if self.parcel_dim else 270  # 默认270个parcels
                )
                logger.info("结构性连接mask处理器初始化完成")
            except Exception as e:
                logger.warning(f"结构性连接mask处理器初始化失败: {e}")
                self.use_structural_mask = False
        
    def load_parcel_info(self) -> None:
        """加载Parcel功能描述信息"""
        if self.parcel_info_path is None:
            logger.warning("未提供Parcel功能描述文件路径，将只使用Parcel ID")
            return
            
        try:
            logger.info(f"加载Parcel功能描述信息: {self.parcel_info_path}")
            with open(self.parcel_info_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'parcel_summaries' in data:
                for parcel in data['parcel_summaries']:
                    parcel_id = parcel.get('parcel_id')
                    if parcel_id is not None:
                        self.parcel_info[parcel_id] = {
                            'function_name': parcel.get('function_name', f'Parcel {parcel_id}'),
                            'function_description': parcel.get('function_description', ''),
                            'model_role': parcel.get('model_role', ''),
                            'keywords': parcel.get('keywords', [])
                        }
                        
            logger.info(f"成功加载 {len(self.parcel_info)} 个Parcel的功能描述信息")
            
        except FileNotFoundError:
            logger.warning(f"Parcel功能描述文件不存在: {self.parcel_info_path}")
        except Exception as e:
            logger.error(f"加载Parcel功能描述信息失败: {e}")
            raise Exception(f"加载Parcel功能描述信息失败: {e}")
    
    def get_parcel_info(self, parcel_id: int) -> Dict:
        """获取指定Parcel的功能描述信息"""
        if parcel_id in self.parcel_info:
            return self.parcel_info[parcel_id]
        else:
            return {
                'function_name': f'Parcel {parcel_id}',
                'function_description': '功能描述不可用',
                'model_role': '角色描述不可用',
                'keywords': []
            }
        
    def load_activation_data(self) -> None:
        """加载激活数据"""
        if self.max_tokens is not None:
            logger.info(f"将只分析前 {self.max_tokens} 个token的Parcel激活数据")
        
        logger.info("开始加载正确样本激活数据...")
        self._load_jsonl_data(self.correct_jsonl_path, self.correct_activations)
        
        logger.info("开始加载幻觉样本激活数据...")
        self._load_jsonl_data(self.incorrect_jsonl_path, self.incorrect_activations)
        
        logger.info(f"正确样本数量: {len(self.correct_activations)}")
        logger.info(f"幻觉样本数量: {len(self.incorrect_activations)}")
        
        if len(self.correct_activations) == 0 or len(self.incorrect_activations) == 0:
            raise ValueError("激活数据为空，请检查输入文件")
        
        # 显示实际使用的token数量
        if len(self.correct_activations) > 0:
            actual_tokens = self.correct_activations[0].shape[0]
            logger.info(f"每个样本实际使用的token数量: {actual_tokens}")
            
    def _load_jsonl_data(self, jsonl_path: str, data_list: List) -> None:
        """加载JSONL格式的激活数据"""
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'token_parcel_acts' not in data:
                            logger.warning(f"第{line_num}行缺少token_parcel_acts字段")
                            continue
                            
                        activations = np.array(data['token_parcel_acts'], dtype=np.float32)
                        
                        # 如果指定了max_tokens，则只保留前max_tokens个token
                        if self.max_tokens is not None and activations.shape[0] > self.max_tokens and self.max_tokens>0:
                            activations = activations[:self.max_tokens, :]
                        if self.parcel_dim is None:
                            self.parcel_dim = activations.shape[1]
                        elif activations.shape[1] != self.parcel_dim:
                            logger.warning(f"第{line_num}行Parcel维度不匹配: {activations.shape[1]} != {self.parcel_dim}")
                            continue
                            
                        data_list.append(activations)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"第{line_num}行JSON解析错误: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"第{line_num}行处理错误: {e}")
                        continue
                        
        except FileNotFoundError:
            raise FileNotFoundError(f"文件不存在: {jsonl_path}")
        except Exception as e:
            raise Exception(f"加载数据失败: {e}")
    
    def _demean_and_standardize(self, activations: np.ndarray) -> np.ndarray:
        """按时间维对每个Parcel去均值并标准化。"""
        mean = activations.mean(axis=0, keepdims=True)
        std = activations.std(axis=0, keepdims=True)
        std = np.maximum(std, self.epsilon)
        return (activations - mean) / std
    
    def compute_connectivity_matrix(self, activations: np.ndarray) -> np.ndarray:
        """
        计算功能连接矩阵（去均值后的皮尔逊相关），并返回 Fisher z 变换结果。
        可选择性地应用结构性连接mask。
        
        Args:
            activations: 形状为 (T, P) 的激活矩阵
            
        Returns:
            形状为 (P, P) 的 Fisher z 相关矩阵
        """
        X = self._demean_and_standardize(activations)
        T = X.shape[0]
        P = X.shape[1]
        if T < 2:
            logger.warning("时间步不足以计算相关(T<2)，返回零矩阵作为连接")
            corr = np.zeros((P, P), dtype=np.float32)
        else:
            # 在标准化后，相关矩阵等价于协方差除以(T-1)
            corr = (X.T @ X) / max(T - 1, 1)
        corr = np.clip(corr, -0.999999, 0.999999)
        z = 0.5 * np.log((1 + corr) / (1 - corr))
        
        # 应用结构性连接mask
        if self.use_structural_mask and self.structural_mask_processor is not None:
            try:
                # 更新parcel维度（如果之前未知）
                if self.parcel_dim is None:
                    self.parcel_dim = P
                    self.structural_mask_processor.parcel_dim = P
                
                # 应用mask
                z = self.structural_mask_processor.apply_mask(
                    z, 
                    threshold=self.structural_threshold,
                    mask_type=self.mask_type
                )
                logger.debug("已应用结构性连接mask到功能连接矩阵")
            except Exception as e:
                logger.warning(f"应用结构性连接mask失败: {e}")
        
        return z
    
    def compute_pca_connectivity(self, activations_list: List[np.ndarray]) -> np.ndarray:
        """
        使用PCA方法计算连接矩阵
        将所有样本的token拼接，进行PCA降维，然后计算连接性
        
        Args:
            activations_list: 激活数据列表，每个元素形状为 (T, P)
            
        Returns:
            形状为 (P, P) 的连接矩阵
        """
        logger.info("使用PCA方法计算连接矩阵...")
        
        # 拼接所有样本的token
        all_tokens = []
        for activations in activations_list:
            all_tokens.append(activations)
        
        # 拼接所有token，形状为 (total_tokens, P)
        concatenated_activations = np.vstack(all_tokens)
        logger.info(f"拼接后激活数据形状: {concatenated_activations.shape}")
        
        # 对每个Parcel进行PCA降维
        parcel_pca_results = []
        pca_components = []
        
        for parcel_idx in range(self.parcel_dim):
            parcel_data = concatenated_activations[:, parcel_idx].reshape(-1, 1)
            
            # 进行PCA，保留指定比例的可解释方差
            pca = PCA(n_components=None)  # 保留所有主成分
            pca.fit(parcel_data)
            
            # 计算累积可解释方差比例
            cumsum_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            
            # 找到保留指定可解释方差所需的主成分数量
            n_components = np.argmax(cumsum_variance_ratio >= self.pca_explained_variance) + 1
            n_components = max(1, n_components)  # 至少保留1个主成分
            
            # 重新拟合PCA
            pca = PCA(n_components=n_components)
            parcel_pca = pca.fit_transform(parcel_data)
            parcel_pca_results.append(parcel_pca)
            pca_components.append(n_components)
        
        logger.info(f"PCA降维后各Parcel主成分数量: {pca_components[:10]}...")  # 显示前10个
        logger.info(f"平均主成分数量: {np.mean(pca_components):.2f}")
        
        # 将PCA结果拼接成矩阵，形状为 (total_tokens, total_components)
        pca_matrix = np.hstack(parcel_pca_results)
        logger.info(f"PCA矩阵形状: {pca_matrix.shape}")
        
        # 标准化PCA结果
        pca_matrix_std = (pca_matrix - np.mean(pca_matrix, axis=0)) / (np.std(pca_matrix, axis=0) + self.epsilon)
        
        # 计算Parcel之间的连接性
        # 由于PCA降维后每个Parcel的主成分数量可能不同，我们需要重新组织数据
        connectivity_matrix = np.zeros((self.parcel_dim, self.parcel_dim))
        
        start_idx = 0
        for i in range(self.parcel_dim):
            end_idx = start_idx + pca_components[i]
            parcel_i_pca = pca_matrix_std[:, start_idx:end_idx]
            
            start_j = 0
            for j in range(self.parcel_dim):
                end_j = start_j + pca_components[j]
                parcel_j_pca = pca_matrix_std[:, start_j:end_j]
                
                # 计算两个Parcel的PCA结果之间的平均相关性
                correlations = []
                for k in range(parcel_i_pca.shape[1]):
                    for l in range(parcel_j_pca.shape[1]):
                        corr = np.corrcoef(parcel_i_pca[:, k], parcel_j_pca[:, l])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
                
                if len(correlations) > 0:
                    connectivity_matrix[i, j] = np.mean(correlations)
                else:
                    connectivity_matrix[i, j] = 0.0
                
                start_j = end_j
            
            start_idx = end_idx
        
        # 应用Fisher z变换
        connectivity_matrix = np.clip(connectivity_matrix, -0.999999, 0.999999)
        z_matrix = 0.5 * np.log((1 + connectivity_matrix) / (1 - connectivity_matrix))
        
        # 应用结构性连接mask
        if self.use_structural_mask and self.structural_mask_processor is not None:
            try:
                if self.parcel_dim is None:
                    self.parcel_dim = z_matrix.shape[0]
                    self.structural_mask_processor.parcel_dim = self.parcel_dim
                
                z_matrix = self.structural_mask_processor.apply_mask(
                    z_matrix, 
                    threshold=self.structural_threshold,
                    mask_type=self.mask_type
                )
                logger.debug("已应用结构性连接mask到PCA连接矩阵")
            except Exception as e:
                logger.warning(f"应用结构性连接mask失败: {e}")
        
        logger.info("PCA连接矩阵计算完成")
        return z_matrix
    
    def compute_baseline_connectivity(self) -> Tuple[np.ndarray, np.ndarray]:
        """计算正确样本的基线连接矩阵（同时计算传统方法和PCA拼接方法）"""
        logger.info("计算基线连接矩阵...")
        
        # 传统方法计算连接矩阵
        logger.info("使用传统方法计算基线连接矩阵...")
        connectivity_matrices = []
        for i, activations in enumerate(self.correct_activations):
            try:
                conn_matrix = self.compute_connectivity_matrix(activations)
                connectivity_matrices.append(conn_matrix)
            except Exception as e:
                logger.warning(f"样本{i}连接矩阵计算失败: {e}")
                continue
        
        if len(connectivity_matrices) == 0:
            raise ValueError("无法计算基线连接矩阵")
        
        # 计算均值
        traditional_baseline = np.mean(connectivity_matrices, axis=0)
        logger.info(f"传统方法基线连接矩阵计算完成，形状: {traditional_baseline.shape}")
        
        # PCA拼接方法计算连接矩阵
        pca_baseline = None
        if self.use_pca_connectivity:
            logger.info("使用PCA拼接方法计算基线连接矩阵...")
            pca_baseline = self.compute_pca_connectivity(self.correct_activations)
            logger.info(f"PCA拼接方法基线连接矩阵计算完成，形状: {pca_baseline.shape}")
        
        return traditional_baseline, pca_baseline
    
    def analyze_activation_anomalies(self) -> Dict:
        """
        分析单Parcel激活异常
        
        Returns:
            包含异常统计的字典
        """
        logger.info("分析单Parcel激活异常...")
        
        # 收集所有样本的激活数据
        correct_all_acts = []
        incorrect_all_acts = []
        for activations in self.correct_activations:
            # 计算每个样本的平均激活
            mean_acts = np.mean(activations, axis=0)
            if np.isnan(mean_acts).any():
                import pdb; pdb.set_trace()
                print(f"mean_acts contains nan at")
                continue
            correct_all_acts.append(mean_acts)
        
        for activations in self.incorrect_activations:
            mean_acts = np.mean(activations, axis=0)
            if np.isnan(mean_acts).any():
                print(f"mean_acts contains nan")
                continue
            incorrect_all_acts.append(mean_acts)
        
        correct_all_acts = np.array(correct_all_acts)
        incorrect_all_acts = np.array(incorrect_all_acts)
        
        # 计算激活差异
        activation_diff = np.mean(incorrect_all_acts, axis=0) - np.mean(correct_all_acts, axis=0)
        
        # 进行t检验，并计算 Welch t 置信区间（默认95%CI）
        t_stats = []
        p_values = []
        ci_lowers = []
        ci_uppers = []
        
        for parcel_idx in range(self.parcel_dim):
            try:
                incorrect_vals = incorrect_all_acts[:, parcel_idx]
                correct_vals = correct_all_acts[:, parcel_idx]

                # 使用 Welch t 检验（方差不等）
                t_stat, p_val = stats.ttest_ind(
                    incorrect_vals,
                    correct_vals,
                    equal_var=False
                )
                t_stats.append(t_stat)
                p_values.append(p_val)

                # 计算 Welch t 95% 置信区间
                n_x = incorrect_vals.shape[0]
                n_y = correct_vals.shape[0]
                if n_x < 2 or n_y < 2:
                    logger.warning(f"Parcel {parcel_idx} 样本数过小，无法计算95%CI")
                    ci_lowers.append(float("nan"))
                    ci_uppers.append(float("nan"))
                    continue

                mean_x = float(np.mean(incorrect_vals))
                mean_y = float(np.mean(correct_vals))
                var_x = float(np.var(incorrect_vals, ddof=1))
                var_y = float(np.var(correct_vals, ddof=1))

                se_sq = var_x / n_x + var_y / n_y
                if se_sq <= 0:
                    logger.warning(f"Parcel {parcel_idx} 标准误为非正数，无法计算95%CI")
                    ci_lowers.append(float("nan"))
                    ci_uppers.append(float("nan"))
                    continue

                se = np.sqrt(se_sq)

                # Welch–Satterthwaite 自由度
                numerator = se_sq ** 2
                denom_part_x = (var_x ** 2) / (n_x ** 2 * (n_x - 1))
                denom_part_y = (var_y ** 2) / (n_y ** 2 * (n_y - 1))
                denominator = denom_part_x + denom_part_y
                if denominator <= 0:
                    logger.warning(f"Parcel {parcel_idx} 自由度计算失败，无法计算95%CI")
                    ci_lowers.append(float("nan"))
                    ci_uppers.append(float("nan"))
                    continue

                dof = numerator / denominator
                # Δ = mean_incorrect - mean_correct，与 activation_diff 定义保持一致
                delta = mean_x - mean_y
                t_crit = stats.t.ppf(0.975, dof)
                ci_low = delta - t_crit * se
                ci_high = delta + t_crit * se
                ci_lowers.append(float(ci_low))
                ci_uppers.append(float(ci_high))
            except Exception as e:
                logger.warning(f"Parcel {parcel_idx} t检验失败: {e}")
                t_stats.append(0.0)
                p_values.append(1.0)
                ci_lowers.append(float("nan"))
                ci_uppers.append(float("nan"))
        
        t_stats = np.array(t_stats)
        p_values = np.array(p_values)
        activation_diff_ci_lower = np.array(ci_lowers)
        activation_diff_ci_upper = np.array(ci_uppers)
        
        # 找出显著异常的Parcel
        significant_mask = p_values < self.significance_threshold
        significant_parcels = np.where(significant_mask)[0]
        
        # 按激活差异绝对值排序
        anomaly_scores = np.abs(activation_diff)
        top_anomalous_indices = np.argsort(anomaly_scores)[::-1]
        # 构建结果
        results = {
            'activation_diff': activation_diff.tolist(),
            'activation_diff_ci_lower': activation_diff_ci_lower.tolist(),
            'activation_diff_ci_upper': activation_diff_ci_upper.tolist(),
            't_stats': t_stats.tolist(),
            'p_values': p_values.tolist(),
            'significant_parcels': significant_parcels.tolist(),
            'top_anomalous_parcels': []
        }
        
        # 前20个最异常的Parcel的索引集合，用于后续排除
        top_20_indices = set(top_anomalous_indices[:20])
        
        # 添加前20个最异常的Parcel（带 rank 与 in_top_20）
        for i, parcel_idx in enumerate(top_anomalous_indices[:20]):
            parcel_info = self.get_parcel_info(int(parcel_idx))
            results['top_anomalous_parcels'].append({
                'parcel_id': int(parcel_idx),
                'rank': i + 1,
                'in_top_20': True,
                'activation_diff_ci_lower': float(activation_diff_ci_lower[parcel_idx]),
                'activation_diff_ci_upper': float(activation_diff_ci_upper[parcel_idx]),
                'function_name': parcel_info['function_name'],
                'function_description': parcel_info['function_description'],
                'model_role': parcel_info['model_role'],
                'keywords': parcel_info['keywords'],
                'activation_diff': float(activation_diff[parcel_idx]),
                't_stat': float(t_stats[parcel_idx]),
                'p_value': float(p_values[parcel_idx]),
                'is_significant': bool(significant_mask[parcel_idx]),
                'anomaly_score': float(anomaly_scores[parcel_idx])
            })
        
        # 添加非 top20 但显著的 Parcel（按 anomaly_score 降序）
        other_significant_indices = [idx for idx in top_anomalous_indices if significant_mask[idx] and idx not in top_20_indices]
        for parcel_idx in other_significant_indices:
            parcel_info = self.get_parcel_info(int(parcel_idx))
            results['top_anomalous_parcels'].append({
                'parcel_id': int(parcel_idx),
                'rank': 0,
                'in_top_20': False,
                'activation_diff_ci_lower': float(activation_diff_ci_lower[parcel_idx]),
                'activation_diff_ci_upper': float(activation_diff_ci_upper[parcel_idx]),
                'function_name': parcel_info['function_name'],
                'function_description': parcel_info['function_description'],
                'model_role': parcel_info['model_role'],
                'keywords': parcel_info['keywords'],
                'activation_diff': float(activation_diff[parcel_idx]),
                't_stat': float(t_stats[parcel_idx]),
                'p_value': float(p_values[parcel_idx]),
                'is_significant': True,
                'anomaly_score': float(anomaly_scores[parcel_idx])
            })
        
        logger.info(f"发现 {len(significant_parcels)} 个显著异常的Parcel，其中 top20 共 20 个，其余显著 {len(other_significant_indices)} 个已一并写入")
        logger.info(f"激活差异范围: [{np.min(activation_diff):.4f}, {np.max(activation_diff):.4f}]")
        
        return results, correct_all_acts, incorrect_all_acts
    
    def analyze_connectivity_anomalies(self, traditional_baseline: np.ndarray, pca_baseline: np.ndarray = None) -> Dict:
        """
        分析连接异常（同时计算传统方法和PCA拼接方法）
        
        Args:
            traditional_baseline: 传统方法基线连接矩阵
            pca_baseline: PCA拼接方法基线连接矩阵（可选）
            
        Returns:
            包含连接异常统计的字典
        """
        logger.info("分析连接异常...")
        
        # 传统方法计算幻觉样本连接矩阵
        logger.info("使用传统方法计算幻觉样本连接矩阵...")
        traditional_hallucination_connectivities = []
        for i, activations in enumerate(self.incorrect_activations):
            try:
                conn_matrix = self.compute_connectivity_matrix(activations)
                traditional_hallucination_connectivities.append(conn_matrix)
            except Exception as e:
                logger.warning(f"幻觉样本{i}连接矩阵计算失败: {e}")
                continue
        
        if len(traditional_hallucination_connectivities) == 0:
            raise ValueError("无法计算幻觉样本连接矩阵")
        
        # 计算平均连接矩阵
        avg_traditional_hallucination_connectivity = np.mean(traditional_hallucination_connectivities, axis=0)
        
        # PCA拼接方法计算幻觉样本连接矩阵
        avg_pca_hallucination_connectivity = None
        if pca_baseline is not None:
            logger.info("使用PCA拼接方法计算幻觉样本连接矩阵...")
            avg_pca_hallucination_connectivity = self.compute_pca_connectivity(self.incorrect_activations)
        
        # 计算连接时间序列用于显著性检验
        logger.info("计算连接时间序列...")
        # 传统方法的时间序列
        traditional_baseline_connectivity_series = []
        for activations in self.correct_activations:
            conn_matrix = self.compute_connectivity_matrix(activations)
            traditional_baseline_connectivity_series.append(conn_matrix)
        traditional_baseline_connectivity_series = np.array(traditional_baseline_connectivity_series)
        
        traditional_hallucination_connectivity_series = np.array(traditional_hallucination_connectivities)
        
        # 计算传统方法的连接差异和显著性
        traditional_connectivity_diff = avg_traditional_hallucination_connectivity - traditional_baseline
        
        # 计算传统方法的连接差异的显著性（t检验）
        logger.info("计算传统方法连接差异显著性...")
        traditional_connectivity_p_values = np.zeros_like(traditional_connectivity_diff)
        traditional_connectivity_significant = np.zeros_like(traditional_connectivity_diff, dtype=bool)
        
        for i in range(self.parcel_dim):
            for j in range(i + 1, self.parcel_dim):  # 只计算上三角部分
                # 获取两个Parcel之间的连接强度时间序列
                baseline_conn = traditional_baseline_connectivity_series[:, i, j]
                hallucination_conn = traditional_hallucination_connectivity_series[:, i, j]
                
                # 进行t检验
                try:
                    t_stat, p_value = stats.ttest_ind(hallucination_conn, baseline_conn)
                    traditional_connectivity_p_values[i, j] = p_value
                    traditional_connectivity_p_values[j, i] = p_value  # 对称矩阵
                    traditional_connectivity_significant[i, j] = p_value < self.significance_threshold
                    traditional_connectivity_significant[j, i] = traditional_connectivity_significant[i, j]
                except Exception as e:
                    logger.warning(f"传统方法连接差异t检验失败 P{i}-P{j}: {e}")
                    traditional_connectivity_p_values[i, j] = 1.0
                    traditional_connectivity_p_values[j, i] = 1.0
                    traditional_connectivity_significant[i, j] = False
                    traditional_connectivity_significant[j, i] = False
        
        # 统计传统方法显著差异
        traditional_significant_count = np.sum(traditional_connectivity_significant) // 2  # 除以2因为是对称矩阵
        total_connections = self.parcel_dim * (self.parcel_dim - 1) // 2
        logger.info(f"传统方法显著连接差异: {traditional_significant_count}/{total_connections} ({traditional_significant_count/total_connections*100:.1f}%)")
        
        # 计算传统方法连接异常分数 (上三角部分的平均绝对值)
        traditional_upper_tri_indices = np.triu_indices_from(traditional_connectivity_diff, k=1)
        traditional_upper_tri_diff = traditional_connectivity_diff[traditional_upper_tri_indices]
        traditional_connectivity_anomaly_score = np.mean(np.abs(traditional_upper_tri_diff))
        
        # 计算传统方法Frobenius范数
        traditional_frobenius_norm = np.linalg.norm(traditional_connectivity_diff, 'fro')
        
        # 构建传统方法结果
        traditional_results = {
            'connectivity_diff': traditional_connectivity_diff.tolist(),
            'connectivity_p_values': traditional_connectivity_p_values.tolist(),
            'connectivity_significant': traditional_connectivity_significant.astype(bool).tolist(),
            'connectivity_anomaly_score': float(traditional_connectivity_anomaly_score),
            'frobenius_norm': float(traditional_frobenius_norm),
            'significant_connections_count': int(traditional_significant_count),
            'total_connections_count': int(total_connections),
            'significant_connections_ratio': float(traditional_significant_count / total_connections),
            'baseline_connectivity': traditional_baseline.tolist(),
            'hallucination_connectivity': avg_traditional_hallucination_connectivity.tolist()
        }
        
        # 处理PCA拼接方法（如果启用）
        pca_results = None
        if avg_pca_hallucination_connectivity is not None:
            logger.info("处理PCA拼接方法结果...")
            pca_connectivity_diff = avg_pca_hallucination_connectivity - pca_baseline
            
            # PCA拼接方法不需要显著性检验，直接按大小排序
            pca_upper_tri_indices = np.triu_indices_from(pca_connectivity_diff, k=1)
            pca_upper_tri_diff = pca_connectivity_diff[pca_upper_tri_indices]
            pca_connectivity_anomaly_score = np.mean(np.abs(pca_upper_tri_diff))
            pca_frobenius_norm = np.linalg.norm(pca_connectivity_diff, 'fro')
            
            pca_results = {
                'connectivity_diff': pca_connectivity_diff.tolist(),
                'connectivity_anomaly_score': float(pca_connectivity_anomaly_score),
                'frobenius_norm': float(pca_frobenius_norm),
                'baseline_connectivity': pca_baseline.tolist(),
                'hallucination_connectivity': avg_pca_hallucination_connectivity.tolist()
            }
            
            logger.info(f"PCA拼接方法连接异常分数: {pca_connectivity_anomaly_score:.4f}")
            logger.info(f"PCA拼接方法Frobenius范数: {pca_frobenius_norm:.4f}")
        
        # 合并结果
        results = {
            'traditional': traditional_results,
            'pca_concate': pca_results
        }
        
        logger.info(f"传统方法连接异常分数: {traditional_connectivity_anomaly_score:.4f}")
        logger.info(f"传统方法Frobenius范数: {traditional_frobenius_norm:.4f}")
        
        return results
    
    def analyze_anomalous_connections(self, connectivity_diff: np.ndarray, 
                                    connectivity_significant: np.ndarray = None,
                                    connectivity_p_values: np.ndarray = None,
                                    top_k: int = 50,
                                    force_all_significant: bool = False,
                                    forced_p_value: float = 0.001) -> Dict:
        """
        分析异常连接关系
        
        Args:
            connectivity_diff: 连接差异矩阵
            connectivity_significant: 连接差异显著性矩阵
            connectivity_p_values: 连接差异p值矩阵
            top_k: 返回前k个最异常的连接
            
        Returns:
            包含异常连接信息的字典
        """
        logger.info(f"分析前{top_k}个最异常的连接关系...")
        
        # 获取上三角矩阵的索引和值
        upper_tri_indices = np.triu_indices_from(connectivity_diff, k=1)
        upper_tri_values = connectivity_diff[upper_tri_indices]
        
        # 如果提供了显著性矩阵，只考虑显著的连接
        if connectivity_significant is not None:
            upper_tri_significant = connectivity_significant[upper_tri_indices]
            # 只保留显著的连接
            significant_mask = upper_tri_significant
            significant_indices = upper_tri_indices[0][significant_mask], upper_tri_indices[1][significant_mask]
            significant_values = upper_tri_values[significant_mask]
            logger.info(f"显著连接数量: {len(significant_values)}")
        else:
            significant_indices = upper_tri_indices
            significant_values = upper_tri_values
        
        anomalous_connections = {"pos_connections": [], "neg_connections": []}

        # 分别选取：最大的 k/2 个正差值 与 最小的 k/2 个负差值
        half_k = max(1, top_k // 2)
        pos_mask = significant_values > 0
        neg_mask = significant_values < 0
        pos_indices_all = np.where(pos_mask)[0]
        neg_indices_all = np.where(neg_mask)[0]
        if pos_indices_all.size > 0:
            pos_sorted_local = np.argsort(significant_values[pos_indices_all])[::-1]
            pos_take = pos_indices_all[pos_sorted_local[:min(half_k, pos_indices_all.size)]]
        else:
            pos_take = np.array([], dtype=int)
        if neg_indices_all.size > 0:
            neg_sorted_local = np.argsort(significant_values[neg_indices_all])
            neg_take = neg_indices_all[neg_sorted_local[:min(half_k, neg_indices_all.size)]]
        else:
            neg_take = np.array([], dtype=int)
        selected_indices_pos = list(pos_take)
        selected_indices_neg = list(neg_take)

        for i, idx in enumerate(selected_indices_pos):
            parcel_i, parcel_j = significant_indices[0][idx], significant_indices[1][idx]
            diff_value = significant_values[idx]
            
            # 获取两个Parcel的功能信息
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            # 获取显著性信息
            if force_all_significant:
                is_significant = True
                p_value = forced_p_value
            else:
                is_significant = False
                p_value = 1.0
                if connectivity_significant is not None:
                    is_significant = connectivity_significant[parcel_i, parcel_j]
                if connectivity_p_values is not None:
                    p_value = connectivity_p_values[parcel_i, parcel_j]
            
            anomalous_connections["pos_connections"].append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connectivity_diff': float(diff_value),
                'abs_connectivity_diff': float(abs(diff_value)),
                'is_significant': bool(is_significant),
                'p_value': float(p_value),
                'rank': i + 1
            })
        for i, idx in enumerate(selected_indices_neg):
            parcel_i, parcel_j = significant_indices[0][idx], significant_indices[1][idx]
            diff_value = significant_values[idx]
            
            # 获取两个Parcel的功能信息
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            # 获取显著性信息
            if force_all_significant:
                is_significant = True
                p_value = forced_p_value
            else:
                is_significant = False
                p_value = 1.0
                if connectivity_significant is not None:
                    is_significant = connectivity_significant[parcel_i, parcel_j]
                if connectivity_p_values is not None:
                    p_value = connectivity_p_values[parcel_i, parcel_j]
            
            anomalous_connections["neg_connections"].append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connectivity_diff': float(diff_value),
                'abs_connectivity_diff': float(abs(diff_value)),
                'is_significant': bool(is_significant),
                'p_value': float(p_value),
                'rank': i + 1
            })
            
        results = {
            'anomalous_connections': anomalous_connections,
            'total_connections_analyzed': len(upper_tri_values),
            'top_k': top_k
        }
        
        logger.info(f"识别出 {len(anomalous_connections)} 个异常连接关系")
        
        return results
    
    def analyze_top_activated_parcels(self, correct_activations: np.ndarray, 
                                    incorrect_activations: np.ndarray, 
                                    top_k: int = 20) -> Dict:
        """
        分析top activated parcels
        
        Args:
            correct_activations: 正确样本的Parcel激活矩阵
            incorrect_activations: 幻觉样本的Parcel激活矩阵
            top_k: 返回前k个最激活的parcels
            
        Returns:
            包含top activated parcels信息的字典
        """
        logger.info(f"分析前{top_k}个最激活的Parcels...")
        
        # 计算平均激活值
        correct_mean_acts = np.mean(correct_activations, axis=0)
        incorrect_mean_acts = np.mean(incorrect_activations, axis=0)
        
        # 按激活值排序
        correct_sorted_indices = np.argsort(correct_mean_acts)[::-1]
        incorrect_sorted_indices = np.argsort(incorrect_mean_acts)[::-1]
        
        # 获取top-k parcels
        top_correct_parcels = []
        top_incorrect_parcels = []
        
        for i in range(min(top_k, len(correct_sorted_indices))):
            parcel_idx = correct_sorted_indices[i]
            parcel_info = self.get_parcel_info(int(parcel_idx))
            
            top_correct_parcels.append({
                'parcel_id': int(parcel_idx),
                'function_name': parcel_info['function_name'],
                'function_description': parcel_info['function_description'],
                'model_role': parcel_info['model_role'],
                'keywords': parcel_info['keywords'],
                'mean_activation': float(correct_mean_acts[parcel_idx]),
                'rank': i + 1
            })
        
        for i in range(min(top_k, len(incorrect_sorted_indices))):
            parcel_idx = incorrect_sorted_indices[i]
            parcel_info = self.get_parcel_info(int(parcel_idx))
            
            top_incorrect_parcels.append({
                'parcel_id': int(parcel_idx),
                'function_name': parcel_info['function_name'],
                'function_description': parcel_info['function_description'],
                'model_role': parcel_info['model_role'],
                'keywords': parcel_info['keywords'],
                'mean_activation': float(incorrect_mean_acts[parcel_idx]),
                'rank': i + 1
            })
        
        results = {
            'top_correct_parcels': top_correct_parcels,
            'top_incorrect_parcels': top_incorrect_parcels,
            'top_k': top_k
        }
        
        logger.info(f"识别出 {len(top_correct_parcels)} 个top正确Parcels")
        logger.info(f"识别出 {len(top_incorrect_parcels)} 个top幻觉Parcels")
        
        return results
    
    def analyze_top_parcel_connections(self, correct_connectivity: np.ndarray, 
                                     incorrect_connectivity: np.ndarray, 
                                     top_k: int = 50) -> Dict:
        """
        分析top parcel connections
        
        Args:
            correct_connectivity: 正确样本的连接矩阵
            incorrect_connectivity: 幻觉样本的连接矩阵
            top_k: 返回前k个最强的连接
            
        Returns:
            包含top connections信息的字典
        """
        logger.info(f"分析前{top_k}个最强的Parcel连接...")
        
        # 获取上三角矩阵的索引和值
        upper_tri_indices = np.triu_indices_from(correct_connectivity, k=1)
        correct_upper_tri = correct_connectivity[upper_tri_indices]
        incorrect_upper_tri = incorrect_connectivity[upper_tri_indices]
        
        # 按连接强度排序
        correct_abs_values = np.abs(correct_upper_tri)
        incorrect_abs_values = np.abs(incorrect_upper_tri)
        
        correct_sorted_indices = np.argsort(correct_abs_values)[::-1]
        incorrect_sorted_indices = np.argsort(incorrect_abs_values)[::-1]
        
        # 获取top-k连接
        top_correct_connections = []
        top_incorrect_connections = []
        
        for i in range(min(top_k, len(correct_sorted_indices))):
            idx = correct_sorted_indices[i]
            parcel_i, parcel_j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            connection_strength = correct_upper_tri[idx]
            
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            top_correct_connections.append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connection_strength': float(connection_strength),
                'abs_connection_strength': float(correct_abs_values[idx]),
                'rank': i + 1
            })
        
        for i in range(min(top_k, len(incorrect_sorted_indices))):
            idx = incorrect_sorted_indices[i]
            parcel_i, parcel_j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            connection_strength = incorrect_upper_tri[idx]
            
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            top_incorrect_connections.append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connection_strength': float(connection_strength),
                'abs_connection_strength': float(incorrect_abs_values[idx]),
                'rank': i + 1
            })
        
        results = {
            'top_correct_connections': top_correct_connections,
            'top_incorrect_connections': top_incorrect_connections,
            'top_k': top_k
        }
        
        logger.info(f"识别出 {len(top_correct_connections)} 个top正确Parcel连接")
        logger.info(f"识别出 {len(top_incorrect_connections)} 个top幻觉Parcel连接")
        
        return results
    
    def visualize_connectivity_graphs(self, correct_connectivity: np.ndarray, 
                                    incorrect_connectivity: np.ndarray,
                                    connectivity_diff: np.ndarray, 
                                    top_k_edges: int = 100,
                                    correct_activations: np.ndarray = None,
                                    incorrect_activations: np.ndarray = None,
                                    connectivity_significant: np.ndarray = None) -> None:
        """
        可视化连接关系矩阵
        
        Args:
            correct_connectivity: 正确样本连接矩阵
            incorrect_connectivity: 幻觉样本连接矩阵
            connectivity_diff: 连接差异矩阵
            top_k_edges: 显示前k个最强的连接
            correct_activations: 正确样本激活值
            incorrect_activations: 幻觉样本激活值
            connectivity_significant: 连接差异显著性矩阵
        """
        logger.info("生成Parcel连接关系可视化图...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建输出目录
        viz_dir = self.output_dir / "connectivity_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. 正确样本连接图
        self._plot_connectivity_graph(
            correct_connectivity, 
            "Correct Sample Parcel Connectivity",
            viz_dir / "correct_parcel_connectivity.html",
            "correct",
            top_k_edges,
            correct_activations
        )
        
        # 2. 幻觉样本连接图
        self._plot_connectivity_graph(
            incorrect_connectivity,
            "Hallucination Sample Parcel Connectivity", 
            viz_dir / "incorrect_parcel_connectivity.html",
            "incorrect",
            top_k_edges,
            incorrect_activations
        )
        
        # 3. 连接差异图
        self._plot_connectivity_diff_graph(
            connectivity_diff,
            "Parcel Connectivity Difference (Hallucination - Correct)",
            viz_dir / "parcel_connectivity_diff.html",
            top_k_edges,
            correct_activations,
            incorrect_activations,
            connectivity_significant
        )
        
        logger.info(f"Parcel连接关系可视化图已保存到: {viz_dir}")
    
    def _plot_connectivity_graph(self, connectivity: np.ndarray, title: str, 
                               save_path: Path, graph_type: str, top_k_edges: int = 100,
                               activations: np.ndarray = None) -> None:
        """绘制连接关系图"""
        # 创建网络图
        G = nx.Graph()
        
        # 获取所有上三角连接的权重
        upper_tri_indices = np.triu_indices_from(connectivity, k=1)
        upper_tri_weights = connectivity[upper_tri_indices]
        
        # 按绝对值排序，获取top-k连接
        abs_weights = np.abs(upper_tri_weights)
        sorted_indices = np.argsort(abs_weights)[::-1]
        top_k_indices = sorted_indices[:top_k_edges]
        
        # 获取top-k连接的节点对
        top_k_edges_list = []
        for idx in top_k_indices:
            i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            weight = connectivity[i, j]
            top_k_edges_list.append((i, j, weight))
        
        if len(top_k_edges_list) == 0:
            logger.warning(f"没有找到连接关系用于{graph_type}图")
            return
        
        # 添加节点（只添加有连接的节点）
        involved_nodes = set()
        for i, j, weight in top_k_edges_list:
            involved_nodes.add(i)
            involved_nodes.add(j)
        
        for node_id in involved_nodes:
            parcel_info = self.get_parcel_info(node_id)
            node_name = f"P{node_id}\n{parcel_info['function_name'].replace('**', '').strip()}"
            G.add_node(node_id, name=node_name, info=parcel_info)
        
        # 添加边
        for i, j, weight in top_k_edges_list:
            G.add_edge(i, j, weight=abs(weight), original_weight=weight)
        
        # 创建HTML可视化
        
        # 使用spring布局获取节点位置
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"P{node}")
            
            # 获取Parcel信息
            parcel_info = G.nodes[node]['info']
            function_name = parcel_info.get('function_name', 'Unknown').replace('**', '').strip()
            node_info.append(f"Parcel {node}<br>Function: {function_name}")
            
            # 计算节点大小（基于激活值）
            if activations is not None:
                # 使用平均激活值
                activation_value = np.mean(activations[:, node])
                node_sizes.append(max(10, min(60, 20 + abs(activation_value) * 100)))
            else:
                # 如果没有激活值，使用连接强度
                node_connections = sum([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)])
                node_sizes.append(max(10, min(50, 20 + node_connections * 10)))
        
        # 准备边数据
        edge_x = []
        edge_y = []
        edge_info = []
        edge_widths = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = G[edge[0]][edge[1]]['weight']
            original_weight = G[edge[0]][edge[1]]['original_weight']
            edge_info.append(f"Connection Strength: {weight:.3f}<br>Original Weight: {original_weight:.3f}")
            edge_widths.append(max(1, weight * 10))
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        if edge_x:  # 只有当有边时才添加
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='lightgray'),
                hoverinfo='none',
                mode='lines',
                name='Connections'
            ))
        
        # 添加节点
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='darkblue'),
                opacity=0.8
            ),
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
            customdata=node_info,
            name='Parcels'
        ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Node size represents connection importance, edge thickness represents connection strength",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # 保存为HTML
        plot(fig, filename=str(save_path), auto_open=False)
        
        logger.info(f"已保存{graph_type}连接图: {save_path}")
    
    def _plot_connectivity_diff_graph(self, connectivity_diff: np.ndarray, 
                                    title: str, save_path: Path, top_k_edges: int = 100,
                                    correct_activations: np.ndarray = None,
                                    incorrect_activations: np.ndarray = None,
                                    connectivity_significant: np.ndarray = None) -> None:
        """绘制连接差异图"""
        # 创建网络图
        G = nx.Graph()
        
        # 获取所有上三角连接的差异值
        upper_tri_indices = np.triu_indices_from(connectivity_diff, k=1)
        upper_tri_diffs = connectivity_diff[upper_tri_indices]
        
        # 如果提供了显著性矩阵，只考虑显著的边
        if connectivity_significant is not None:
            upper_tri_significant = connectivity_significant[upper_tri_indices]
            # 只保留显著的连接
            significant_mask = upper_tri_significant
            significant_diffs = upper_tri_diffs[significant_mask]
            significant_indices = upper_tri_indices[0][significant_mask], upper_tri_indices[1][significant_mask]
            logger.info(f"显著连接差异数量: {len(significant_diffs)}")
        else:
            significant_diffs = upper_tri_diffs
            significant_indices = upper_tri_indices
        
        # 分别选取：最大的 half_k 个正差值 与 最小的 half_k 个负差值
        half_k = max(1, top_k_edges // 2)
        pos_mask = significant_diffs > 0
        neg_mask = significant_diffs < 0
        pos_idx_all = np.where(pos_mask)[0]
        neg_idx_all = np.where(neg_mask)[0]
        if pos_idx_all.size > 0:
            pos_sorted_local = np.argsort(significant_diffs[pos_idx_all])[::-1]
            pos_take = pos_idx_all[pos_sorted_local[:min(half_k, pos_idx_all.size)]]
        else:
            pos_take = np.array([], dtype=int)
        if neg_idx_all.size > 0:
            neg_sorted_local = np.argsort(significant_diffs[neg_idx_all])
            neg_take = neg_idx_all[neg_sorted_local[:min(half_k, neg_idx_all.size)]]
        else:
            neg_take = np.array([], dtype=int)
        take_indices = list(pos_take) + list(neg_take)
        
        # 获取选择的节点对
        top_k_edges_list = []
        for idx in take_indices:
            i, j = significant_indices[0][idx], significant_indices[1][idx]
            diff = significant_diffs[idx]
            top_k_edges_list.append((i, j, diff))
        
        if len(top_k_edges_list) == 0:
            logger.warning("没有找到连接差异用于差异图")
            return
        
        # 添加节点（只添加有差异的节点）
        involved_nodes = set()
        for i, j, diff in top_k_edges_list:
            involved_nodes.add(i)
            involved_nodes.add(j)
        
        for node_id in involved_nodes:
            parcel_info = self.get_parcel_info(node_id)
            node_name = f"P{node_id}\n{parcel_info['function_name'].replace('**', '').strip()}"
            G.add_node(node_id, name=node_name, info=parcel_info)
        
        # 添加边并分类
        positive_edges = []
        negative_edges = []
        
        for i, j, diff in top_k_edges_list:
            weight = abs(diff)
            G.add_edge(i, j, weight=weight, diff=diff)
            if diff > 0:
                positive_edges.append((i, j))
            else:
                negative_edges.append((i, j))
        
        # 创建HTML可视化
        
        # 使用spring布局获取节点位置
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 计算激活差异
        if correct_activations is not None and incorrect_activations is not None:
            # 计算激活差异：incorrect - correct
            activation_diff = np.mean(incorrect_activations, axis=0) - np.mean(correct_activations, axis=0)
        else:
            # 如果没有提供激活数据，使用连接差异的对角线元素
            activation_diff = np.diag(connectivity_diff)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        node_sizes = []
        node_colors = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"P{node}")
            
            # 获取Parcel信息
            parcel_info = G.nodes[node]['info']
            function_name = parcel_info.get('function_name', 'Unknown').replace('**', '').strip()
            node_info.append(f"Parcel {node}<br>Function: {function_name}")
            
            # 计算节点大小（基于激活差异的绝对值）
            diff_abs = abs(activation_diff[node])
            node_sizes.append(max(10, min(60, 20 + diff_abs * 100)))
            
            # 计算节点颜色（基于激活差异的正负）
            diff_value = activation_diff[node]
            if diff_value > 0:
                node_colors.append('red')  # 正差异：红色
            elif diff_value < 0:
                node_colors.append('blue')  # 负差异：蓝色
            else:
                node_colors.append('gray')  # 无差异：灰色
        
        # 准备边数据（分别处理正负差异）
        pos_edge_x = []
        pos_edge_y = []
        pos_edge_info = []
        pos_edge_widths = []
        
        neg_edge_x = []
        neg_edge_y = []
        neg_edge_info = []
        neg_edge_widths = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            diff = G[edge[0]][edge[1]]['diff']
            weight = G[edge[0]][edge[1]]['weight']
            
            if diff > 0:  # 正差异
                pos_edge_x.extend([x0, x1, None])
                pos_edge_y.extend([y0, y1, None])
                pos_edge_info.append(f"Positive Difference: +{diff:.3f}<br>Strength: {weight:.3f}")
                pos_edge_widths.append(max(1, weight * 10))
            else:  # 负差异
                neg_edge_x.extend([x0, x1, None])
                neg_edge_y.extend([y0, y1, None])
                neg_edge_info.append(f"Negative Difference: {diff:.3f}<br>Strength: {weight:.3f}")
                neg_edge_widths.append(max(1, weight * 10))
        
        # 创建图形
        fig = go.Figure()
        
        # 添加正差异边（红色）
        if pos_edge_x:
            fig.add_trace(go.Scatter(
                x=pos_edge_x, y=pos_edge_y,
                line=dict(width=3, color='red'),
                hoverinfo='none',
                mode='lines',
                name='Positive Differences (Enhanced Connections)',
                opacity=0.8
            ))
        
        # 添加负差异边（蓝色）
        if neg_edge_x:
            fig.add_trace(go.Scatter(
                x=neg_edge_x, y=neg_edge_y,
                line=dict(width=3, color='blue'),
                hoverinfo='none',
                mode='lines',
                name='Negative Differences (Weakened Connections)',
                opacity=0.8
            ))
        
        # 添加节点（按颜色分组）
        # 正差异节点（红色）
        pos_nodes = [i for i, color in enumerate(node_colors) if color == 'red']
        if pos_nodes:
            fig.add_trace(go.Scatter(
                x=[node_x[i] for i in pos_nodes],
                y=[node_y[i] for i in pos_nodes],
                mode='markers+text',
                marker=dict(
                    size=[node_sizes[i] for i in pos_nodes],
                    color='red',
                    line=dict(width=2, color='darkred'),
                    opacity=0.8
                ),
                text=[node_text[i] for i in pos_nodes],
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
                customdata=[node_info[i] for i in pos_nodes],
                name='Positive Difference Parcels'
            ))
        
        # 负差异节点（蓝色）
        neg_nodes = [i for i, color in enumerate(node_colors) if color == 'blue']
        if neg_nodes:
            fig.add_trace(go.Scatter(
                x=[node_x[i] for i in neg_nodes],
                y=[node_y[i] for i in neg_nodes],
                mode='markers+text',
                marker=dict(
                    size=[node_sizes[i] for i in neg_nodes],
                    color='blue',
                    line=dict(width=2, color='darkblue'),
                    opacity=0.8
                ),
                text=[node_text[i] for i in neg_nodes],
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
                customdata=[node_info[i] for i in neg_nodes],
                name='Negative Difference Parcels'
            ))
        
        # 无差异节点（灰色）
        neutral_nodes = [i for i, color in enumerate(node_colors) if color == 'gray']
        if neutral_nodes:
            fig.add_trace(go.Scatter(
                x=[node_x[i] for i in neutral_nodes],
                y=[node_y[i] for i in neutral_nodes],
                mode='markers+text',
                marker=dict(
                    size=[node_sizes[i] for i in neutral_nodes],
                    color='gray',
                    line=dict(width=2, color='darkgray'),
                    opacity=0.8
                ),
                text=[node_text[i] for i in neutral_nodes],
                textposition="middle center",
                textfont=dict(size=10, color='black'),
                hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
                customdata=[node_info[i] for i in neutral_nodes],
                name='Neutral Parcels'
            ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Red nodes: Parcels with positive activation differences<br>Blue nodes: Parcels with negative activation differences<br>Gray nodes: Parcels with neutral differences<br>Node size represents activation difference magnitude<br>Red edges: Enhanced connections, Blue edges: Weakened connections",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # 保存为HTML
        plot(fig, filename=str(save_path), auto_open=False)
        
        logger.info(f"已保存连接差异图: {save_path}")
    
    def save_connectivity_matrices(self, traditional_results: Dict, pca_results: Dict = None) -> None:
        """
        保存连接矩阵和高连接、异常连接信息
        
        Args:
            traditional_results: 传统方法结果
            pca_results: PCA拼接方法结果（可选）
        """
        logger.info("保存连接矩阵和高连接、异常连接信息...")
        
        # 创建连接矩阵保存目录
        conn_dir = self.output_dir / "connectivity_matrices"
        conn_dir.mkdir(exist_ok=True)
        
        # 保存传统方法结果
        logger.info("保存传统方法连接矩阵...")
        traditional_correct = np.array(traditional_results['baseline_connectivity'])
        traditional_incorrect = np.array(traditional_results['hallucination_connectivity'])
        traditional_diff = np.array(traditional_results['connectivity_diff'])
        traditional_significant = np.array(traditional_results['connectivity_significant'])
        
        np.save(conn_dir / "correct_connectivity_matrix.npy", traditional_correct)
        np.save(conn_dir / "incorrect_connectivity_matrix.npy", traditional_incorrect)
        np.save(conn_dir / "connectivity_difference_matrix.npy", traditional_diff)
        np.save(conn_dir / "connectivity_significance_matrix.npy", traditional_significant)


        # 保存节点名称（对应行/列名称）
        try:
            parcel_names = []
            for idx in range(self.parcel_dim):
                info = self.get_parcel_info(idx)
                name = info.get('function_name', f'Parcel {idx}')
                name = name.replace('**', '').strip()
                parcel_names.append({"id": int(idx), "name": name})
            with open(conn_dir / "parcel_node_names.json", 'w', encoding='utf-8') as f:
                json.dump(parcel_names, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"保存Parcel节点名称失败: {e}")
        
        # 保存传统方法高连接和异常连接信息
        self._save_high_connections(traditional_correct, traditional_incorrect, conn_dir)
        self._save_anomaly_connections(traditional_diff, conn_dir, traditional_significant)
        
        # 保存PCA拼接方法结果（如果存在）
        if pca_results is not None:
            logger.info("保存PCA拼接方法连接矩阵...")
            pca_correct = np.array(pca_results['baseline_connectivity'])
            pca_incorrect = np.array(pca_results['hallucination_connectivity'])
            pca_diff = np.array(pca_results['connectivity_diff'])
            
            np.save(conn_dir / "correct_connectivity_matrix_concate.npy", pca_correct)
            np.save(conn_dir / "incorrect_connectivity_matrix_concate.npy", pca_incorrect)
            np.save(conn_dir / "connectivity_difference_matrix_concate.npy", pca_diff)
            
            # 保存PCA拼接方法高连接和异常连接信息（不需要显著性）
            self._save_high_connections(pca_correct, pca_incorrect, conn_dir, suffix="_concate")
            self._save_anomaly_connections(pca_diff, conn_dir, None, suffix="_concate")
        
        logger.info(f"连接矩阵已保存到: {conn_dir}")
    
    def _save_high_connections(self, correct_connectivity: np.ndarray, 
                             incorrect_connectivity: np.ndarray, 
                             output_dir: Path, top_k: int = 100, suffix: str = "") -> None:
        """保存高连接信息"""
        logger.info(f"保存前{top_k}个高连接信息...")
        
        # 获取上三角矩阵的索引和值
        upper_tri_indices = np.triu_indices_from(correct_connectivity, k=1)
        correct_upper_tri = correct_connectivity[upper_tri_indices]
        incorrect_upper_tri = incorrect_connectivity[upper_tri_indices]
        
        # 按绝对值排序
        correct_abs_values = np.abs(correct_upper_tri)
        incorrect_abs_values = np.abs(incorrect_upper_tri)
        
        correct_sorted_indices = np.argsort(correct_abs_values)[::-1]
        incorrect_sorted_indices = np.argsort(incorrect_abs_values)[::-1]
        
        # 保存正确样本的高连接
        high_correct_connections = []
        for i in range(min(top_k, len(correct_sorted_indices))):
            idx = correct_sorted_indices[i]
            parcel_i, parcel_j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            connection_strength = correct_upper_tri[idx]
            
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            high_correct_connections.append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connection_strength': float(connection_strength),
                'abs_connection_strength': float(correct_abs_values[idx]),
                'rank': i + 1
            })
        
        # 保存幻觉样本的高连接
        high_incorrect_connections = []
        for i in range(min(top_k, len(incorrect_sorted_indices))):
            idx = incorrect_sorted_indices[i]
            parcel_i, parcel_j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
            connection_strength = incorrect_upper_tri[idx]
            
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            high_incorrect_connections.append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connection_strength': float(connection_strength),
                'abs_connection_strength': float(incorrect_abs_values[idx]),
                'rank': i + 1
            })
        
        # 保存到文件
        high_connections = {
            'correct_high_connections': high_correct_connections,
            'incorrect_high_connections': high_incorrect_connections,
            'top_k': top_k
        }
        
        filename = f"high_connections{suffix}.json"
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(high_connections, f, indent=2, ensure_ascii=False)
        
        logger.info(f"高连接信息已保存到: {output_dir / filename}")
    
    def _save_anomaly_connections(self, connectivity_diff: np.ndarray, 
                                output_dir: Path, 
                                connectivity_significant: np.ndarray = None,
                                top_k: int = 100, suffix: str = "") -> None:
        """保存异常连接信息"""
        logger.info(f"保存前{top_k}个异常连接信息...")
        
        # 获取上三角矩阵的索引和值
        upper_tri_indices = np.triu_indices_from(connectivity_diff, k=1)
        upper_tri_values = connectivity_diff[upper_tri_indices]
        
        # 如果提供了显著性矩阵，只考虑显著的连接
        if connectivity_significant is not None:
            upper_tri_significant = connectivity_significant[upper_tri_indices]
            significant_mask = upper_tri_significant
            significant_indices = upper_tri_indices[0][significant_mask], upper_tri_indices[1][significant_mask]
            significant_values = upper_tri_values[significant_mask]
        else:
            # 对于PCA拼接方法，不需要显著性检验，直接使用所有连接
            significant_indices = upper_tri_indices
            significant_values = upper_tri_values
        
        # 分别选取正负差异
        half_k = max(1, top_k // 2)
        pos_mask = significant_values > 0
        neg_mask = significant_values < 0
        pos_indices_all = np.where(pos_mask)[0]
        neg_indices_all = np.where(neg_mask)[0]
        
        if pos_indices_all.size > 0:
            pos_sorted_local = np.argsort(significant_values[pos_indices_all])[::-1]
            pos_take = pos_indices_all[pos_sorted_local[:min(half_k, pos_indices_all.size)]]
        else:
            pos_take = np.array([], dtype=int)
        
        if neg_indices_all.size > 0:
            neg_sorted_local = np.argsort(significant_values[neg_indices_all])
            neg_take = neg_indices_all[neg_sorted_local[:min(half_k, neg_indices_all.size)]]
        else:
            neg_take = np.array([], dtype=int)
        
        # 构建异常连接信息
        anomaly_connections = {"positive_anomalies": [], "negative_anomalies": []}
        
        # 正异常连接
        for i, idx in enumerate(pos_take):
            parcel_i, parcel_j = significant_indices[0][idx], significant_indices[1][idx]
            diff_value = significant_values[idx]
            
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            is_significant = False
            if connectivity_significant is not None:
                is_significant = connectivity_significant[parcel_i, parcel_j]
            
            anomaly_connections["positive_anomalies"].append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connectivity_diff': float(diff_value),
                'abs_connectivity_diff': float(abs(diff_value)),
                'is_significant': bool(is_significant),
                'rank': i + 1
            })
        
        # 负异常连接
        for i, idx in enumerate(neg_take):
            parcel_i, parcel_j = significant_indices[0][idx], significant_indices[1][idx]
            diff_value = significant_values[idx]
            
            parcel_i_info = self.get_parcel_info(parcel_i)
            parcel_j_info = self.get_parcel_info(parcel_j)
            
            is_significant = False
            if connectivity_significant is not None:
                is_significant = connectivity_significant[parcel_i, parcel_j]
            
            anomaly_connections["negative_anomalies"].append({
                'parcel_i': {
                    'id': int(parcel_i),
                    'function_name': parcel_i_info['function_name'],
                    'function_description': parcel_i_info['function_description'],
                    'model_role': parcel_i_info['model_role']
                },
                'parcel_j': {
                    'id': int(parcel_j),
                    'function_name': parcel_j_info['function_name'],
                    'function_description': parcel_j_info['function_description'],
                    'model_role': parcel_j_info['model_role']
                },
                'connectivity_diff': float(diff_value),
                'abs_connectivity_diff': float(abs(diff_value)),
                'is_significant': bool(is_significant),
                'rank': i + 1
            })
        
        # 保存到文件
        filename = f"anomaly_connections_detailed{suffix}.json"
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(anomaly_connections, f, indent=2, ensure_ascii=False)
        
        logger.info(f"异常连接信息已保存到: {output_dir / filename}")
    
    def check_existing_files(self) -> bool:
        """检查结果文件是否已存在"""
        if not self.skip_existing:
            return False
            
        required_files = [
            "parcel_activation_diff.json",
            "parcel_connectivity_diff.npy", 
            "top_anomalous_parcels.json",
            "anomalous_connections.json",
            "parcel_level_analysis_complete.json"
        ]
        if self.use_pca_connectivity:
            required_files.append("anomalous_connections_pca.json")
        
        for filename in required_files:
            file_path = self.output_dir / filename
            if not file_path.exists():
                return False
                
        logger.info("所有结果文件已存在，跳过分析")
        return True
    
    def save_results(self, activation_results: Dict, connectivity_results: Dict, 
                    anomalous_connections: Dict = None, anomalous_connections_pca: Dict = None,
                    top_activated_parcels: Dict = None, top_parcel_connections: Dict = None) -> None:
        """保存分析结果"""
        logger.info("保存分析结果...")
        
        # 保存激活异常结果
        activation_file = self.output_dir / "parcel_activation_diff.json"
        with open(activation_file, 'w', encoding='utf-8') as f:
            json.dump(activation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"激活异常结果已保存到: {activation_file}")
        
        # 保存连接异常矩阵（传统方法）
        traditional_results = connectivity_results['traditional']
        connectivity_file = self.output_dir / "parcel_connectivity_diff.npy"
        np.save(connectivity_file, np.array(traditional_results['connectivity_diff']))
        logger.info(f"连接异常矩阵已保存到: {connectivity_file}")
        
        # 如果存在PCA拼接方法结果，也保存
        if connectivity_results.get('pca_concate') is not None:
            pca_results = connectivity_results['pca_concate']
            pca_connectivity_file = self.output_dir / "parcel_connectivity_diff_concate.npy"
            np.save(pca_connectivity_file, np.array(pca_results['connectivity_diff']))
            logger.info(f"PCA拼接连接异常矩阵已保存到: {pca_connectivity_file}")
        
        # 保存异常Parcel排名
        top_anomalous_file = self.output_dir / "top_anomalous_parcels.json"
        with open(top_anomalous_file, 'w', encoding='utf-8') as f:
            json.dump(activation_results['top_anomalous_parcels'], f, indent=2, ensure_ascii=False)
        logger.info(f"异常Parcel排名已保存到: {top_anomalous_file}")
        
        # 保存异常连接关系
        if anomalous_connections is not None:
            anomalous_conn_file = self.output_dir / "anomalous_connections.json"
            with open(anomalous_conn_file, 'w', encoding='utf-8') as f:
                json.dump(anomalous_connections, f, indent=2, ensure_ascii=False)
            logger.info(f"异常连接关系已保存到: {anomalous_conn_file}")
        
        # 保存PCA异常连接关系（格式与traditional一致）
        if anomalous_connections_pca is not None:
            anomalous_conn_pca_file = self.output_dir / "anomalous_connections_pca.json"
            with open(anomalous_conn_pca_file, 'w', encoding='utf-8') as f:
                json.dump(anomalous_connections_pca, f, indent=2, ensure_ascii=False)
            logger.info(f"PCA异常连接关系已保存到: {anomalous_conn_pca_file}")
        
        # 保存top activated parcels
        if top_activated_parcels is not None:
            top_activated_file = self.output_dir / "top_activated_parcels.json"
            with open(top_activated_file, 'w', encoding='utf-8') as f:
                json.dump(top_activated_parcels, f, indent=2, ensure_ascii=False)
            logger.info(f"Top激活Parcels已保存到: {top_activated_file}")
        
        # 保存top parcel connections
        if top_parcel_connections is not None:
            top_connections_file = self.output_dir / "top_parcel_connections.json"
            with open(top_connections_file, 'w', encoding='utf-8') as f:
                json.dump(top_parcel_connections, f, indent=2, ensure_ascii=False)
            logger.info(f"Top Parcel连接已保存到: {top_connections_file}")
        
        # 保存完整结果
        complete_results = {
            'activation_analysis': activation_results,
            'connectivity_analysis': connectivity_results,
            'parameters': {
                'parcel_dim': self.parcel_dim,
                'correct_samples': len(self.correct_activations),
                'incorrect_samples': len(self.incorrect_activations),
                'significance_threshold': self.significance_threshold,
                'epsilon': self.epsilon,
                'max_tokens': self.max_tokens,
                'actual_tokens_per_sample': self.correct_activations[0].shape[0] if len(self.correct_activations) > 0 else None
            }
        }
        
        if anomalous_connections is not None:
            complete_results['anomalous_connections'] = anomalous_connections
        if anomalous_connections_pca is not None:
            complete_results['anomalous_connections_pca'] = anomalous_connections_pca
        if top_activated_parcels is not None:
            complete_results['top_activated_parcels'] = top_activated_parcels
        if top_parcel_connections is not None:
            complete_results['top_parcel_connections'] = top_parcel_connections
        
        complete_file = self.output_dir / "parcel_level_analysis_complete.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, indent=2, ensure_ascii=False)
        logger.info(f"完整分析结果已保存到: {complete_file}")
    
    def run_analysis(self) -> None:
        """运行完整的Parcel级别分析"""
        try:
            logger.info("开始Parcel级别异常分析...")
            
            # 检查是否跳过已存在的文件
            if self.check_existing_files():
                return
            
            # 1. 加载Parcel功能描述信息
            self.load_parcel_info()
            
            # 2. 加载数据
            self.load_activation_data()
            
            # 3. 分析激活异常
            activation_results, correct_all_acts, incorrect_all_acts = self.analyze_activation_anomalies()
            
            # 4. 计算基线连接矩阵
            traditional_baseline, pca_baseline = self.compute_baseline_connectivity()
            
            # 5. 分析连接异常
            connectivity_results = self.analyze_connectivity_anomalies(traditional_baseline, pca_baseline)
            
            # 6. 分析异常连接关系（传统方法）
            traditional_results = connectivity_results['traditional']
            anomalous_connections = self.analyze_anomalous_connections(
                np.array(traditional_results['connectivity_diff']), 
                np.array(traditional_results['connectivity_significant']),
                np.array(traditional_results['connectivity_p_values']),
                top_k=50
            )
            
            # 7. 分析异常连接关系（PCA方法，若启用则不做显著性检验，按需求写死显著性）
            anomalous_connections_pca = None
            pca_results = connectivity_results.get('pca_concate')
            if pca_results is not None:
                anomalous_connections_pca = self.analyze_anomalous_connections(
                    np.array(pca_results['connectivity_diff']),
                    connectivity_significant=None,
                    connectivity_p_values=None,
                    top_k=50,
                    force_all_significant=True,
                    forced_p_value=0.001
                )
            
            # 8. 分析top activated parcels
            top_activated_parcels = self.analyze_top_activated_parcels(
                correct_all_acts, incorrect_all_acts, top_k=300
            )
            
            # 9. 分析top parcel connections（传统方法）
            traditional_baseline_conn = np.array(traditional_results['baseline_connectivity'])
            traditional_hallucination_conn = np.array(traditional_results['hallucination_connectivity'])
            top_parcel_connections = self.analyze_top_parcel_connections(
                traditional_baseline_conn, traditional_hallucination_conn, top_k=50
            )
            
            # 10. 生成连接关系可视化图（传统方法）
            traditional_connectivity_diff = np.array(traditional_results['connectivity_diff'])
            
            # 获取激活值用于节点大小计算
            correct_acts = np.array(correct_all_acts)
            incorrect_acts = np.array(incorrect_all_acts)
            
            self.visualize_connectivity_graphs(traditional_baseline_conn, traditional_hallucination_conn, traditional_connectivity_diff, 
                                            top_k_edges=self.top_k_edges, 
                                            correct_activations=correct_acts,
                                            incorrect_activations=incorrect_acts,
                                            connectivity_significant=np.array(traditional_results['connectivity_significant']))
            
            # 11. 保存连接矩阵和高连接、异常连接信息
            self.save_connectivity_matrices(traditional_results, pca_results)
            
            # 12. 保存结果
            self.save_results(activation_results, connectivity_results, anomalous_connections,
                            anomalous_connections_pca,
                            top_activated_parcels, top_parcel_connections)
            
            logger.info("Parcel级别分析完成！")
            
        except Exception as e:
            logger.error(f"分析过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Parcel级别异常分析')
    parser.add_argument('--correct_jsonl', type=str, required=True,
                       help='正确样本激活数据路径 (JSONL格式)')
    parser.add_argument('--incorrect_jsonl', type=str, required=True,
                       help='幻觉样本激活数据路径 (JSONL格式)')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--parcel_info', type=str, default=None,
                       help='Parcel功能描述信息文件路径 (JSON格式)')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                       help='L2归一化的小常数 (默认: 1e-8)')
    parser.add_argument('--significance_threshold', type=float, default=0.05,
                       help='统计显著性阈值 (默认: 0.05)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='如果结果文件已存在则跳过分析')
    parser.add_argument('--top_k_edges', type=int, default=500,
                       help='可视化时显示前k个最强的连接 (默认: 100)')
    parser.add_argument('--anomalous_select_mode', type=str, default='posneg', choices=['posneg','abs'],
                       help="异常连接Top-K挑选策略：'posneg'（正负各取一半）或 'abs'（绝对值最大）")
    parser.add_argument('--use_structural_mask', action='store_true',
                       help='是否使用结构性连接矩阵作为mask')
    parser.add_argument('--structural_matrix_path', type=str, default=None,
                       help='结构性连接矩阵文件路径')
    parser.add_argument('--structural_threshold', type=float, default=0.0,
                       help='结构性连接阈值 (默认: 0.0)')
    parser.add_argument('--mask_type', type=str, default='binary', choices=['binary', 'weighted'],
                       help="Mask类型：'binary'（二进制mask）或 'weighted'（加权mask）")
    parser.add_argument('--use_pca_connectivity', action='store_true',
                       help='是否使用PCA方法计算连接性（拼接所有token后PCA降维）')
    parser.add_argument('--pca_explained_variance', type=float, default=0.8,
                       help='PCA保留的可解释方差比例 (默认: 0.8)')
    parser.add_argument('--max_tokens', type=int, default=None,
                       help='最大token数量，如果指定则只分析前max_tokens个token (默认: None表示使用所有token)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.correct_jsonl):
        logger.error(f"正确样本文件不存在: {args.correct_jsonl}")
        sys.exit(1)
    
    if not os.path.exists(args.incorrect_jsonl):
        logger.error(f"幻觉样本文件不存在: {args.incorrect_jsonl}")
        sys.exit(1)
    
    # 检查Parcel功能描述文件是否存在
    if args.parcel_info and not os.path.exists(args.parcel_info):
        logger.warning(f"Parcel功能描述文件不存在: {args.parcel_info}")
        args.parcel_info = None
    
    # 创建分析器并运行分析
    analyzer = ParcelLevelAnalyzer(
        correct_jsonl_path=args.correct_jsonl,
        incorrect_jsonl_path=args.incorrect_jsonl,
        output_dir=args.out_dir,
        parcel_info_path=args.parcel_info,
        epsilon=args.epsilon,
        significance_threshold=args.significance_threshold,
        skip_existing=args.skip_existing,
        top_k_edges=args.top_k_edges,
        anomalous_select_mode=args.anomalous_select_mode,
        use_structural_mask=args.use_structural_mask,
        structural_matrix_path=args.structural_matrix_path,
        structural_threshold=args.structural_threshold,
        mask_type=args.mask_type,
        use_pca_connectivity=args.use_pca_connectivity,
        pca_explained_variance=args.pca_explained_variance,
        max_tokens=args.max_tokens
    )
    
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
