#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Representational Similarity Analysis (RSA) for Human Parcels

重构版本：逻辑清晰，模块化设计

1. 加载1-100个Human Parcel，以及LLM Parcel text embedding和Human brain激活embedding
2. 按照parcel_descriptions.json的顺序（parcel_id 1-100）组织数据
3. 根据筛选原则筛选区域，按LH/RH分成左右脑两个矩阵
4. 使用带normalization的cosine_similarity计算RSM，画热力图并存储
5. 计算左右脑分别的两个RSM矩阵的Pearson相关系数
"""

import argparse
import json
import os
import logging
import re
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr, spearmanr, ttest_1samp, norm, t
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from common import RESULT_DIR, DATA_DIR, ensure_output_dir, set_nature_style, should_skip

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def reload_font_manager() -> None:
    """重新加载字体管理器，确保可识别系统中最新安装的字体。"""
    try:
        fm.fontManager = fm.FontManager()
    except Exception as e:
        logger.warning(f"字体管理器重载失败: {e}")


# ============================================================================
# 工具函数
# ============================================================================

def corr_to_t_value(r: float, n: int) -> float:
    """
    将相关系数 r 转换为 t 统计量（双侧检验常用形式）。

    对 Pearson/Spearman（在这里均以 r 近似）使用：
        t = r * sqrt((n - 2) / (1 - r^2)),  df = n - 2

    Args:
        r: 相关系数
        n: 样本量（这里对应 vector 长度，即 n_pairs）

    Returns:
        float: t 值；若 n < 3 则抛异常；若 |r|==1 则返回 +/-inf
    """
    if n < 3:
        raise ValueError(f"n 必须 >= 3 才能计算 t 值，但得到 n={n}")
    r = float(r)
    if abs(r) >= 1.0:
        return float(np.sign(r) * np.inf)
    denom = 1.0 - r * r
    if denom <= 0:
        # 数值误差下的保护；此时 t 将非常大
        return float(np.sign(r) * np.inf)
    return float(r * np.sqrt((n - 2) / denom))


def extract_parcel_id(parcel_label: str) -> int:
    """
    从parcel标签中安全地提取ID
    
    Args:
        parcel_label: parcel标签字符串，如 'LLM_Parcel_244' 或 '7Networks_LH_Vis_1'
        
    Returns:
        int: parcel ID
        
    Raises:
        ValueError: 如果无法从标签中提取parcel_id
    """
    match = re.search(r'(\d+)$', parcel_label)
    if match:
        return int(match.group(1))
    raise ValueError(f"无法从标签中提取parcel_id: {parcel_label}")


# ============================================================================
# 步骤1: 加载数据
# ============================================================================

def load_parcel_descriptions(parcel_desc_path: Path) -> List[Dict]:
    """
    加载parcel_descriptions.json文件，按照parcel_id 1-100的顺序
    
    Args:
        parcel_desc_path: parcel_descriptions.json文件路径
        
    Returns:
        List[Dict]: 按照parcel_id 1-100排序的parcel描述列表
    """
    logger.info(f"加载parcel描述文件: {parcel_desc_path}")
    if not parcel_desc_path.exists():
        raise FileNotFoundError(f"找不到parcel描述文件: {parcel_desc_path}")
    
    with open(parcel_desc_path, 'r', encoding='utf-8') as f:
        parcel_descriptions = json.load(f)
    
    # 确保按照parcel_id排序（文件应该已经是排序的，但为了安全起见）
    parcel_descriptions = sorted(parcel_descriptions, key=lambda x: x.get('parcel_id', 9999))
    
    logger.info(f"加载了 {len(parcel_descriptions)} 个parcel描述（parcel_id 1-100）")
    return parcel_descriptions


def load_llm_parcel_embeddings(
    csv_file: Path,
    llm_parcel_json: Path,
    top_k: int,
    model_name: str,
    batch_size: int,
    device: Optional[str],
    use_kth: bool = False,
    agg_mode: str = "mean"
) -> Dict[str, np.ndarray]:
    """
    加载LLM Parcel的text embedding
    
    从CSV文件中获取每个human parcel对应的top k个LLM parcels，
    然后计算这些LLM parcels的text embedding
    
    Args:
        csv_file: 包含top LLM Parcels的CSV文件
        llm_parcel_json: LLM Parcel功能描述JSON文件
        top_k: top k值
        model_name: 嵌入模型名称
        batch_size: 批处理大小
        device: 设备（cuda/cpu）
        use_kth: 是否使用k-th模式
        
    Returns:
        Dict[str, np.ndarray]: {human_parcel_name: embedding_vector}
            每个human parcel对应一个聚合后的embedding（mean或max）
    """
    logger.info("=" * 60)
    logger.info("步骤1.1: 加载LLM Parcel text embeddings")
    logger.info("=" * 60)
    
    # 加载CSV文件
    logger.info(f"加载CSV文件: {csv_file}")
    df = pd.read_csv(csv_file)
    required_cols = {'human_parcel_name', 'llm_parcel', 'semantic_similarity'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"CSV文件缺少必要列: {sorted(missing_cols)}。"
            f"当前列: {df.columns.tolist()}。"
            "请确保CSV包含 human_parcel_name / llm_parcel / semantic_similarity。"
        )
    # 按 human_parcel_name 分组后，根据 semantic_similarity 从大到小选择（而非 rank_by_acc）
    df = df.sort_values(['human_parcel_name', 'semantic_similarity'], ascending=[True, False])
    
    # 提取每个human parcel对应的LLM parcel IDs 及对应的 semantic_similarity（用于加权聚合）
    human_parcel_llm_parcels = {}
    human_parcel_llm_weights = {}
    all_llm_parcel_ids = set()
    
    for human_parcel_name in df['human_parcel_name'].unique():
        name_df = df[df['human_parcel_name'] == human_parcel_name]

        # name_df 已按 semantic_similarity 降序排好
        if use_kth:
            # 取 semantic_similarity 排序后的第 k 名（1-indexed）
            if top_k <= 0:
                raise ValueError(f"--top_k 必须为正整数，但得到: {top_k}")
            if len(name_df) < top_k:
                logger.warning(
                    f"{human_parcel_name} 只有 {len(name_df)} 行，无法取第 {top_k} 名（按semantic_similarity排序），跳过"
                )
                continue
            selected_df = name_df.iloc[[top_k - 1]]
        else:
            selected_df = name_df.head(top_k)

        llm_parcel_ids = []
        llm_weights = []
        for _, row in selected_df.iterrows():
            label = row['llm_parcel']
            sim = float(row['semantic_similarity'])
            try:
                parcel_id = extract_parcel_id(label)
                llm_parcel_ids.append(parcel_id)
                llm_weights.append(sim)
            except ValueError as e:
                logger.warning(f"跳过无效的LLM Parcel标签 '{label}': {e}")
        
        if len(llm_parcel_ids) == 0:
            logger.warning(f"{human_parcel_name} 没有有效的LLM Parcel IDs，跳过")
            continue
        
        human_parcel_llm_parcels[human_parcel_name] = llm_parcel_ids
        human_parcel_llm_weights[human_parcel_name] = llm_weights
        all_llm_parcel_ids.update(llm_parcel_ids)
    
    logger.info(f"找到 {len(human_parcel_llm_parcels)} 个human parcels")
    logger.info(f"需要计算 {len(all_llm_parcel_ids)} 个唯一的LLM Parcel embeddings")
    
    # 加载LLM Parcel描述
    logger.info(f"加载LLM Parcel描述: {llm_parcel_json}")
    with open(llm_parcel_json, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    
    llm_parcel_descriptions = {}
    for parcel_id in all_llm_parcel_ids:
        parcel_id_str = str(parcel_id)
        if parcel_id_str in llm_data:
            item = llm_data[parcel_id_str]
            if 'functionality_description' in item:
                llm_parcel_descriptions[parcel_id] = item['functionality_description']
    
    logger.info(f"成功加载 {len(llm_parcel_descriptions)} 个LLM Parcel描述")
    
    # 计算LLM Parcel embeddings
    logger.info(f"初始化嵌入模型: {model_name}")
    embedding_computer = EmbeddingComputer(model_name=model_name, device=device)
    embedding_computer.load_model()
    
    # 按parcel_id排序计算embeddings
    sorted_ids = sorted(llm_parcel_descriptions.keys())
    texts = [llm_parcel_descriptions[pid] for pid in sorted_ids]
    embeddings = embedding_computer.compute_embeddings(texts, batch_size=batch_size)
    
    # 构建parcel_id -> embedding的映射
    llm_parcel_embeddings = {pid: embeddings[i] for i, pid in enumerate(sorted_ids)}
    
    # 为每个human parcel聚合对应的LLM parcel embeddings
    human_parcel_embeddings = {}
    for human_parcel_name, llm_parcel_ids in human_parcel_llm_parcels.items():
        weights_for_name = human_parcel_llm_weights.get(human_parcel_name, [])

        valid_embeddings = []
        valid_weights = []
        for pid, w in zip(llm_parcel_ids, weights_for_name):
            if pid in llm_parcel_embeddings:
                valid_embeddings.append(llm_parcel_embeddings[pid])
                valid_weights.append(w)

        if not valid_embeddings:
            logger.warning(f"{human_parcel_name} 没有有效的LLM Parcel embeddings")
            continue

        emb_mat = np.stack(valid_embeddings, axis=0)

        if agg_mode == "sim_weighted" and len(valid_weights) > 0:
            w = np.array(valid_weights, dtype=np.float32)
            # 最大最小归一化权重，再归一化为概率；若退化则回退到简单平均
            w_min, w_max = float(w.min()), float(w.max())
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)
            w = np.clip(w, 0.0, None)
            if float(w.sum()) == 0.0:
                logger.warning(f"{human_parcel_name} 的加权系数全为0，回退为简单平均聚合")
                human_parcel_embeddings[human_parcel_name] = emb_mat.mean(axis=0)
            else:
                w = w / w.sum()
                human_parcel_embeddings[human_parcel_name] = np.sum(emb_mat * w[:, None], axis=0)
        else:
            # 默认：简单平均
            human_parcel_embeddings[human_parcel_name] = emb_mat.mean(axis=0)
    
    logger.info(f"计算了 {len(human_parcel_embeddings)} 个human parcel的LLM text embeddings")
    return human_parcel_embeddings


def load_brain_activation_embeddings(
    story_name: str,
    data_dir: Optional[Path] = None
) -> np.ndarray:
    """
    加载Human brain激活数据
    
    Args:
        story_name: story名称
        data_dir: 数据目录（data4draw），如果为None则使用默认的DATA_DIR
        
    Returns:
        np.ndarray: 形状为 (n_stimuli, 100) 的激活强度数据
    """
    logger.info("=" * 60)
    logger.info("步骤1.2: 加载Human brain激活数据")
    logger.info("=" * 60)
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    story_data_path = data_dir / story_name / f"{story_name}.npy"
    if not story_data_path.exists():
        raise FileNotFoundError(f"找不到story parcel数据文件: {story_data_path}")
    
    logger.info(f"加载激活数据: {story_data_path}")
    parcel_data = np.load(story_data_path)
    logger.info(f"激活数据形状: {parcel_data.shape}")
    
    if parcel_data.shape[1] != 100:
        raise ValueError(f"期望100个parcels，但得到{parcel_data.shape[1]}个")
    
    return parcel_data


def load_cognition_terms_embeddings(
    csv_path: Path
) -> Dict[str, np.ndarray]:
    """
    加载Human Parcel的cognition terms表征向量
    
    Args:
        csv_path: ns_scale100.csv文件路径
        
    Returns:
        Dict[str, np.ndarray]: {parcel_name: cognition_terms_vector}
    """
    logger.info("=" * 60)
    logger.info("步骤1.3: 加载Human Parcel的cognition terms表征向量")
    logger.info("=" * 60)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到cognition terms文件: {csv_path}")
    
    logger.info(f"加载cognition terms数据: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)  # 第一列作为索引（parcel名称）
    
    logger.info(f"CSV文件形状: {df.shape}")
    logger.info(f"前5个parcels: {df.index[:5].tolist()}")
    logger.info(f"前5个cognition terms: {df.columns[:5].tolist()}")
    
    # 转换为字典，每个parcel对应一个向量
    cognition_embeddings = {}
    for parcel_name in df.index:
        vector = df.loc[parcel_name].values.astype(np.float32)
        cognition_embeddings[parcel_name] = vector
    
    logger.info(f"成功加载 {len(cognition_embeddings)} 个parcels的cognition terms向量")
    logger.info(f"每个向量的维度: {len(df.columns)}")
    
    return cognition_embeddings


def load_function_description_embeddings(
    parcel_descriptions: List[Dict],
    model_name: str,
    batch_size: int,
    device: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    加载Human Parcel的function_description文本并计算embedding
    
    Args:
        parcel_descriptions: parcel描述列表（包含function_description字段）
        model_name: 嵌入模型名称
        batch_size: 批处理大小
        device: 设备（cuda/cpu）
        
    Returns:
        Dict[str, np.ndarray]: {parcel_name: embedding_vector}
    """
    logger.info("=" * 60)
    logger.info("步骤1.4: 加载Human Parcel的function_description embeddings")
    logger.info("=" * 60)
    
    # 提取function_description文本
    parcel_texts = {}
    texts_list = []
    parcel_names_list = []
    
    for item in parcel_descriptions:
        parcel_name = item.get('parcel_name', '')
        function_desc = item.get('function_description', '')
        
        if parcel_name and function_desc:
            parcel_texts[parcel_name] = function_desc
            texts_list.append(function_desc)
            parcel_names_list.append(parcel_name)
    
    if len(texts_list) == 0:
        raise ValueError("没有找到任何parcel的function_description")
    
    logger.info(f"找到 {len(texts_list)} 个parcels的function_description")
    
    # 使用EmbeddingComputer计算embeddings
    embedding_computer = EmbeddingComputer(model_name=model_name, device=device)
    embedding_computer.load_model()
    
    embeddings_array = embedding_computer.compute_embeddings(texts_list, batch_size=batch_size)
    
    # 转换为字典
    function_desc_embeddings = {}
    for i, parcel_name in enumerate(parcel_names_list):
        function_desc_embeddings[parcel_name] = embeddings_array[i]
    
    logger.info(f"成功计算 {len(function_desc_embeddings)} 个parcels的function_description embeddings")
    logger.info(f"每个embedding的维度: {embeddings_array.shape[1]}")
    
    return function_desc_embeddings


# ============================================================================
# 步骤2: 按照parcel_descriptions.json的顺序组织数据
# ============================================================================

def organize_parcels_by_description_order(
    parcel_descriptions: List[Dict],
    human_parcel_llm_embeddings: Dict[str, np.ndarray],
    filter_networks: Optional[List[str]] = None
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    按照parcel_descriptions.json的顺序组织parcels，并根据网络类型过滤
        
        Args:
        parcel_descriptions: parcel描述列表（已按parcel_id 1-100排序）
        human_parcel_llm_embeddings: {human_parcel_name: embedding_vector}
        filter_networks: 要保留的网络列表，如果为None则不过滤
            
        Returns:
        Tuple[List[str], Dict[str, np.ndarray]]: 
            (排序后的parcel名称列表, {parcel_name: embedding})
    """
    logger.info("=" * 60)
    logger.info("步骤2: 按照parcel_descriptions.json的顺序组织数据")
    logger.info("=" * 60)
    
    # 按照parcel_descriptions.json的顺序提取parcel名称
    ordered_parcel_names = []
    ordered_embeddings = {}
    
    for item in parcel_descriptions:
        parcel_name = item.get('parcel_name', '')
        if not parcel_name:
            continue
        
        # 根据网络类型过滤
        if filter_networks is not None:
            parts = parcel_name.split('_')
            if len(parts) >= 3:
                network = parts[2]
                if network not in filter_networks:
                    continue
        
        # 只保留有embedding的parcels
        if parcel_name in human_parcel_llm_embeddings:
            ordered_parcel_names.append(parcel_name)
            ordered_embeddings[parcel_name] = human_parcel_llm_embeddings[parcel_name]
    
    logger.info(f"按照描述文件顺序组织后，共有 {len(ordered_parcel_names)} 个parcels")
    logger.info(f"前5个parcels: {ordered_parcel_names[:5]}")
    
    return ordered_parcel_names, ordered_embeddings


# ============================================================================
# 步骤3: 按半球分离
# ============================================================================

def separate_by_hemisphere(parcel_names: List[str]) -> Dict[str, List[str]]:
    """
    按半球分离parcels
    
    Args:
        parcel_names: parcel名称列表
        
    Returns:
        Dict[str, List[str]]: {'LH': [LH_parcel_names], 'RH': [RH_parcel_names]}
    """
    logger.info("=" * 60)
    logger.info("步骤3: 按半球分离parcels")
    logger.info("=" * 60)
    
    lh_parcels = []
    rh_parcels = []
    
    for name in parcel_names:
        parts = name.split('_')
        if len(parts) >= 2:
            hemisphere = parts[1]
            if hemisphere == 'LH':
                lh_parcels.append(name)
            elif hemisphere == 'RH':
                rh_parcels.append(name)
    
    logger.info(f"分离结果: LH={len(lh_parcels)} 个parcels, RH={len(rh_parcels)} 个parcels")
    return {'LH': lh_parcels, 'RH': rh_parcels}


# ============================================================================
# 步骤4: 计算RSM
# ============================================================================

def compute_rsm_with_normalized_cosine(
    embeddings: Dict[str, np.ndarray],
    parcel_names: List[str]
) -> pd.DataFrame:
    """
    使用带normalization的cosine similarity计算RSM
    
    Args:
        embeddings: {parcel_name: embedding_vector}
        parcel_names: parcel名称列表（按顺序）
        
    Returns:
        pd.DataFrame: RSM矩阵，行列都是parcel_names
    """
    logger.info(f"计算RSM，parcels数量: {len(parcel_names)}")
    
    # 提取embeddings矩阵
    embedding_matrix = np.array([embeddings[name] for name in parcel_names])
    
    # L2归一化
    embedding_matrix_normalized = normalize(embedding_matrix, axis=1, norm='l2')
    
    # 计算余弦相似度（归一化后的点积）
    rsm = cosine_similarity(embedding_matrix_normalized)
    
    # 创建DataFrame
    rsm_df = pd.DataFrame(rsm, index=parcel_names, columns=parcel_names)
    
    logger.info(f"RSM形状: {rsm_df.shape}, 值范围: [{rsm_df.values.min():.4f}, {rsm_df.values.max():.4f}]")
    return rsm_df


def compute_activation_rsm_with_normalized_cosine(
    activation_data: np.ndarray,
    parcel_names: List[str],
    hemisphere: str,
    parcel_descriptions: List[Dict],
    similarity_method: str = 'cosine'
) -> pd.DataFrame:
    """
    从激活数据计算RSM（使用cosine similarity或Pearson correlation）
    
    Args:
        activation_data: 形状为 (n_stimuli, 100) 的激活数据
        parcel_names: parcel名称列表（按顺序，已经是该半球的parcels）
        hemisphere: 半球标识（'LH'或'RH'）
        parcel_descriptions: parcel描述列表（用于建立parcel_name到parcel_id的映射）
        similarity_method: 相似度计算方法，'cosine' 或 'pearson'（默认: 'cosine'）
        
    Returns:
        pd.DataFrame: RSM矩阵
    """
    logger.info(f"计算{hemisphere}的激活RSM")
    
    # 建立parcel_name -> parcel_id的映射
    parcel_name_to_id = {}
    for item in parcel_descriptions:
        parcel_id = item.get('parcel_id')
        parcel_name = item.get('parcel_name', '')
        if parcel_id is not None and parcel_name:
            parcel_name_to_id[parcel_name] = parcel_id
    
    # 根据半球选择数据
    if hemisphere == 'LH':
        # 左脑：列索引0-49对应parcel_id 1-50
        hemisphere_data = activation_data[:, 0:50]
    elif hemisphere == 'RH':
        # 右脑：列索引50-99对应parcel_id 51-100
        hemisphere_data = activation_data[:, 50:100]
    else:
        raise ValueError(f"不支持的半球: {hemisphere}")
    
    logger.info(f"{hemisphere}激活数据形状: {hemisphere_data.shape}")
    
    # 按照parcel_names的顺序映射列索引
    # parcel_names已经按照parcel_id排序，并且已经是该半球的parcels
    ordered_indices = []
    ordered_names = []
    
    for name in parcel_names:
        if name not in parcel_name_to_id:
            logger.warning(f"找不到{name}对应的parcel_id，跳过")
            continue
        
        parcel_id = parcel_name_to_id[name]
        
        # 计算对应的列索引
        if hemisphere == 'LH':
            # parcel_id 1-50 -> 列索引 0-49
            if 1 <= parcel_id <= 50:
                col_idx = parcel_id - 1
            else:
                continue
        else:  # RH
            # parcel_id 51-100 -> 列索引 50-99
            if 51 <= parcel_id <= 100:
                col_idx = parcel_id - 51  # 在hemisphere_data中的索引（0-49）
            else:
                continue
        
        if 0 <= col_idx < hemisphere_data.shape[1]:
            ordered_indices.append(col_idx)
            ordered_names.append(name)
    
    if len(ordered_names) == 0:
        raise ValueError(f"没有找到{hemisphere}对应的parcels")
    
    logger.info(f"成功映射{len(ordered_names)}个{hemisphere} parcels")
    
    # 提取对应的列
    hemisphere_data_ordered = hemisphere_data[:, ordered_indices]
    
    # 根据相似度方法计算RSM
    if similarity_method == 'pearson':
        # 使用Pearson相关性（转置后计算列之间的相关性）
        # 每一列是一个parcel在不同stimuli上的激活模式
        rsm = np.corrcoef(hemisphere_data_ordered.T)
        logger.info(f"使用Pearson相关性计算{hemisphere}激活RSM")
    elif similarity_method == 'cosine':
        # L2归一化（对每一列，即每个parcel的激活模式）
        hemisphere_data_normalized = normalize(hemisphere_data_ordered, axis=0, norm='l2')
        # 计算余弦相似度（转置后计算列之间的相似度）
        rsm = cosine_similarity(hemisphere_data_normalized.T)
        logger.info(f"使用余弦相似度计算{hemisphere}激活RSM")
    else:
        raise ValueError(f"不支持的相似度方法: {similarity_method}，请使用 'cosine' 或 'pearson'")
    
    # 创建DataFrame
    rsm_df = pd.DataFrame(rsm, index=ordered_names, columns=ordered_names)
    
    logger.info(f"{hemisphere}激活RSM形状: {rsm_df.shape}")
    return rsm_df


def compute_cognition_terms_rsm(
    cognition_embeddings: Dict[str, np.ndarray],
    parcel_names: List[str],
    similarity_method: str = 'cosine'
) -> pd.DataFrame:
    """
    基于cognition terms向量计算RSM
    
    Args:
        cognition_embeddings: {parcel_name: cognition_terms_vector}
        parcel_names: parcel名称列表（按顺序，已经是该半球的parcels）
        similarity_method: 相似度计算方法，'cosine' 或 'pearson'（默认: 'cosine'）
        
    Returns:
        pd.DataFrame: RSM矩阵
    """
    logger.info(f"计算基于cognition terms的RSM，parcels数量: {len(parcel_names)}")
    
    # 提取embeddings矩阵
    embedding_matrix = np.array([cognition_embeddings[name] for name in parcel_names])
    
    logger.info(f"Cognition terms向量矩阵形状: {embedding_matrix.shape}")
    logger.info(f"数据值范围: [{embedding_matrix.min():.4f}, {embedding_matrix.max():.4f}]")
    logger.info(f"数据包含负值: {np.any(embedding_matrix < 0)}")
    
    # 根据相似度方法计算RSM
    if similarity_method == 'pearson':
        # 使用Pearson相关性
        rsm = np.corrcoef(embedding_matrix)
        logger.info(f"使用Pearson相关性计算cognition terms RSM")
    elif similarity_method == 'cosine':
        # L2归一化
        embedding_matrix_normalized = normalize(embedding_matrix, axis=1, norm='l2')
        # 计算余弦相似度
        rsm = cosine_similarity(embedding_matrix_normalized)
        logger.info(f"使用余弦相似度计算cognition terms RSM")
    else:
        raise ValueError(f"不支持的相似度方法: {similarity_method}，请使用 'cosine' 或 'pearson'")
    
    # 创建DataFrame
    rsm_df = pd.DataFrame(rsm, index=parcel_names, columns=parcel_names)
    
    logger.info(f"Cognition terms RSM形状: {rsm_df.shape}, 值范围: [{rsm_df.values.min():.4f}, {rsm_df.values.max():.4f}]")
    return rsm_df


def compute_function_description_rsm(
    function_desc_embeddings: Dict[str, np.ndarray],
    parcel_names: List[str],
    similarity_method: str = 'cosine'
) -> pd.DataFrame:
    """
    基于function_description embedding计算RSM
    
    Args:
        function_desc_embeddings: {parcel_name: embedding_vector}
        parcel_names: parcel名称列表（按顺序，已经是该半球的parcels）
        similarity_method: 相似度计算方法，'cosine' 或 'pearson'（默认: 'cosine'）
        
    Returns:
        pd.DataFrame: RSM矩阵
    """
    logger.info(f"计算基于function_description的RSM，parcels数量: {len(parcel_names)}")
    
    # 提取embeddings矩阵
    embedding_matrix = np.array([function_desc_embeddings[name] for name in parcel_names])
    
    logger.info(f"Function description embedding矩阵形状: {embedding_matrix.shape}")
    
    # 根据相似度方法计算RSM
    if similarity_method == 'pearson':
        # 使用Pearson相关性
        rsm = np.corrcoef(embedding_matrix)
        logger.info(f"使用Pearson相关性计算function_description RSM")
    elif similarity_method == 'cosine':
        # L2归一化
        embedding_matrix_normalized = normalize(embedding_matrix, axis=1, norm='l2')
        # 计算余弦相似度
        rsm = cosine_similarity(embedding_matrix_normalized)
        logger.info(f"使用余弦相似度计算function_description RSM")
    else:
        raise ValueError(f"不支持的相似度方法: {similarity_method}，请使用 'cosine' 或 'pearson'")
    
    # 创建DataFrame
    rsm_df = pd.DataFrame(rsm, index=parcel_names, columns=parcel_names)
    
    logger.info(f"Function description RSM形状: {rsm_df.shape}, 值范围: [{rsm_df.values.min():.4f}, {rsm_df.values.max():.4f}]")
    return rsm_df


# ============================================================================
# 步骤5: 绘制热力图
# ============================================================================

def plot_rsa_heatmap(
    similarity_df: pd.DataFrame,
    output_path: Path,
    overwrite: bool,
    dpi: int = 300,
    fig_width: float = 12.0,
    fig_height: float = 12.0,
    hemisphere: str = None,
    split_by_network: bool = False,
    network_order: Optional[List[str]] = None,
    triangle: str = "none",
    highlight_special_regions: bool = False
):
    """
    绘制RSA热力图
    
    Args:
        similarity_df: 相似度矩阵DataFrame
        output_path: 输出文件路径
        overwrite: 是否覆盖已存在的文件
        dpi: 图片分辨率
        fig_width: 图片宽度（英寸）
        fig_height: 图片高度（英寸）
        hemisphere: 半球标识（用于标题）
    """
    if should_skip(output_path, overwrite):
        return
    
    set_nature_style()
    
    parcel_names = similarity_df.index.tolist()
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_white_red_reference",
        [
            (0.00, "#3B4CC0"),
            (0.35, "#8FB0F2"),
            (0.50, "#F7F7F7"),
            (0.70, "#F2B8A0"),
            (1.00, "#B40426"),
        ],
        N=256,
    )
    try:
        custom_cmap.set_bad("#FFFFFF")
    except Exception:
        pass

    finite_vals = similarity_df.to_numpy()
    vmin_raw = np.nanmin(finite_vals)
    vmax_raw = np.nanmax(finite_vals)
    if vmax_raw > vmin_raw:
        data = (similarity_df - vmin_raw) / (vmax_raw - vmin_raw)
    else:
        data = similarity_df.copy()
    finite_vals_norm = data.to_numpy()
    vmin = float(np.nanmin(finite_vals_norm))
    vmax = float(np.nanmax(finite_vals_norm))

    def build_mask(df: pd.DataFrame, tri: str) -> np.ndarray:
        mask = df.isna().to_numpy()
        # 按用户最新要求，所有 RSA 图统一绘制完整矩阵，不再仅显示上/下三角。
        # 因此这里忽略 triangle 参数，只根据 NaN 做遮罩。
        return mask

    def _indices_by_hemi_network(names: List[str], hemi: str, networks: List[str]) -> List[int]:
        idxs = []
        for i, nm in enumerate(names):
            parts = nm.split("_")
            if len(parts) >= 3 and parts[1] == hemi and parts[2] in networks:
                idxs.append(i)
        return idxs

    def _contiguous_segments(indices: List[int]) -> List[Tuple[int, int]]:
        if len(indices) == 0:
            return []
        segs = []
        s = indices[0]
        p = indices[0]
        for x in indices[1:]:
            if x == p + 1:
                p = x
            else:
                segs.append((s, p))
                s = x
                p = x
        segs.append((s, p))
        return segs

    def _draw_special_boxes(ax_obj, names: List[str]) -> None:
        n = len(names)
        # RH: Cont + SalVentAttn 对应行带
        rh_cont_sal_idx = _indices_by_hemi_network(names, "RH", ["Cont", "SalVentAttn"])
        for st, ed in _contiguous_segments(rh_cont_sal_idx):
            ax_obj.add_patch(Rectangle((0, st), n, ed - st + 1, fill=False, edgecolor="#DE7D82", linewidth=0.6))

        # RH: Cont 自相关子矩阵
        rh_cont_idx = _indices_by_hemi_network(names, "RH", ["Cont"])
        if len(rh_cont_idx) > 0:
            r0, r1 = min(rh_cont_idx), max(rh_cont_idx)
            ax_obj.add_patch(Rectangle((r0, r0), r1 - r0 + 1, r1 - r0 + 1, fill=False, edgecolor="#DE7D82", linewidth=0.6))

        # LH: Default 对应行带
        lh_default_idx = _indices_by_hemi_network(names, "LH", ["Default"])
        for st, ed in _contiguous_segments(lh_default_idx):
            ax_obj.add_patch(Rectangle((0, st), n, ed - st + 1, fill=False, edgecolor="#579FCA", linewidth=0.6))

        # LH: Default 自相关子矩阵
        if len(lh_default_idx) > 0:
            r0, r1 = min(lh_default_idx), max(lh_default_idx)
            ax_obj.add_patch(Rectangle((r0, r0), r1 - r0 + 1, r1 - r0 + 1, fill=False, edgecolor="#579FCA", linewidth=0.6))

    if not split_by_network:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        sns.heatmap(
            data,
            ax=ax,
            cmap=custom_cmap,
            vmin=vmin,
            vmax=vmax,
            mask=build_mask(data, triangle),
            cbar_kws={"shrink": 0.8, "label": "Representational Similarity"},
            xticklabels=parcel_names,
            yticklabels=parcel_names,
            square=True
        )
        # Nature-style axis formatting
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=5)
        ax.tick_params(axis='both', labelsize=5, width=0.6, length=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        title = "Representational Similarity Analysis (RSA) of Human Parcels"
        if hemisphere:
            title = f"Representational Similarity Analysis (RSA) of Human Parcels - {hemisphere}"
        ax.set_title(title, fontsize=5, pad=4, fontweight='bold')
        ax.set_xlabel("Human Parcels", fontsize=5, fontweight='bold')
        ax.set_ylabel("Human Parcels", fontsize=5, fontweight='bold')
        if highlight_special_regions:
            _draw_special_boxes(ax, parcel_names)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"保存图片到: {output_path}")
        return

    network_to_names: Dict[str, List[str]] = {}
    for name in parcel_names:
        net = extract_network_from_parcel_name(name)
        if net is None:
            continue
        network_to_names.setdefault(net, []).append(name)
    if network_order is None:
        network_order = list(network_to_names.keys())
    ordered_networks = [n for n in network_order if n in network_to_names]
    if len(ordered_networks) == 0:
        logger.warning("split_by_network=True 但没有可用 network，回退到整体热力图")
        plot_rsa_heatmap(
            similarity_df=similarity_df,
            output_path=output_path,
            overwrite=overwrite,
            dpi=dpi,
            fig_width=fig_width,
            fig_height=fig_height,
            hemisphere=hemisphere,
            split_by_network=False,
            network_order=network_order,
            triangle=triangle
        )
        return

    n_panels = len(ordered_networks)
    ncols = min(4, n_panels)
    nrows = math.ceil(n_panels / ncols)
    # 根据每个 network 中 parcel 数量自适应调整子图尺寸，避免矩阵和坐标轴过于拥挤
    max_parcels = max(len(names) for names in network_to_names.values())
    base_size = 1.4  # 最小宽高（inch）
    scale_per_parcel = 0.10  # 每个 parcel 额外预留的 inch
    panel_w = max(base_size, scale_per_parcel * max_parcels)
    panel_h = max(base_size, scale_per_parcel * max_parcels)
    # 用户反馈过于拥挤，这里整体把宽高再放大一倍
    panel_w *= 2.0
    panel_h *= 2.0
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        constrained_layout=True
    )
    axes_arr = np.array(axes).reshape(-1)
    cbar_added = False

    for i, network in enumerate(ordered_networks):
        ax = axes_arr[i]
        names = network_to_names[network]
        sub_data = data.loc[names, names]
        sub_mask = build_mask(sub_data, triangle)
        show_cbar = (not cbar_added) and (i == n_panels - 1)
        sns.heatmap(
            sub_data,
            ax=ax,
            cmap=custom_cmap,
            vmin=vmin,
            vmax=vmax,
            mask=sub_mask,
            cbar=show_cbar,
            cbar_kws={"shrink": 0.8, "label": "Representational Similarity"} if show_cbar else None,
            xticklabels=names,
            yticklabels=names,
            square=True
        )
        if show_cbar:
            cbar_added = True
        ax.set_title(network, fontsize=5, fontweight='bold', pad=2)
        ax.set_xlabel("Parcels", fontsize=5, fontweight='bold')
        ax.set_ylabel("Parcels", fontsize=5, fontweight='bold')
        ax.tick_params(axis='x', labelrotation=90, labelsize=5, width=0.6, length=2.0)
        ax.tick_params(axis='y', labelrotation=0, labelsize=5, width=0.6, length=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    for j in range(n_panels, len(axes_arr)):
        axes_arr[j].axis('off')

    title = "Representational Similarity Analysis (RSA) by Network"
    if hemisphere:
        title = f"Representational Similarity Analysis (RSA) by Network - {hemisphere}"
    fig.suptitle(title, fontsize=5, fontweight='bold')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"保存图片到: {output_path}")


def p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def plot_correlation_bar(
    correlation_result: Dict[str, float],
    output_path: Path,
    overwrite: bool,
    network_order: Optional[List[str]] = None,
    dpi: int = 450,
    metric: str = "pearson"
) -> None:
    """Plot network-level correlation bar chart (with significance stars).

    If JSON contains both:
      - by_network:        Pearson correlation from pairwise RSM (within-network upper triangle)
      - by_network_rowwise: row-wise profile similarity (Fisher z + t/Z)
    then draw two bars per network side by side.
    """
    if should_skip(output_path, overwrite):
        return
    set_nature_style()
    # 强制 Nature 规范：全局字体 Arial + 5pt（局部再用 fontsize=5 保证一致）
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 5

    metric = metric.lower()
    if metric not in ("pearson", "spearman"):
        raise ValueError(f"Unsupported metric for bar plot: {metric}")

    labels: List[str] = []

    by_network = correlation_result.get("by_network", {})
    by_network_rowwise = correlation_result.get("by_network_rowwise", {})

    # If no by_network, fall back to a single global bar
    if not by_network:
        labels = ["All"]
        if metric == "pearson":
            main_values = [float(correlation_result["pearson_correlation"])]
            main_pvals = [float(correlation_result["pearson_p_value"])]
        else:
            main_values = [float(correlation_result["spearman_correlation"])]
            main_pvals = [float(correlation_result["spearman_p_value"])]
        row_values = []
        row_pvals = []
        main_ci_lows = [float("nan")]
        main_ci_highs = [float("nan")]
        row_ci_lows = []
        row_ci_highs = []
        main_points = [[]]
        row_points = []
    else:
        # 按 JSON 中 by_network 的原始顺序绘制；network_order 仅用于过滤，不用于重排
        network_filter = set(network_order) if network_order is not None else None
        for network in by_network.keys():
            if network_filter is not None and network not in network_filter:
                continue
            labels.append(network)

        main_values = []
        main_pvals = []
        row_values = []
        row_pvals = []
        main_ci_lows = []
        main_ci_highs = []
        row_ci_lows = []
        row_ci_highs = []
        main_points = []
        row_points = []

        for network in labels:
            net_res = by_network[network]
            if metric == "pearson":
                main_values.append(float(net_res["pearson_correlation"]))
                main_pvals.append(float(net_res["pearson_p_value"]))
                ci = net_res.get("pearson_ci95", [float("nan"), float("nan")])
            else:
                main_values.append(float(net_res["spearman_correlation"]))
                main_pvals.append(float(net_res["spearman_p_value"]))
                ci = net_res.get("spearman_ci95", [float("nan"), float("nan")])
            main_ci_lows.append(float(ci[0]) if ci and len(ci) == 2 else float("nan"))
            main_ci_highs.append(float(ci[1]) if ci and len(ci) == 2 else float("nan"))
            parcel_points_main = net_res.get("parcel_points", [])
            if metric == "pearson":
                main_points.append([float(p["pearson_correlation"]) for p in parcel_points_main if "pearson_correlation" in p])
            else:
                main_points.append([float(p["spearman_correlation"]) for p in parcel_points_main if "spearman_correlation" in p])

            if network in by_network_rowwise:
                row_res = by_network_rowwise[network]
                # 平均相关性：优先使用 Fisher z 变换后的平均值，其次回退到 mean_row_pearson_correlation
                row_mean = row_res.get("fisher_z_mean", row_res.get("mean_row_pearson_correlation", None))
                if row_mean is None:
                    logger.warning(f"row-wise 结果中缺少平均相关性字段 (fisher_z_mean / mean_row_pearson_correlation): network={network}，跳过该网络的行级柱")
                    row_values.append(float("nan"))
                    row_pvals.append(float("nan"))
                    row_ci_lows.append(float("nan"))
                    row_ci_highs.append(float("nan"))
                    row_points.append([])
                    continue
                row_values.append(float(row_mean))
                # 显著性：优先使用 Fisher z 一元 t 检验的 p 值，其次回退到 mean_row_pearson_p_value
                p_row = row_res.get("fisher_z_t_p_value", row_res.get("mean_row_pearson_p_value", None))
                if p_row is None:
                    logger.warning(f"row-wise 结果中缺少 p 值字段 (fisher_z_t_p_value / mean_row_pearson_p_value): network={network}，跳过该网络的行级柱")
                    row_pvals.append(float("nan"))
                else:
                    row_pvals.append(float(p_row))
                row_ci = row_res.get("fisher_z_ci95", [float("nan"), float("nan")])
                row_ci_lows.append(float(row_ci[0]) if row_ci and len(row_ci) == 2 else float("nan"))
                row_ci_highs.append(float(row_ci[1]) if row_ci and len(row_ci) == 2 else float("nan"))
                parcel_points_row = row_res.get("parcel_points", [])
                row_points.append([float(p["fisher_z"]) for p in parcel_points_row if "fisher_z" in p])
            else:
                row_values.append(float("nan"))
                row_pvals.append(float("nan"))
                row_ci_lows.append(float("nan"))
                row_ci_highs.append(float("nan"))
                row_points.append([])

    has_rowwise = len(row_values) == len(labels) and len(row_values) > 0 and np.any(np.isfinite(np.array(row_values, dtype=float)))

    if len(labels) == 0:
        raise ValueError("没有可用于柱状图的数据（labels 为空）")

    # Colors: main bar uses Nature blue, row-wise uses warm yellow
    main_color = "#579FCA"   # medium_blue
    row_color = "#F7DC7C"    # warm_yellow

    # Figure width: 1/3 of A4 width
    fig_w = (210.0 / 25.4) / 3.0  # ≈ 2.76 inch
    # Increase height to better accommodate point clouds and significance stars
    fig_h = max(2.8, 0.28 * len(labels) + 1.9)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    x_positions = np.arange(len(labels), dtype=float)
    if has_rowwise:
        width = 0.35
        main_pos = x_positions - width / 2.0
        row_pos = x_positions + width / 2.0
        main_yerr = np.vstack([
            np.maximum(np.array(main_values) - np.array(main_ci_lows), 0.0),
            np.maximum(np.array(main_ci_highs) - np.array(main_values), 0.0)
        ])
        main_bars = ax.bar(
            main_pos,
            main_values,
            width=width,
            color=main_color,
            edgecolor="black",
            linewidth=0.4,
            label="Within-network structure similarity (pairwise RSM)",
        )
        row_yerr = np.vstack([
            np.maximum(np.array(row_values) - np.array(row_ci_lows), 0.0),
            np.maximum(np.array(row_ci_highs) - np.array(row_values), 0.0)
        ])
        row_bars = ax.bar(
            row_pos,
            row_values,
            width=width,
            color=row_color,
            edgecolor="black",
            linewidth=0.4,
            label="Node-wise whole-brain profile similarity (row-wise)",
        )
    else:
        width = 0.5
        main_pos = x_positions
        main_yerr = np.vstack([
            np.maximum(np.array(main_values) - np.array(main_ci_lows), 0.0),
            np.maximum(np.array(main_ci_highs) - np.array(main_values), 0.0)
        ])
        main_bars = ax.bar(
            main_pos,
            main_values,
            width=width,
            color=main_color,
            edgecolor="black",
            linewidth=0.4,
            label="Within-network structure similarity (pairwise RSM)",
        )
        row_bars = None

    ax.axhline(0.0, color='black', linewidth=0.5)

    # ------------------------------------------------------------------
    # Y 轴范围：
    #   - 如果所有柱子为正，则从 0 开始到一个稍大于最大值的上界；
    #   - 若存在负值，则围绕 0 对称，限制在 [-1, 1] 内。
    # ------------------------------------------------------------------
    data_vals = list(main_values)
    if has_rowwise:
        data_vals += list(row_values)
    data_vals.append(0.0)
    data_arr = np.asarray(data_vals, dtype=float)
    # 若全部为 NaN，则回退到 [-0.1, 0.1]
    if not np.any(np.isfinite(data_arr)):
        y_min, y_max = -0.1, 0.1
    else:
        finite = data_arr[np.isfinite(data_arr)]
        if finite.size == 0:
            y_min, y_max = -0.1, 0.1
        else:
            min_val = float(np.min(finite))
            max_val = float(np.max(finite))
            if min_val >= 0.0:
                # 柱状图全为正：从 0 开始，略高于最大值，且不超过 1
                upper = min(max_val + 0.05, 1.0)
                # 保证有一定高度
                if upper < 0.1:
                    upper = 0.1
                y_min, y_max = 0.0, upper
            else:
                # 存在负值：围绕 0 对称，限制在 [-1, 1]
                max_abs = float(np.max(np.abs(finite)))
                max_abs = max(max_abs, 0.1)
                max_abs = min(max_abs + 0.02, 1.0)
                y_min, y_max = -max_abs, max_abs
    ax.set_ylim(y_min, y_max)

    # 给两组柱子分别加显著性星号
    for bar, value, p_val in zip(main_bars, main_values, main_pvals):
        stars = p_to_stars(p_val)
        if stars:
            x = bar.get_x() + bar.get_width() / 2
            y = value + 0.02 if value >= 0 else value - 0.04
            va = 'bottom' if value >= 0 else 'top'
            ax.text(x, y, stars, ha='center', va=va, fontsize=5, fontweight='bold')

    if has_rowwise and row_bars is not None:
        for bar, value, p_val in zip(row_bars, row_values, row_pvals):
            stars = p_to_stars(p_val)
            if stars:
                x = bar.get_x() + bar.get_width() / 2
                y = value + 0.02 if value >= 0 else value - 0.04
                va = 'bottom' if value >= 0 else 'top'
                ax.text(x, y, stars, ha='center', va=va, fontsize=5, fontweight='bold')

    # 当前版本不再在图中叠加 parcel-level 散点，仅保留聚合后的bar和显著性标记。

    # Axes text use 5pt as required（Nature-style axis）
    if metric == "pearson":
        ax.set_ylabel("Pearson correlation (r)", fontsize=5, fontweight='bold')
    else:
        ax.set_ylabel("Spearman correlation (ρ)", fontsize=5, fontweight='bold')
    ax.set_xlabel("Functional network", fontsize=5, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=5, rotation=30, ha='right')
    # 刻度线宽统一 0.6，长度设置为 2.0，确保有清晰刻度线；关闭上/右坐标轴，符合 nature-axis-style 规则
    from matplotlib.ticker import MultipleLocator
    # Y 轴使用 0.1 为间隔的主刻度，增加刻度线密度以便阅读
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis='x', labelsize=5, width=0.6, length=2.0, direction='out')
    ax.tick_params(axis='y', labelsize=5, width=0.6, length=2.0, direction='out')
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.6)

    title_metric = "Pearson" if metric == "pearson" else "Spearman"
    ax.set_title(f"Network-level representational similarity ({title_metric})", fontsize=5, fontweight='bold', pad=4)
    ax.legend(fontsize=5, frameon=False)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"相关性柱状图已保存到: {output_path}")


def plot_by_network_rowwise_strips(
    correlation_result: Dict,
    output_path: Path,
    overwrite: bool,
    network_order: Optional[List[str]] = None,
    dpi: int = 450,
    value_key: str = "fisher_z",
    center: float = 0.0,
) -> None:
    """
    可视化 correlation_result["by_network_rowwise"]：
    每个 network 输出为“横向一行”的 strip heatmap（network 为行，parcel 为列）。

    Args:
        correlation_result: 相关性结果 JSON（dict），需包含 by_network_rowwise
        output_path: 输出 SVG 路径
        overwrite: 是否覆盖
        network_order: network 过滤/顺序（仅用于过滤；顺序跟 JSON 保持一致）
        value_key: parcel_points 中取值字段，默认 fisher_z
        center: 热图中心值（默认 0），用于对称色条
    """
    if should_skip(output_path, overwrite):
        return
    set_nature_style()

    by_network_rowwise = correlation_result.get("by_network_rowwise", {})
    if not by_network_rowwise:
        logger.warning("by_network_rowwise 为空，跳过 rowwise strips 绘图")
        return

    # network 顺序：保持 JSON 原始顺序；network_order 仅用于过滤
    network_filter = set(network_order) if network_order is not None else None
    networks = []
    for net in by_network_rowwise.keys():
        if network_filter is not None and net not in network_filter:
            continue
        networks.append(net)
    if len(networks) == 0:
        logger.warning("by_network_rowwise 过滤后无 network，跳过 rowwise strips 绘图")
        return

    # 构造 networks x parcels 的矩阵（每个 network 的 parcel_points 只包含该 network 内的 parcels）
    net_to_parcels: Dict[str, List[str]] = {}
    net_to_values: Dict[str, List[float]] = {}
    all_parcels: List[str] = []
    for net in networks:
        pts = by_network_rowwise.get(net, {}).get("parcel_points", [])
        parcels = [p.get("parcel_name") for p in pts if p.get("parcel_name") is not None]
        vals = [float(p.get(value_key)) for p in pts if p.get("parcel_name") is not None and value_key in p]
        # 对齐：只保留同时有 name 和 value 的点
        keep = []
        keep_vals = []
        for p in pts:
            nm = p.get("parcel_name", None)
            if nm is None or value_key not in p:
                continue
            keep.append(nm)
            keep_vals.append(float(p[value_key]))
        net_to_parcels[net] = keep
        net_to_values[net] = keep_vals
        all_parcels.extend(keep)

    # 统一列集合（按出现顺序去重）
    seen = set()
    cols: List[str] = []
    for nm in all_parcels:
        if nm in seen:
            continue
        seen.add(nm)
        cols.append(nm)
    if len(cols) == 0:
        logger.warning("by_network_rowwise 中未找到任何 parcel_points，跳过 rowwise strips 绘图")
        return

    mat = np.full((len(networks), len(cols)), np.nan, dtype=np.float64)
    for i, net in enumerate(networks):
        parcels = net_to_parcels.get(net, [])
        vals = net_to_values.get(net, [])
        for nm, v in zip(parcels, vals):
            if nm in seen:
                j = cols.index(nm)
                mat[i, j] = float(v)

    df = pd.DataFrame(mat, index=networks, columns=cols)

    # 对称色条范围（围绕 center）
    finite = df.to_numpy()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        v = 1.0
    else:
        v = float(np.max(np.abs(finite - center)))
        if v <= 0:
            v = 1.0
    vmin = center - v
    vmax = center + v

    # 自适应宽度：每个 parcel 给 0.10 inch，再加上边距
    fig_w = max(6.0, 0.10 * len(cols) + 2.5)
    # 高度：每个 network 一行，给足够空间写标签
    fig_h = max(1.6, 0.35 * len(networks) + 1.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_white_red_reference",
        [
            (0.00, "#3B4CC0"),
            (0.35, "#8FB0F2"),
            (0.50, "#F7F7F7"),
            (0.70, "#F2B8A0"),
            (1.00, "#B40426"),
        ],
        N=256,
    )
    try:
        custom_cmap.set_bad("#FFFFFF")
    except Exception:
        pass
    sns.heatmap(
        df,
        ax=ax,
        cmap=custom_cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": value_key},
        xticklabels=True,
        yticklabels=True,
        linewidths=0.0,
        linecolor="white",
    )
    ax.set_xlabel("Parcels", fontsize=5, fontweight="bold")
    ax.set_ylabel("Network", fontsize=5, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=90, labelsize=5, width=0.6, length=2.0)
    ax.tick_params(axis="y", labelrotation=0, labelsize=5, width=0.6, length=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    ax.set_title("by_network_rowwise strip heatmap", fontsize=5, fontweight="bold", pad=4)

    # 同步保存一个 CSV，方便后续排版/复用
    csv_path = output_path.with_suffix(".csv")
    try:
        df.to_csv(csv_path)
        logger.info(f"rowwise strips 数据已保存到: {csv_path}")
    except Exception as e:
        logger.warning(f"保存 rowwise strips CSV 失败: {e}")

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"rowwise strips SVG 已保存到: {output_path}")


def plot_rsm_network_rows(
    rsm: pd.DataFrame,
    output_path: Path,
    overwrite: bool,
    networks: Optional[List[str]] = None,
    dpi: int = 450,
    title: Optional[str] = None,
) -> None:
    """
    从完整 RSM 矩阵中提取指定 network 的所有 parcel 行（每个 parcel 一行），
    列保持为完整的 RSM 列顺序，用于可视化“某个 network 的所有节点在全半球上的 profile”。

    Args:
        rsm: RSM 矩阵（行列同为 parcel_name）
        output_path: 输出 SVG 文件
        overwrite: 是否覆盖
        networks: 要包含的 network 列表；None 表示使用全部 network
        dpi: 分辨率
        title: 可选标题；None 时使用默认
    """
    if should_skip(output_path, overwrite):
        return
    set_nature_style()

    # 根据 network 顺序构造行顺序：先按 network，再按原始 index 顺序
    row_names: List[str] = []
    if networks is None:
        # 所有 network，按出现顺序
        for name in rsm.index:
            row_names.append(name)
    else:
        for net in networks:
            for name in rsm.index:
                if extract_network_from_parcel_name(name) == net:
                    row_names.append(name)

    if len(row_names) == 0:
        logger.warning(f"plot_rsm_network_rows: 指定的 networks 在 RSM 中没有任何行，跳过: {networks}")
        return

    sub = rsm.loc[row_names, rsm.columns]

    vals = sub.to_numpy(dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1e-8

    # 自适应尺寸：保证 cell 为正方形（长宽一致），fig 宽高比约等于 n_cols/n_rows
    n_rows, n_cols = sub.shape
    # 每个 cell 的物理尺寸（inch）；适中即可，避免图过大
    cell = 0.12
    fig_w = max(4.0, cell * n_cols + 1.5)
    fig_h = max(2.0, cell * n_rows + 1.2)

    from matplotlib.colors import LinearSegmentedColormap

    custom_cmap = LinearSegmentedColormap.from_list(
        "blue_white_red_reference",
        [
            (0.00, "#3B4CC0"),
            (0.35, "#8FB0F2"),
            (0.50, "#F7F7F7"),
            (0.70, "#F2B8A0"),
            (1.00, "#B40426"),
        ],
        N=256,
    )
    try:
        custom_cmap.set_bad("#FFFFFF")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    sns.heatmap(
        sub,
        ax=ax,
        cmap=custom_cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        cbar_kws={"shrink": 0.8, "label": "Representational Similarity"},
        xticklabels=sub.columns.tolist(),
        yticklabels=row_names,
        square=True,
    )
    # 再显式设置一次 aspect，确保导出 SVG 时 cell 为正方形
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=5)
    ax.tick_params(axis='both', labelsize=5, width=0.6, length=2.0)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_linewidth(0.6)

    if title is None:
        title = "Network parcel rows from full RSM"
    ax.set_title(title, fontsize=5, fontweight='bold', pad=4)
    ax.set_xlabel("Parcels", fontsize=5, fontweight='bold')
    ax.set_ylabel("Parcels (selected networks)", fontsize=5, fontweight='bold')

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"network rows RSM 已保存到: {output_path}")


# ============================================================================
# 步骤6: 计算Pearson和Spearman相关系数
# ============================================================================

def extract_network_from_parcel_name(parcel_name: str) -> Optional[str]:
    """
    从parcel名称中提取network
    
    Args:
        parcel_name: parcel名称，如 '7Networks_LH_Vis_1'
        
    Returns:
        str: network名称，如 'Vis'，如果无法提取则返回None
    """
    parts = parcel_name.split('_')
    if len(parts) >= 3:
        return parts[2]
    return None


def compute_ci95(values: np.ndarray) -> Tuple[float, float]:
    """基于 t 分布计算均值的 95% CI。"""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        v = float(arr[0])
        return (v, v)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    margin = float(t.ppf(0.975, df=arr.size - 1) * sem)
    return (mean - margin, mean + margin)


def bootstrap_corr_ci95(
    values1: np.ndarray,
    values2: np.ndarray,
    method: str,
    n_boot: int = 1000
) -> Tuple[float, float]:
    """对相关系数做 bootstrap 95% CI。"""
    x = np.asarray(values1, dtype=np.float64)
    y = np.asarray(values2, dtype=np.float64)
    if x.size != y.size or x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    n = x.size
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        try:
            if method == "pearson":
                r, _ = pearsonr(xb, yb)
            elif method == "spearman":
                r, _ = spearmanr(xb, yb)
            else:
                raise ValueError(f"不支持的相关方法: {method}")
            if np.isfinite(r):
                stats.append(float(r))
        except Exception:
            continue
    if len(stats) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def compute_rsm_correlation_by_network(
    rsm1: pd.DataFrame,
    rsm2: pd.DataFrame,
    networks: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    按network分组计算两个RSM之间的相关性
    
    Args:
        rsm1: 第一个RSM矩阵
        rsm2: 第二个RSM矩阵
        networks: 要分析的network列表，如果为None则分析所有network
        
    Returns:
        dict: {network: {pearson_correlation, pearson_p_value, pearson_t_value,
              spearman_correlation, spearman_p_value, spearman_t_value, n_pairs}}
    """
    # 对齐矩阵
    common_indices = rsm1.index.intersection(rsm2.index)
    common_cols = rsm1.columns.intersection(rsm2.columns)
    common_indices = common_indices.intersection(common_cols)
    
    if len(common_indices) == 0:
        raise ValueError("两个RSM矩阵没有共同的索引")
    
    # 对齐矩阵
    rsm1_aligned = rsm1.loc[common_indices, common_indices]
    rsm2_aligned = rsm2.loc[common_indices, common_indices]
    
    # 按network分组
    network_parcels = {}
    for parcel_name in common_indices:
        network = extract_network_from_parcel_name(parcel_name)
        if network is None:
            continue
        if networks is not None and network not in networks:
            continue
        if network not in network_parcels:
            network_parcels[network] = []
        network_parcels[network].append(parcel_name)
    
    results = {}
    for network, parcel_list in network_parcels.items():
        if len(parcel_list) < 2:
            logger.warning(f"Network {network} 只有 {len(parcel_list)} 个parcels，跳过")
            continue
        
        # 提取该network的子矩阵
        network_rsm1 = rsm1_aligned.loc[parcel_list, parcel_list]
        network_rsm2 = rsm2_aligned.loc[parcel_list, parcel_list]
        
        # 提取上三角矩阵（不包括对角线）
        n = len(network_rsm1)
        upper_tri_indices = np.triu_indices(n, k=1)
        
        values1 = network_rsm1.values[upper_tri_indices]
        values2 = network_rsm2.values[upper_tri_indices]
        
        # 排除NaN值
        valid_mask = ~(np.isnan(values1) | np.isnan(values2))
        values1 = values1[valid_mask]
        values2 = values2[valid_mask]
        
        # 检查是否有足够的数据点（至少需要2个）
        if len(values1) < 2:
            logger.warning(f"Network {network} 只有 {len(values1)} 个有效数据点（需要至少2个），跳过")
            continue
        
        # 计算Pearson correlation
        pearson_corr, pearson_p = pearsonr(values1, values2)
        
        # 计算Spearman correlation
        spearman_corr, spearman_p = spearmanr(values1, values2)

        n_pairs = int(len(values1))
        pearson_t = corr_to_t_value(float(pearson_corr), n_pairs)
        spearman_t = corr_to_t_value(float(spearman_corr), n_pairs)
        
        pearson_ci_low, pearson_ci_high = bootstrap_corr_ci95(values1, values2, method="pearson", n_boot=1000)
        spearman_ci_low, spearman_ci_high = bootstrap_corr_ci95(values1, values2, method="spearman", n_boot=1000)

        # 计算 parcel-level 点：每个 parcel 在该 network 内（去除自身）的一行 profile 相关
        parcel_points = []
        if n >= 3:
            for parcel_name in parcel_list:
                row1 = network_rsm1.loc[parcel_name, parcel_list].to_numpy(dtype=np.float64)
                row2 = network_rsm2.loc[parcel_name, parcel_list].to_numpy(dtype=np.float64)
                self_idx = parcel_list.index(parcel_name)
                row1 = np.delete(row1, self_idx)
                row2 = np.delete(row2, self_idx)
                valid = ~(np.isnan(row1) | np.isnan(row2))
                row1 = row1[valid]
                row2 = row2[valid]
                if len(row1) < 2:
                    continue
                try:
                    p_r, _ = pearsonr(row1, row2)
                    s_r, _ = spearmanr(row1, row2)
                    parcel_points.append({
                        "parcel_name": parcel_name,
                        "pearson_correlation": float(p_r),
                        "spearman_correlation": float(s_r),
                    })
                except Exception:
                    continue

        parcel_pearsons = np.array([p["pearson_correlation"] for p in parcel_points], dtype=np.float64) if parcel_points else np.array([])
        parcel_spearmans = np.array([p["spearman_correlation"] for p in parcel_points], dtype=np.float64) if parcel_points else np.array([])
        parcel_pearson_ci = compute_ci95(parcel_pearsons) if parcel_points else (float("nan"), float("nan"))
        parcel_spearman_ci = compute_ci95(parcel_spearmans) if parcel_points else (float("nan"), float("nan"))

        results[network] = {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'pearson_t_value': float(pearson_t),
            'pearson_ci95': [float(pearson_ci_low), float(pearson_ci_high)],
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'spearman_t_value': float(spearman_t),
            'spearman_ci95': [float(spearman_ci_low), float(spearman_ci_high)],
            'n_pairs': n_pairs,
            'n_parcels': int(n),
            'parcel_points': parcel_points,
            'parcel_pearson_ci95': [float(parcel_pearson_ci[0]), float(parcel_pearson_ci[1])],
            'parcel_spearman_ci95': [float(parcel_spearman_ci[0]), float(parcel_spearman_ci[1])]
        }
        
        logger.info(f"Network {network}: Pearson={pearson_corr:.4f}, Spearman={spearman_corr:.4f}, n_parcels={n}")
    
    return results


def compute_rowwise_network_similarity(
    rsm1: pd.DataFrame,
    rsm2: pd.DataFrame,
    networks: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    按 network 计算“行级”相似性：
    对每个 network 内的每个 parcel，取其在 RSM 中的整行向量（保持原始列顺序），
    分别计算 rsm1-row vs rsm2-row 的 Pearson / Spearman，再在 network 内做平均。

    Args:
        rsm1: 第一个 RSM 矩阵
        rsm2: 第二个 RSM 矩阵
        networks: 要分析的 network 列表；None 表示全部

    Returns:
        dict: {network: {
            mean_row_pearson_correlation, mean_row_pearson_p_value,
            mean_row_spearman_correlation, mean_row_spearman_p_value, n_rows
        }}
    """
    common_indices = rsm1.index.intersection(rsm2.index)
    common_cols = rsm1.columns.intersection(rsm2.columns)
    common_names = common_indices.intersection(common_cols)
    if len(common_names) == 0:
        raise ValueError("两个RSM矩阵没有共同的索引/列，无法计算行级network相似性")

    rsm1_aligned = rsm1.loc[common_names, common_names]
    rsm2_aligned = rsm2.loc[common_names, common_names]

    network_parcels: Dict[str, List[str]] = {}
    for parcel_name in common_names:
        network = extract_network_from_parcel_name(parcel_name)
        if network is None:
            continue
        if networks is not None and network not in networks:
            continue
        network_parcels.setdefault(network, []).append(parcel_name)

    results: Dict[str, Dict[str, float]] = {}
    for network, parcel_list in network_parcels.items():
        if len(parcel_list) == 0:
            continue

        row_pearsons: List[float] = []
        row_pearson_ps: List[float] = []
        row_spearmans: List[float] = []
        row_spearman_ps: List[float] = []
        parcel_points: List[Dict[str, float]] = []

        for parcel_name in parcel_list:
            row1 = rsm1_aligned.loc[parcel_name, :].to_numpy(dtype=np.float64)
            row2 = rsm2_aligned.loc[parcel_name, :].to_numpy(dtype=np.float64)
            valid_mask = ~(np.isnan(row1) | np.isnan(row2))
            row1 = row1[valid_mask]
            row2 = row2[valid_mask]

            if len(row1) < 2:
                logger.warning(f"{network}/{parcel_name} 有效数据点不足(<2)，跳过该行")
                continue

            pearson_corr, pearson_p = pearsonr(row1, row2)
            spearman_corr, spearman_p = spearmanr(row1, row2)

            row_pearsons.append(float(pearson_corr))
            row_pearson_ps.append(float(pearson_p))
            row_spearmans.append(float(spearman_corr))
            row_spearman_ps.append(float(spearman_p))
            parcel_points.append({
                "parcel_name": parcel_name,
                "pearson_correlation": float(pearson_corr),
                "spearman_correlation": float(spearman_corr),
                "fisher_z": float(np.arctanh(np.clip(float(pearson_corr), -1.0 + 1e-7, 1.0 - 1e-7)))
            })

        if len(row_pearsons) == 0:
            logger.warning(f"Network {network} 没有可用的行级相关结果，跳过")
            continue

        # Fisher z 变换 + 一元 t 检验 / 近似 Z 检验
        # 只对 Pearson 相关做 Fisher z（更标准），Spearman 可以视为近似 Pearson时同理，但这里只对 row_pearsons 处理
        r_arr = np.asarray(row_pearsons, dtype=np.float64)
        # 防止 |r|=1 导致 arctanh 溢出
        eps = 1e-7
        r_arr_clipped = np.clip(r_arr, -1.0 + eps, 1.0 - eps)
        z_arr = np.arctanh(r_arr_clipped)
        # H0: mean(z) = 0
        t_stat, p_t = ttest_1samp(z_arr, popmean=0.0, alternative="two-sided")
        # 近似 Z：df 大时 t ~ N(0,1)
        df = max(len(z_arr) - 1, 1)
        z_stat = float(t_stat)  # 直接把 t 视作近似 Z
        p_z = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        fisher_z_ci = compute_ci95(z_arr)

        results[network] = {
            "mean_row_pearson_correlation": float(np.mean(row_pearsons)),
            "mean_row_pearson_p_value": float(np.mean(row_pearson_ps)),
            "mean_row_spearman_correlation": float(np.mean(row_spearmans)),
            "mean_row_spearman_p_value": float(np.mean(row_spearman_ps)),
            "n_rows": int(len(row_pearsons)),
            # Fisher z + t / Z 检验结果（整体显著性）
            "fisher_z_mean": float(np.mean(z_arr)),
            "fisher_z_ci95": [float(fisher_z_ci[0]), float(fisher_z_ci[1])],
            "fisher_z_t_value": float(t_stat),
            "fisher_z_t_p_value": float(p_t),
            "fisher_z_z_value": float(z_stat),
            "fisher_z_z_p_value": float(p_z),
            "fisher_z_df": int(df),
            "parcel_points": parcel_points,
        }

        logger.info(
            f"[Rowwise] Network {network}: "
            f"mean Pearson={results[network]['mean_row_pearson_correlation']:.4f}, "
            f"mean Spearman={results[network]['mean_row_spearman_correlation']:.4f}, "
            f"n_rows={results[network]['n_rows']}"
        )

    return results


def compute_rsm_correlation(
    rsm1: pd.DataFrame, 
    rsm2: pd.DataFrame,
    exclude_cross_hemisphere: bool = False,
    lh_parcels: Optional[List[str]] = None,
    rh_parcels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算两个RSM之间的Pearson和Spearman correlation
    
    Args:
        rsm1: 第一个RSM矩阵
        rsm2: 第二个RSM矩阵
        exclude_cross_hemisphere: 如果为True，则排除左右脑之间的数据点
        lh_parcels: 左脑parcels列表（仅在exclude_cross_hemisphere=True时使用）
        rh_parcels: 右脑parcels列表（仅在exclude_cross_hemisphere=True时使用）
        
    Returns:
        dict: 包含 pearson_correlation / pearson_p_value / pearson_t_value，
              spearman_correlation / spearman_p_value / spearman_t_value，以及 n_pairs
    """
    # 对齐矩阵
    common_indices = rsm1.index.intersection(rsm2.index)
    common_cols = rsm1.columns.intersection(rsm2.columns)
    common_indices = common_indices.intersection(common_cols)
    
    if len(common_indices) == 0:
        raise ValueError("两个RSM矩阵没有共同的索引")
    
    logger.info(f"对齐后的共同parcels数量: {len(common_indices)}")
    
    # 对齐矩阵
    rsm1_aligned = rsm1.loc[common_indices, common_indices]
    rsm2_aligned = rsm2.loc[common_indices, common_indices]
    
    # 提取上三角矩阵（不包括对角线）
    n = len(rsm1_aligned)
    upper_tri_indices = np.triu_indices(n, k=1)
    
    values1 = rsm1_aligned.values[upper_tri_indices]
    values2 = rsm2_aligned.values[upper_tri_indices]
    
    # 如果排除左右脑之间的数据点
    if exclude_cross_hemisphere and lh_parcels is not None and rh_parcels is not None:
        # 创建掩码：只保留左脑内部和右脑内部的数据点（排除NaN值）
        row_indices, col_indices = upper_tri_indices
        mask = np.ones(len(row_indices), dtype=bool)
        
        for i, (row_idx, col_idx) in enumerate(zip(row_indices, col_indices)):
            row_parcel = common_indices[row_idx]
            col_parcel = common_indices[col_idx]
            # 排除左右脑之间的数据点（一个在左脑，一个在右脑）
            if (row_parcel in lh_parcels and col_parcel in rh_parcels) or \
               (row_parcel in rh_parcels and col_parcel in lh_parcels):
                mask[i] = False
        
        values1 = values1[mask]
        values2 = values2[mask]
        logger.info(f"排除左右脑之间的数据点后，剩余 {len(values1)} 个数据点")
    
    # 排除NaN值
    valid_mask = ~(np.isnan(values1) | np.isnan(values2))
    values1 = values1[valid_mask]
    values2 = values2[valid_mask]
    
    if len(values1) < 2:
        raise ValueError(f"有效数据点数量不足（{len(values1)} < 2），无法计算相关性")
    
    # 计算Pearson correlation
    pearson_corr, pearson_p = pearsonr(values1, values2)
    
    # 计算Spearman correlation
    spearman_corr, spearman_p = spearmanr(values1, values2)

    n_pairs = int(len(values1))
    pearson_t = corr_to_t_value(float(pearson_corr), n_pairs)
    spearman_t = corr_to_t_value(float(spearman_corr), n_pairs)
    
    logger.info(f"RSM之间的Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    logger.info(f"RSM之间的Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    
    # 转换为Python原生类型以确保JSON序列化
    return {
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'pearson_t_value': float(pearson_t),
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'spearman_t_value': float(spearman_t),
        'n_pairs': n_pairs
    }


# ============================================================================
# 步骤6: 基于 RSM 的聚类 / 块结构识别
# ============================================================================

def cluster_rsm(
    rsm: pd.DataFrame,
    method: str = "ward",
    metric: str = "euclidean",
    n_clusters: int = 6
) -> Dict[str, Dict]:
    """
    对 RSM 做层次聚类，帮助识别块结构。

    Args:
        rsm: 相似度矩阵（行列相同、对称）
        method: linkage 方法（如 'ward', 'average', 'complete'）
        metric: 距离度量，默认用 (1 - similarity) 转为距离
        n_clusters: 聚类个数

    Returns:
        dict，包含：
            - order: 重新排序后的 parcel 名称列表
            - labels: {parcel_name: cluster_id}
    """
    names = list(rsm.index)
    sim_mat = rsm.values.astype(np.float64)

    # 将相似度转为距离（1 - sim），并确保非负
    dist_mat = 1.0 - sim_mat
    dist_mat = np.clip(dist_mat, 0.0, None)

    # squareform 只接受上三角
    dist_condensed = squareform(dist_mat, checks=False)
    Z = linkage(dist_condensed, method=method)

    # 生成叶子顺序和聚类标签
    leaf_order = leaves_list(Z)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    ordered_names = [names[i] for i in leaf_order]
    label_dict = {names[i]: int(labels[i]) for i in range(len(names))}
    return {"order": ordered_names, "labels": label_dict}


# ============================================================================
# EmbeddingComputer类
# ============================================================================

class EmbeddingComputer:
    """使用Qwen3-8B-embedding模型计算文本嵌入"""
    
    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-8B', device=None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # 设置设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"GPU available, using: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                logger.warning("GPU not available, using CPU")
        else:
            self.device = torch.device(device)
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """加载模型"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def compute_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量计算文本嵌入
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            numpy array: (n_texts, embedding_dim)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Computing embeddings for {len(texts)} texts...")
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # 按照官方推荐方式：tokenize → last token pooling → L2 归一化
            batch_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            )
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                last_hidden_state = outputs.last_hidden_state  # [B, T, D]
                attention_mask = batch_inputs.get("attention_mask", None)

                if attention_mask is None:
                    # 若没有 attention_mask，则退化为使用最后一个 token
                    logger.warning("attention_mask 未找到，退化为使用最后一个 token 做 pooling")
                    token_embeddings = last_hidden_state[:, -1, :]
                else:
                    # 官方推荐：使用最后一个非 padding token 作为句向量
                    # attention_mask: [B, T]，有效 token 为 1，padding 为 0
                    token_counts = attention_mask.sum(dim=1)  # [B]
                    # 防止出现全 0 的情况（极端异常输入），为 0 时用第一个 token 兜底
                    if torch.any(token_counts == 0):
                        logger.warning("存在 attention_mask 全 0 的样本，使用第一个 token 作为兜底 pooling")
                    last_indices = torch.clamp(token_counts - 1, min=0)  # [B]
                    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                    token_embeddings = last_hidden_state[batch_indices, last_indices, :]  # [B, D]

                # 在 PyTorch 中先做 L2 归一化（与官方示例一致）
                token_embeddings = F.normalize(token_embeddings, p=2, dim=1)
                batch_embeddings = token_embeddings.cpu().numpy()

            embeddings.append(batch_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")

        result = np.vstack(embeddings)
        logger.info(f"Embeddings shape: {result.shape}")
        return result


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='计算并绘制Human Parcel之间的Representational Similarity Analysis'
    )
    parser.add_argument(
        '--csv_file',
        type=Path,
        required=True,
        help='包含top LLM Parcels的CSV文件路径'
    )
    parser.add_argument(
        '--llm_parcel_json',
        type=Path,
        required=True,
        help='LLM Parcel功能描述JSON文件路径'
    )
    parser.add_argument(
        '--parcel_desc_json',
        type=Path,
        default=Path("/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"),
        help='Human Parcel描述JSON文件路径'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=1,
        help='每个human_parcel_name选取的k值（默认: 1）'
    )
    parser.add_argument(
        '--use_kth',
        action='store_true',
        help='如果指定，则使用k-th模式（选择第k个），否则使用topk模式（选择前k个）'
    )
    parser.add_argument(
        '--agg_mode',
        type=str,
        default='mean',
        choices=['mean', 'sim_weighted'],
        help='对同一human parcel的多个LLM parcels进行聚合的方式：mean（简单平均，默认）或 sim_weighted（按semantic_similarity加权）'
    )
    parser.add_argument(
        '--cluster_rsa',
        action='store_true',
        help='如果指定，则对全脑 RSM 进行层次聚类，输出聚类顺序和块结构 heatmap'
    )
    parser.add_argument(
        '--cluster_n',
        type=int,
        default=6,
        help='层次聚类划分的簇数（默认: 6）'
    )
    parser.add_argument(
        '--cluster_method',
        type=str,
        default='ward',
        choices=['ward', 'average', 'complete', 'single'],
        help='层次聚类 linkage 方法（默认: ward）'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen3-Embedding-8B',
        help='嵌入模型名称（默认: Qwen/Qwen3-Embedding-8B）'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='批处理大小（默认: 32）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='设备（cuda或cpu，默认: None自动检测）'
    )
    parser.add_argument(
        '--output_file',
        type=Path,
        default=None,
        help='输出目录路径（默认: draw_result/rsa_files）'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='覆盖已存在的输出文件'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='图片分辨率（默认: 300）'
    )
    parser.add_argument(
        '--fig_width',
        type=float,
        default=12.0,
        help='图片宽度（英寸，默认: 12.0）'
    )
    parser.add_argument(
        '--fig_height',
        type=float,
        default=12.0,
        help='图片高度（英寸，默认: 12.0）'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='如果结果文件已存在则跳过（默认: False）'
    )
    parser.add_argument(
        '--filter_networks',
        type=str,
        nargs='+',
        default=None,
        help='要保留的网络列表（默认: Default SalVentAttn Limbic Cont）。设置为空列表则不过滤'
    )
    parser.add_argument(
        '--compare_activation',
        action='store_true',
        help='如果指定，则计算基于激活强度的RSM并与embedding RSM比较（已废弃，建议使用--compare_cognition_terms）'
    )
    parser.add_argument(
        '--story_name',
        type=str,
        default=None,
        help='Story名称（用于加载激活数据，仅在--compare_activation时使用）'
    )
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=None,
        help='数据目录（data4draw），如果为None则使用默认的DATA_DIR'
    )
    parser.add_argument(
        '--similarity_method',
        type=str,
        default='cosine',
        choices=['cosine', 'pearson'],
        help='相似度计算方法：cosine（余弦相似度，默认）或pearson（Pearson相关性）'
    )
    parser.add_argument(
        '--compare_cognition_terms',
        action='store_true',
        help='如果指定，则计算基于cognition terms的RSM并与embedding RSM比较'
    )
    parser.add_argument(
        '--cognition_terms_csv',
        type=Path,
        default=None,
        help='Cognition terms CSV文件路径（ns_scale100.csv），仅在--compare_cognition_terms时使用'
    )
    parser.add_argument(
        '--compare_function_description',
        action='store_true',
        help='如果指定，则计算基于function_description的RSM并与embedding RSM比较'
    )
    parser.add_argument(
        '--split_by_network_plots',
        action='store_true',
        help='如果指定，则将非cluster的RSA图按network拆分为多个子图'
    )
    parser.add_argument(
        '--plot_correlation_bar',
        action='store_true',
        help='如果指定，则为相关性JSON额外绘制spearman柱状图并标注显著性'
    )
    parser.add_argument(
        '--plot_only',
        action='store_true',
        help='如果指定，则基于已有的 RSM / 相关性 JSON 直接重新绘图，而不重新计算嵌入和RSM'
    )
    
    args = parser.parse_args()
    reload_font_manager()
    
    # 设置输出路径
    ensure_output_dir()
    if args.output_file is None:
        prefix = f"kth{args.top_k}" if args.use_kth else f"top{args.top_k}"
        output_dir = RESULT_DIR / "rsa_files"
    else:
        output_dir = args.output_file
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"kth{args.top_k}" if args.use_kth else f"top{args.top_k}"
    
    # ========================================================================
    # plot_only 模式：仅基于已有结果文件重新绘图，不做任何重新计算
    # ========================================================================
    if args.plot_only:
        logger.info("开启 plot_only 模式：基于已有 RSM / 相关性 JSON 重新绘图，不重新计算。")

        # 1) 重新绘制 LLM / Function description RSM（LH / RH）
        for hemi in ['LH', 'RH']:
            llm_csv = output_dir / f"rsa_llm_embedding_{hemi}_{prefix}.csv"
            if llm_csv.exists():
                logger.info(f"[plot_only] 重新绘制 {hemi} LLM RSM 热图，使用已有 CSV: {llm_csv}")
                llm_df = pd.read_csv(llm_csv, index_col=0)
                llm_svg = output_dir / f"rsa_llm_embedding_{hemi}_{prefix}.svg"
                plot_rsa_heatmap(
                    llm_df,
                    llm_svg,
                    args.overwrite,
                    dpi=args.dpi,
                    fig_width=args.fig_width,
                    fig_height=args.fig_height,
                    hemisphere=f"{hemi} (LLM Embedding-based)",
                    split_by_network=args.split_by_network_plots,
                    network_order=args.filter_networks,
                    triangle="upper" if args.split_by_network_plots else "none"
                )

            func_csv = output_dir / f"rsa_function_description_{hemi}_{args.similarity_method}.csv"
            if func_csv.exists():
                logger.info(f"[plot_only] 重新绘制 {hemi} Function description RSM 热图，使用已有 CSV: {func_csv}")
                func_df = pd.read_csv(func_csv, index_col=0)
                func_svg = output_dir / f"rsa_function_description_{hemi}_{args.similarity_method}.svg"
                plot_rsa_heatmap(
                    func_df,
                    func_svg,
                    args.overwrite,
                    dpi=args.dpi,
                    fig_width=args.fig_width,
                    fig_height=args.fig_height,
                    hemisphere=f"{hemi} (Function Description-based)",
                    # 与 LLM 图保持一致：在需要时也可以按 network 拆成多个矩阵块
                    split_by_network=args.split_by_network_plots,
                    network_order=args.filter_networks,
                    triangle="lower"
                )

        # 全脑 RSM
        global_llm_csv = output_dir / f"rsa_llm_embedding_all_{prefix}.csv"
        if global_llm_csv.exists():
            logger.info(f"[plot_only] 重新绘制全脑 LLM RSM 热图，使用已有 CSV: {global_llm_csv}")
            global_llm_df = pd.read_csv(global_llm_csv, index_col=0)
            global_llm_svg = output_dir / f"rsa_llm_embedding_all_{prefix}.svg"
            plot_rsa_heatmap(
                global_llm_df,
                global_llm_svg,
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere="All Parcels (LLM Embedding-based)",
                # 全脑 LLM 图在 plot_only 模式下也绘制完整矩阵
                split_by_network=False,
                network_order=args.filter_networks,
                triangle="upper",
                highlight_special_regions=True
            )

        global_func_csv = output_dir / f"rsa_function_description_all_{args.similarity_method}.csv"
        if global_func_csv.exists():
            logger.info(f"[plot_only] 重新绘制全脑 Function description RSM 热图，使用已有 CSV: {global_func_csv}")
            global_func_df = pd.read_csv(global_func_csv, index_col=0)
            global_func_svg = output_dir / f"rsa_function_description_all_{args.similarity_method}.svg"
            plot_rsa_heatmap(
                global_func_df,
                global_func_svg,
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere="All Parcels (Function Description-based)",
                split_by_network=False,
                network_order=args.filter_networks,
                triangle="lower",
                highlight_special_regions=True
            )

        # 2) 基于已有相关性 JSON 重新绘制柱状图（仅在需要时）
        if args.plot_correlation_bar:
            for hemi in ['LH', 'RH', 'all']:
                if hemi == 'all':
                    corr_json = output_dir / f"rsa_correlation_function_desc_all_{prefix}.json"
                    bar_svg_pearson = output_dir / f"rsa_correlation_function_desc_all_{prefix}_pearson_bar.svg"
                    bar_svg_spearman = output_dir / f"rsa_correlation_function_desc_all_{prefix}_spearman_bar.svg"
                else:
                    corr_json = output_dir / f"rsa_correlation_function_desc_{hemi}_{prefix}.json"
                    bar_svg_pearson = output_dir / f"rsa_correlation_function_desc_{hemi}_{prefix}_pearson_bar.svg"
                    bar_svg_spearman = output_dir / f"rsa_correlation_function_desc_{hemi}_{prefix}_spearman_bar.svg"

                if corr_json.exists():
                    logger.info(f"[plot_only] 重新绘制 {hemi} 相关性柱状图，使用已有 JSON: {corr_json}")
                    with open(corr_json, 'r', encoding='utf-8') as f:
                        corr_result = json.load(f)
                    # Pearson version
                    plot_correlation_bar(
                        corr_result,
                        bar_svg_pearson,
                        args.overwrite,
                        network_order=args.filter_networks,
                        metric="pearson"
                    )
                    # Spearman version
                    plot_correlation_bar(
                        corr_result,
                        bar_svg_spearman,
                        args.overwrite,
                        network_order=args.filter_networks,
                        metric="spearman"
                    )
                    # by_network_rowwise strip heatmap（network 一行）
                    rowwise_svg = output_dir / f"rsa_by_network_rowwise_{hemi}_{prefix}.svg"
                    plot_by_network_rowwise_strips(
                        corr_result,
                        rowwise_svg,
                        args.overwrite,
                        network_order=args.filter_networks,
                        value_key="fisher_z",
                        center=0.0
                    )

        logger.info("plot_only 模式下的所有重绘已完成，程序结束。")
        return

    # ========================================================================
    # 步骤1: 加载数据
    # ========================================================================
    logger.info("=" * 60)
    logger.info("开始计算RSA")
    logger.info("=" * 60)
    
    # 1.1 加载LLM Parcel text embeddings
    human_parcel_llm_embeddings = load_llm_parcel_embeddings(
        args.csv_file, 
        args.llm_parcel_json,
        args.top_k,
        args.model_name,
        args.batch_size,
        args.device,
        use_kth=args.use_kth,
        agg_mode=args.agg_mode
    )
    
    # 1.2 加载Human brain激活数据（如果需要，已废弃）
    activation_data = None
    if args.compare_activation:
        if args.story_name is None:
            # 尝试从CSV文件路径中提取story_name
            csv_path_parts = Path(args.csv_file).parts
            if 'draw_result' in csv_path_parts:
                idx = csv_path_parts.index('draw_result')
                if idx + 1 < len(csv_path_parts):
                    args.story_name = csv_path_parts[idx + 1]
                    logger.info(f"从CSV文件路径中提取story_name: {args.story_name}")
        
        if args.story_name is None:
            raise ValueError("需要指定--story_name或确保CSV文件路径包含story名称")
        
        activation_data = load_brain_activation_embeddings(args.story_name, args.data_dir)
    
    # 1.3 加载Human Parcel的cognition terms表征向量（如果需要）
    cognition_embeddings = None
    if args.compare_cognition_terms:
        if args.cognition_terms_csv is None:
            # 使用默认路径
            default_path = Path("/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/ns_scale100.csv")
            if default_path.exists():
                args.cognition_terms_csv = default_path
                logger.info(f"使用默认cognition terms文件: {args.cognition_terms_csv}")
            else:
                raise ValueError("需要指定--cognition_terms_csv或确保默认路径存在")
        
        cognition_embeddings = load_cognition_terms_embeddings(args.cognition_terms_csv)
    
    # ========================================================================
    # 步骤2: 按照parcel_descriptions.json的顺序组织数据
    # ========================================================================
    parcel_descriptions = load_parcel_descriptions(args.parcel_desc_json)
    
    # 1.4 加载Human Parcel的function_description embeddings（如果需要）
    function_desc_embeddings = None
    if args.compare_function_description:
        function_desc_embeddings = load_function_description_embeddings(
            parcel_descriptions,
            args.model_name,
            args.batch_size,
            args.device
        )
    
    ordered_parcel_names, ordered_llm_embeddings = organize_parcels_by_description_order(
        parcel_descriptions,
        human_parcel_llm_embeddings,
        filter_networks=args.filter_networks if args.filter_networks else None
    )
    
    # ========================================================================
    # 步骤3: 按半球分离
    # ========================================================================
    hemisphere_parcels = separate_by_hemisphere(ordered_parcel_names)
    
    # ========================================================================
    # 步骤4: 计算RSM并绘制热力图
    # ========================================================================
    logger.info("=" * 60)
    logger.info("步骤4: 计算RSM并绘制热力图")
    logger.info("=" * 60)
    
    correlation_results = {}
    
    # 存储左右脑的 RSM，用于后续合并计算
    hemisphere_rsms = {
        'LH': {'llm': None, 'cognition_terms': None, 'function_desc': None, 'activation': None},
        'RH': {'llm': None, 'cognition_terms': None, 'function_desc': None, 'activation': None}
    }
    
    for hemisphere in ['LH', 'RH']:
        if len(hemisphere_parcels[hemisphere]) == 0:
            logger.warning(f"{hemisphere} 没有parcels，跳过")
            continue
        
        logger.info(f"\n处理 {hemisphere}...")
        
        # 4.1 计算LLM text embedding RSM
        hemisphere_llm_embeddings = {name: ordered_llm_embeddings[name] 
                                  for name in hemisphere_parcels[hemisphere]}
        llm_rsm = compute_rsm_with_normalized_cosine(
            hemisphere_llm_embeddings,
            hemisphere_parcels[hemisphere]
        )
        
        # 保存到字典中，用于后续合并计算
        hemisphere_rsms[hemisphere]['llm'] = llm_rsm
        
        # 保存LLM RSM
        llm_rsm_csv = output_dir / f"rsa_llm_embedding_{hemisphere}_{prefix}.csv"
        llm_rsm.to_csv(llm_rsm_csv)
        logger.info(f"{hemisphere} LLM embedding RSM已保存到: {llm_rsm_csv}")
        
        # 绘制LLM RSM热力图
        llm_rsm_svg = output_dir / f"rsa_llm_embedding_{hemisphere}_{prefix}.svg"
        plot_rsa_heatmap(
            llm_rsm,
            llm_rsm_svg,
            args.overwrite,
            dpi=args.dpi,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            hemisphere=f"{hemisphere} (LLM Embedding-based)",
            split_by_network=args.split_by_network_plots,
            network_order=args.filter_networks,
            triangle="upper" if args.split_by_network_plots else "none"
        )

        # 如指定，对当前半球的 RSM 也做层次聚类，输出半球级别的块结构
        if args.cluster_rsa:
            logger.info(f"\n对 {hemisphere} RSM 进行层次聚类以识别块结构 ...")
            hemi_cluster_info = cluster_rsm(
                llm_rsm,
                method=args.cluster_method,
                metric="euclidean",
                n_clusters=args.cluster_n
            )
            hemi_order = hemi_cluster_info["order"]
            hemi_labels = hemi_cluster_info["labels"]

            llm_rsm_clustered_hemi = llm_rsm.loc[hemi_order, hemi_order]

            hemi_clustered_csv = output_dir / f"rsa_llm_embedding_{hemisphere}_{prefix}_clustered.csv"
            llm_rsm_clustered_hemi.to_csv(hemi_clustered_csv)
            logger.info(f"{hemisphere} LLM embedding RSM（聚类排序）已保存到: {hemi_clustered_csv}")

            # 保存半球聚类标签（仅该半球的 parcels）
            hemi_cluster_json = output_dir / f"rsa_llm_embedding_{hemisphere}_{prefix}_clusters.json"
            hemi_cluster_meta = {}
            for name in llm_rsm.index:
                hemi_cluster_meta[name] = {
                    "cluster_id": int(hemi_labels.get(name, -1))
                }
            with open(hemi_cluster_json, "w", encoding="utf-8") as f:
                json.dump(hemi_cluster_meta, f, indent=2, ensure_ascii=False)
            logger.info(f"{hemisphere} RSM 聚类结果（cluster_id）已保存到: {hemi_cluster_json}")

            # 绘制半球聚类排序后的 heatmap
            hemi_clustered_svg = output_dir / f"rsa_llm_embedding_{hemisphere}_{prefix}_clustered.svg"
            plot_rsa_heatmap(
                llm_rsm_clustered_hemi,
                hemi_clustered_svg,
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere=f"{hemisphere} (Clustered, {args.cluster_method}, k={args.cluster_n})"
            )
        
        # 4.2 计算基于cognition terms的RSM（如果指定了）
        if args.compare_cognition_terms and cognition_embeddings is not None:
            # 提取该半球的cognition embeddings
            hemisphere_cognition_embeddings = {
                name: cognition_embeddings[name] 
                for name in hemisphere_parcels[hemisphere]
                if name in cognition_embeddings
            }
            
            if len(hemisphere_cognition_embeddings) == 0:
                logger.warning(f"{hemisphere} 没有找到对应的cognition terms数据，跳过")
                continue
            
            cognition_rsm = compute_cognition_terms_rsm(
                hemisphere_cognition_embeddings,
                hemisphere_parcels[hemisphere],
                similarity_method=args.similarity_method
            )
            
            # 确保两个矩阵顺序一致
            cognition_rsm = cognition_rsm.reindex(
                index=llm_rsm.index,
                columns=llm_rsm.columns
            )
            
            # 保存到字典中，用于后续合并计算
            hemisphere_rsms[hemisphere]['cognition_terms'] = cognition_rsm
            
            # 保存cognition terms RSM（不需要prefix，因为不受top_k影响，但包含相似度方法信息）
            cognition_rsm_csv = output_dir / f"rsa_cognition_terms_{hemisphere}_{args.similarity_method}.csv"
            cognition_rsm.to_csv(cognition_rsm_csv)
            logger.info(f"{hemisphere} Cognition terms RSM已保存到: {cognition_rsm_csv}")
            
            # 绘制cognition terms RSM热力图
            cognition_rsm_svg = output_dir / f"rsa_cognition_terms_{hemisphere}_{args.similarity_method}.svg"
            plot_rsa_heatmap(
                cognition_rsm,
                cognition_rsm_svg,
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere=f"{hemisphere} (Cognition Terms-based)"
            )
            
            # ========================================================================
            # 步骤5: 计算Pearson和Spearman相关系数
            # ========================================================================
            logger.info(f"\n计算{hemisphere}的两个RSM之间的相关性...")
            correlation_result = compute_rsm_correlation(llm_rsm, cognition_rsm)
            correlation_results[hemisphere] = correlation_result
            
            # 按network分组计算相关性
            if args.filter_networks:
                logger.info(f"\n按network分组计算{hemisphere}的两个RSM之间的相关性...")
                network_correlations = compute_rsm_correlation_by_network(
                    llm_rsm, cognition_rsm, networks=args.filter_networks
                )
                correlation_result['by_network'] = network_correlations
                
                # 打印按network分组的结果
                print(f"\n{hemisphere} RSM比较结果 (LLM Embedding vs Cognition Terms) - 按Network分组:")
                for network, net_result in network_correlations.items():
                    print(f"  {network}:")
                    print(f"    Pearson: r={net_result['pearson_correlation']:.4f}, p={net_result['pearson_p_value']:.4e}")
                    print(f"    Spearman: ρ={net_result['spearman_correlation']:.4f}, p={net_result['spearman_p_value']:.4e}")
                    print(f"    Parcels: {net_result['n_parcels']}, Pairs: {net_result['n_pairs']}")
            
            # 保存相关性结果
            correlation_json = output_dir / f"rsa_correlation_{hemisphere}_{prefix}.json"
            with open(correlation_json, 'w', encoding='utf-8') as f:
                json.dump(correlation_result, f, indent=2, ensure_ascii=False)
            logger.info(f"{hemisphere} 相关性结果已保存到: {correlation_json}")
            
            # 打印结果
            print(f"\n{hemisphere} RSM比较结果 (LLM Embedding vs Cognition Terms) - 整体:")
            print(f"  Pearson correlation: {correlation_result['pearson_correlation']:.4f}")
            print(f"  Pearson P-value: {correlation_result['pearson_p_value']:.4e}")
            print(f"  Spearman correlation: {correlation_result['spearman_correlation']:.4f}")
            print(f"  Spearman P-value: {correlation_result['spearman_p_value']:.4e}")
            print(f"  配对数: {correlation_result['n_pairs']}")
        
        # 4.3 计算基于function_description的RSM（如果指定了）
        if args.compare_function_description and function_desc_embeddings is not None:
            # 提取该半球的function_description embeddings
            hemisphere_function_desc_embeddings = {
                name: function_desc_embeddings[name] 
                for name in hemisphere_parcels[hemisphere]
                if name in function_desc_embeddings
            }
            
            if len(hemisphere_function_desc_embeddings) == 0:
                logger.warning(f"{hemisphere} 没有找到对应的function_description数据，跳过")
                continue
            
            function_desc_rsm = compute_function_description_rsm(
                hemisphere_function_desc_embeddings,
                hemisphere_parcels[hemisphere],
                similarity_method=args.similarity_method
            )
            
            # 确保两个矩阵顺序一致
            function_desc_rsm = function_desc_rsm.reindex(
                index=llm_rsm.index,
                columns=llm_rsm.columns
            )
            
            # 保存到字典中，用于后续合并计算
            hemisphere_rsms[hemisphere]['function_desc'] = function_desc_rsm
            
            # 保存function_description RSM（不需要prefix，因为不受top_k影响，但包含相似度方法信息）
            function_desc_rsm_csv = output_dir / f"rsa_function_description_{hemisphere}_{args.similarity_method}.csv"
            function_desc_rsm.to_csv(function_desc_rsm_csv)
            logger.info(f"{hemisphere} Function description RSM已保存到: {function_desc_rsm_csv}")
            
            # 绘制function_description RSM热力图（在需要时也可以按 network 拆分为多个矩阵块）
            function_desc_rsm_svg = output_dir / f"rsa_function_description_{hemisphere}_{args.similarity_method}.svg"
            plot_rsa_heatmap(
                function_desc_rsm,
                function_desc_rsm_svg,
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere=f"{hemisphere} (Function Description-based)",
                split_by_network=args.split_by_network_plots,
                network_order=args.filter_networks,
                triangle="lower"
            )
            # 从完整 RSM 中提取指定 networks 的所有 parcel 行（每个 parcel 一行），列为完整半球 parcels
            if args.filter_networks:
                llm_rows_svg = output_dir / f"rsa_llm_embedding_{hemisphere}_{prefix}_network_rows.svg"
                plot_rsm_network_rows(
                    llm_rsm,
                    llm_rows_svg,
                    args.overwrite,
                    networks=args.filter_networks,
                    dpi=args.dpi,
                    title=f"{hemisphere} LLM RSM rows (networks={','.join(args.filter_networks)})"
                )
                func_rows_svg = output_dir / f"rsa_function_description_{hemisphere}_{args.similarity_method}_network_rows.svg"
                plot_rsm_network_rows(
                    function_desc_rsm,
                    func_rows_svg,
                    args.overwrite,
                    networks=args.filter_networks,
                    dpi=args.dpi,
                    title=f"{hemisphere} Function-desc RSM rows (networks={','.join(args.filter_networks)})"
                )
            
            # ========================================================================
            # 步骤5: 计算Pearson和Spearman相关系数
            # ========================================================================
            logger.info(f"\n计算{hemisphere}的两个RSM之间的相关性...")
            correlation_result = compute_rsm_correlation(llm_rsm, function_desc_rsm)
            correlation_results[hemisphere] = correlation_result
            
            # 按network分组计算相关性
            if args.filter_networks:
                logger.info(f"\n按network分组计算{hemisphere}的两个RSM之间的相关性...")
                network_correlations = compute_rsm_correlation_by_network(
                    llm_rsm, function_desc_rsm, networks=args.filter_networks
                )
                correlation_result['by_network'] = network_correlations
                # 新增：按 network 的“行级整行向量”相似性（先逐行相关，再network内取平均）
                rowwise_network_similarity = compute_rowwise_network_similarity(
                    llm_rsm, function_desc_rsm, networks=args.filter_networks
                )
                correlation_result['by_network_rowwise'] = rowwise_network_similarity
                
                # 打印按network分组的结果
                print(f"\n{hemisphere} RSM比较结果 (LLM Embedding vs Function Description) - 按Network分组:")
                for network, net_result in network_correlations.items():
                    print(f"  {network}:")
                    print(f"    Pearson: r={net_result['pearson_correlation']:.4f}, p={net_result['pearson_p_value']:.4e}")
                    print(f"    Spearman: ρ={net_result['spearman_correlation']:.4f}, p={net_result['spearman_p_value']:.4e}")
                    print(f"    Parcels: {net_result['n_parcels']}, Pairs: {net_result['n_pairs']}")
                print(f"\n{hemisphere} 行级Network平均相关（整行向量）:")
                for network, net_row_result in rowwise_network_similarity.items():
                    print(f"  {network}:")
                    print(
                        "    mean(row Pearson)={:.4f}, mean(row Spearman)={:.4f}, n_rows={}".format(
                            net_row_result["mean_row_pearson_correlation"],
                            net_row_result["mean_row_spearman_correlation"],
                            net_row_result["n_rows"]
                        )
                    )
            
            # 保存相关性结果
            correlation_json = output_dir / f"rsa_correlation_function_desc_{hemisphere}_{prefix}.json"
            with open(correlation_json, 'w', encoding='utf-8') as f:
                json.dump(correlation_result, f, indent=2, ensure_ascii=False)
            logger.info(f"{hemisphere} 相关性结果已保存到: {correlation_json}")
            if args.plot_correlation_bar:
                # Pearson version（仅保留 SVG）
                bar_svg_pearson = output_dir / f"rsa_correlation_function_desc_{hemisphere}_{prefix}_pearson_bar.svg"
                plot_correlation_bar(
                    correlation_result,
                    bar_svg_pearson,
                    args.overwrite,
                    network_order=args.filter_networks,
                    metric="pearson"
                )
                # Spearman version（仅保留 SVG）
                bar_svg_spearman = output_dir / f"rsa_correlation_function_desc_{hemisphere}_{prefix}_spearman_bar.svg"
                plot_correlation_bar(
                    correlation_result,
                    bar_svg_spearman,
                    args.overwrite,
                    network_order=args.filter_networks,
                    metric="spearman"
                )
                # by_network_rowwise strip heatmap（network 一行）
                rowwise_svg = output_dir / f"rsa_by_network_rowwise_{hemisphere}_{prefix}.svg"
                plot_by_network_rowwise_strips(
                    correlation_result,
                    rowwise_svg,
                    args.overwrite,
                    network_order=args.filter_networks,
                    value_key="fisher_z",
                    center=0.0
                )
            
            # 打印结果
            print(f"\n{hemisphere} RSM比较结果 (LLM Embedding vs Function Description) - 整体:")
            print(f"  Pearson correlation: {correlation_result['pearson_correlation']:.4f}")
            print(f"  Pearson P-value: {correlation_result['pearson_p_value']:.4e}")
            print(f"  Spearman correlation: {correlation_result['spearman_correlation']:.4f}")
            print(f"  Spearman P-value: {correlation_result['spearman_p_value']:.4e}")
            print(f"  配对数: {correlation_result['n_pairs']}")
        
        # 4.4 计算激活RSM（如果指定了，已废弃，保留向后兼容）
        elif args.compare_activation and activation_data is not None:
            activation_rsm = compute_activation_rsm_with_normalized_cosine(
                activation_data,
                hemisphere_parcels[hemisphere],
                hemisphere,
                parcel_descriptions,
                similarity_method=args.similarity_method
            )
            
            # 确保两个矩阵顺序一致
            activation_rsm = activation_rsm.reindex(
                index=llm_rsm.index,
                columns=llm_rsm.columns
            )
            
            # 保存激活RSM（不需要prefix，因为不受top_k影响，但包含相似度方法信息）
            activation_rsm_csv = output_dir / f"rsa_activation_{hemisphere}_{args.similarity_method}.csv"
            activation_rsm.to_csv(activation_rsm_csv)
            logger.info(f"{hemisphere} 激活RSM已保存到: {activation_rsm_csv}")
            
            # 绘制激活RSM热力图（不需要prefix，因为不受top_k影响，但包含相似度方法信息）
            activation_rsm_svg = output_dir / f"rsa_activation_{hemisphere}_{args.similarity_method}.svg"
            plot_rsa_heatmap(
                activation_rsm,
                activation_rsm_svg,
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere=f"{hemisphere} (Activation-based)"
            )
            
            # ========================================================================
            # 步骤5: 计算Pearson和Spearman相关系数
            # ========================================================================
            logger.info(f"\n计算{hemisphere}的两个RSM之间的相关性...")
            correlation_result = compute_rsm_correlation(llm_rsm, activation_rsm)
            correlation_results[hemisphere] = correlation_result
            
            # 保存相关性结果
            correlation_json = output_dir / f"rsa_correlation_{hemisphere}_{prefix}.json"
            with open(correlation_json, 'w', encoding='utf-8') as f:
                json.dump(correlation_result, f, indent=2, ensure_ascii=False)
            logger.info(f"{hemisphere} 相关性结果已保存到: {correlation_json}")
            
            # 打印结果
            print(f"\n{hemisphere} RSM比较结果 (LLM Embedding vs Activation):")
            print(f"  Pearson correlation: {correlation_result['pearson_correlation']:.4f}")
            print(f"  Pearson P-value: {correlation_result['pearson_p_value']:.4e}")
            print(f"  Spearman correlation: {correlation_result['spearman_correlation']:.4f}")
            print(f"  Spearman P-value: {correlation_result['spearman_p_value']:.4e}")
            print(f"  配对数: {correlation_result['n_pairs']}")
    
    # ========================================================================
    # 步骤6: 计算「全脑」RSM（按 parcel_id 顺序，不再通过 NaN 拼接）
    # ========================================================================
    logger.info("=" * 60)
    logger.info("步骤6: 计算全脑 RSM（按 parcel_id 顺序）")
    logger.info("=" * 60)

    if len(ordered_parcel_names) > 0:
        logger.info("\n计算全脑 LLM embedding RSM ...")
        global_llm_rsm = compute_rsm_with_normalized_cosine(
            ordered_llm_embeddings,
            ordered_parcel_names
        )

        global_llm_rsm_csv = output_dir / f"rsa_llm_embedding_all_{prefix}.csv"
        global_llm_rsm.to_csv(global_llm_rsm_csv)
        logger.info(f"全脑 LLM embedding RSM 已保存到: {global_llm_rsm_csv}")

        global_llm_rsm_svg = output_dir / f"rsa_llm_embedding_all_{prefix}.svg"
        plot_rsa_heatmap(
            global_llm_rsm,
            global_llm_rsm_svg,
            args.overwrite,
            dpi=args.dpi,
            fig_width=args.fig_width,
            fig_height=args.fig_height,
            hemisphere="All Parcels (LLM Embedding-based)",
            highlight_special_regions=True
        )

        # 如果有 function_description 的 embedding，则计算全脑 function_description RSM 并与 LLM RSM 做整体相关
        if args.compare_function_description and function_desc_embeddings is not None:
            logger.info("\n计算全脑 Function description RSM ...")
            # 只保留在 ordered_parcel_names 中且有 function_desc 的 parcels
            global_func_embeddings = {
                name: function_desc_embeddings[name]
                for name in ordered_parcel_names
                if name in function_desc_embeddings
            }
            if len(global_func_embeddings) == 0:
                logger.warning("没有任何parcel同时拥有LLM embedding和function_description embedding，跳过全脑 Function description RSM")
            else:
                func_parcel_order = list(global_func_embeddings.keys())
                global_func_rsm = compute_function_description_rsm(
                    global_func_embeddings,
                    func_parcel_order,
                    similarity_method=args.similarity_method
                )
                # 对齐到全脑 LLM RSM 的行列顺序
                global_func_rsm = global_func_rsm.reindex(
                    index=global_llm_rsm.index,
                    columns=global_llm_rsm.columns
                )

                # 保存全脑 function_description RSM
                global_func_rsm_csv = output_dir / f"rsa_function_description_all_{args.similarity_method}.csv"
                global_func_rsm.to_csv(global_func_rsm_csv)
                logger.info(f"全脑 Function description RSM 已保存到: {global_func_rsm_csv}")

                # 绘制全脑 function_description RSM 热力图（仅 SVG）
                plot_rsa_heatmap(
                    global_func_rsm,
                    output_dir / f"rsa_function_description_all_{args.similarity_method}.svg",
                    args.overwrite,
                    dpi=args.dpi,
                    fig_width=args.fig_width,
                    fig_height=args.fig_height,
                    hemisphere="All Parcels (Function Description-based)",
                    highlight_special_regions=True
                )
                global_func_rsm_svg = output_dir / f"rsa_function_description_all_{args.similarity_method}.svg"
                plot_rsa_heatmap(
                    global_func_rsm,
                    global_func_rsm_svg,
                    args.overwrite,
                    dpi=args.dpi,
                    fig_width=args.fig_width,
                    fig_height=args.fig_height,
                    hemisphere="All Parcels (Function Description-based)",
                    highlight_special_regions=True
                )

                # 计算全脑层面的 LLM vs Function description RSM 相关性
                logger.info("\n计算全脑 LLM RSM 与 Function description RSM 的相关性 ...")
                global_corr = compute_rsm_correlation(global_llm_rsm, global_func_rsm)
                # 按 network 分组（与 LH/RH 输出保持一致）
                if args.filter_networks:
                    logger.info("\n按network分组计算全脑两个RSM之间的相关性...")
                    global_net_corr = compute_rsm_correlation_by_network(
                        global_llm_rsm, global_func_rsm, networks=args.filter_networks
                    )
                    global_corr["by_network"] = global_net_corr
                    global_rowwise_net = compute_rowwise_network_similarity(
                        global_llm_rsm, global_func_rsm, networks=args.filter_networks
                    )
                    global_corr["by_network_rowwise"] = global_rowwise_net
                global_corr_json = output_dir / f"rsa_correlation_function_desc_all_{prefix}.json"
                with open(global_corr_json, 'w', encoding='utf-8') as f:
                    json.dump(global_corr, f, indent=2, ensure_ascii=False)
                logger.info(f"全脑 LLM vs Function description RSM 相关性结果已保存到: {global_corr_json}")
                if args.plot_correlation_bar:
                    # Pearson version（仅 SVG）
                    all_bar_svg_pearson = output_dir / f"rsa_correlation_function_desc_all_{prefix}_pearson_bar.svg"
                    plot_correlation_bar(
                        global_corr,
                        all_bar_svg_pearson,
                        args.overwrite,
                        network_order=args.filter_networks,
                        metric="pearson"
                    )
                    # Spearman version（仅 SVG）
                    all_bar_svg_spearman = output_dir / f"rsa_correlation_function_desc_all_{prefix}_spearman_bar.svg"
                    plot_correlation_bar(
                        global_corr,
                        all_bar_svg_spearman,
                        args.overwrite,
                        network_order=args.filter_networks,
                        metric="spearman"
                    )
                    # by_network_rowwise strip heatmap（network 一行）
                    all_rowwise_svg = output_dir / f"rsa_by_network_rowwise_all_{prefix}.svg"
                    plot_by_network_rowwise_strips(
                        global_corr,
                        all_rowwise_svg,
                        args.overwrite,
                        network_order=args.filter_networks,
                        value_key="fisher_z",
                        center=0.0
                    )
        
        # 如指定，对全脑 RSM 做层次聚类，帮助识别块结构
        if args.cluster_rsa:
            logger.info("\n对全脑 RSM 进行层次聚类以识别块结构 ...")
            cluster_info = cluster_rsm(
                global_llm_rsm,
                method=args.cluster_method,
                metric="euclidean",
                n_clusters=args.cluster_n
            )
            clustered_order = cluster_info["order"]
            clustered_labels = cluster_info["labels"]

            # 重新排序 RSM
            global_llm_rsm_clustered = global_llm_rsm.loc[clustered_order, clustered_order]

            # 保存聚类后的 RSM
            clustered_csv = output_dir / f"rsa_llm_embedding_all_{prefix}_clustered.csv"
            global_llm_rsm_clustered.to_csv(clustered_csv)
            logger.info(f"全脑 LLM embedding RSM（聚类排序）已保存到: {clustered_csv}")

            # 保存聚类标签（包含 parcel 基本信息）
            cluster_json = output_dir / f"rsa_llm_embedding_all_{prefix}_clusters.json"
            parcel_cluster_meta = {}
            # 构造 parcel -> (id, hemisphere, network) 的映射
            parcel_meta = {}
            for item in parcel_descriptions:
                name = item.get("parcel_name", "")
                if name in global_llm_rsm.index:
                    pid = item.get("parcel_id")
                    parts = name.split("_")
                    hemi = parts[1] if len(parts) > 1 else None
                    net = parts[2] if len(parts) > 2 else None
                    parcel_meta[name] = {
                        "parcel_id": pid,
                        "hemisphere": hemi,
                        "network": net,
                    }
            for name in global_llm_rsm.index:
                parcel_cluster_meta[name] = {
                    "cluster_id": int(clustered_labels.get(name, -1)),
                    **parcel_meta.get(name, {})
                }
            with open(cluster_json, "w", encoding="utf-8") as f:
                json.dump(parcel_cluster_meta, f, indent=2, ensure_ascii=False)
            logger.info(f"全脑 RSM 聚类结果（cluster_id）已保存到: {cluster_json}")

            # 绘制聚类排序后的 heatmap（仅 SVG，可更清楚看到块结构）
            plot_rsa_heatmap(
                global_llm_rsm_clustered,
                output_dir / f"rsa_llm_embedding_all_{prefix}_clustered.svg",
                args.overwrite,
                dpi=args.dpi,
                fig_width=args.fig_width,
                fig_height=args.fig_height,
                hemisphere=f"All Parcels (Clustered, {args.cluster_method}, k={args.cluster_n})"
            )
    
    # 打印总结
    if (args.compare_cognition_terms or args.compare_function_description or args.compare_activation) and correlation_results:
        print("\n" + "=" * 60)
        print("RSM比较总结")
        print("=" * 60)
        if args.compare_function_description:
            comparison_type = "Function Description"
        elif args.compare_cognition_terms:
            comparison_type = "Cognition Terms"
        else:
            comparison_type = "Activation"
        print(f"比较类型: LLM Embedding vs {comparison_type}")
        for hemisphere, result in correlation_results.items():
            print(f"{hemisphere}:")
            print(f"  Pearson: r={result['pearson_correlation']:.4f}, p={result['pearson_p_value']:.4e}")
            print(f"  Spearman: ρ={result['spearman_correlation']:.4f}, p={result['spearman_p_value']:.4e}")
        print("=" * 60)
    
    logger.info("完成！")


if __name__ == "__main__":
    main()
