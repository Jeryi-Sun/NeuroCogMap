#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始结果文件夹构建 LLM Parcel 对 Human Parcel 的预测结果矩阵

直接从 results_* 文件夹中读取数据，构建 (H, L) 矩阵
H: Human Parcels (按 parcel_id 排序，去除第 0 和第 101 个，共 100 个，与 parcel_descriptions.json 保持一致)
L: LLM Parcels (按 entity_id 排序)
值: correlations 预测准确率（去除第 0 和第 101 个 Human Parcel）
"""

import json
import pandas as pd
import numpy as np
import argparse
import os
import pickle
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_run_data(run_dir, id_key="layer_idx"):
    """
    加载单个 run 目录的数据
    
    Args:
        run_dir: run 目录路径
        id_key: hyperparams.json 中用于标识 ID 的字段名
        
    Returns:
        包含 entity_id 和 correlations 的字典，如果加载失败返回 None
    """
    try:
        # 读取 hyperparams.json
        hyperparams_path = run_dir / "hyperparams.json"
        if not hyperparams_path.exists():
            logger.warning(f"警告: {hyperparams_path} 不存在，跳过")
            return None
            
        with open(hyperparams_path, 'r', encoding='utf-8') as f:
            hyperparams = json.load(f)
        
        entity_id = hyperparams.get(id_key)
        if entity_id is None:
            logger.warning(f"警告: {hyperparams_path} 中没有 {id_key}，跳过")
            return None
        
        # 读取 metrics.pkl
        metrics_path = run_dir / "metrics.pkl"
        if not metrics_path.exists():
            logger.warning(f"警告: {metrics_path} 不存在，跳过")
            return None
            
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
        
        correlations = metrics.get('correlations')
        if correlations is None:
            logger.warning(f"警告: {metrics_path} 中没有 correlations，跳过")
            return None
        
        # 转换为 numpy 数组
        correlations = np.asarray(correlations, dtype=np.float64)
        
        return {
            "entity_id": entity_id,
            "correlations": correlations
        }
        
    except Exception as e:
        logger.error(f"错误: 加载 {run_dir} 时出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def load_prediction_data_from_results(results_dir, id_key="layer_idx", skip_existing=False, output_file=None):
    """
    从原始结果文件夹加载预测数据
    
    Args:
        results_dir: results 目录路径（包含多个 run_* 子目录）
        id_key: hyperparams.json 中用于标识 ID 的字段名
        skip_existing: 如果输出文件已存在，是否跳过
        output_file: 输出文件路径（用于检查是否已存在）
        
    Returns:
        list: 包含所有 LLM Parcel 预测结果的列表
    """
    if skip_existing and output_file and os.path.exists(output_file):
        logger.info(f"输出文件已存在，跳过处理: {output_file}")
        return None
    
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"目录不存在: {results_dir}")
    
    logger.info(f"从原始结果文件夹加载数据: {results_dir}")
    
    # 遍历所有 run_* 目录
    run_dirs = sorted(results_dir.glob("run_*"))
    
    # 如果根目录下没有找到，则在子目录中递归查找（最多一层深度）
    if not run_dirs:
        run_dirs = sorted(results_dir.glob("*/run_*"))
        if run_dirs:
            logger.info(f"在子目录中找到 {len(run_dirs)} 个 run 目录")
    
    if not run_dirs:
        raise ValueError(f"在 {results_dir} 中未找到任何 run_* 目录")
    
    logger.info(f"找到 {len(run_dirs)} 个 run 目录")
    
    # 存储每个 entity 的所有结果（可能有多个 run 对应同一个 entity）
    entity_results = {}
    
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        
        data = load_run_data(run_dir, id_key)
        if data is None:
            continue
        
        entity_id = data["entity_id"]
        correlations = data["correlations"]
        
        # 检查 correlations 长度
        if len(correlations) != 102:
            logger.warning(f"LLM Parcel {entity_id} (run: {run_dir.name}) 有 {len(correlations)} 个 correlations，期望 102，跳过")
            continue
        
        # 存储该 entity 的结果（如果有多个 run，保留最后一个）
        # 如果需要保留最佳结果，可以在这里添加逻辑
        entity_results[entity_id] = {
            "entity_id": entity_id,
            "correlations": correlations
        }
    
    # 转换为列表格式
    prediction_data = list(entity_results.values())
    
    logger.info(f"加载了 {len(prediction_data)} 个 LLM Parcel 的预测结果")
    
    return prediction_data


def extract_correlations(prediction_data):
    """
    提取 correlations 数据并构建矩阵（去除第 0 和第 101 个 Human Parcel）
    
    Args:
        prediction_data: 预测结果数据列表
        
    Returns:
        tuple: (llm_parcel_ids, human_parcel_ids, correlation_matrix)
            - llm_parcel_ids: LLM Parcel ID 列表（已排序）
            - human_parcel_ids: Human Parcel ID 列表（去除第 0 和第 101 个，共 100 个，与 parcel_descriptions.json 保持一致）
            - correlation_matrix: numpy 数组，形状为 (100, n_llm_parcels)
    """
    logger.info("提取 correlations 并构建矩阵...")
    
    # 收集所有 LLM Parcel 的数据
    llm_data = []
    for item in prediction_data:
        entity_id = item.get('entity_id')
        if entity_id is None:
            logger.warning("跳过没有 entity_id 的项")
            continue
        
        correlations = item.get('correlations', [])
        
        if len(correlations) != 102:
            logger.warning(f"LLM Parcel {entity_id} 有 {len(correlations)} 个 correlations，期望 102，跳过")
            continue
        
        # 去除第 0 和第 101 个 Human Parcel（索引 0 和 101）
        # parcel_descriptions.json 已经去除了第 0 和第 101 个，只包含 parcel_id 1-100
        # 索引映射：原始索引 i -> human parcel_id = i + 1
        # 索引 0: 跳过（第 0 个 Human Parcel，parcel_id 0，已从 parcel_descriptions.json 中去除）
        # 索引 1-100: 对应 human parcel_id 1-100（与 parcel_descriptions.json 中的 parcel_id 对应）
        # 索引 101: 跳过（第 101 个 Human Parcel，parcel_id 101，已从 parcel_descriptions.json 中去除）
        human_correlations = []
        
        for i in range(102):
            if i == 0 or i == 101:  # 跳过第 0 个（索引 0）和第 101 个（索引 101）
                continue
            
            human_correlations.append(float(correlations[i]))
        
        if len(human_correlations) != 100:
            logger.warning(f"LLM Parcel {entity_id} 过滤后有 {len(human_correlations)} 个 correlations，期望 100，跳过")
            continue
        
        llm_data.append({
            'entity_id': entity_id,
            'correlations': human_correlations
        })
    
    # 按 entity_id 排序
    llm_data.sort(key=lambda x: x['entity_id'])
    
    # 构建矩阵
    n_llm_parcels = len(llm_data)
    n_human_parcels = 100
    
    correlation_matrix = np.zeros((n_human_parcels, n_llm_parcels))
    llm_parcel_ids = [item['entity_id'] for item in llm_data]
    
    # Human Parcel IDs: 1-100（去除第 0 和第 101 个）
    # parcel_descriptions.json 已经去除了第 0 和第 101 个，只包含 parcel_id 1-100
    # 注意：原始索引 i 对应 parcel_id = i + 1，所以：
    # - 索引 0 -> parcel_id 0（已去除，不在 parcel_descriptions.json 中）
    # - 索引 1-100 -> parcel_id 1-100（与 parcel_descriptions.json 中的 parcel_id 对应）
    # - 索引 101 -> parcel_id 101（已去除，不在 parcel_descriptions.json 中）
    human_parcel_ids = list(range(1, 101))
    
    # 填充矩阵
    for j, llm_item in enumerate(llm_data):
        correlations = llm_item['correlations']
        for i in range(n_human_parcels):
            correlation_matrix[i, j] = correlations[i]
    
    logger.info(f"相关性矩阵形状: {correlation_matrix.shape} (H={n_human_parcels}, L={n_llm_parcels})")
    logger.info(f"LLM Parcel IDs 范围: {min(llm_parcel_ids)} - {max(llm_parcel_ids)}")
    logger.info(f"Human Parcel IDs 范围: {min(human_parcel_ids)} - {max(human_parcel_ids)}")
    logger.info(f"已去除第 0 和第 101 个 Human Parcel（与 parcel_descriptions.json 保持一致）")
    
    return llm_parcel_ids, human_parcel_ids, correlation_matrix


def load_human_parcel_names(human_parcel_file):
    """
    加载 Human Parcel 的 function_name，用于矩阵的行名
    
    Args:
        human_parcel_file: Human Parcel descriptions JSON 文件路径
        
    Returns:
        dict: {parcel_id: function_name}
    """
    logger.info(f"加载 Human Parcel 名称: {human_parcel_file}")
    with open(human_parcel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Human parcel 文件应为列表，实际为 {type(data)}")
    
    parcel_names = {}
    for item in data:
        parcel_id = item.get('parcel_id')
        function_name = item.get('function_name', '')
        if parcel_id is not None:
            parcel_names[parcel_id] = function_name
    
    logger.info(f"加载了 {len(parcel_names)} 个 Human Parcel 名称")
    return parcel_names


def load_llm_parcel_names(llm_parcel_file):
    """
    加载 LLM Parcel 的 function_name，用于矩阵的列名
    
    Args:
        llm_parcel_file: LLM Parcel functionality summary JSON 文件路径
        
    Returns:
        dict: {parcel_id: function_name}
    """
    logger.info(f"加载 LLM Parcel 名称: {llm_parcel_file}")
    with open(llm_parcel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 parcel_summaries
    if isinstance(data, dict) and 'parcel_summaries' in data:
        parcel_summaries = data['parcel_summaries']
    elif isinstance(data, list):
        parcel_summaries = data
    else:
        raise ValueError(f"意外的 LLM parcel 文件格式: {type(data)}")
    
    parcel_names = {}
    for item in parcel_summaries:
        parcel_id = item.get('parcel_id')
        function_name = item.get('function_name', '')
        if parcel_id is not None:
            # 清理 function_name（去除可能的 ** 标记）
            function_name = function_name.replace('**', '').strip()
            function_name = ' '.join(function_name.split())
            parcel_names[parcel_id] = function_name
    
    logger.info(f"加载了 {len(parcel_names)} 个 LLM Parcel 名称")
    return parcel_names


def build_prediction_matrix_from_results(
    results_dir,
    output_file,
    human_parcel_file=None,
    llm_parcel_file=None,
    id_key="layer_idx",
    skip_existing=False
):
    """
    从原始结果文件夹构建预测结果矩阵并保存为 CSV
    
    Args:
        results_dir: 原始结果文件夹路径（包含多个 run_* 子目录）
        output_file: 输出 CSV 文件路径
        human_parcel_file: Human Parcel descriptions JSON 文件路径（可选，用于行名）
        llm_parcel_file: LLM Parcel functionality summary JSON 文件路径（可选，用于列名）
        id_key: hyperparams.json 中用于标识 ID 的字段名
        skip_existing: 如果输出文件已存在，是否跳过
    """
    # 从原始结果文件夹加载预测数据
    prediction_data = load_prediction_data_from_results(
        results_dir,
        id_key=id_key,
        skip_existing=skip_existing,
        output_file=output_file
    )
    
    if prediction_data is None:
        logger.info("已跳过处理（输出文件已存在）")
        return None
    
    # 提取 correlations
    llm_parcel_ids, human_parcel_ids, correlation_matrix = extract_correlations(prediction_data)
    
    # 使用 parcel_id 作为行名和列名
    row_names = [f"Human_Parcel_{pid}" for pid in human_parcel_ids]
    col_names = [f"LLM_Parcel_{pid}" for pid in llm_parcel_ids]
    
    # 加载 function_name 用于映射文件（如果提供了文件）
    human_parcel_id_to_name = {}
    llm_parcel_id_to_name = {}
    
    if human_parcel_file:
        try:
            human_parcel_names = load_human_parcel_names(human_parcel_file)
            human_parcel_id_to_name = {pid: human_parcel_names.get(pid, f"Human_Parcel_{pid}") for pid in human_parcel_ids}
        except Exception as e:
            logger.warning(f"加载 Human Parcel 名称失败: {e}")
    
    if llm_parcel_file:
        try:
            llm_parcel_names = load_llm_parcel_names(llm_parcel_file)
            llm_parcel_id_to_name = {pid: llm_parcel_names.get(pid, f"LLM_Parcel_{pid}") for pid in llm_parcel_ids}
        except Exception as e:
            logger.warning(f"加载 LLM Parcel 名称失败: {e}")
    
    # 创建 DataFrame（使用 parcel_id 作为行名和列名）
    df = pd.DataFrame(
        correlation_matrix,
        index=row_names,
        columns=col_names
    )
    
    # 保存为 CSV
    logger.info(f"保存预测矩阵到: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file)
    
    # 保存映射文件（parcel_id -> function_name）
    mapping_file = output_file.replace('.csv', '_parcel_id_to_function_name.json')
    mapping_data = {
        'human_parcels': human_parcel_id_to_name,
        'llm_parcels': llm_parcel_id_to_name
    }
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Parcel ID 到 function name 的映射已保存到: {mapping_file}")
    
    logger.info(f"预测矩阵已保存: 形状 {df.shape}")
    logger.info(f"行名（前 3 个）: {row_names[:3]}")
    logger.info(f"列名（前 3 个）: {col_names[:3]}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='从原始结果文件夹构建预测矩阵')
    parser.add_argument('--results_dir',
                       required=True,
                       help='原始结果文件夹路径（包含多个 run_* 子目录）')
    parser.add_argument('--output_file',
                       required=True,
                       help='输出 CSV 文件路径')
    parser.add_argument('--human_parcel_file',
                       default=None,
                       help='Human Parcel descriptions JSON 文件路径（可选，用于行名）')
    parser.add_argument('--llm_parcel_file',
                       default=None,
                       help='LLM Parcel functionality summary JSON 文件路径（可选，用于列名）')
    parser.add_argument('--id_key',
                       default='layer_idx',
                       help='hyperparams.json 中用于标识 ID 的字段名（默认: layer_idx）')
    parser.add_argument('--skip_existing',
                       action='store_true',
                       help='如果输出文件已存在，则跳过处理')
    
    args = parser.parse_args()
    
    # 构建矩阵
    df = build_prediction_matrix_from_results(
        results_dir=args.results_dir,
        output_file=args.output_file,
        human_parcel_file=args.human_parcel_file,
        llm_parcel_file=args.llm_parcel_file,
        id_key=args.id_key,
        skip_existing=args.skip_existing
    )
    
    if df is not None:
        print(f"\n预测矩阵形状: {df.shape}")
        print(f"结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()
