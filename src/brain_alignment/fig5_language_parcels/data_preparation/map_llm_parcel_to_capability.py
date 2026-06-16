#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LLM Parcel→Human Parcel 的预测矩阵，通过 Capability-Parcel 映射，
转换为 LLM Capability→Human Parcel 的预测矩阵。

输入:
- 一个形如 prediction_matrix_gemma2_2b.csv 的矩阵:
    行: Human_Parcel_x
    列: LLM_Parcel_i

映射:
- 使用 capability-parcel 映射 JSON (如 final_capability_parcel_all.json)，
  其结构与 analysis_capability_level.py 中使用的一致:
  {
    "Some capability": {
      "ranking": [
        ["parcel_0", weight0],
        ["parcel_1", weight1],
        ...
      ],
      ...
    },
    ...
  }

输出:
- 一个新的 CSV，行仍然是 Human_Parcel_x，列为 Capability 名称，
  每个元素是该 Human Parcel 与该 Capability 的聚合预测值。
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_capability_parcel_mapping(mapping_json_path: str) -> Tuple[Dict, List[str]]:
    """加载 Capability-Parcel 映射 JSON。

    Returns:
        capability_parcel_mapping: 原始字典
        capability_names: capability 名称列表（有序）
    """
    logger.info(f"加载 Capability-Parcel 映射: {mapping_json_path}")
    if not os.path.exists(mapping_json_path):
        raise FileNotFoundError(f"映射文件不存在: {mapping_json_path}")

    try:
        with open(mapping_json_path, "r", encoding="utf-8") as f:
            capability_parcel_mapping = json.load(f)
    except Exception as e:
        logger.error(f"加载映射文件失败: {e}")
        raise

    if not isinstance(capability_parcel_mapping, dict):
        raise ValueError(f"映射文件内容必须是 dict，得到: {type(capability_parcel_mapping)}")

    capability_names = list(capability_parcel_mapping.keys())
    logger.info(f"加载了 {len(capability_names)} 个 Capability")
    logger.info(f"前 5 个 Capability: {capability_names[:5]}")
    return capability_parcel_mapping, capability_names


def build_mapping_matrix(
    capability_parcel_mapping: Dict,
    capability_names: List[str],
    parcel_dim: int,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    构建与 analysis_capability_level.py 相同逻辑的 Capability-Parcel 映射矩阵。

    Args:
        capability_parcel_mapping: Capability→Parcel 映射字典
        capability_names: Capability 名称列表
        parcel_dim: Parcel 维度（应与 LLM Parcel 数量一致）
        epsilon: 归一化时避免除零的小常数

    Returns:
        映射矩阵，形状为 (parcel_dim, capability_dim)
    """
    logger.info("构建 Capability-Parcel 映射矩阵...")
    capability_dim = len(capability_names)
    mapping_matrix = np.zeros((parcel_dim, capability_dim), dtype=np.float32)

    for cap_idx, cap_name in enumerate(capability_names):
        if cap_name not in capability_parcel_mapping:
            logger.warning(f"Capability {cap_name} 在映射中不存在，跳过")
            continue

        cap_data = capability_parcel_mapping[cap_name]
        if not isinstance(cap_data, dict):
            logger.warning(f"Capability {cap_name} 的映射数据不是 dict，跳过")
            continue

        if "ranking" not in cap_data:
            logger.warning(f"Capability {cap_name} 缺少 ranking 字段，跳过")
            continue

        ranking = cap_data["ranking"]
        if not isinstance(ranking, list):
            logger.warning(f"Capability {cap_name} ranking 不是列表，跳过")
            continue

        weights: List[float] = []
        parcel_indices: List[int] = []

        for item in ranking:
            if not isinstance(item, list) or len(item) != 2:
                logger.warning(f"Capability {cap_name} ranking 项格式错误: {item}")
                continue
            parcel_name, weight = item
            try:
                if isinstance(parcel_name, str) and parcel_name.startswith("parcel_"):
                    parcel_idx = int(parcel_name.split("_")[1])
                    if 0 <= parcel_idx < parcel_dim:
                        weights.append(float(weight))
                        parcel_indices.append(parcel_idx)
                    else:
                        logger.warning(f"Capability {cap_name} 的 Parcel 索引超出范围: {parcel_idx}")
                else:
                    logger.warning(f"Capability {cap_name} 出现未知的 parcel 名称格式: {parcel_name}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Capability {cap_name} 解析 parcel 名称失败: {parcel_name}, 错误: {e}")
                continue

        if len(weights) == 0:
            logger.warning(f"Capability {cap_name} 没有有效的 parcel 映射，跳过")
            continue

        # 归一化权重，小于 0 的值先截断为 0
        weights_arr = np.array(weights, dtype=np.float32)
        weights_arr = np.maximum(weights_arr, 0.0)
        weights_arr = weights_arr / (np.sum(weights_arr) + float(epsilon))

        for parcel_idx, w in zip(parcel_indices, weights_arr):
            mapping_matrix[parcel_idx, cap_idx] = w
    # 检查映射矩阵有效性
    valid_capabilities = np.sum(mapping_matrix, axis=0) > 0
    valid_count = int(np.sum(valid_capabilities))
    if valid_count == 0:
        raise ValueError("没有任何有效的 Capability-Parcel 映射，请检查映射文件与 parcel 维度是否匹配")

    logger.info(f"映射矩阵构建完成，形状: {mapping_matrix.shape}")
    logger.info(f"具有有效映射的 Capability 数量: {valid_count}/{mapping_matrix.shape[1]}")
    return mapping_matrix


def load_prediction_matrix(input_file: str) -> pd.DataFrame:
    """加载 Human_Parcel×LLM_Parcel 预测矩阵 CSV。"""
    logger.info(f"加载预测矩阵: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入矩阵文件不存在: {input_file}")

    try:
        # 第一列是行名（Human_Parcel_x）
        df = pd.read_csv(input_file, index_col=0)
    except Exception as e:
        logger.error(f"加载预测矩阵失败: {e}")
        raise

    if df.shape[1] == 0:
        raise ValueError(f"预测矩阵没有任何列: {input_file}")

    logger.info(f"预测矩阵形状: {df.shape} (H={df.shape[0]}, L={df.shape[1]})")
    logger.info(f"前 3 行索引: {df.index[:3].tolist()}")
    logger.info(f"前 3 列名: {df.columns[:3].tolist()}")
    return df


def load_human_parcel_info(parcel_desc_file: str) -> Dict[str, Dict[str, str]]:
    """
    加载 Human Parcel 的功能名和 7Networks parcel_name，用于 top-k 输出。

    Returns:
        dict: key 为 "Human_Parcel_{id}"，value 为:
            {
              "function_name": str,
              "parcel_name": str,
            }
    """
    logger.info(f"加载 Human Parcel 描述: {parcel_desc_file}")
    if not os.path.exists(parcel_desc_file):
        raise FileNotFoundError(f"Human Parcel 描述文件不存在: {parcel_desc_file}")

    try:
        with open(parcel_desc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载 Human Parcel 描述失败: {e}")
        raise

    if not isinstance(data, list):
        raise ValueError(f"Human Parcel 描述文件应为 list, 得到: {type(data)}")

    info: Dict[str, Dict[str, str]] = {}
    for item in data:
        parcel_id = item.get("parcel_id")
        if parcel_id is None:
            continue
        key = f"Human_Parcel_{parcel_id}"
        info[key] = {
            "function_name": item.get("function_name", ""),
            "parcel_name": item.get("parcel_name", ""),
        }

    logger.info(f"加载了 {len(info)} 条 Human Parcel 描述信息")
    return info


def build_topk_capabilities_per_human(
    df_cap: pd.DataFrame,
    parcel_desc_file: str,
    top_k: int,
    output_topk_file: str,
) -> pd.DataFrame:
    """
    对于每个 Human Parcel，从 Human×Capability 矩阵中选出相关性最高的 top-k 能力，
    并输出类似 fig2_top_llm_parcels_per_human_human.csv 结构的 CSV。

    输出列:
    - human_parcel: 行名，例如 "Human_Parcel_1"
    - human_function: 对应的人脑功能名称 (parcel_descriptions.json 中的 function_name)
    - human_parcel_name: 7Networks parcel_name，例如 "7Networks_LH_Vis_1"
    - llm_capability: LLM capability 名称 (来自列名)
    - llm_function: 这里暂时与 llm_capability 相同
    - rank_by_corr: 相关性排序 (1 表示相关性最高)
    - selection_type: 固定为 "top"
    - prediction_correlation: 相关性数值 (来自 df_cap)

    为兼容已有绘图代码，也可以在后处理时重命名列。
    """
    logger.info(
        f"开始为每个 Human Parcel 计算 top-{top_k} LLM capability 相关性: {output_topk_file}"
    )

    if top_k <= 0:
        raise ValueError(f"top_k 必须为正整数，得到: {top_k}")

    human_info = load_human_parcel_info(parcel_desc_file)

    records: List[Dict[str, object]] = []

    for human_parcel, row in df_cap.iterrows():
        # 取该 Human Parcel 对所有 capability 的相关性
        series = row.dropna()
        if series.empty:
            continue

        # 按相关性从大到小排序，取前 top_k 个
        top_series = series.sort_values(ascending=False).head(top_k)

        h_info = human_info.get(human_parcel, {})
        human_function = h_info.get("function_name", "")
        human_parcel_name = h_info.get("parcel_name", "")

        for rank, (cap_name, corr) in enumerate(top_series.items(), start=1):
            records.append(
                {
                    "human_parcel": human_parcel,
                    "human_function": human_function,
                    "human_parcel_name": human_parcel_name,
                    "llm_capability": cap_name,
                    "llm_function": cap_name,  # 暂用 capability 名称
                    "rank_by_corr": rank,
                    "selection_type": "top",
                    "prediction_correlation": float(corr),
                }
            )

    if not records:
        logger.warning("未生成任何 top-k 记录，请检查输入矩阵是否为空")
        df_topk = pd.DataFrame()
    else:
        df_topk = pd.DataFrame.from_records(records)

    out_path = Path(output_topk_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_topk.to_csv(out_path, index=False)
    logger.info(
        f"top-{top_k} LLM capability 结果已保存到: {output_topk_file}，总行数: {len(df_topk)}"
    )

    return df_topk


def map_llm_parcel_to_capability(
    input_file: str,
    mapping_json_path: str,
    output_file: str,
    epsilon: float = 1e-8,
    skip_if_exists: bool = False,
    parcel_desc_file: str = None,
    top_k: int = 0,
    topk_output_file: str = None,
) -> pd.DataFrame:
    """
    将 LLM Parcel 维度映射为 Capability 维度，并输出新的 CSV。

    Args:
        input_file: 原始 prediction_matrix_gemma2_2b.csv 路径
        mapping_json_path: Capability-Parcel 映射 JSON 路径
        output_file: 输出 CSV 路径
        epsilon: 归一化权重时使用的小常数
        skip_if_exists: 若输出文件已存在且为 True，则跳过运算
    """
    output_path = Path(output_file)
    if output_path.exists() and skip_if_exists:
        logger.info(f"输出文件已存在且 skip_if_exists=True，跳过计算: {output_file}")
        return pd.DataFrame()

    # 1. 加载原始 Human×LLM Parcel 预测矩阵
    df_parcel = load_prediction_matrix(input_file)
    human_dim, parcel_dim = df_parcel.shape

    # 2. 加载 Capability-Parcel 映射并构建映射矩阵
    capability_parcel_mapping, capability_names = load_capability_parcel_mapping(mapping_json_path)
    mapping_matrix = build_mapping_matrix(
        capability_parcel_mapping=capability_parcel_mapping,
        capability_names=capability_names,
        parcel_dim=parcel_dim,
        epsilon=epsilon,
    )

    # 3. 执行矩阵乘法: (H×P) · (P×C) = (H×C)
    logger.info("开始进行矩阵乘法，将 LLM Parcel 映射到 Capability 空间...")
    parcel_matrix = df_parcel.values.astype(np.float32)
    if parcel_matrix.shape[1] != mapping_matrix.shape[0]:
        raise ValueError(
            f"矩阵维度不匹配: 预测矩阵 L 维度为 {parcel_matrix.shape[1]}，"
            f"映射矩阵 Parcel 维度为 {mapping_matrix.shape[0]}"
        )

    capability_matrix = np.matmul(parcel_matrix, mapping_matrix)
    logger.info(
        f"Capability 矩阵形状: {capability_matrix.shape} "
        f"(H={human_dim}, C={capability_matrix.shape[1]})"
    )

    # 4. 构建 DataFrame
    df_cap = pd.DataFrame(
        capability_matrix,
        index=df_parcel.index.copy(),
        columns=capability_names,
    )

    # 5. 保存 Human×Capability 矩阵
    logger.info(f"保存 Capability 级别预测矩阵到: {output_file}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cap.to_csv(output_file)
    logger.info("Capability 级别预测矩阵保存完成")

    # 6. 如果指定了 top-k 输出, 则为每个 Human Parcel 计算 top-k capability
    if parcel_desc_file and top_k and topk_output_file:
        try:
            build_topk_capabilities_per_human(
                df_cap=df_cap,
                parcel_desc_file=parcel_desc_file,
                top_k=top_k,
                output_topk_file=topk_output_file,
            )
        except Exception as e:
            # 不影响主输出, 但要打印异常
            logger.error(f"生成 Human Parcel top-{top_k} capability 文件失败: {e}")

    return df_cap


def main():
    parser = argparse.ArgumentParser(
        description="将 LLM Parcel 预测矩阵映射到 Capability 维度，生成新的 CSV。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/adventuresinsayingyes/prediction_matrix_gemma2_2b.csv",
        help="输入的 Human×LLM Parcel 预测矩阵 CSV 路径",
    )
    parser.add_argument(
        "--mapping_json",
        type=str,
        default="/path/to/project_root/neural_area/connect_cap_parcel/results/aggrate_final_9b/final_capability_parcel_all.json",
        help="Capability-Parcel 映射 JSON 路径",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data4draw/adventuresinsayingyes/prediction_matrix_gemma2_2b_capability.csv",
        help="输出的 Human×Capability 预测矩阵 CSV 路径",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="归一化权重时使用的小常数",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="若输出文件已存在，则跳过计算",
    )
    parser.add_argument(
        "--parcel_desc_file",
        type=str,
        default="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json",
        help="Human Parcel 描述文件 (用于 top-k 输出 human_function 和 parcel_name)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="为每个 Human Parcel 选择相关性最高的前 top_k 个 LLM capability",
    )
    parser.add_argument(
        "--topk_output_file",
        type=str,
        default="",
        help="若指定, 生成类似 fig2_top_llm_parcels_per_human_human.csv 的 top-k 结果 CSV",
    )

    args = parser.parse_args()

    try:
        map_llm_parcel_to_capability(
            input_file=args.input_file,
            mapping_json_path=args.mapping_json,
            output_file=args.output_file,
            epsilon=args.epsilon,
            skip_if_exists=args.skip_if_exists,
            parcel_desc_file=args.parcel_desc_file,
            top_k=args.top_k,
            topk_output_file=args.topk_output_file or None,
        )
    except Exception as e:
        # 遵循用户约定: 不静默吞掉异常，至少打印出来
        logger.error(f"映射过程中发生异常: {e}")
        raise


if __name__ == "__main__":
    main()


