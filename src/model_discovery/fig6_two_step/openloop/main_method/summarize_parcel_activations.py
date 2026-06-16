#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
功能：
    汇总 two-step 认知实验中 LLM parcel 激活数据，并结合 parcel 功能描述，
    生成用于分析的 Top parcel 列表。

数据来源：
    1) LLM parcel 激活：
       /path/to/project_root/Human_LLM_align/\
Llama-3.1-Centaur-70B-main/openloop/analysis_code/results/participant_parcel_activations.json
       - key 形如 baseline_125 或 gemma2b_max1024_17
       - value 形状约为 (n_trials, 3, 270):
           0: 第一次选择时刻激活
           1: 第二次选择时刻激活
           2: 获得 reward 之后激活

    2) Parcel 功能描述：
       /path/to/project_root/neural_area/\
divide_area_by_sae_act/cluster_output_2b_pt/\
clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/\
latent_parcel_topsamples_functionality_summary.json

输出：
    将不同对比条件下（Baseline_LLM_Inter、LLM>Baseline，以及二者差值）的：
        - 第一步决策、第二步决策、reward 三个时刻
        - 各自 Top-K（默认 5）平均激活最高的 parcel
      以及对应的 function_name、function_description
    存储到 comparison_results 目录下的一个 JSON 文件中，默认路径为：
        openloop/results/comparison_results/parcel_activation_summary.json

注意：
    - 默认会根据 key 前缀自动把所有 baseline_* 归为 baseline 组，
      所有 gemma2b_max1024_* 归为 LLM 组。
    - 如果你之后有更精确的「top10 实验轮次」集合（例如 Baseline_LLM_Inter、LLM>Baseline），
      可以直接修改下方的 BASELINE_LLM_INTER_KEYS / LLM_GT_BASELINE_KEYS 配置，
      或者改成从外部 JSON/文本中读取。
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

import numpy as np


# === 用户可修改的简单配置 ======================================================

# 如果你已经根据行为结果选出了「两者都表现很好」的实验 key，
# 可以在这里人工填入，例如：
# BASELINE_LLM_INTER_KEYS = ["baseline_125", "baseline_120", "gemma2b_max1024_17", ...]
BASELINE_LLM_INTER_KEYS: List[str] = []

# 如果你已经选出了「LLM 明显优于 Baseline」的实验 key，
# 可以在这里人工填入，例如：
# LLM_GT_BASELINE_KEYS = ["gemma2b_max1024_17", "gemma2b_max1024_22", ...]
LLM_GT_BASELINE_KEYS: List[str] = []

# 默认统计每个条件下 Top-K parcel
TOP_K = 10


@dataclass
class ParcelInfo:
    parcel_id: int
    function_name: str
    function_description: str


def load_json(path: str) -> Any:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"读取 JSON 失败：{path}")
            raise e


def load_parcel_descriptions(path: str) -> Dict[int, ParcelInfo]:
    data = load_json(path)
    summaries = data.get("parcel_summaries")
    if summaries is None:
        raise ValueError("parcel 功能描述 JSON 中缺少 'parcel_summaries' 字段")

    parcel_map: Dict[int, ParcelInfo] = {}
    for item in summaries:
        pid = item.get("parcel_id")
        if pid is None:
            print(f"警告：发现没有 parcel_id 的条目，已跳过：{item}")
            continue
        parcel_map[int(pid)] = ParcelInfo(
            parcel_id=int(pid),
            function_name=item.get("function_name", "").strip(),
            function_description=item.get("function_description", "").strip(),
        )
    return parcel_map


def load_capability_parcel_mapping(mapping_json_path: str) -> Tuple[Dict, List[str]]:
    """加载 Capability-Parcel 映射 JSON。

    Returns:
        capability_parcel_mapping: 原始字典
        capability_names: capability 名称列表（有序）
    """
    print(f"加载 Capability-Parcel 映射: {mapping_json_path}")
    if not os.path.exists(mapping_json_path):
        raise FileNotFoundError(f"映射文件不存在: {mapping_json_path}")

    try:
        with open(mapping_json_path, "r", encoding="utf-8") as f:
            capability_parcel_mapping = json.load(f)
    except Exception as e:
        print(f"加载映射文件失败: {e}")
        raise

    if not isinstance(capability_parcel_mapping, dict):
        raise ValueError(f"映射文件内容必须是 dict，得到: {type(capability_parcel_mapping)}")

    capability_names = list(capability_parcel_mapping.keys())
    print(f"加载了 {len(capability_names)} 个 Capability")
    return capability_parcel_mapping, capability_names


def build_capability_parcel_mapping_matrix(
    capability_parcel_mapping: Dict,
    capability_names: List[str],
    parcel_dim: int,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    构建 Capability-Parcel 映射矩阵。

    Args:
        capability_parcel_mapping: Capability→Parcel 映射字典
        capability_names: Capability 名称列表
        parcel_dim: Parcel 维度（应与 LLM Parcel 数量一致，270）
        epsilon: 归一化时避免除零的小常数

    Returns:
        映射矩阵，形状为 (parcel_dim, capability_dim)
    """
    print("构建 Capability-Parcel 映射矩阵...")
    capability_dim = len(capability_names)
    mapping_matrix = np.zeros((parcel_dim, capability_dim), dtype=np.float32)

    for cap_idx, cap_name in enumerate(capability_names):
        if cap_name not in capability_parcel_mapping:
            print(f"警告：Capability {cap_name} 在映射中不存在，跳过")
            continue

        cap_data = capability_parcel_mapping[cap_name]
        if not isinstance(cap_data, dict):
            print(f"警告：Capability {cap_name} 的映射数据不是 dict，跳过")
            continue

        if "ranking" not in cap_data:
            print(f"警告：Capability {cap_name} 缺少 ranking 字段，跳过")
            continue

        ranking = cap_data["ranking"]
        if not isinstance(ranking, list):
            print(f"警告：Capability {cap_name} ranking 不是列表，跳过")
            continue

        weights: List[float] = []
        parcel_indices: List[int] = []

        for item in ranking:
            if not isinstance(item, list) or len(item) != 2:
                print(f"警告：Capability {cap_name} ranking 项格式错误: {item}")
                continue
            parcel_name, weight = item
            try:
                if isinstance(parcel_name, str) and parcel_name.startswith("parcel_"):
                    parcel_idx = int(parcel_name.split("_")[1])
                    if 0 <= parcel_idx < parcel_dim:
                        weights.append(float(weight))
                        parcel_indices.append(parcel_idx)
                    else:
                        print(f"警告：Capability {cap_name} 的 Parcel 索引超出范围: {parcel_idx}")
                else:
                    print(f"警告：Capability {cap_name} 出现未知的 parcel 名称格式: {parcel_name}")
            except (ValueError, IndexError) as e:
                print(f"警告：Capability {cap_name} 解析 parcel 名称失败: {parcel_name}, 错误: {e}")
                continue

        if len(weights) == 0:
            print(f"警告：Capability {cap_name} 没有有效的 parcel 映射，跳过")
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

    print(f"映射矩阵构建完成，形状: {mapping_matrix.shape}")
    print(f"具有有效映射的 Capability 数量: {valid_count}/{mapping_matrix.shape[1]}")
    return mapping_matrix


def map_parcel_to_capability(
    parcel_vector: np.ndarray,
    mapping_matrix: np.ndarray,
) -> np.ndarray:
    """
    将 Parcel 级别的激活向量映射到 Capability 级别。

    Args:
        parcel_vector: Parcel 激活向量，形状为 (270,)
        mapping_matrix: Capability-Parcel 映射矩阵，形状为 (270, n_capabilities)

    Returns:
        Capability 激活向量，形状为 (n_capabilities,)
    """
    if parcel_vector.ndim != 1:
        raise ValueError(f"parcel_vector 应该是一维向量，得到形状: {parcel_vector.shape}")
    if parcel_vector.shape[0] != mapping_matrix.shape[0]:
        raise ValueError(
            f"parcel_vector 长度 ({parcel_vector.shape[0]}) 与映射矩阵 Parcel 维度 "
            f"({mapping_matrix.shape[0]}) 不匹配"
        )

    # 矩阵乘法: (270,) · (270, n_cap) = (n_cap,)
    capability_vector = np.matmul(parcel_vector, mapping_matrix)
    return capability_vector


def group_keys_by_prefix(activation_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    baseline_keys = [k for k in activation_dict.keys() if k.startswith("baseline_")]
    llm_keys = [k for k in activation_dict.keys() if k.startswith("gemma2b_max1024_")]

    if not baseline_keys:
        print("警告：在激活文件中没有找到前缀为 'baseline_' 的 key。")
    if not llm_keys:
        print("警告：在激活文件中没有找到前缀为 'gemma2b_max1024_' 的 key。")

    return baseline_keys, llm_keys


def stack_activations(
    activation_dict: Dict[str, Any],
    keys: List[str],
) -> np.ndarray:
    """
    将若干 key 对应的激活堆叠在一起。
    返回形状为 (总 trial 数, 3, 270) 的数组。
    
    注意：由于数据中有些 trial 的第二层长度不一致（有些是3，有些可能是1），
    需要逐个处理每个 trial，只保留形状为 (3, 270) 的 trial。
    """
    valid_trials: List[np.ndarray] = []
    skipped_trials_count = 0
    
    for k in keys:
        if k not in activation_dict:
            print(f"警告：key '{k}' 不在激活数据中，已跳过。")
            continue
        
        value = activation_dict[k]
        
        # 逐个处理每个 trial
        for trial_idx, trial_data in enumerate(value):
            try:
                # 尝试转换为 numpy 数组
                trial_arr = np.array(trial_data, dtype=float)
                
                # 检查形状
                if trial_arr.ndim == 2:
                    # 形状应该是 (3, 270) 或 (n_steps, 270)
                    if trial_arr.shape[1] != 270:
                        print(f"警告：key '{k}' trial {trial_idx} 的 parcel 维度为 {trial_arr.shape[1]}，预期为 270，已跳过。")
                        skipped_trials_count += 1
                        continue
                    
                    if trial_arr.shape[0] == 3:
                        # 形状正确，直接添加
                        valid_trials.append(trial_arr)
                    elif trial_arr.shape[0] == 1:
                        # 只有1个时间点，需要填充为3个（复制或填充0）
                        # 这里我们复制第一个时间点3次，或者填充0
                        # 根据需求，我们填充0到3个时间点
                        padded_trial = np.zeros((3, 270), dtype=float)
                        padded_trial[0] = trial_arr[0]  # 第一个时间点使用原始数据
                        valid_trials.append(padded_trial)
                        print(f"警告：key '{k}' trial {trial_idx} 只有1个时间点，已填充为3个时间点。")
                    else:
                        print(f"警告：key '{k}' trial {trial_idx} 的时间点维度为 {trial_arr.shape[0]}，预期为 3，已跳过。")
                        skipped_trials_count += 1
                        continue
                elif trial_arr.ndim == 1:
                    # 如果是一维数组，可能是 (270,)，需要扩展为 (3, 270)
                    if trial_arr.shape[0] == 270:
                        padded_trial = np.zeros((3, 270), dtype=float)
                        padded_trial[0] = trial_arr  # 只填充第一个时间点
                        valid_trials.append(padded_trial)
                        print(f"警告：key '{k}' trial {trial_idx} 是一维数组，已扩展为 (3, 270)。")
                    else:
                        print(f"警告：key '{k}' trial {trial_idx} 的形状为 {trial_arr.shape}，无法处理，已跳过。")
                        skipped_trials_count += 1
                        continue
                else:
                    print(f"警告：key '{k}' trial {trial_idx} 的维度为 {trial_arr.ndim}，预期为 2，已跳过。")
                    skipped_trials_count += 1
                    continue
                    
            except (ValueError, TypeError) as e:
                print(f"警告：key '{k}' trial {trial_idx} 转换失败: {e}，已跳过。")
                skipped_trials_count += 1
                continue
    
    if not valid_trials:
        raise ValueError("没有找到有效的 trial 数据。所有 trial 都被跳过或数据格式不正确。")
    
    if skipped_trials_count > 0:
        print(f"信息：共跳过了 {skipped_trials_count} 个形状不规则的 trial。")
    
    # 堆叠所有有效的 trial
    stacked = np.stack(valid_trials, axis=0)  # (n_trials, 3, 270)
    
    print(f"信息：成功堆叠了 {stacked.shape[0]} 个 trial，形状为 {stacked.shape}。")
    
    return stacked


def compute_mean_by_step(stacked: np.ndarray) -> Dict[str, np.ndarray]:
    """
    输入形状 (N_trials, 3, 270)，返回：
        {
            "step1": (270,),
            "step2": (270,),
            "reward": (270,)
        }
    """
    if stacked.ndim != 3 or stacked.shape[1] != 3:
        raise ValueError(f"输入数组形状异常：{stacked.shape}，预期为 (N, 3, 270)")

    mean_all = stacked.mean(axis=0)  # (3, 270)
    return {
        "step1": mean_all[0],  # type: ignore[index]
        "step2": mean_all[1],  # type: ignore[index]
        "reward": mean_all[2],  # type: ignore[index]
    }


def top_k_parcels(
    mean_vector: np.ndarray,
    parcel_desc: Dict[int, ParcelInfo],
    k: int,
) -> List[Dict[str, Any]]:
    if mean_vector.ndim != 1:
        raise ValueError(f"top_k_parcels 期望输入一维向量，但得到形状 {mean_vector.shape}")

    if mean_vector.size != 270:
        raise ValueError(f"top_k_parcels 期望长度为 270 的向量，但得到长度 {mean_vector.size}")

    k = min(k, mean_vector.size)
    # 从大到小排序
    indices = np.argsort(mean_vector)[::-1][:k]

    results: List[Dict[str, Any]] = []
    for pid in indices:
        pid_int = int(pid)
        info = parcel_desc.get(
            pid_int,
            ParcelInfo(parcel_id=pid_int, function_name="", function_description=""),
        )
        results.append(
            {
                "parcel_id": pid_int,
                "mean_activation": float(mean_vector[pid_int]),
                "function_name": info.function_name,
                "function_description": info.function_description,
            }
        )
    return results


def top_k_capabilities(
    capability_vector: np.ndarray,
    capability_names: List[str],
    k: int,
) -> List[Dict[str, Any]]:
    """
    计算 Top-K Capability。

    Args:
        capability_vector: Capability 激活向量，形状为 (n_capabilities,)
        capability_names: Capability 名称列表
        k: Top-K 数量

    Returns:
        Top-K Capability 列表，每个元素包含 capability_name 和 mean_activation
    """
    if capability_vector.ndim != 1:
        raise ValueError(f"top_k_capabilities 期望输入一维向量，但得到形状 {capability_vector.shape}")

    if len(capability_names) != capability_vector.size:
        raise ValueError(
            f"capability_names 长度 ({len(capability_names)}) 与 "
            f"capability_vector 长度 ({capability_vector.size}) 不匹配"
        )

    k = min(k, capability_vector.size)
    # 从大到小排序
    indices = np.argsort(capability_vector)[::-1][:k]

    results: List[Dict[str, Any]] = []
    for cap_idx in indices:
        cap_idx_int = int(cap_idx)
        if cap_idx_int >= len(capability_names):
            print(f"警告：Capability 索引 {cap_idx_int} 超出范围，跳过")
            continue
        results.append(
            {
                "capability_name": capability_names[cap_idx_int],
                "mean_activation": float(capability_vector[cap_idx_int]),
            }
        )
    return results


def build_summary_for_condition(
    name: str,
    activation_dict: Dict[str, Any],
    keys: List[str],
    parcel_desc: Dict[int, ParcelInfo],
    top_k: int,
    mapping_matrix: np.ndarray = None,
    capability_names: List[str] = None,
) -> Dict[str, Any]:
    """
    构建条件摘要，同时包含 Parcel 和 Capability 级别的信息。

    Args:
        name: 条件名称
        activation_dict: 激活数据字典
        keys: 该条件使用的 key 列表
        parcel_desc: Parcel 描述字典
        top_k: Top-K 数量
        mapping_matrix: Capability-Parcel 映射矩阵，如果提供则同时计算 Capability 级别
        capability_names: Capability 名称列表，如果提供则同时计算 Capability 级别
    """
    print(f"正在处理条件：{name}，包含 key 数量：{len(keys)}")
    stacked = stack_activations(activation_dict, keys)
    mean_by_step = compute_mean_by_step(stacked)

    summary: Dict[str, Any] = {
        "condition_name": name,
        "num_keys": len(keys),
        "num_trials_total": int(stacked.shape[0]),
        "top_k": top_k,
        "steps": {},
    }

    for step_name, vec in mean_by_step.items():
        step_summary: Dict[str, Any] = {
            "mean_activation_vector_shape": list(vec.shape),
            "top_parcels": top_k_parcels(vec, parcel_desc, top_k),
        }

        # 如果提供了映射矩阵，同时计算 Capability 级别
        if mapping_matrix is not None and capability_names is not None:
            capability_vec = map_parcel_to_capability(vec, mapping_matrix)
            step_summary["capability_activation_vector_shape"] = list(capability_vec.shape)
            step_summary["top_capabilities"] = top_k_capabilities(
                capability_vec, capability_names, top_k
            )

        summary["steps"][step_name] = step_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="汇总 two-step 实验中 LLM parcel 激活，并输出带功能描述的 Top parcel 列表。"
    )
    parser.add_argument(
        "--activations",
        type=str,
        default=(
            "/path/to/project_root/"
            "Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/analysis_code/"
            "results/participant_parcel_activations.json"
        ),
        help="LLM parcel 激活 JSON 文件路径。",
    )
    parser.add_argument(
        "--parcel-desc",
        type=str,
        default=(
            "/path/to/project_root/neural_area/"
            "divide_area_by_sae_act/cluster_output_2b_pt/"
            "clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/"
            "latent_parcel_topsamples_functionality_summary.json"
        ),
        help="parcel 功能描述 JSON 文件路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=(
            "/path/to/project_root/"
            "Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/results/"
            "comparison_results/parcel_activation_summary.json"
        ),
        help="输出 JSON 文件路径。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标输出文件已存在，是否允许覆盖（默认不覆盖，直接跳过）。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="每个条件/时间点选取的 Top-K parcel 数量。",
    )
    parser.add_argument(
        "--capability-mapping",
        type=str,
        default=(
            "/path/to/project_root/neural_area/"
            "connect_cap_parcel/results/aggrate_final/final_capability_parcel_all.json"
        ),
        help="Capability-Parcel 映射 JSON 文件路径。如果提供，将同时计算 Capability 级别的 Top-K。",
    )
    args = parser.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print(f"输出文件已存在且未指定 --overwrite，跳过生成：{args.output}")
        return

    print("加载激活数据...")
    activations = load_json(args.activations)
    if not isinstance(activations, dict):
        raise ValueError("激活文件的顶层结构应为 dict。")

    print("加载 parcel 功能描述...")
    parcel_desc = load_parcel_descriptions(args.parcel_desc)

    # 加载 Capability-Parcel 映射（如果提供）
    mapping_matrix = None
    capability_names = None
    if args.capability_mapping and os.path.exists(args.capability_mapping):
        print("加载 Capability-Parcel 映射...")
        capability_parcel_mapping, capability_names = load_capability_parcel_mapping(
            args.capability_mapping
        )
        mapping_matrix = build_capability_parcel_mapping_matrix(
            capability_parcel_mapping=capability_parcel_mapping,
            capability_names=capability_names,
            parcel_dim=270,  # LLM Parcel 数量固定为 270
        )
        print("Capability 映射已加载，将同时计算 Capability 级别的 Top-K")
    else:
        print("未提供 Capability 映射文件或文件不存在，仅计算 Parcel 级别")

    # 自动分组：根据前缀找到所有 baseline_* 和 gemma2b_max1024_* key
    baseline_all_keys, llm_all_keys = group_keys_by_prefix(activations)

    # 如果用户没有在配置中填入更精细的 key 集合，就使用「所有」作为默认集合，
    # 并打印提示，不会静默降级。
    if not BASELINE_LLM_INTER_KEYS:
        print(
            "提示：BASELINE_LLM_INTER_KEYS 为空，默认使用所有 baseline_* key 作为 "
            "Baseline_LLM_Inter 条件。之后可以在脚本顶部修改为更精确的集合。"
        )
        baseline_llm_inter_keys = baseline_all_keys
    else:
        baseline_llm_inter_keys = BASELINE_LLM_INTER_KEYS

    if not LLM_GT_BASELINE_KEYS:
        print(
            "提示：LLM_GT_BASELINE_KEYS 为空，默认使用所有 gemma2b_max1024_* key 作为 "
            "LLM>Baseline 条件。之后可以在脚本顶部修改为更精确的集合。"
        )
        llm_gt_baseline_keys = llm_all_keys
    else:
        llm_gt_baseline_keys = LLM_GT_BASELINE_KEYS

    if not baseline_llm_inter_keys:
        raise ValueError("Baseline_LLM_Inter 条件下没有任何 key，请检查配置或激活文件。")
    if not llm_gt_baseline_keys:
        raise ValueError("LLM>Baseline 条件下没有任何 key，请检查配置或激活文件。")

    print("开始计算各条件下的平均激活与 Top parcel ...")
    baseline_llm_inter_summary = build_summary_for_condition(
        name="Baseline_LLM_Inter",
        activation_dict=activations,
        keys=baseline_llm_inter_keys,
        parcel_desc=parcel_desc,
        top_k=args.top_k,
        mapping_matrix=mapping_matrix,
        capability_names=capability_names,
    )

    llm_gt_baseline_summary = build_summary_for_condition(
        name="LLM>Baseline",
        activation_dict=activations,
        keys=llm_gt_baseline_keys,
        parcel_desc=parcel_desc,
        top_k=args.top_k,
        mapping_matrix=mapping_matrix,
        capability_names=capability_names,
    )

    # 计算「LLM>Baseline - Baseline_LLM_Inter」在激活空间上的差异，并同样取 Top-K
    print("计算 LLM>Baseline 与 Baseline_LLM_Inter 之间的差值 Top parcel ...")
    stacked_baseline = stack_activations(activations, baseline_llm_inter_keys)
    stacked_llm = stack_activations(activations, llm_gt_baseline_keys)
    mean_baseline = compute_mean_by_step(stacked_baseline)
    mean_llm = compute_mean_by_step(stacked_llm)

    diff_summary: Dict[str, Any] = {
        "condition_name": "LLM>Baseline_minus_Baseline_LLM_Inter",
        "description": "对不同 key 集合（LLM>Baseline 与 Baseline_LLM_Inter）在 "
        "parcel 激活空间上的平均差值，并选取差值最大的 Top-K parcel。",
        "top_k": args.top_k,
        "steps": {},
    }

    for step_name in ["step1", "step2", "reward"]:
        vec_baseline = mean_baseline[step_name]
        vec_llm = mean_llm[step_name]
        if vec_baseline.shape != vec_llm.shape:
            raise ValueError(
                f"差值计算时 {step_name} 的向量形状不一致："
                f"baseline={vec_baseline.shape}, llm={vec_llm.shape}"
            )
        diff_vec = vec_llm - vec_baseline
        step_diff: Dict[str, Any] = {
            "diff_vector_shape": list(diff_vec.shape),
            "top_parcels": top_k_parcels(diff_vec, parcel_desc, args.top_k),
        }

        # 如果提供了映射矩阵，同时计算 Capability 级别的差值
        if mapping_matrix is not None and capability_names is not None:
            cap_vec_baseline = map_parcel_to_capability(vec_baseline, mapping_matrix)
            cap_vec_llm = map_parcel_to_capability(vec_llm, mapping_matrix)
            cap_diff_vec = cap_vec_llm - cap_vec_baseline
            step_diff["capability_diff_vector_shape"] = list(cap_diff_vec.shape)
            step_diff["top_capabilities"] = top_k_capabilities(
                cap_diff_vec, capability_names, args.top_k
            )

        diff_summary["steps"][step_name] = step_diff

    # 计算「Baseline_LLM_Inter - LLM>Baseline」在激活空间上的差异，并同样取 Top-K
    print("计算 Baseline_LLM_Inter 与 LLM>Baseline 之间的反向差值 Top parcel ...")
    reverse_diff_summary: Dict[str, Any] = {
        "condition_name": "Baseline_LLM_Inter_minus_LLM>Baseline",
        "description": "对不同 key 集合（Baseline_LLM_Inter 与 LLM>Baseline）在 "
        "parcel 激活空间上的平均差值（Baseline - LLM），并选取差值最大的 Top-K parcel。",
        "top_k": args.top_k,
        "steps": {},
    }

    for step_name in ["step1", "step2", "reward"]:
        vec_baseline = mean_baseline[step_name]
        vec_llm = mean_llm[step_name]
        if vec_baseline.shape != vec_llm.shape:
            raise ValueError(
                f"反向差值计算时 {step_name} 的向量形状不一致："
                f"baseline={vec_baseline.shape}, llm={vec_llm.shape}"
            )
        reverse_diff_vec = vec_baseline - vec_llm  # 反向差值：baseline - llm
        step_reverse_diff: Dict[str, Any] = {
            "diff_vector_shape": list(reverse_diff_vec.shape),
            "top_parcels": top_k_parcels(reverse_diff_vec, parcel_desc, args.top_k),
        }

        # 如果提供了映射矩阵，同时计算 Capability 级别的反向差值
        if mapping_matrix is not None and capability_names is not None:
            cap_vec_baseline = map_parcel_to_capability(vec_baseline, mapping_matrix)
            cap_vec_llm = map_parcel_to_capability(vec_llm, mapping_matrix)
            cap_reverse_diff_vec = cap_vec_baseline - cap_vec_llm  # 反向差值
            step_reverse_diff["capability_diff_vector_shape"] = list(cap_reverse_diff_vec.shape)
            step_reverse_diff["top_capabilities"] = top_k_capabilities(
                cap_reverse_diff_vec, capability_names, args.top_k
            )

        reverse_diff_summary["steps"][step_name] = step_reverse_diff

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = {
        "meta": {
            "activations_path": args.activations,
            "parcel_desc_path": args.parcel_desc,
            "capability_mapping_path": args.capability_mapping if mapping_matrix is not None else None,
            "top_k": args.top_k,
            "has_capability_mapping": mapping_matrix is not None,
        },
        "Baseline_LLM_Inter": baseline_llm_inter_summary,
        "LLM>Baseline": llm_gt_baseline_summary,
        "LLM>Baseline_minus_Baseline_LLM_Inter": diff_summary,
        "Baseline_LLM_Inter_minus_LLM>Baseline": reverse_diff_summary,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()


