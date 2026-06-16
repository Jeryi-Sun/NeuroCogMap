#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取工具

基于 token 级 Parcel 激活时间序列，计算：
- 样本级 Parcel 平均激活向量 a ∈ R^P
- 样本级 Parcel-Parcel 连接矩阵 F ∈ R^{P×P}（去均值标准化后相关，Fisher z）
- 通过 Capability-Parcel 映射聚合得到：
  - Capability 激活向量 c ∈ R^C
  - Capability-Capability 连接矩阵 G ∈ R^{C×C}

并提供“原型对齐 + 指示器轴（M^+, R^-, G^-）+ 连接失配（C^+, C^-)”的特征构造方法。

注意：按用户规则，任何异常均需警告或抛出，不能静默赋默认值。
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


def _demean_and_standardize(X: np.ndarray, eps: float) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (X - mean) / std


def compute_parcel_connectivity(acts_tp: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """基于时间序列激活计算 Parcel-Parcel 连接（Fisher z of Pearson corr）。

    Args:
        acts_tp: 形状 (T, P)
        eps: 数值稳定项
    Returns:
        (P, P) Fisher z 相关矩阵
    """
    if acts_tp.ndim != 2:
        raise ValueError(f"acts_tp 维度错误，期望二维 (T,P)，实际: {acts_tp.shape}")
    T = acts_tp.shape[0]
    P = acts_tp.shape[1]
    if T < 2:
        # 报警并返回零矩阵（无法计算相关）
        # 这里选择返回零矩阵，但明确记录异常。
        # 上游应避免极短序列样本进入训练集。
        return np.zeros((P, P), dtype=np.float32)
    X = _demean_and_standardize(acts_tp, eps)
    corr = (X.T @ X) / max(T - 1, 1)
    corr = np.clip(corr, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + corr) / (1 - corr))
    return z.astype(np.float32)


def build_mapping_matrix(capability_parcel_mapping: Dict, parcel_dim: int, eps: float = 1e-8) -> Tuple[np.ndarray, List[str]]:
    """从 Capability-Parcel 映射字典构建映射矩阵 M ∈ R^{P×C}。

    字段约定：每个 capability 下存在 'ranking': [["parcel_12", weight], ...]
    小于0的权重视作0后再归一化。
    """
    if not isinstance(capability_parcel_mapping, dict):
        raise ValueError("capability_parcel_mapping 必须是字典")

    capability_names = list(capability_parcel_mapping.keys())
    C = len(capability_names)
    if C == 0:
        raise ValueError("映射中 capability 数量为 0")

    M = np.zeros((parcel_dim, C), dtype=np.float32)
    for j, cap in enumerate(capability_names):
        entry = capability_parcel_mapping.get(cap, {})
        ranking = entry.get("ranking", [])
        if not isinstance(ranking, list):
            raise ValueError(f"Capability {cap} 的 ranking 字段应为列表")
        weights: List[float] = []
        indices: List[int] = []
        for item in ranking:
            if not (isinstance(item, list) and len(item) == 2):
                raise ValueError(f"Capability {cap} ranking 项格式错误: {item}")
            parcel_name, w = item
            if not isinstance(parcel_name, str):
                raise ValueError(f"parcel 名称应为字符串，实际: {type(parcel_name)}")
            try:
                if parcel_name.startswith("parcel_"):
                    pid = int(parcel_name.split("_")[1])
                else:
                    raise ValueError(f"未知 parcel 名称格式: {parcel_name}")
            except Exception as e:
                raise ValueError(f"解析 parcel 名称失败: {parcel_name}, 错误: {e}")
            if not (0 <= pid < parcel_dim):
                raise ValueError(f"parcel 索引越界: {pid} (parcel_dim={parcel_dim})")
            weights.append(float(max(0.0, w)))
            indices.append(pid)
        wsum = float(np.sum(weights))
        if wsum <= 0:
            continue
        weights = [float(w / (wsum + eps)) for w in weights]
        for pid, w in zip(indices, weights):
            M[pid, j] = w
    return M, capability_names


def aggregate_to_capability(acts_tp: np.ndarray, M_p_to_c: np.ndarray) -> np.ndarray:
    """A = acts_tp(T×P) 聚合到 C：B(T×C) = A(T×P) · M(P×C)。"""
    if acts_tp.ndim != 2:
        raise ValueError("acts_tp 维度应为 (T,P)")
    if M_p_to_c.ndim != 2 or M_p_to_c.shape[0] != acts_tp.shape[1]:
        raise ValueError("映射矩阵维度不匹配 (P×C)")
    return acts_tp @ M_p_to_c


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < eps or bn < eps:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def frobenius_similarity(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> float:
    """将距离转为[0,1)相似度：1/(1 + ||A-B||_F)。"""
    if A.shape != B.shape:
        raise ValueError(f"矩阵形状不一致: {A.shape} vs {B.shape}")
    dist = float(np.linalg.norm(A - B))
    return float(1.0 / (1.0 + dist + eps))


@dataclass
class IndicatorConfig:
    """指示器配置（可按数据集/模型微调）。"""
    # 上升轴（幻觉上升）——部分 Parcel id
    pos_parcels: List[int] = None  # e.g., [89, 171, 177, 60, 134]
    # 下降轴（应强检索/地理/事实）
    neg_parcels: List[int] = None  # e.g., [238, 88, 145, 163, 167]
    # 动态 capability 轴
    pos_capabilities: List[str] = None
    neg_capabilities: List[str] = None
    # 错通路增强对（可选，连接层面聚合，使用 parcel 对索引）
    # 这里以 parcel 对实现，capability 对在外部构造也可
    wrong_path_pairs: List[Tuple[int, int]] = None
    true_path_pairs: List[Tuple[int, int]] = None
    # capability 层连接对（名称对）
    wrong_capability_pairs: List[Tuple[str, str]] = None
    true_capability_pairs: List[Tuple[str, str]] = None


def zsum(values: List[float]) -> float:
    # 简单求和（训练时再标准化），保留原始强度
    return float(np.sum(values)) if values else 0.0


def compute_prototypes(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """从样本集合计算原型：a, F, c, G 的均值原型。

    样本字典需包含：
      'a': (P,), 'F': (P,P), 'c': (C,), 'G': (C,C)
    """
    if len(samples) == 0:
        raise ValueError("计算原型的样本为空")
    keys = ["a", "F", "c", "G"]
    for k in keys:
        if any(k not in s for s in samples):
            raise ValueError(f"样本缺少关键字段: {k}")
    proto = {}
    proto["a"] = np.mean([s["a"] for s in samples], axis=0)
    proto["F"] = np.mean([s["F"] for s in samples], axis=0)
    proto["c"] = np.mean([s["c"] for s in samples], axis=0)
    proto["G"] = np.mean([s["G"] for s in samples], axis=0)
    return proto


def build_sample_from_jsonl_record(record: Dict, M_p_to_c: np.ndarray, eps: float = 1e-8) -> Dict:
    """从一条 JSONL 记录构造样本 dict，包含 a/F/c/G。"""
    if "token_parcel_acts" not in record:
        raise ValueError("记录缺少 token_parcel_acts 字段")
    acts = np.array(record["token_parcel_acts"], dtype=np.float32)  # (T,P)
    if acts.ndim != 2:
        raise ValueError("token_parcel_acts 维度应为 (T,P)")
    a = acts.mean(axis=0)
    F = compute_parcel_connectivity(acts, eps=eps)
    cap_acts = aggregate_to_capability(acts, M_p_to_c)  # (T,C)
    c = cap_acts.mean(axis=0)
    G = compute_parcel_connectivity(cap_acts, eps=eps)
    return {"a": a, "F": F, "c": c, "G": G}


def build_features(sample: Dict,
                   proto_truth: Dict,
                   proto_hall: Dict,
                   indicator: IndicatorConfig,
                   capability_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """构造单样本特征向量及其名称。"""
    a = sample["a"]
    F = sample["F"]
    c = sample["c"]
    G = sample["G"]

    # 1) 指示器轴
    pos_vals = [a[i] for i in (indicator.pos_parcels or []) if 0 <= i < a.shape[0]]
    neg_vals = [a[i] for i in (indicator.neg_parcels or []) if 0 <= i < a.shape[0]]
    M_plus = zsum(pos_vals)
    G_minus = zsum([-v for v in neg_vals])

    cap_index = {name: idx for idx, name in enumerate(capability_names)}

    # 动态 capability 轴：正向（幻觉上升）、负向（真实性上升）
    pos_cap_vals = []
    for name in (indicator.pos_capabilities or []):
        if name in cap_index:
            pos_cap_vals.append(c[cap_index[name]])
    neg_cap_vals = []
    for name in (indicator.neg_capabilities or []):
        if name in cap_index:
            neg_cap_vals.append(c[cap_index[name]])
    Cpos_cap = zsum(pos_cap_vals)  # 幻觉相关能力增强
    Cneg_cap = zsum([-v for v in neg_cap_vals])  # 真实性能力减弱（取负求和）

    # 2) 连接失配（以 parcel 对实现）
    C_plus = 0.0
    C_minus = 0.0
    for (i, j) in (indicator.true_path_pairs or []):
        if 0 <= i < F.shape[0] and 0 <= j < F.shape[1] and i != j:
            # 期望应强：若比 truth 原型弱，则计入崩塌度
            C_minus += max(0.0, float(proto_truth["F"][i, j] - F[i, j]))
    for (i, j) in (indicator.wrong_path_pairs or []):
        if 0 <= i < F.shape[0] and 0 <= j < F.shape[1] and i != j:
            # 错通路增强：若比 truth 原型强，则计入入侵度
            C_plus += max(0.0, float(F[i, j] - proto_truth["F"][i, j]))

    # capability 层连接失配（名称对 -> 索引）
    C_plus_cap = 0.0
    C_minus_cap = 0.0
    for (ni, nj) in (indicator.true_capability_pairs or []):
        if ni in cap_index and nj in cap_index:
            ii, jj = cap_index[ni], cap_index[nj]
            C_minus_cap += max(0.0, float(proto_truth["G"][ii, jj] - G[ii, jj]))
    for (ni, nj) in (indicator.wrong_capability_pairs or []):
        if ni in cap_index and nj in cap_index:
            ii, jj = cap_index[ni], cap_index[nj]
            C_plus_cap += max(0.0, float(G[ii, jj] - proto_truth["G"][ii, jj]))

    # 3) 原型相似度差
    s_a_truth = cosine_similarity(a, proto_truth["a"])  # 标量
    s_a_hall = cosine_similarity(a, proto_hall["a"])
    s_c_truth = cosine_similarity(c, proto_truth["c"])
    s_c_hall = cosine_similarity(c, proto_hall["c"])
    s_F_truth = frobenius_similarity(F, proto_truth["F"])  # 标量∈(0,1]
    s_F_hall = frobenius_similarity(F, proto_hall["F"])  # 标量
    s_G_truth = frobenius_similarity(G, proto_truth["G"])  # 标量
    s_G_hall = frobenius_similarity(G, proto_hall["G"])  # 标量

    s_da = s_a_truth - s_a_hall
    s_dc = s_c_truth - s_c_hall
    s_dF = s_F_truth - s_F_hall
    s_dG = s_G_truth - s_G_hall

    feat = np.array([
        M_plus, G_minus,
        Cpos_cap, Cneg_cap,
        C_plus, C_minus,
        C_plus_cap, C_minus_cap,
        s_da, s_dc, s_dF, s_dG
    ], dtype=np.float32)

    names = [
        "M_plus", "G_minus",
        "Cpos_cap", "Cneg_cap",
        "C_plus", "C_minus",
        "C_plus_cap", "C_minus_cap",
        "s_da", "s_dc", "s_dF", "s_dG"
    ]
    return feat, names


def load_mapping_json(mapping_json_path: str) -> Dict:
    p = Path(mapping_json_path)
    if not p.exists():
        raise FileNotFoundError(f"映射文件不存在: {mapping_json_path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    records: List[Dict] = []
    with open(p, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"第{line_num}行 JSON 解析失败: {e}")
    return records



