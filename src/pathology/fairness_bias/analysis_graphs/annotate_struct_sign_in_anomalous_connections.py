#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 fairness_bias 项目下的 parcel-level anomalous_connections.json 中每条连边
增加一个「结构连接正负」标记字段，基于 parcel_connection_matrix.csv 中的 M[i, j] 符号。

参考：
- hallucination 项目的 circle_graph_edge2node.py 中对结构矩阵的读取与使用方式

行为说明：
- 在指定 root_results_dir 下递归查找所有名为 anomalous_connections.json 的文件
  （通常位于 */parcel_level/anomalous_connections.json）
- 对其中的每条 edge（pos_connections / neg_connections），根据 parcel_i.id / parcel_j.id
  查表 M[id_i, id_j]：
    >0  -> "positive"
    <0  -> "negative"
    ==0 -> "zero"
- 将标记写入新字段：
    "struct_connectivity_sign": "positive" | "negative" | "zero"
    "struct_connectivity_value": float(M[i, j])
- 结果保存为同目录下的新文件：
    anomalous_connections_with_struct_sign.json

命令行使用示例：
    python safety_explanation/fairness_bias/analysis_graphs/annotate_struct_sign_in_anomalous_connections.py \\
        --root_results_dir /path/to/project_root/safety_explanation/fairness_bias/results/analysis_output \\
        --parcel_matrix_csv /path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv \\
        --skip_if_exists
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _load_parcel_matrix(csv_path: Path) -> np.ndarray:
    """
    从 parcel_connection_matrix.csv 读取结构连接矩阵 M。

    约定格式（与 circle_graph_edge2node.py 保持一致）：
    - 首行：空单元 + 列索引 0..N-1
    - 首列：行索引 0..N-1
    - 其余为数值矩阵
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Parcel connection matrix CSV 不存在: {csv_path}")

    try:
        raw = np.genfromtxt(str(csv_path), delimiter=",")
    except Exception as e:
        print(f"[ERROR] 读取 parcel 结构矩阵失败: {csv_path} ({e})")
        raise

    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError(f"parcel 结构矩阵形状异常: {raw.shape}，期望 >= (2, 2)。")

    # 去掉首行首列索引，只保留数值矩阵
    mat = raw[1:, 1:]
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"parcel 结构矩阵不是方阵: {mat.shape}。")
    return mat.astype(float)


def _annotate_edges_for_file(
    json_path: Path,
    struct_mat: np.ndarray,
    skip_if_exists: bool,
    output_suffix: str = "_with_struct_sign",
) -> None:
    """
    对单个 anomalous_connections.json 文件进行结构正负标注。
    """
    if not json_path.exists():
        print(f"[WARN] 找不到 JSON 文件，跳过: {json_path}")
        return

    output_path = json_path.with_name(json_path.stem + f"{output_suffix}.json")

    if skip_if_exists and output_path.exists():
        print(f"[INFO] 结果已存在且 skip_if_exists=True，跳过: {output_path}")
        return

    print(f"[INFO] 处理文件: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    root = data.get("anomalous_connections", {})
    pos_list: List[Dict[str, Any]] = root.get("pos_connections", [])
    neg_list: List[Dict[str, Any]] = root.get("neg_connections", [])

    n_nodes = struct_mat.shape[0]

    def annotate_list(edges: List[Dict[str, Any]], list_name: str) -> None:
        for idx, item in enumerate(edges):
            try:
                node_i = item["parcel_i"]
                node_j = item["parcel_j"]
                nid_i = int(node_i["id"])
                nid_j = int(node_j["id"])
            except Exception as e:
                print(
                    f"[WARN] {json_path.name} 中 {list_name}[{idx}] 解析 parcel_i/parcel_j 失败: {e}"
                )
                continue

            if not (0 <= nid_i < n_nodes and 0 <= nid_j < n_nodes):
                raise ValueError(
                    f"在 {json_path} 中发现越界的 parcel id: "
                    f"parcel_i.id={nid_i}, parcel_j.id={nid_j}, "
                    f"但结构矩阵尺寸为 {struct_mat.shape}。"
                )

            m_val = float(struct_mat[nid_i, nid_j])
            if m_val > 0:
                sign = "positive"
            elif m_val < 0:
                sign = "negative"
            else:
                sign = "zero"

            item["struct_connectivity_sign"] = sign
            item["struct_connectivity_value"] = m_val

    annotate_list(pos_list, "pos_connections")
    annotate_list(neg_list, "neg_connections")

    # 写回新的 JSON 文件
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 已写出带结构正负标记的文件: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "为 fairness_bias/results/analysis_output 下的 anomalous_connections.json "
            "增加结构连接正负标记（基于 parcel_connection_matrix.csv）。"
        )
    )
    parser.add_argument(
        "--root_results_dir",
        type=str,
        default=(
            "/path/to/project_root/"
            "safety_explanation/fairness_bias/results/analysis_output"
        ),
        help="fairness_bias 分析结果根目录（递归查找 anomalous_connections.json）。",
    )
    parser.add_argument(
        "--parcel_matrix_csv",
        type=str,
        default=(
            "/path/to/project_root/"
            "neural_area/global_weight/outputs/parcel_connection_matrix.csv"
        ),
        help="Parcel-level 结构连接矩阵 CSV 路径（与 hallucination 绘图脚本保持一致）。",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_with_struct_sign",
        help=(
            "输出文件名后缀。例如原始为 anomalous_connections.json，"
            "若 suffix='_with_struct_sign'，则输出为 anomalous_connections_with_struct_sign.json。"
        ),
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="若目标输出文件已存在，则跳过该 JSON 的处理。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_results_dir).resolve()
    csv_path = Path(args.parcel_matrix_csv).resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"root_results_dir 不存在: {root_dir}")

    struct_mat = _load_parcel_matrix(csv_path)
    print(f"[INFO] 已加载结构矩阵，形状: {struct_mat.shape}，来源: {csv_path}")

    # 递归查找所有 anomalous_connections.json
    json_files = sorted(root_dir.rglob("anomalous_connections.json"))
    if not json_files:
        print(f"[WARN] 在目录 {root_dir} 下未找到任何 anomalous_connections.json 文件。")
        return

    print(f"[INFO] 共检测到 {len(json_files)} 个 anomalous_connections.json，将逐一处理。")

    for jp in json_files:
        _annotate_edges_for_file(
            json_path=jp,
            struct_mat=struct_mat,
            skip_if_exists=bool(args.skip_if_exists),
            output_suffix=str(args.output_suffix),
        )

    print("[INFO] 所有 JSON 文件的结构正负标记处理完成。")


if __name__ == "__main__":
    main()

