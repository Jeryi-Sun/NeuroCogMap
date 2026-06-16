"""
通用数据加载与绘图辅助函数。

该模块集中处理：
1. 数据路径解析与加载；
2. 结果目录管理与重复运行控制；
3. Nature 风格的全局样式配置；
4. 常用数学操作（如行排名等）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

DATA_ROOT = Path(
    "/path/to/project_root/"
    "Human_LLM_align/litcoder_core/data_analysis/draw_graphs"
).resolve()
DATA_DIR = DATA_ROOT / "data4draw"
RESULT_DIR = DATA_ROOT / "draw_result"


def set_nature_style() -> None:
    """设置 Nature 期刊常用的干净绘图风格。"""
    sns.set_theme(
        context="talk",
        style="white",
        font_scale=0.9,
        rc={
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#111111",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        },
    )


def load_matrix(csv_path: Path | str) -> pd.DataFrame:
    """读取 CSV 矩阵并保证行列标签均存在。"""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到矩阵文件：{path}")
    df = pd.read_csv(path, index_col=0)
    if df.index.hasnans or df.columns.hasnans:
        raise ValueError(f"{path.name} 中存在缺失的行列标签")
    return df


def load_label_mapping(json_path: Path | str) -> dict:
    """加载 parcel id -> 功能描述 的映射。"""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到映射文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} 格式异常，期望顶层 dict")
    return data


def ensure_output_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def should_skip(output_path: Path, overwrite: bool) -> bool:
    """
    当输出已存在且未显式允许覆盖时返回 True。

    Args:
        output_path: 目标输出文件路径
        overwrite:   若为 True 则强制重绘
    """
    if output_path.exists() and not overwrite:
        print(f"[Skip] {output_path} 已存在，使用 --overwrite 可重新绘制。")
        return True
    return False


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """为各脚本注入通用 CLI 参数。"""
    parser.add_argument(
        "--prediction-matrix",
        type=Path,
        default=DATA_DIR / "prediction_matrix_gemma2_2b.csv",
        help="预测准确性矩阵 CSV 路径（默认：data4draw/prediction_matrix_gemma2_2b.csv）",
    )
    parser.add_argument(
        "--semantic-matrix",
        type=Path,
        default=DATA_DIR / "semantic_matrix_gemma2_2b.csv",
        help="功能相似度矩阵 CSV 路径（默认：data4draw/semantic_matrix_gemma2_2b.csv）",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=DATA_DIR / "gemma2_2b_parcel_id_to_function_name.json",
        help="parcel id -> 功能描述 JSON 路径（默认：data4draw/gemma2_2b_parcel_id_to_function_name.json）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定则覆盖已有结果，否则检测到文件后会跳过。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="输出图片 DPI（默认 300，兼容 Nature 排版）",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6.0,
        help="图像宽度（英寸）",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.0,
        help="图像高度（英寸）",
    )


def compute_rowwise_rank_scores(sim_matrix: pd.DataFrame, best_idx: np.ndarray) -> np.ndarray:
    """计算每个脑区的归一化排名分数 R_i。"""
    ranks = np.empty(sim_matrix.shape[0], dtype=float)
    sim_values = sim_matrix.to_numpy()
    for i, j_star in enumerate(best_idx):
        row = sim_values[i]
        # rank: 1 表示最大 (降序排名)
        order = np.argsort(-row)
        rank_position = np.where(order == j_star)[0][0] + 1
        ranks[i] = 1.0 - (rank_position - 1) / (sim_matrix.shape[1] - 1)
    return ranks


def get_argmax_indices(acc_matrix: pd.DataFrame, axis: int = 1) -> np.ndarray:
    """
    求解沿给定 axis 的 argmax。

    axis=1: 针对每个脑区（行）寻找最佳 LLM parcel。
    axis=0: 针对每个 LLM parcel（列）寻找最佳脑区。
    """
    values = acc_matrix.to_numpy()
    if axis == 1:
        return values.argmax(axis=1)
    if axis == 0:
        return values.argmax(axis=0)
    raise ValueError("axis 仅支持 0 或 1")


def flatten_pair(acc_matrix: pd.DataFrame, sim_matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """返回向量化后的 (A, S) 数组。"""
    a_vec = acc_matrix.to_numpy().ravel()
    s_vec = sim_matrix.to_numpy().ravel()
    if a_vec.shape != s_vec.shape:
        raise ValueError("A 与 S 的形状不一致，无法比较")
    return a_vec, s_vec


def reorder_by_matching(
    matrix: pd.DataFrame,
    row_order: Iterable[int],
    col_order: Iterable[int],
) -> pd.DataFrame:
    """根据给定顺序重新排列矩阵。"""
    return matrix.iloc[list(row_order), list(col_order)]


def zscore_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """对每个列（LLM parcel）执行标准化，避免列间尺度差异。"""
    values = df.to_numpy(dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=0, keepdims=True)
    std[std == 0] = 1.0
    normalized = (values - mean) / std
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


