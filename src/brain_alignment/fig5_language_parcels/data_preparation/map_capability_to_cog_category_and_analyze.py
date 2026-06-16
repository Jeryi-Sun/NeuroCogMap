#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能: 使用 capability→认知层级(category) 的映射, 将
`Human_Parcel × Capability` 的矩阵转换为
`Human_Parcel × 认知层级(category)` 的矩阵, 并做归一化与统计分析。

输入:
- capability 级矩阵 CSV:
    行: Human_Parcel_1, Human_Parcel_2, ...
    列: 各个 LLM capability 名称 (与 capability_cog_mapping_flat.json 对应)
- capability_cog_mapping_flat.json:
    每个 capability 对应一个认知类别:
    {
      "Causal Reasoning Capability": {
        "category": "C1",
        "category_name": "...",
        "level": "...",
        ...
      },
      ...
    }
- parcel_descriptions.json:
    每个 parcel 的 7Networks 名称, 用于解析左右脑和 Yeo7 网络:
    {
      "parcel_id": 1,
      "parcel_name": "7Networks_LH_Vis_1",
      ...
    }

输出:
1) Human Parcel × 认知层级(category) 的矩阵 (已按类别内能力数目做平均):
   - 行: Human_Parcel_x
   - 列: 各个 category (如 A1, B1, C1, ...)
2) 按左右脑 + Yeo7 网络聚合后的认知层级均值矩阵:
   - 行: "LH_Vis", "RH_Default", 等 (hemisphere_network)
   - 列: 各个 category
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


def _normalize_capability_name(name: str) -> str:
    """
    统一 capability 名称格式, 以便在 CSV 列名和 JSON key 之间做鲁棒匹配。
    - 小写
    - 去掉首尾空格
    - 合并中间多余空白
    """
    if not isinstance(name, str):
        return ""
    return " ".join(name.strip().lower().split())


def load_capability_category_mapping(
    mapping_file: str,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    加载 capability→category 映射。

    Returns:
        cap_to_cat: 规范化 capability 名称 → category (如 "a1", "b1")
        cat_cap_counts: category → 该类别下能力数量
    """
    logger.info(f"加载 capability→认知层级(category) 映射: {mapping_file}")
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"capability 认知层级映射文件不存在: {mapping_file}")

    try:
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载 capability cogn mapping 失败: {e}")
        raise

    if not isinstance(data, dict):
        raise ValueError(f"capability cogn mapping 文件必须是 dict, 得到: {type(data)}")

    cap_to_cat: Dict[str, str] = {}
    cat_cap_counts: Dict[str, int] = {}

    for raw_cap_name, info in data.items():
        if not isinstance(info, dict):
            logger.warning(f"capability 映射项格式异常, 跳过: {raw_cap_name}")
            continue
        category = info.get("category")
        if category is None:
            logger.warning(f"capability 缺少 category 字段, 跳过: {raw_cap_name}")
            continue
        norm_cap = _normalize_capability_name(raw_cap_name)
        norm_cat = str(category).strip()
        if not norm_cat:
            logger.warning(f"capability 的 category 为空字符串, 跳过: {raw_cap_name}")
            continue
        cap_to_cat[norm_cap] = norm_cat
        cat_cap_counts[norm_cat] = cat_cap_counts.get(norm_cat, 0) + 1

    logger.info(f"有效 capability 数量: {len(cap_to_cat)}")
    logger.info(f"category 分布: {cat_cap_counts}")
    return cap_to_cat, cat_cap_counts


def load_capability_matrix(capability_matrix_file: str) -> pd.DataFrame:
    """加载 Human_Parcel × Capability 矩阵。"""
    logger.info(f"加载 capability 矩阵: {capability_matrix_file}")
    if not os.path.exists(capability_matrix_file):
        raise FileNotFoundError(f"capability 矩阵文件不存在: {capability_matrix_file}")

    try:
        df = pd.read_csv(capability_matrix_file, index_col=0)
    except Exception as e:
        logger.error(f"加载 capability 矩阵失败: {e}")
        raise

    if df.shape[1] == 0:
        raise ValueError(f"capability 矩阵没有任何列: {capability_matrix_file}")

    logger.info(f"capability 矩阵形状: {df.shape}")
    logger.info(f"前 3 行索引: {df.index[:3].tolist()}")
    logger.info(f"前 3 列名: {df.columns[:3].tolist()}")
    return df


def map_capability_to_categories(
    df_cap: pd.DataFrame,
    cap_to_cat: Dict[str, str],
    cat_cap_counts_from_mapping: Dict[str, int],
    use_first_letter_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    将 capability 级矩阵聚合为 category 级矩阵。

    对每个 category:
      对该类别下所有 capability 的数值按列求平均:
      cat_value(parcel, category) = mean_{cap ∈ category}( value(parcel, cap) )

    如果 use_first_letter_only=True:
      在细分类别聚合后, 再按第一个字母 (A, B, C, D) 做一次聚合,
      例如 A1, A2 等合并到 A; B1, B2 等合并到 B。

    注意:
    - 如果某个 category 在当前矩阵中没有任何对应的列, 会输出 0 并打印 warning。
    - 归一化时使用的“能力数量”是 **实际在矩阵中出现的 capability 数量**,
      而不是 mapping 中理论上的数量, 防止 mismatch 时出现偏差。

    Returns:
        df_cat: Human_Parcel × category 的 DataFrame
        cat_actual_counts: category → 在矩阵中实际使用到的 capability 数量
    """
    logger.info("开始将 capability 聚合到认知层级(category)...")

    # 为 CSV 中的 capability 列建立 norm 名称
    col_norm_map: Dict[str, str] = {
        col: _normalize_capability_name(col) for col in df_cap.columns
    }

    # category → 对应的列名列表
    cat_cols: Dict[str, List[str]] = {}
    unmapped_caps: List[str] = []

    for col, norm_col in col_norm_map.items():
        cat = cap_to_cat.get(norm_col)
        if cat is None:
            unmapped_caps.append(col)
            continue
        cat_cols.setdefault(cat, []).append(col)

    if unmapped_caps:
        logger.warning(
            f"有 {len(unmapped_caps)} 个 capability 在认知层级映射中找不到对应项, "
            f"示例: {unmapped_caps[:10]}"
        )

    categories = sorted(set(cat_cols.keys()) | set(cat_cap_counts_from_mapping.keys()))
    logger.info(f"最终 category 列表: {categories}")

    cat_actual_counts: Dict[str, int] = {}
    cat_matrices: Dict[str, pd.Series] = {}

    for cat in categories:
        cols = cat_cols.get(cat, [])
        if not cols:
            logger.warning(
                f"category {cat} 在当前 capability 矩阵中没有任何对应列, 将输出 0"
            )
            # 输出全 0 列
            cat_matrices[cat] = pd.Series(
                np.zeros(len(df_cap), dtype=np.float32), index=df_cap.index
            )
            cat_actual_counts[cat] = 0
            continue

        # 对该 category 下的所有 capability 按列求平均
        sub_df = df_cap[cols]
        cat_matrices[cat] = sub_df.mean(axis=1)
        cat_actual_counts[cat] = len(cols)
        logger.info(
            f"category {cat}: 使用 {len(cols)} 个 capability 计算平均, "
            f"mapping 中该类理论 capability 数量为 {cat_cap_counts_from_mapping.get(cat, 0)}"
        )

    # 拼成 DataFrame
    df_cat = pd.DataFrame(cat_matrices, index=df_cap.index)
    # 按列名排序
    df_cat = df_cat.reindex(columns=sorted(df_cat.columns))

    # 如果指定只按第一个字母聚类, 则再做一次聚合
    if use_first_letter_only:
        logger.info("按第一个字母 (A, B, C, D) 进一步聚合 category...")
        letter_to_cols: Dict[str, List[str]] = {}
        for col in df_cat.columns:
            if col and isinstance(col, str):
                first_letter = col[0].upper()
                if first_letter.isalpha():
                    letter_to_cols.setdefault(first_letter, []).append(col)
        
        letter_matrices: Dict[str, pd.Series] = {}
        letter_actual_counts: Dict[str, int] = {}
        
        for letter, cols in letter_to_cols.items():
            if not cols:
                continue
            # 对属于同一字母的 category 列求平均
            sub_df = df_cat[cols]
            letter_matrices[letter] = sub_df.mean(axis=1)
            # 统计该字母下实际使用的 capability 数量 (累加)
            letter_actual_counts[letter] = sum(cat_actual_counts.get(col, 0) for col in cols)
            logger.info(
                f"字母 {letter}: 聚合了 {len(cols)} 个 category ({cols}), "
                f"共使用 {letter_actual_counts[letter]} 个 capability"
            )
        
        df_cat = pd.DataFrame(letter_matrices, index=df_cap.index)
        df_cat = df_cat.reindex(columns=sorted(df_cat.columns))
        cat_actual_counts = letter_actual_counts

    return df_cat, cat_actual_counts


def row_normalize_by_parcel(
    df: pd.DataFrame,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """
    在 Human Parcel 维度上对每一行做归一化, 使得每个 Human Parcel 上所有
    维度(例如 LLM Parcel / capability / category) 的取值被压缩到 [0, 1] 区间。

    这里采用按行的最大最小归一化:
        v_norm = (v - min(v)) / (max(v) - min(v) + epsilon)

    注意:
    - 如果一行所有元素都相同, 则分母接近 0, 此时会得到全 0 行 (因为 v - min(v) 为 0)。
    """
    if df.shape[0] == 0 or df.shape[1] == 0:
        return df

    values = df.values.astype(float)
    row_min = np.min(values, axis=1, keepdims=True)
    row_max = np.max(values, axis=1, keepdims=True)
    denom = row_max - row_min
    denom = denom + epsilon
    normalized = (values - row_min) / denom

    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def add_row_and_col_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    在矩阵上添加行平均和列平均。
    
    添加:
    - 一列 "Row_Mean": 每行的平均值
    - 一行 "Col_Mean": 每列的平均值
    - 右下角元素: 整个矩阵的平均值 (在 "Col_Mean" 行的 "Row_Mean" 列)
    
    Args:
        df: 输入的 DataFrame
        
    Returns:
        添加了行平均和列平均的 DataFrame
    """
    if df.shape[0] == 0 or df.shape[1] == 0:
        return df
    
    df_with_means = df.copy()
    # 为防止后续添加字符串索引行导致索引类型混合, 统一将现有索引转为字符串
    df_with_means.index = df_with_means.index.map(str)
    
    # 添加行平均列 (对原始矩阵的每一行计算列的平均值)
    row_means = df.mean(axis=1)
    df_with_means["Row_Mean"] = row_means
    
    # 添加列平均行 (对原始矩阵的每一列计算行的平均值)
    col_means = df.mean(axis=0)
    col_means_series = pd.Series(col_means, name="Col_Mean")
    # "Row_Mean" 列的平均值 = 所有行平均值的平均值 = 整个矩阵的平均值
    overall_mean = df.values.mean()
    col_means_series["Row_Mean"] = overall_mean
    
    # 将列平均行添加到 DataFrame
    df_with_means = pd.concat([df_with_means, col_means_series.to_frame().T])
    
    return df_with_means


def load_parcel_metadata(parcel_desc_file: str) -> pd.DataFrame:
    """
    加载 parcel 描述, 提取 hemisphere (LH/RH) 和 Yeo7 network。

    Returns:
        DataFrame, index 为 parcel_id, 列包括:
        - parcel_name
        - hemisphere: "LH"/"RH"/"Unknown"
        - network: 如 "Vis", "Default" 等
    """
    logger.info(f"加载 parcel 描述: {parcel_desc_file}")
    if not os.path.exists(parcel_desc_file):
        raise FileNotFoundError(f"parcel 描述文件不存在: {parcel_desc_file}")

    try:
        with open(parcel_desc_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"加载 parcel 描述失败: {e}")
        raise

    if not isinstance(data, list):
        raise ValueError(f"parcel 描述文件应该是 list, 得到: {type(data)}")

    records = []
    for item in data:
        parcel_id = item.get("parcel_id")
        parcel_name = item.get("parcel_name", "")
        if parcel_id is None:
            logger.warning(f"跳过没有 parcel_id 的记录: {item}")
            continue

        hemisphere = "Unknown"
        network = "Unknown"
        if isinstance(parcel_name, str):
            parts = parcel_name.split("_")
            # 典型格式: 7Networks_LH_Vis_1
            if len(parts) >= 3:
                hemi_candidate = parts[1]
                net_candidate = parts[2]
                if hemi_candidate in ("LH", "RH"):
                    hemisphere = hemi_candidate
                network = net_candidate

        records.append(
            {
                "parcel_id": parcel_id,
                "parcel_name": parcel_name,
                "hemisphere": hemisphere,
                "network": network,
            }
        )

    df_meta = pd.DataFrame(records).set_index("parcel_id")
    logger.info(f"parcel 描述数据量: {df_meta.shape}")
    return df_meta


def attach_parcel_meta_to_category_matrix(
    df_cat: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    将 parcel 的左右脑 & 网络信息附着到 Human_Parcel × category 矩阵上。

    df_cat 的 index 形如 "Human_Parcel_1", "Human_Parcel_2", ...
    对应 parcel_id 为最后的数字。
    """
    parcel_ids: List[int] = []
    for idx in df_cat.index:
        try:
            # 形如 "Human_Parcel_1"
            pid = int(str(idx).split("_")[-1])
        except (ValueError, IndexError) as e:
            logger.warning(f"无法从行名解析 parcel_id: {idx}, 错误: {e}")
            parcel_ids.append(None)
            continue
        parcel_ids.append(pid)

    df_cat_with_meta = df_cat.copy()
    df_cat_with_meta["parcel_id"] = parcel_ids

    # 合并元信息
    df_cat_with_meta = df_cat_with_meta.merge(
        df_meta,
        how="left",
        left_on="parcel_id",
        right_index=True,
    )

    missing_meta = df_cat_with_meta["parcel_name"].isna().sum()
    if missing_meta > 0:
        logger.warning(f"有 {missing_meta} 个 parcel 无法在描述文件中找到元信息")

    # 方便后续 groupby
    df_cat_with_meta["hemi_net"] = (
        df_cat_with_meta["hemisphere"].fillna("Unknown")
        + "_"
        + df_cat_with_meta["network"].fillna("Unknown")
    )

    return df_cat_with_meta


def compute_yeo7_category_distribution(
    df_cat_with_meta: pd.DataFrame,
    category_cols: List[str],
    ignore_hemisphere: bool = False,
) -> pd.DataFrame:
    """
    按 hemisphere + Yeo7 网络聚合, 计算各个认知层级(category) 的平均值。

    Args:
        ignore_hemisphere: 如果为 True, 则不区分左右脑, 只按 Yeo7 网络名称聚合
            (例如 LH_Vis 和 RH_Vis 合并为 Vis)

    Returns:
        DataFrame:
            index: "LH_Vis", "RH_Default", ... (如果 ignore_hemisphere=False)
                   或 "Vis", "Default", ... (如果 ignore_hemisphere=True)
            columns: category 列
    """
    if ignore_hemisphere:
        # 只按 network 分组, 不区分左右脑
        valid_mask = df_cat_with_meta["network"].notna()
        df_valid = df_cat_with_meta.loc[valid_mask].copy()
        grouped = df_valid.groupby("network")[category_cols].mean()
        logger.info(f"Yeo7 (network only, 不区分左右脑) × category 聚合结果形状: {grouped.shape}")
    else:
        # 按 hemi_net (左右脑 + 网络) 分组
        valid_mask = df_cat_with_meta["hemi_net"].notna()
        df_valid = df_cat_with_meta.loc[valid_mask].copy()
        grouped = df_valid.groupby("hemi_net")[category_cols].mean()
        logger.info(f"Yeo7 (hemi_net) × category 聚合结果形状: {grouped.shape}")
    
    return grouped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "将 capability 级矩阵映射到认知层级(category), "
            "生成 Human Parcel × category 的矩阵, 并按左右脑和 Yeo7 网络做统计"
        )
    )

    parser.add_argument(
        "--capability_matrix_file",
        type=str,
        default=(
            "/path/to/project_root/"
            "Human_LLM_align/litcoder_core/data_analysis/draw_graphs/"
            "data4draw/adventuresinsayingyes/prediction_matrix_gemma2_2b_capability.csv"
        ),
        help="Human Parcel × Capability 的输入矩阵 CSV 文件路径",
    )
    parser.add_argument(
        "--capability_cog_mapping_file",
        type=str,
        default=(
            "/path/to/project_root/"
            "capability_analysis/data/capability_cog_mapping_flat.json"
        ),
        help="capability → 认知层级(category) 映射 JSON 文件路径",
    )
    parser.add_argument(
        "--parcel_desc_file",
        type=str,
        default=(
            "/path/to/project_root/"
            "Human_LLM_align/litcoder_core/dataset/brain_parcel_description/"
            "parcel_descriptions.json"
        ),
        help="Human Parcel 描述文件 (含 7Networks parcel_name) 路径",
    )
    parser.add_argument(
        "--output_category_matrix_file",
        type=str,
        default=(
            "/path/to/project_root/"
            "Human_LLM_align/litcoder_core/data_analysis/draw_graphs/"
            "data4draw/adventuresinsayingyes/prediction_matrix_gemma2_2b_cog_category.csv"
        ),
        help="输出的 Human Parcel × category 矩阵 CSV 路径",
    )
    parser.add_argument(
        "--output_yeo7_summary_file",
        type=str,
        default=(
            "/path/to/project_root/"
            "Human_LLM_align/litcoder_core/data_analysis/draw_graphs/"
            "data4draw/adventuresinsayingyes/yeo7_cog_category_distribution.csv"
        ),
        help="输出的 (左右脑+Yeo7 网络) × category 聚合结果 CSV 路径",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="如果输出文件已存在, 则跳过计算 (两个输出文件都存在时才跳过)",
    )
    parser.add_argument(
        "--row_norm_epsilon",
        type=float,
        default=1e-8,
        help="Human Parcel 维度行归一化时使用的 epsilon, 避免除以 0",
    )
    parser.add_argument(
        "--use_first_letter_only",
        action="store_true",
        help="如果指定, 则只按 category 的第一个字母 (A, B, C, D) 聚类, "
        "而不是使用完整的细分类别 (A1, B1, B2, C1, C2, D1, D2, D3, D4)",
    )
    parser.add_argument(
        "--ignore_hemisphere",
        action="store_true",
        help="如果指定, 则不区分左右脑, 只按 Yeo7 网络名称聚合 "
        "(例如 LH_Vis 和 RH_Vis 合并为 Vis)",
    )

    args = parser.parse_args()

    out_cat_path = Path(args.output_category_matrix_file)
    out_yeo_path = Path(args.output_yeo7_summary_file)

    if args.skip_if_exists and out_cat_path.exists() and out_yeo_path.exists():
        logger.info(
            "两个输出文件均已存在且 --skip_if_exists 为 True, 跳过全部计算"
        )
        return

    try:
        # 1. 加载输入数据
        df_cap = load_capability_matrix(args.capability_matrix_file)
        cap_to_cat, cat_cap_counts_mapping = load_capability_category_mapping(
            args.capability_cog_mapping_file
        )

        # 2. capability → category 聚合 & 归一化 (按类别内能力数量做平均)
        df_cat, cat_actual_counts = map_capability_to_categories(
            df_cap, cap_to_cat, cat_cap_counts_mapping,
            use_first_letter_only=args.use_first_letter_only
        )

        logger.info("各 category 在当前矩阵中实际使用到的 capability 数量: "
                    f"{cat_actual_counts}")

        # 3. 在 Human Parcel 维度上对 category 矩阵做行归一化
        logger.info("在 Human Parcel 维度上对 category 矩阵做行归一化(按 L1 范数)...")
        #df_cat = row_normalize_by_parcel(df_cat, epsilon=args.row_norm_epsilon)

        # 4. 保存原始矩阵用于后续处理
        df_cat_original = df_cat.copy()

        # 5. 添加行平均和列平均
        logger.info("为 Human Parcel × category 矩阵添加行平均和列平均...")
        df_cat_with_means = add_row_and_col_means(df_cat)

        # 6. 保存 Human Parcel × category 矩阵 (带行平均和列平均)
        logger.info(f"保存 Human Parcel × category 矩阵到: {out_cat_path}")
        out_cat_path.parent.mkdir(parents=True, exist_ok=True)
        df_cat_with_means.to_csv(out_cat_path)

        # 7. 加载 parcel 元信息, 附加左右脑 & Yeo7 网络
        # 使用原始的 df_cat (没有行平均和列平均), 因为 attach_parcel_meta_to_category_matrix
        # 需要从行名解析 parcel_id
        df_meta = load_parcel_metadata(args.parcel_desc_file)
        df_cat_with_meta = attach_parcel_meta_to_category_matrix(df_cat_original, df_meta)

        # 8. 按左右脑 + Yeo7 网络聚合
        category_cols = sorted(df_cat_original.columns.tolist())
        df_yeo = compute_yeo7_category_distribution(
            df_cat_with_meta, category_cols,
            ignore_hemisphere=args.ignore_hemisphere
        )
        #df_yeo = row_normalize_by_parcel(df_yeo, epsilon=args.row_norm_epsilon)
        
        # 9. 添加行平均和列平均
        logger.info("为 Yeo7 × category 矩阵添加行平均和列平均...")
        df_yeo = add_row_and_col_means(df_yeo)
        
        # 10. 保存 Yeo7 聚合结果
        if args.ignore_hemisphere:
            logger.info(f"保存 (Yeo7 网络, 不区分左右脑) × category 聚合结果到: {out_yeo_path}")
        else:
            logger.info(f"保存 (左右脑+Yeo7 网络) × category 聚合结果到: {out_yeo_path}")
        out_yeo_path.parent.mkdir(parents=True, exist_ok=True)
        df_yeo.to_csv(out_yeo_path)

        # 10. 简要打印一些统计信息
        logger.info("==== 示例: 前 5 个 Human Parcel 的认知层级分布 ====")
        logger.info("\n%s", df_cat.head().to_string())

        if args.ignore_hemisphere:
            logger.info("==== 示例: 前 10 个 Yeo7 网络(不区分左右脑) 的认知层级分布 ====")
        else:
            logger.info("==== 示例: 前 10 个 Yeo7(hemi_net) 的认知层级分布 ====")
        logger.info("\n%s", df_yeo.head(10).to_string())

    except Exception as e:
        # 按用户要求, 不静默吞掉异常, 至少打印出来
        logger.error(f"capability → 认知层级分析过程中发生异常: {e}")
        raise


if __name__ == "__main__":
    main()


