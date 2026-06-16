#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量分析 parcel feature 与预测指标（nll/aic）的关系。

功能：
1) 在 test 集上，计算每个 feature 与 nll/aic 的 Pearson 相关系数（PCC）并排序；
2) 在 train 集上训练线性模型，并在 test 集上评估相关性：
   - 单特征一元线性回归（每个 feature 一个模型）；
   - 多特征多元线性回归（每个目标一个模型，使用全部特征）。

自动匹配规则：
- activations 文件名：{dataset_key}_reformatted_parcel_activations.json
- predictions 文件名：llm_prediction_{model_tag}_{dataset_key}_filtered.csv
其中 {dataset_key} 通过文件名自动提取并匹配。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 parcel feature 与 nll/aic 的关系")
    parser.add_argument(
        "--activations-root",
        type=str,
        required=True,
        help="activations 根目录，需包含 train/ 和 test/ 子目录",
    )
    parser.add_argument(
        "--predictions-root",
        type=str,
        required=True,
        help="predictions 根目录，需包含 train/ 和 test/ 子目录",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="输出目录（每个数据集会生成独立子目录）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若目标数据集输出已存在则跳过",
    )
    return parser.parse_args()


def normalize_participant_id(value) -> str:
    text = str(value).strip()
    if text == "":
        raise ValueError("participant id 为空")
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def safe_pearsonr(x: np.ndarray, y: np.ndarray, pair_name: str) -> float:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{pair_name} 长度不一致: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[0] < 2:
        print(f"[WARN] {pair_name} 样本数小于2，无法计算PCC")
        return np.nan
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        print(f"[WARN] {pair_name} 存在零方差，无法计算PCC")
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def extract_dataset_key(activation_file: Path) -> str:
    suffix = "_reformatted_parcel_activations.json"
    if not activation_file.name.endswith(suffix):
        raise ValueError(f"无法解析数据集key: {activation_file.name}")
    return activation_file.name[: -len(suffix)]


def find_prediction_file(pred_dir: Path, dataset_key: str) -> Optional[Path]:
    candidates = sorted(pred_dir.glob(f"llm_prediction_*_{dataset_key}_filtered.csv"))
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        print(
            f"[WARN] {pred_dir} 下 {dataset_key} 匹配到多个预测文件，使用第一个: {candidates[0].name}"
        )
    return candidates[0]


def load_activations(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    result: Dict[str, np.ndarray] = {}
    for pid, feats in raw.items():
        pid_norm = normalize_participant_id(pid)
        arr = np.asarray(feats, dtype=np.float64)
        result[pid_norm] = arr
    return result


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"participant", "nll", "aic"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    df = df.copy()
    df["participant"] = df["participant"].apply(normalize_participant_id)
    return df


def build_aligned_matrices(
    activations: Dict[str, np.ndarray], predictions: pd.DataFrame, split_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    pred_map = predictions.set_index("participant")
    common_ids = sorted(set(activations.keys()) & set(pred_map.index.tolist()))
    if not common_ids:
        raise ValueError(f"{split_name} 无交集参与者，无法分析")
    x_list: List[np.ndarray] = []
    nll_list: List[float] = []
    aic_list: List[float] = []
    for pid in common_ids:
        x_list.append(activations[pid])
        row = pred_map.loc[pid]
        nll_list.append(float(row["nll"]))
        aic_list.append(float(row["aic"]))
    x = np.vstack(x_list)
    nll = np.asarray(nll_list, dtype=np.float64)
    aic = np.asarray(aic_list, dtype=np.float64)
    return x, nll, aic, common_ids


def fit_simple_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # 闭式解：y = w*x + b
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0:
        raise ValueError("训练特征方差为0，无法拟合一元线性回归")
    w = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    b = y_mean - w * x_mean
    return w, b


def predict_simple_linear(x: np.ndarray, w: float, b: float) -> np.ndarray:
    return w * x + b


def fit_multivariate_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    # 最小二乘闭式解：带偏置项
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    x_aug = np.hstack([x, ones])
    coeffs, residuals, rank, singular_vals = np.linalg.lstsq(x_aug, y, rcond=None)
    if rank < x_aug.shape[1]:
        print("[WARN] 多元线性回归设计矩阵秩不足，解可能不唯一")
    if np.isnan(coeffs).any():
        raise ValueError("多元线性回归出现NaN系数")
    w = coeffs[:-1]
    b = float(coeffs[-1])
    _ = residuals, singular_vals
    return w, b


def predict_multivariate_linear(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return x @ w + b


def analyze_single_dataset(
    dataset_key: str,
    act_train_file: Path,
    act_test_file: Path,
    pred_train_file: Path,
    pred_test_file: Path,
    output_dir: Path,
) -> Dict[str, float]:
    print(f"[INFO] 开始分析: {dataset_key}")
    act_train = load_activations(act_train_file)
    act_test = load_activations(act_test_file)
    pred_train = load_predictions(pred_train_file)
    pred_test = load_predictions(pred_test_file)

    x_train, y_train_nll, y_train_aic, train_ids = build_aligned_matrices(
        act_train, pred_train, "train"
    )
    x_test, y_test_nll, y_test_aic, test_ids = build_aligned_matrices(
        act_test, pred_test, "test"
    )

    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(f"{dataset_key} train/test 特征维度不一致")

    num_features = x_train.shape[1]
    print(
        f"[INFO] {dataset_key}: train={x_train.shape[0]}人, test={x_test.shape[0]}人, features={num_features}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 在 train / test 上，feature 与 nll/aic 的直接 PCC（不经过任何映射/训练）
    pcc_rows = []
    for idx in range(num_features):
        x_col = x_test[:, idx]
        pcc_nll = safe_pearsonr(x_col, y_test_nll, f"{dataset_key}/feature_{idx}/test_vs_nll")
        pcc_aic = safe_pearsonr(x_col, y_test_aic, f"{dataset_key}/feature_{idx}/test_vs_aic")
        pcc_rows.append(
            {
                "feature_idx": idx,
                "pcc_with_test_nll": pcc_nll,
                "abs_pcc_with_test_nll": np.abs(pcc_nll) if not np.isnan(pcc_nll) else np.nan,
                "pcc_with_test_aic": pcc_aic,
                "abs_pcc_with_test_aic": np.abs(pcc_aic) if not np.isnan(pcc_aic) else np.nan,
            }
        )
    pcc_df = pd.DataFrame(pcc_rows)
    pcc_df.sort_values(by="abs_pcc_with_test_nll", ascending=False).to_csv(
        output_dir / "step1_test_feature_pcc_ranked_by_nll.csv", index=False
    )
    pcc_df.sort_values(by="abs_pcc_with_test_aic", ascending=False).to_csv(
        output_dir / "step1_test_feature_pcc_ranked_by_aic.csv", index=False
    )

    pcc_train_rows = []
    for idx in range(num_features):
        x_col = x_train[:, idx]
        pcc_nll = safe_pearsonr(x_col, y_train_nll, f"{dataset_key}/feature_{idx}/train_vs_nll")
        pcc_aic = safe_pearsonr(x_col, y_train_aic, f"{dataset_key}/feature_{idx}/train_vs_aic")
        pcc_train_rows.append(
            {
                "feature_idx": idx,
                "pcc_with_train_nll": pcc_nll,
                "abs_pcc_with_train_nll": np.abs(pcc_nll) if not np.isnan(pcc_nll) else np.nan,
                "pcc_with_train_aic": pcc_aic,
                "abs_pcc_with_train_aic": np.abs(pcc_aic) if not np.isnan(pcc_aic) else np.nan,
            }
        )
    pcc_train_df = pd.DataFrame(pcc_train_rows)
    pcc_train_df.sort_values(by="abs_pcc_with_train_nll", ascending=False).to_csv(
        output_dir / "step1_train_feature_pcc_ranked_by_nll.csv", index=False
    )
    pcc_train_df.sort_values(by="abs_pcc_with_train_aic", ascending=False).to_csv(
        output_dir / "step1_train_feature_pcc_ranked_by_aic.csv", index=False
    )

    # Step 2A: 单特征模型 train->test
    uni_rows = []
    for idx in range(num_features):
        x_train_col = x_train[:, idx]
        x_test_col = x_test[:, idx]

        if np.std(x_train_col) == 0:
            print(
                f"[WARN] {dataset_key}/feature_{idx} 在train中零方差，"
                "跳过一元线性训练并写入NaN"
            )
            uni_rows.append(
                {
                    "feature_idx": idx,
                    "train_coef_for_nll": np.nan,
                    "train_intercept_for_nll": np.nan,
                    "test_pred_pcc_with_nll": np.nan,
                    "abs_test_pred_pcc_with_nll": np.nan,
                    "train_coef_for_aic": np.nan,
                    "train_intercept_for_aic": np.nan,
                    "test_pred_pcc_with_aic": np.nan,
                    "abs_test_pred_pcc_with_aic": np.nan,
                }
            )
            continue

        w_nll, b_nll = fit_simple_linear(x_train_col, y_train_nll)
        pred_nll = predict_simple_linear(x_test_col, w_nll, b_nll)
        test_pcc_nll = safe_pearsonr(
            pred_nll, y_test_nll, f"{dataset_key}/feature_{idx}/uni_model_pred_vs_test_nll"
        )

        w_aic, b_aic = fit_simple_linear(x_train_col, y_train_aic)
        pred_aic = predict_simple_linear(x_test_col, w_aic, b_aic)
        test_pcc_aic = safe_pearsonr(
            pred_aic, y_test_aic, f"{dataset_key}/feature_{idx}/uni_model_pred_vs_test_aic"
        )

        uni_rows.append(
            {
                "feature_idx": idx,
                "train_coef_for_nll": w_nll,
                "train_intercept_for_nll": b_nll,
                "test_pred_pcc_with_nll": test_pcc_nll,
                "abs_test_pred_pcc_with_nll": np.abs(test_pcc_nll)
                if not np.isnan(test_pcc_nll)
                else np.nan,
                "train_coef_for_aic": w_aic,
                "train_intercept_for_aic": b_aic,
                "test_pred_pcc_with_aic": test_pcc_aic,
                "abs_test_pred_pcc_with_aic": np.abs(test_pcc_aic)
                if not np.isnan(test_pcc_aic)
                else np.nan,
            }
        )
    uni_df = pd.DataFrame(uni_rows)
    uni_df.sort_values(by="abs_test_pred_pcc_with_nll", ascending=False).to_csv(
        output_dir / "step2A_univariate_train_test_ranked_by_nll.csv", index=False
    )
    uni_df.sort_values(by="abs_test_pred_pcc_with_aic", ascending=False).to_csv(
        output_dir / "step2A_univariate_train_test_ranked_by_aic.csv", index=False
    )

    # Step 2B: 多特征模型 train->test
    w_nll_multi, b_nll_multi = fit_multivariate_linear(x_train, y_train_nll)
    pred_nll_multi = predict_multivariate_linear(x_test, w_nll_multi, b_nll_multi)
    multi_pcc_nll = safe_pearsonr(
        pred_nll_multi, y_test_nll, f"{dataset_key}/multi_model_pred_vs_test_nll"
    )

    w_aic_multi, b_aic_multi = fit_multivariate_linear(x_train, y_train_aic)
    pred_aic_multi = predict_multivariate_linear(x_test, w_aic_multi, b_aic_multi)
    multi_pcc_aic = safe_pearsonr(
        pred_aic_multi, y_test_aic, f"{dataset_key}/multi_model_pred_vs_test_aic"
    )

    coef_df = pd.DataFrame(
        {
            "feature_idx": np.arange(num_features, dtype=int),
            "coef_nll": w_nll_multi,
            "abs_coef_nll": np.abs(w_nll_multi),
            "coef_aic": w_aic_multi,
            "abs_coef_aic": np.abs(w_aic_multi),
        }
    )
    coef_df.sort_values(by="abs_coef_nll", ascending=False).to_csv(
        output_dir / "step2B_multivariate_coeff_ranked_by_nll_coef_abs.csv", index=False
    )
    coef_df.sort_values(by="abs_coef_aic", ascending=False).to_csv(
        output_dir / "step2B_multivariate_coeff_ranked_by_aic_coef_abs.csv", index=False
    )

    with (output_dir / "meta_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_key": dataset_key,
                "num_train_participants": int(x_train.shape[0]),
                "num_test_participants": int(x_test.shape[0]),
                "num_features": int(num_features),
                "train_participants": train_ids,
                "test_participants": test_ids,
                "train_activation_file": str(act_train_file),
                "test_activation_file": str(act_test_file),
                "train_prediction_file": str(pred_train_file),
                "test_prediction_file": str(pred_test_file),
                "step2B_test_pcc_nll": multi_pcc_nll,
                "step2B_test_pcc_aic": multi_pcc_aic,
                "step2B_intercept_nll": b_nll_multi,
                "step2B_intercept_aic": b_aic_multi,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "dataset_key": dataset_key,
        "num_train_participants": float(x_train.shape[0]),
        "num_test_participants": float(x_test.shape[0]),
        "num_features": float(num_features),
        "step2B_test_pcc_nll": multi_pcc_nll,
        "step2B_test_pcc_aic": multi_pcc_aic,
    }


def main() -> None:
    args = parse_args()

    activations_root = Path(args.activations_root).resolve()
    predictions_root = Path(args.predictions_root).resolve()
    output_root = Path(args.output_root).resolve()

    act_train_dir = activations_root / "train"
    act_test_dir = activations_root / "test"
    pred_train_dir = predictions_root / "train"
    pred_test_dir = predictions_root / "test"

    for path in [act_train_dir, act_test_dir, pred_train_dir, pred_test_dir]:
        if not path.exists():
            raise FileNotFoundError(f"目录不存在: {path}")

    test_activation_files = sorted(act_test_dir.glob("*_reformatted_parcel_activations.json"))
    if not test_activation_files:
        raise ValueError(f"{act_test_dir} 下没有激活文件")

    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float]] = []
    for act_test_file in test_activation_files:
        dataset_key = extract_dataset_key(act_test_file)
        act_train_file = act_train_dir / act_test_file.name
        pred_test_file = find_prediction_file(pred_test_dir, dataset_key)
        pred_train_file = find_prediction_file(pred_train_dir, dataset_key)

        if not act_train_file.exists():
            print(f"[WARN] 跳过 {dataset_key}: 缺少 train activation 文件 {act_train_file.name}")
            continue
        if pred_test_file is None:
            print(f"[WARN] 跳过 {dataset_key}: test predictions 未匹配")
            continue
        if pred_train_file is None:
            print(f"[WARN] 跳过 {dataset_key}: train predictions 未匹配")
            continue

        dataset_output_dir = output_root / dataset_key
        marker_file = dataset_output_dir / "meta_info.json"
        if args.skip_existing and marker_file.exists():
            # 旧版本可能只生成了 step1_test_*，此处允许补齐 step1_train_*。
            required_step1_train_files = [
                dataset_output_dir / "step1_train_feature_pcc_ranked_by_aic.csv",
                dataset_output_dir / "step1_train_feature_pcc_ranked_by_nll.csv",
            ]
            missing_required = [p for p in required_step1_train_files if not p.exists()]
            if not missing_required:
                print(f"[SKIP] {dataset_key} 输出已存在，跳过")
                continue
            print(
                f"[REGEN] {dataset_key} 检测到缺少训练集 Step1 输出，"
                f"将重新计算这些结果: {[p.name for p in missing_required]}"
            )

        row = analyze_single_dataset(
            dataset_key=dataset_key,
            act_train_file=act_train_file,
            act_test_file=act_test_file,
            pred_train_file=pred_train_file,
            pred_test_file=pred_test_file,
            output_dir=dataset_output_dir,
        )
        summary_rows.append(row)

    if not summary_rows:
        print("[WARN] 没有完成任何数据集分析")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="dataset_key")
    summary_path = output_root / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[DONE] 分析完成，汇总结果: {summary_path}")


if __name__ == "__main__":
    main()
