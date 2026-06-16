#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5 折交叉验证的幻觉检测训练脚本（可解释特征 + 线性模型）。

输入：
- correct.jsonl 与 incorrect.jsonl（JSONL，每行包含 token_parcel_acts）
- capability_parcel 映射 JSON（构造 capability 聚合）

流程：
1) 读取两类样本，构建 M(P×C)，将每条记录转为样本 {a,F,c,G}
2) 基于训练折构建 truth/hall 两个原型
3) 用指示器 + 原型差 + 连接失配构造特征
4) 使用逻辑回归（L2 或 L1 可选）做 5 折交叉验证，输出 Accuracy/AUROC/AUPRC 等
5) 训练全量模型并保存（含标准化器、特征名）

用户规则：
- 不静默吞异常；
- 提供 --skip_existing 以在已有结果时跳过；
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# 允许脚本直接运行（非包环境）
import sys
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from feature_extractor import (
    load_mapping_json,
    build_mapping_matrix,
    load_jsonl,
    build_sample_from_jsonl_record,
    compute_prototypes,
    build_features,
    IndicatorConfig,
)
from config_builder import build_auto_indicator


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def default_indicator_config() -> IndicatorConfig:
    # 保留回退配置（当未提供分析结果目录时）
    return IndicatorConfig(
        pos_parcels=[],
        neg_parcels=[],
        true_path_pairs=[],
        wrong_path_pairs=[],
        pos_capabilities=[],
        neg_capabilities=[],
        true_capability_pairs=[],
        wrong_capability_pairs=[],
    )


def build_dataset(correct_records: List[Dict], incorrect_records: List[Dict], M: np.ndarray, eps: float = 1e-8) -> Tuple[List[Dict], np.ndarray]:
    X_samples: List[Dict] = []
    y_labels: List[int] = []
    for r in correct_records:
        X_samples.append(build_sample_from_jsonl_record(r, M, eps=eps))
        y_labels.append(0)
    for r in incorrect_records:
        X_samples.append(build_sample_from_jsonl_record(r, M, eps=eps))
        y_labels.append(1)
    return X_samples, np.array(y_labels, dtype=np.int32)


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray, verbose: bool = False) -> float:
    """找到最佳阈值（基于 F1 分数）。scores 需在 [0,1] 或可单调映射到概率。"""
    from sklearn.metrics import precision_recall_fscore_support
    # 若 scores 不在 [0,1]，做 min-max 缩放到 [0,1]
    s_min, s_max = float(np.min(scores)), float(np.max(scores))
    if s_max - s_min > 1e-10:
        scores_norm = (np.asarray(scores, dtype=np.float64) - s_min) / (s_max - s_min)
    else:
        scores_norm = np.full_like(scores, 0.5, dtype=np.float64)
    thresholds = np.linspace(0, 1, 501)
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in thresholds:
        pred = (scores_norm >= threshold).astype(int)
        if len(np.unique(pred)) > 1:
            _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    if verbose:
        print(f"  最佳阈值: {best_threshold:.4f}, F1: {best_f1:.4f}")
    return best_threshold


def _scores_to_proba(clf, X: np.ndarray, model_type: str, fit_scores_min: float = None, fit_scores_max: float = None) -> np.ndarray:
    """统一得到 (n_samples,) 的类 1 概率，用于 AUROC/阈值等。"""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    # Ridge / LinearSVC 等：用 decision_function 并缩放到 [0,1]
    raw = clf.decision_function(X)
    if raw.ndim > 1:
        raw = raw.ravel()
    if fit_scores_min is not None and fit_scores_max is not None and fit_scores_max - fit_scores_min > 1e-10:
        proba = (raw - fit_scores_min) / (fit_scores_max - fit_scores_min)
    else:
        proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-10)
    return np.clip(proba, 0.0, 1.0).astype(np.float64)


def _fit_clf(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    tune_hyperparams: bool,
    random_state: int,
    lr_penalty: str = "l2",
    lr_solver: str = "lbfgs",
    lr_C: float = 1.0,
    class_weight: str = "balanced",
):
    """根据 model_type 创建并拟合分类器。返回 (clf, scores_min, scores_max)。"""
    if model_type == "lr":
        if tune_hyperparams:
            param_grid = [
                {"penalty": ["l1"], "solver": ["liblinear"], "C": [0.01, 0.1, 1.0, 10.0], "class_weight": ["balanced", None]},
                {"penalty": ["l2"], "solver": ["lbfgs"], "C": [0.01, 0.1, 1.0, 10.0], "class_weight": ["balanced", None]},
            ]
            base = LogisticRegression(max_iter=2000, random_state=random_state)
            clf = GridSearchCV(base, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, refit=True)
            clf.fit(X, y)
            clf = clf.best_estimator_
        else:
            clf = LogisticRegression(
                max_iter=2000,
                solver=lr_solver,
                penalty=lr_penalty,
                C=lr_C,
                class_weight=class_weight,
                random_state=random_state,
            )
            clf.fit(X, y)
        return clf, None, None

    if model_type == "ridge":
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=random_state)
        clf.fit(X, y)
        dtr = clf.decision_function(X)
        return clf, float(np.min(dtr)), float(np.max(dtr))

    if model_type == "svm":
        clf = LinearSVC(max_iter=5000, class_weight="balanced", random_state=random_state, dual="auto")
        clf.fit(X, y)
        dtr = clf.decision_function(X)
        return clf, float(np.min(dtr)), float(np.max(dtr))

    if model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight="balanced", random_state=random_state, n_jobs=-1)
        clf.fit(X, y)
        return clf, None, None

    raise ValueError(f"不支持的 model_type: {model_type}，可选: lr, ridge, svm, rf")


def dataset_to_features(
    X_samples: List[Dict],
    y: np.ndarray,
    indicator: IndicatorConfig,
    cap_names: List[str],
    n_splits: int = 5,
    random_state: int = 42,
    model_type: str = "lr",
    tune_hyperparams: bool = False,
    lr_penalty: str = "l2",
    lr_solver: str = "lbfgs",
    lr_C: float = 1.0,
    class_weight: str = "balanced",
    verbose: bool = False,
):
    """跑一遍 CV，返回 (metrics, feature_names)。verbose 为 True 时打印第 1 折的阈值信息。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(y, y), 1):
        X_tr = [X_samples[i] for i in tr_idx]
        X_te = [X_samples[i] for i in te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        # 用训练折计算 truth/hall 原型
        proto_truth = compute_prototypes([s for s, label in zip(X_tr, y_tr) if label == 0])
        proto_hall = compute_prototypes([s for s, label in zip(X_tr, y_tr) if label == 1])

        # 提取特征
        Fe_tr = []
        Fe_te = []
        for s in X_tr:
            fv, names = build_features(s, proto_truth, proto_hall, indicator, cap_names)
            Fe_tr.append(fv)
        for s in X_te:
            fv, _ = build_features(s, proto_truth, proto_hall, indicator, cap_names)
            Fe_te.append(fv)
        Fe_tr = np.stack(Fe_tr, axis=0)
        Fe_te = np.stack(Fe_te, axis=0)

        # 标准化
        scaler = StandardScaler()
        Fe_tr_std = scaler.fit_transform(Fe_tr)
        Fe_te_std = scaler.transform(Fe_te)

        # 模型拟合（含可选超参搜索）
        clf, dmin, dmax = _fit_clf(
            model_type,
            Fe_tr_std,
            y_tr,
            tune_hyperparams,
            random_state,
            lr_penalty=lr_penalty,
            lr_solver=lr_solver,
            lr_C=lr_C,
            class_weight=class_weight,
        )

        # 训练集分数 → 找最佳阈值
        prob_tr = _scores_to_proba(clf, Fe_tr_std, model_type, dmin, dmax)
        threshold = find_best_threshold(y_tr, prob_tr, verbose=(verbose and fold == 1))

        # 测试集概率与预测
        prob_te = _scores_to_proba(clf, Fe_te_std, model_type, dmin, dmax)
        pred_te = (prob_te >= threshold).astype(int)

        # 评估指标
        acc = accuracy_score(y_te, pred_te)
        from sklearn.metrics import precision_recall_fscore_support
        prec, rec, f1, _ = precision_recall_fscore_support(y_te, pred_te, average="binary", zero_division=0)
        try:
            auroc = roc_auc_score(y_te, prob_te)
        except Exception as e:
            print(f"Fold {fold} AUROC计算失败: {e}")
            auroc = float("nan")
        try:
            auprc = average_precision_score(y_te, prob_te)
        except Exception as e:
            print(f"Fold {fold} AUPRC计算失败: {e}")
            auprc = float("nan")

        metrics.append({
            "fold": fold,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "threshold": float(threshold),
        })

    return metrics, names


def train_full_and_save(
    X_samples: List[Dict],
    y: np.ndarray,
    indicator: IndicatorConfig,
    cap_names: List[str],
    out_dir: Path,
    model_type: str = "lr",
    tune_hyperparams: bool = False,
    lr_penalty: str = "l2",
    lr_solver: str = "lbfgs",
    lr_C: float = 1.0,
    class_weight: str = "balanced",
    random_state: int = 42,
    model_filename: str = "hallucination_detector.joblib",
):
    # 全量数据计算原型
    proto_truth = compute_prototypes([s for s, label in zip(X_samples, y) if label == 0])
    proto_hall = compute_prototypes([s for s, label in zip(X_samples, y) if label == 1])

    # 特征
    Fe = []
    for s in X_samples:
        fv, names = build_features(s, proto_truth, proto_hall, indicator, cap_names)
        Fe.append(fv)
    Fe = np.stack(Fe, axis=0)

    scaler = StandardScaler()
    Fe_std = scaler.fit_transform(Fe)

    clf, dmin, dmax = _fit_clf(
        model_type,
        Fe_std,
        y,
        tune_hyperparams,
        random_state,
        lr_penalty=lr_penalty,
        lr_solver=lr_solver,
        lr_C=lr_C,
        class_weight=class_weight,
    )
    prob_scores = _scores_to_proba(clf, Fe_std, model_type, dmin, dmax)
    threshold = find_best_threshold(y, prob_scores, verbose=True)

    # Ridge/SVM 无 predict_proba，用校准包装以便推理时统一接口
    if model_type in ("ridge", "svm") and not hasattr(clf, "predict_proba"):
        clf = CalibratedClassifierCV(clf, cv=3, method="isotonic")
        clf.fit(Fe_std, y)
        prob_scores = clf.predict_proba(Fe_std)[:, 1]
        threshold = find_best_threshold(y, prob_scores, verbose=True)

    # 保存（含 model_type 供推理识别）
    ensure_dir(out_dir)
    job = {
        "scaler": scaler,
        "clf": clf,
        "threshold": threshold,
        "feature_names": names,
        "indicator": indicator,
        "proto_truth": proto_truth,
        "proto_hall": proto_hall,
        "model_type": model_type,
        "model_params": {
            "tune_hyperparams": tune_hyperparams,
            "lr_penalty": lr_penalty,
            "lr_solver": lr_solver,
            "lr_C": lr_C,
            "class_weight": class_weight,
        },
    }
    joblib.dump(job, out_dir / model_filename)


def main():
    parser = argparse.ArgumentParser(description="5折交叉验证的幻觉检测训练")
    parser.add_argument("--correct", type=str, required=True, help="正确样本 JSONL")
    parser.add_argument("--incorrect", type=str, required=True, help="幻觉样本 JSONL")
    parser.add_argument("--mapping_json", type=str, required=True, help="Capability-Parcel 映射 JSON")
    parser.add_argument("--out_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--skip_existing", action="store_true", help="若已有结果则跳过")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--analysis_output_root", type=str, default=None, help="分析输出根目录（自动构建指示器配置）")
    parser.add_argument("--output_suffix", type=str, default="", help="输出文件名后缀，如 updated 则得到 cv_metrics_updated.json、hallucination_detector_updated.joblib")
    parser.add_argument("--model_type", type=str, default="lr", choices=["lr", "ridge", "svm", "rf"], help="手动指定检测模型")
    parser.add_argument("--tune_hyperparams", action="store_true", help="仅对 LogisticRegression 启用内部超参数搜索")
    parser.add_argument("--lr_penalty", type=str, default="l2", choices=["l1", "l2"], help="LogisticRegression 正则项")
    parser.add_argument("--lr_solver", type=str, default="lbfgs", help="LogisticRegression solver；L1 通常使用 liblinear")
    parser.add_argument("--lr_C", type=float, default=1.0, help="LogisticRegression 正则强度倒数 C")
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["balanced", "none"], help="分类权重；none 表示不使用 class_weight")

    args = parser.parse_args()
    class_weight = None if args.class_weight == "none" else args.class_weight

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    suf = args.output_suffix.strip()
    metrics_path = out_dir / ("cv_metrics_" + suf + ".json") if suf else out_dir / "cv_metrics.json"
    model_path = out_dir / ("hallucination_detector_" + suf + ".joblib") if suf else out_dir / "hallucination_detector.joblib"

    if args.skip_existing and metrics_path.exists() and model_path.exists():
        print(f"检测到已存在结果，跳过：{metrics_path} & {model_path}")
        return

    # 读取映射
    mapping = load_mapping_json(args.mapping_json)

    # 读取数据
    correct_records = load_jsonl(args.correct)
    incorrect_records = load_jsonl(args.incorrect)
    if len(correct_records) == 0 or len(incorrect_records) == 0:
        raise ValueError("正确/幻觉样本不能为空")

    # 估计 parcel 维度
    sample0 = correct_records[0] if len(correct_records) > 0 else incorrect_records[0]
    acts0 = np.array(sample0.get("token_parcel_acts"))
    if acts0.ndim != 2:
        raise ValueError("token_parcel_acts 应为二维 (T,P)")
    P = acts0.shape[1]

    # 构建映射矩阵与 capability 名称
    M, cap_names = build_mapping_matrix(mapping, parcel_dim=P)

    # 组装数据集
    X_samples, y = build_dataset(correct_records, incorrect_records, M)

    # 指示器配置
    indicator = default_indicator_config()
    if args.analysis_output_root is not None:
        # 基于分析输出构建更贴合当前模型数据的动态指示器
        auto_cfg = build_auto_indicator(args.analysis_output_root)
        # 将自动配置映射到 IndicatorConfig
        indicator.pos_parcels = auto_cfg.pos_parcels
        indicator.neg_parcels = auto_cfg.neg_parcels
        indicator.pos_capabilities = auto_cfg.pos_capabilities
        indicator.neg_capabilities = auto_cfg.neg_capabilities
        indicator.true_path_pairs = auto_cfg.neg_parcel_pairs  # 真实性应强 → true_path
        indicator.wrong_path_pairs = auto_cfg.pos_parcel_pairs
        indicator.true_capability_pairs = auto_cfg.neg_capability_pairs
        indicator.wrong_capability_pairs = auto_cfg.pos_capability_pairs

    if args.model_type != "lr" and args.tune_hyperparams:
        raise ValueError("--tune_hyperparams 目前只支持 model_type=lr")
    print(
        "选用配置: "
        f"model_type={args.model_type}, "
        f"tune_hyperparams={args.tune_hyperparams}, "
        f"lr_penalty={args.lr_penalty}, "
        f"lr_solver={args.lr_solver}, "
        f"lr_C={args.lr_C}, "
        f"class_weight={class_weight}"
    )

    metrics, feat_names = dataset_to_features(
        X_samples, y, indicator, cap_names,
        n_splits=args.folds,
        random_state=args.random_state,
        model_type=args.model_type,
        tune_hyperparams=args.tune_hyperparams,
        lr_penalty=args.lr_penalty,
        lr_solver=args.lr_solver,
        lr_C=args.lr_C,
        class_weight=class_weight,
        verbose=False,
    )

    # 按原格式写入 cv_metrics.json（与之前脚本输出一致）
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "fold_metrics": metrics,
            "mean_accuracy": float(np.mean([m["accuracy"] for m in metrics])),
            "mean_precision": float(np.mean([m["precision"] for m in metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in metrics])),
            "mean_auroc": float(np.nanmean([m["auroc"] for m in metrics])),
            "mean_auprc": float(np.nanmean([m["auprc"] for m in metrics])),
            "feature_names": feat_names,
            "model_type": args.model_type,
            "model_params": {
                "tune_hyperparams": args.tune_hyperparams,
                "lr_penalty": args.lr_penalty,
                "lr_solver": args.lr_solver,
                "lr_C": args.lr_C,
                "class_weight": class_weight,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"CV 指标已保存到: {metrics_path}")

    # 用同一显式配置训练全量模型并保存
    train_full_and_save(
        X_samples, y, indicator, cap_names, out_dir,
        model_type=args.model_type,
        tune_hyperparams=args.tune_hyperparams,
        lr_penalty=args.lr_penalty,
        lr_solver=args.lr_solver,
        lr_C=args.lr_C,
        class_weight=class_weight,
        random_state=args.random_state,
        model_filename=model_path.name,
    )
    print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
