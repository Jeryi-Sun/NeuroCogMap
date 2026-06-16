#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 !human_kool2017cost_exp2 的不同方法在每个 participant/param 行上的 AIC：
- 更低 AIC: win
- 相同 AIC: tie
- 更高 AIC: loss

默认将 (participant, param) 作为行对齐主键，并跳过 param == 'mean' 的汇总行。
输出：
- 终端打印汇总
- openloop/results/ 下写入逐行明细与汇总 csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FULL_METHOD = "GPT5.2+NeuroCogMap"
BASELINE_METHOD = "GPT5.2"
SIMPLE_METHOD = "Cognitive Model"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件: {path}")
    df = pd.read_csv(path)
    return df


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    required = {"participant", "aic"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"文件缺少必要列 {missing}: {path} (columns={list(df.columns)})")


def _normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # participant 统一成字符串便于过滤；并额外生成可用于稳定对齐的整数 participant_id
    out["participant"] = out["participant"].astype(str)
    out["participant_id"] = pd.to_numeric(out["participant"], errors="coerce")
    # 注意：participant_id 可能包含 NaN（如 'mean'），也可能是非整数的小数（如均值汇总行）
    if "Unnamed: 0" in out.columns:
        # 行号列用 int，便于稳定对齐（mean 行通常是 100）
        out["Unnamed: 0"] = pd.to_numeric(out["Unnamed: 0"], errors="raise").astype(int)
    return out


def _drop_mean_rows(df: pd.DataFrame) -> pd.DataFrame:
    # 兼容两种汇总行：
    # 1) participant == 'mean'
    # 2) participant 是均值数字（如 102.6111...），表现为 participant_id 非整数
    d = df.copy()
    is_mean_str = d["participant"].astype(str).str.lower() == "mean"
    pid = d.get("participant_id")
    if pid is None:
        return d[~is_mean_str].copy()
    is_numeric = pid.notna()
    is_integer = is_numeric & (pid % 1 == 0)
    # 仅保留 participant_id 为整数的正常行，且排除 'mean'
    keep = (~is_mean_str) & is_integer
    out = d[keep].copy()
    # participant_id 转成 int 便于后续对齐
    out["participant_id"] = out["participant_id"].astype(int)
    return out


@dataclass(frozen=True)
class CompareResult:
    win: int
    tie: int
    loss: int
    n: int
    win_rate: float
    tie_rate: float
    loss_rate: float


def _compare_series(aic_full: pd.Series, aic_other: pd.Series, eps: float) -> pd.Series:
    # 返回字符串标签：win/tie/loss（以 full 相对 other）
    diff = aic_full - aic_other
    label = pd.Series(index=diff.index, dtype="object")
    label[diff < -eps] = "win"
    label[diff > eps] = "loss"
    label[(diff >= -eps) & (diff <= eps)] = "tie"
    return label


def _summarize(labels: pd.Series) -> CompareResult:
    vc = labels.value_counts(dropna=False)
    win = int(vc.get("win", 0))
    tie = int(vc.get("tie", 0))
    loss = int(vc.get("loss", 0))
    n = int(labels.shape[0])
    if n <= 0:
        return CompareResult(win=0, tie=0, loss=0, n=0, win_rate=0.0, tie_rate=0.0, loss_rate=0.0)
    return CompareResult(
        win=win,
        tie=tie,
        loss=loss,
        n=n,
        win_rate=win / n,
        tie_rate=tie / n,
        loss_rate=loss / n,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, type=Path, help="full_fix_bug.csv (GPT5.2+NeuroCogMap)")
    ap.add_argument("--baseline", required=True, type=Path, help="baseline_fix_bug.csv (GPT5.2)")
    ap.add_argument("--simple", required=True, type=Path, help="simple_fix_bug.csv (Cognitive Model)")
    ap.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="输出目录（建议 openloop/results/）",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="若输出文件已存在则跳过（不覆盖）",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="AIC 视为相同的容差（默认 0；若想处理浮点误差可设为 1e-9 等）",
    )
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    detail_out = outdir / "!human_kool2017cost_exp2.aic_compare.detail.csv"
    summary_out = outdir / "!human_kool2017cost_exp2.aic_compare.summary.csv"
    if args.skip_existing and detail_out.exists() and summary_out.exists():
        print(f"[skip_existing] 输出已存在，跳过：{detail_out} , {summary_out}")
        return

    df_full = _read_csv(args.full)
    df_base = _read_csv(args.baseline)
    df_simple = _read_csv(args.simple)
    _validate_columns(df_full, args.full)
    _validate_columns(df_base, args.baseline)
    _validate_columns(df_simple, args.simple)

    df_full = _drop_mean_rows(_normalize_key_cols(df_full))
    df_base = _drop_mean_rows(_normalize_key_cols(df_base))
    df_simple = _drop_mean_rows(_normalize_key_cols(df_simple))

    # 这里的 param 是模型拟合出来的连续参数值，不是“参数编号”，不同方法之间不会一致；
    # 因此对齐请用稳定的行号列（若存在 Unnamed: 0），否则用 participant_id（整数 participant 编号）。
    if "Unnamed: 0" in df_full.columns and "Unnamed: 0" in df_base.columns and "Unnamed: 0" in df_simple.columns:
        key_cols = ["Unnamed: 0"]
    else:
        key_cols = ["participant_id"]

    keep_cols = key_cols + ["participant", "aic"]
    df_full_k = df_full[keep_cols].rename(columns={"aic": "aic_full"})
    df_base_k = df_base[keep_cols].rename(columns={"aic": "aic_baseline"})
    df_simple_k = df_simple[keep_cols].rename(columns={"aic": "aic_simple"})

    merged = (
        df_full_k.merge(df_base_k, on=key_cols, how="inner", suffixes=("", "_dup"))
        .merge(df_simple_k, on=key_cols, how="inner", suffixes=("", "_dup2"))
    )

    # 清理重复的 participant 列（如果 key_cols 不是 participant，会出现 participant_dup/participant_dup2）
    for c in ["participant_dup", "participant_dup2"]:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    # 检查是否有行未对齐成功
    expected = min(len(df_full_k), len(df_base_k), len(df_simple_k))
    if len(merged) != expected:
        # 不直接吞掉问题：打印差异方便定位
        full_keys = set(map(tuple, df_full_k[key_cols].to_numpy()))
        base_keys = set(map(tuple, df_base_k[key_cols].to_numpy()))
        simple_keys = set(map(tuple, df_simple_k[key_cols].to_numpy()))
        inter = full_keys & base_keys & simple_keys
        print(
            "[warn] 合并后行数与预期不一致："
            f" merged={len(merged)} expected(min)={expected} intersection={len(inter)}"
        )
        print(f"[warn] full-only keys: {len(full_keys - inter)} ; baseline-only: {len(base_keys - inter)} ; simple-only: {len(simple_keys - inter)}")

    merged["full_vs_baseline"] = _compare_series(merged["aic_full"], merged["aic_baseline"], eps=args.eps)
    merged["full_vs_simple"] = _compare_series(merged["aic_full"], merged["aic_simple"], eps=args.eps)
    merged["delta_full_minus_baseline"] = merged["aic_full"] - merged["aic_baseline"]
    merged["delta_full_minus_simple"] = merged["aic_full"] - merged["aic_simple"]

    r_base = _summarize(merged["full_vs_baseline"])
    r_simple = _summarize(merged["full_vs_simple"])

    # 写逐行明细
    merged_sorted = merged.sort_values(key_cols).reset_index(drop=True)
    merged_sorted.to_csv(detail_out, index=False)

    # 写汇总
    summary = pd.DataFrame(
        [
            {
                "dataset": "!human_kool2017cost_exp2",
                "full_method": FULL_METHOD,
                "other_method": BASELINE_METHOD,
                "win": r_base.win,
                "tie": r_base.tie,
                "loss": r_base.loss,
                "n": r_base.n,
                "win_rate": r_base.win_rate,
                "tie_rate": r_base.tie_rate,
                "loss_rate": r_base.loss_rate,
            },
            {
                "dataset": "!human_kool2017cost_exp2",
                "full_method": FULL_METHOD,
                "other_method": SIMPLE_METHOD,
                "win": r_simple.win,
                "tie": r_simple.tie,
                "loss": r_simple.loss,
                "n": r_simple.n,
                "win_rate": r_simple.win_rate,
                "tie_rate": r_simple.tie_rate,
                "loss_rate": r_simple.loss_rate,
            },
        ]
    )
    summary.to_csv(summary_out, index=False)

    print("=== AIC 对比（越低越好）===")
    print(f"- full: {FULL_METHOD}")
    print(f"- baseline: {BASELINE_METHOD}")
    print(f"- simple: {SIMPLE_METHOD}")
    print("")
    print(
        f"[full vs baseline] win/tie/loss = {r_base.win}/{r_base.tie}/{r_base.loss} "
        f"(n={r_base.n}, rate={r_base.win_rate:.3f}/{r_base.tie_rate:.3f}/{r_base.loss_rate:.3f})"
    )
    print(
        f"[full vs simple]   win/tie/loss = {r_simple.win}/{r_simple.tie}/{r_simple.loss} "
        f"(n={r_simple.n}, rate={r_simple.win_rate:.3f}/{r_simple.tie_rate:.3f}/{r_simple.loss_rate:.3f})"
    )
    print("")
    print(f"逐行明细: {detail_out}")
    print(f"汇总表:   {summary_out}")


if __name__ == "__main__":
    main()

