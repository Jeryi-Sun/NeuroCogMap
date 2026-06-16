#!/usr/bin/env python3
"""
panel_a：对指定的 fits JSON 文件做统计汇总，并将结果写到 result/panel_a 目录：
- 每个输入文件生成一个独立 summary JSON（包含逐 participant 字段，便于显著性检验）
- 同时生成一个整合的 CSV（每个输入文件一行：Fisher Z 聚合均值、std、95% bootstrap CI 等）

注意：
- Pearson 相关系数聚合使用 Fisher Z 变换：z=atanh(r)，聚合后回变换 r=tanh(mean(z))
- bootstrap 在 participant 维度重采样，CI 使用 percentile (2.5%, 97.5%)
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev


DEFAULT_INPUTS = [
    Path(
        "/path/to/project_root/"
        "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits/"
        "saeact_model/model=google-gemma-2-2b_extractor=saeact_model_all_parcels_n=270_roi=all_rois.json"
    ),
    Path(
        "/path/to/project_root/"
        "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits/"
        "language_model_attention/model=google-gemma-2-2b_extractor=language_model_attention_layer=12_roi=all_rois.json"
    ),
    Path(
        "/path/to/project_root/"
        "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits/"
        "language_model/model=google-gemma-2-2b_extractor=language_model_layer=12_roi=all_rois.json"
    ),
    Path(
        "/path/to/project_root/"
        "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits/"
        "embeddings/model=google-gemma-2-2b_extractor=embeddings_general_roi=all_rois.json"
    ),
    Path(
        "/path/to/project_root/"
        "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits/"
        "bert_model/model=google-gemma-2-2b_extractor=bert_model_layer=12_roi=all_rois.json"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="panel_a: analyze selected fit JSON files")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="*",
        default=DEFAULT_INPUTS,
        help="待分析的 JSON 文件路径列表（默认使用脚本内置的 5 个文件）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/path/to/project_root/"
            "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/analysis_results/result/panel_a"
        ),
        help="输出目录（默认: analysis_results/result/panel_a）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="若输出 summary 已存在则跳过该文件（默认关闭；建议首次跑关闭以覆盖）",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000, help="bootstrap 次数（默认 5000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument(
        "--participant-first-n",
        type=int,
        default=None,
        help="仅使用前 N 个 participant（按 participant_ 数字顺序；默认不启用）",
    )
    parser.add_argument(
        "--participant-ids",
        type=str,
        default=None,
        help="仅使用指定 participant_id（逗号分隔，如 '0,1,2'；默认不启用）",
    )
    parser.add_argument(
        "--participant-ids-file",
        type=Path,
        default=None,
        help="从文件读取 participant_id（每行一个，允许空行和 # 注释；默认不启用）",
    )
    return parser.parse_args()


def _participant_sort_key(participant_key: str) -> tuple[int, str]:
    suffix = participant_key.replace("participant_", "", 1)
    if suffix.isdigit():
        return int(suffix), participant_key
    return 10**9, participant_key


def extract_participants(payload: dict, source_path: Path) -> list[dict]:
    participants: list[dict] = []
    participant_keys = sorted(
        [k for k in payload.keys() if k.startswith("participant_")],
        key=_participant_sort_key,
    )
    for key in participant_keys:
        value = payload[key]
        if not isinstance(value, dict):
            print(f"[ERROR] {source_path}: {key} 的值不是字典，已跳过该 participant")
            continue
        if "mean_score" not in value:
            print(f"[ERROR] {source_path}: {key} 缺少 mean_score 字段，已跳过该 participant")
            continue
        mean_score = value["mean_score"]
        if not isinstance(mean_score, (int, float)):
            print(
                f"[ERROR] {source_path}: {key}.mean_score 不是数值类型 "
                f"(当前类型: {type(mean_score)}), 已跳过该 participant"
            )
            continue
        participants.append(
            {
                "participant_key": key,
                "participant_id": value.get("participant_id"),
                "mean_score": float(mean_score),
                "median_score": value.get("median_score"),
                "std_score": value.get("std_score"),
                "min_score": value.get("min_score"),
                "max_score": value.get("max_score"),
            }
        )
    return participants


def parse_participant_id_set(participant_ids: str | None, participant_ids_file: Path | None) -> set[str] | None:
    ids: set[str] = set()
    if participant_ids:
        for token in participant_ids.split(","):
            token = token.strip()
            if token:
                ids.add(token)
    if participant_ids_file:
        if not participant_ids_file.exists():
            raise FileNotFoundError(f"participant ids 文件不存在: {participant_ids_file}")
        for line in participant_ids_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            ids.add(stripped)
    return ids if ids else None


def filter_participants(
    participants: list[dict],
    *,
    first_n: int | None,
    include_ids: set[str] | None,
    source_path: Path,
) -> list[dict]:
    filtered = participants
    if include_ids is not None:
        available_ids = {str(p.get("participant_id")) for p in participants if p.get("participant_id") is not None}
        missing = sorted([pid for pid in include_ids if pid not in available_ids])
        if missing:
            raise ValueError(
                f"{source_path}: 指定 participant_id 不存在于该文件中: {missing[:20]}"
                + (" ..." if len(missing) > 20 else "")
            )
        filtered = [p for p in filtered if str(p.get("participant_id")) in include_ids]
    if first_n is not None:
        if first_n <= 0:
            raise ValueError(f"{source_path}: --participant-first-n 必须为正整数，当前: {first_n}")
        filtered = filtered[:first_n]
    return filtered


def fisher_z_transform(r: float, eps: float = 1e-7) -> float:
    if r >= 1.0:
        clipped = 1.0 - eps
        print(f"[WARN] 发现 r={r} >= 1，已裁剪为 {clipped} 后进行 Fisher Z")
        r = clipped
    elif r <= -1.0:
        clipped = -1.0 + eps
        print(f"[WARN] 发现 r={r} <= -1，已裁剪为 {clipped} 后进行 Fisher Z")
        r = clipped
    return math.atanh(r)


def percentile(sorted_values: list[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q 必须在 [0, 1] 区间内，当前 q={q}")
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile 输入为空")
    if n == 1:
        return sorted_values[0]
    position = (n - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def bootstrap_ci_fisher_mean(mean_scores: list[float], n_bootstrap: int, seed: int) -> tuple[float, float]:
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap 必须 > 0，当前: {n_bootstrap}")
    rng = random.Random(seed)
    n = len(mean_scores)
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        sampled = [mean_scores[rng.randrange(n)] for _ in range(n)]
        sampled_z = [fisher_z_transform(x) for x in sampled]
        boot_means.append(math.tanh(mean(sampled_z)))
    boot_means.sort()
    return percentile(boot_means, 0.025), percentile(boot_means, 0.975)


def build_out_summary_path(out_dir: Path, source_json: Path) -> Path:
    # 保持“原文件名 + _summary.json”，但写到 panel_a 目录里
    return out_dir / f"{source_json.stem}_summary{source_json.suffix}"


def derive_model_name_from_path(source_json: Path) -> str:
    """
    从文件名中提取紧凑模型名：
    例如：
    - model=..._extractor=language_model_attention_layer=12_roi=all_rois.json
      -> language_model_attention_layer=12_roi
    - model=..._extractor=bert_model_layer=12_roi=all_rois.json
      -> bert_model_layer=12_roi
    - model=..._extractor=embeddings_general_roi=all_rois.json
      -> embeddings_general_roi
    - model=..._extractor=saeact_model_all_parcels_n=270_roi=all_rois.json
      -> saeact_model_all_parcels_n=270_roi
    """
    stem = source_json.stem  # 不含扩展名
    # 优先基于 'extractor=' 分割
    if "extractor=" in stem:
        model_part = stem.split("extractor=", 1)[1]
    else:
        # 回退：去掉前缀 'model=' 如存在
        model_part = stem
        if model_part.startswith("model="):
            model_part = model_part[len("model=") :]
    # 将末尾的 `_roi=...` 归一化为 `_roi`
    # 例如 '_roi=all_rois' -> '_roi'
    roi_idx = model_part.find("_roi=")
    if roi_idx != -1:
        model_part = model_part[: roi_idx + len("_roi")]
    return model_part

def summarize_one(
    source_json: Path,
    *,
    out_dir: Path,
    skip_existing: bool,
    n_bootstrap: int,
    seed: int,
    participant_first_n: int | None,
    participant_id_set: set[str] | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = build_out_summary_path(out_dir, source_json)
    if skip_existing and summary_path.exists():
        print(f"[SKIP] {summary_path} 已存在，跳过")
        return {}

    if not source_json.exists():
        raise FileNotFoundError(f"输入文件不存在: {source_json}")

    with source_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{source_json} 顶层 JSON 不是对象，无法解析 participant 数据")

    participants_all = extract_participants(payload, source_json)
    participants = filter_participants(
        participants_all,
        first_n=participant_first_n,
        include_ids=participant_id_set,
        source_path=source_json,
    )
    mean_scores = [float(p["mean_score"]) for p in participants]
    if not mean_scores:
        raise ValueError(f"{source_json} 经过筛选后未剩余任何 participant mean_score")

    z_scores = [fisher_z_transform(x) for x in mean_scores]
    fisher_z_mean = mean(z_scores)
    fisher_z_std = stdev(z_scores) if len(z_scores) >= 2 else None
    raw_std = stdev(mean_scores) if len(mean_scores) >= 2 else None
    if raw_std is None:
        print(f"[WARN] {source_json}: participant 数量 < 2，无法计算原始相关系数 std")
    if fisher_z_std is None:
        print(f"[WARN] {source_json}: participant 数量 < 2，无法计算 Fisher Z std")

    ci_low, ci_high = bootstrap_ci_fisher_mean(mean_scores, n_bootstrap=n_bootstrap, seed=seed)

    model_name = derive_model_name_from_path(source_json)
    summary_payload = {
        "source_file": str(source_json),
        "model_name": model_name,
        "participant_count": len(participants),
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
        "participant_filter": {
            "participant_first_n": participant_first_n,
            "participant_ids": sorted(list(participant_id_set)) if participant_id_set is not None else None,
        },
        "participant_mean_score_average_raw": mean(mean_scores),
        "participant_mean_score_std_raw": raw_std,
        "participant_mean_score_average_fisher_z": math.tanh(fisher_z_mean),
        "participant_mean_score_std_fisher_z": fisher_z_std,
        "participant_mean_score_bootstrap_ci_95": {"lower": ci_low, "upper": ci_high},
        "participant_mean_scores": mean_scores,
        "participants": participants,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"[OK] {source_json.name} -> {summary_path.name} | "
        f"participant_count={len(mean_scores)}, "
        f"fisher_mean={summary_payload['participant_mean_score_average_fisher_z']:.6f}, "
        f"95%CI=[{ci_low:.6f}, {ci_high:.6f}]"
    )
    return summary_payload


def write_integrated_csv(out_dir: Path, summaries: list[dict]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "panel_a_integrated_summary.csv"
    fieldnames = [
        "model",
        "participant_count",
        "participant_first_n",
        "participant_ids",
        "participant_mean_score_average_raw",
        "participant_mean_score_std_raw",
        "participant_mean_score_average_fisher_z",
        "participant_mean_score_std_fisher_z",
        "ci95_lower",
        "ci95_upper",
        "n_bootstrap",
        "bootstrap_seed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            filt = s.get("participant_filter", {}) if isinstance(s, dict) else {}
            ci = s.get("participant_mean_score_bootstrap_ci_95", {}) if isinstance(s, dict) else {}
            writer.writerow(
                {
                    "model": s.get("model_name"),
                    "participant_count": s.get("participant_count"),
                    "participant_first_n": filt.get("participant_first_n"),
                    "participant_ids": ";".join(filt.get("participant_ids") or []),
                    "participant_mean_score_average_raw": s.get("participant_mean_score_average_raw"),
                    "participant_mean_score_std_raw": s.get("participant_mean_score_std_raw"),
                    "participant_mean_score_average_fisher_z": s.get("participant_mean_score_average_fisher_z"),
                    "participant_mean_score_std_fisher_z": s.get("participant_mean_score_std_fisher_z"),
                    "ci95_lower": ci.get("lower"),
                    "ci95_upper": ci.get("upper"),
                    "n_bootstrap": s.get("n_bootstrap"),
                    "bootstrap_seed": s.get("bootstrap_seed"),
                }
            )
    return csv_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    participant_id_set = parse_participant_id_set(args.participant_ids, args.participant_ids_file)

    inputs = [p.resolve() for p in args.inputs]
    if not inputs:
        raise ValueError("未提供任何 --inputs")

    summaries: list[dict] = []
    failed = 0
    for p in inputs:
        try:
            summary = summarize_one(
                p,
                out_dir=out_dir,
                skip_existing=args.skip_existing,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                participant_first_n=args.participant_first_n,
                participant_id_set=participant_id_set,
            )
            if summary:
                summaries.append(summary)
        except Exception as exc:
            failed += 1
            print(f"[ERROR] 处理失败: {p}\n        异常: {exc}")

    if not summaries:
        raise RuntimeError("没有成功生成任何 summary（可能全部被 skip 或全部失败）")

    csv_path = write_integrated_csv(out_dir, summaries)
    print(f"[DONE] summary_count={len(summaries)}, failed={failed}, integrated_csv={csv_path}")


if __name__ == "__main__":
    main()

