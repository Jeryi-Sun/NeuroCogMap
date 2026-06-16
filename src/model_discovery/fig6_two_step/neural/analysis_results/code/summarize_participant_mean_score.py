#!/usr/bin/env python3
"""
批量提取 neural/fits 各子文件夹下 JSON 文件中所有 participant 的 mean_score，
使用 Fisher Z 变换后聚合 Pearson 相关性，并输出 std、95% bootstrap CI，
同时保存每个 participant 的关键分数字段，写入同目录 *_summary.json 文件。
"""

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize participant mean_score for fit result JSON files."
    )
    parser.add_argument(
        "--fits-root",
        type=Path,
        default=Path(
            "/path/to/project_root/"
            "Human_LLM_align/Llama-3.1-Centaur-70B-main/neural/fits"
        ),
        help="fits 根目录路径",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="若 summary 文件已存在则跳过（默认开启）",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="关闭已存在结果跳过逻辑，强制重算并覆盖",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="bootstrap 重采样次数（默认: 5000）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="bootstrap 随机种子（默认: 42）",
    )
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


def build_summary_path(source_json: Path) -> Path:
    return source_json.with_name(f"{source_json.stem}_summary{source_json.suffix}")


def _participant_sort_key(participant_key: str) -> tuple[int, str]:
    suffix = participant_key.replace("participant_", "", 1)
    if suffix.isdigit():
        return int(suffix), participant_key
    return 10**9, participant_key


def extract_participants(
    payload: dict, source_path: Path
) -> tuple[list[dict], list[float]]:
    participants: list[dict] = []
    mean_scores: list[float] = []
    participant_keys = sorted(
        [k for k in payload.keys() if k.startswith("participant_")],
        key=_participant_sort_key,
    )
    for key in participant_keys:
        value = payload[key]
        if not key.startswith("participant_"):
            continue
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
        participant_record = {
            "participant_key": key,
            "participant_id": value.get("participant_id"),
            "mean_score": float(mean_score),
            "median_score": value.get("median_score"),
            "std_score": value.get("std_score"),
            "min_score": value.get("min_score"),
            "max_score": value.get("max_score"),
        }
        participants.append(participant_record)
        mean_scores.append(float(mean_score))
    return participants, mean_scores


def parse_participant_id_set_from_args(args: argparse.Namespace) -> set[str] | None:
    ids: set[str] = set()
    if args.participant_ids:
        for token in args.participant_ids.split(","):
            token = token.strip()
            if token:
                ids.add(token)
    if args.participant_ids_file:
        if not args.participant_ids_file.exists():
            raise FileNotFoundError(f"participant ids 文件不存在: {args.participant_ids_file}")
        for line in args.participant_ids_file.read_text(encoding="utf-8").splitlines():
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


def bootstrap_ci_fisher_mean(
    mean_scores: list[float], n_bootstrap: int, seed: int
) -> tuple[float, float]:
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


def process_one_file(
    source_json: Path,
    skip_existing: bool,
    n_bootstrap: int,
    seed: int,
    participant_first_n: int | None,
    participant_id_set: set[str] | None,
) -> None:
    summary_path = build_summary_path(source_json)
    if skip_existing and summary_path.exists():
        print(f"[SKIP] {summary_path} 已存在，跳过")
        return

    with source_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"{source_json} 顶层 JSON 不是对象，无法解析 participant 数据")

    participants_all, _ = extract_participants(payload, source_json)
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

    ci_low, ci_high = bootstrap_ci_fisher_mean(
        mean_scores=mean_scores, n_bootstrap=n_bootstrap, seed=seed
    )

    summary_payload = {
        "source_file": str(source_json),
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
        "participant_mean_score_bootstrap_ci_95": {
            "lower": ci_low,
            "upper": ci_high,
        },
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


def main() -> None:
    args = parse_args()
    fits_root = args.fits_root.resolve()
    participant_id_set = parse_participant_id_set_from_args(args)

    if not fits_root.exists():
        raise FileNotFoundError(f"fits 根目录不存在: {fits_root}")

    json_files = sorted(
        [
            path
            for path in fits_root.rglob("*.json")
            if path.is_file() and not path.name.endswith("_summary.json")
        ]
    )
    if not json_files:
        raise FileNotFoundError(f"在 {fits_root} 下未找到可处理的 JSON 文件")

    print(f"[INFO] 扫描到 {len(json_files)} 个 JSON 文件，开始处理...")

    success = 0
    failed = 0
    for json_file in json_files:
        try:
            process_one_file(
                json_file,
                skip_existing=args.skip_existing,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                participant_first_n=args.participant_first_n,
                participant_id_set=participant_id_set,
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"[ERROR] 处理失败: {json_file}\n        异常: {exc}")

    print(f"[DONE] success={success}, failed={failed}")


if __name__ == "__main__":
    main()
