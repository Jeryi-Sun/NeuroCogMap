#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan eval jsonl files and split each into correct/incorrect groups, "
            "organizing outputs by model name derived from filename."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/path/to/project_root/safety_explanation/hallucination/results/",
        help="Directory containing eval jsonl files (filenames include 'eval').",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help=(
            "Root directory to write outputs. Default: same as --results-dir. "
            "Final structure: <output-root>/<model_name>/(correct.jsonl, incorrect.jsonl)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If set, skip processing a file when both target outputs already exist."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; only print planned actions.",
    )
    return parser.parse_args()


def derive_model_name(filename: str) -> str:
    """
    Derive model name from an eval filename.

    Strategy: take the basename without extension, then remove a trailing
    "_eval" suffix if present. Example: "dolly_close_gemma-2-2b_eval.jsonl"
    -> "dolly_close_gemma-2-2b".
    """
    base = os.path.basename(filename)
    stem = base
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    if stem.endswith("_eval"):
        stem = stem[: -len("_eval")]
    return stem


def ensure_dirs(path: str, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] mkdir -p {path}")
        return
    os.makedirs(path, exist_ok=True)


def target_paths(output_root: str, model_name: str) -> Tuple[str, str]:
    model_dir = os.path.join(output_root, model_name)
    return (
        os.path.join(model_dir, "correct.jsonl"),
        os.path.join(model_dir, "incorrect.jsonl"),
    )


def outputs_exist(correct_path: str, incorrect_path: str) -> bool:
    return os.path.exists(correct_path) and os.path.exists(incorrect_path)


def split_file(
    src_path: str,
    correct_out_path: str,
    incorrect_out_path: str,
    dry_run: bool = False,
) -> Tuple[int, int, int]:
    total = 0
    num_correct = 0
    num_incorrect = 0

    if dry_run:
        print(
            f"[DRY-RUN] Would split {src_path} -> {correct_out_path} / {incorrect_out_path}"
        )
        return (total, num_correct, num_incorrect)

    # Ensure parent directories exist (now the model directory only)
    ensure_dirs(os.path.dirname(correct_out_path), dry_run=False)
    ensure_dirs(os.path.dirname(incorrect_out_path), dry_run=False)

    # Stream process to avoid large memory usage
    try:
        with open(src_path, "r", encoding="utf-8") as fin, \
             open(correct_out_path, "w", encoding="utf-8") as fcorrect, \
             open(incorrect_out_path, "w", encoding="utf-8") as finncorrect:
            for line_idx, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except Exception as ex:
                    print(
                        f"[ERROR] JSON parse failed in {src_path} line {line_idx}: {ex}",
                        file=sys.stderr,
                    )
                    continue

                if "is_correct" not in record and "refusal_detected" not in record:
                    print(
                        f"[WARN] Missing 'is_correct' in {src_path} line {line_idx}; skipping",
                        file=sys.stderr,
                    )
                    continue

                is_correct = record["is_correct"] if "is_correct" in record else record["refusal_detected"]
                # Only accept boolean truth values; non-bool treated as warning and skipped
                if isinstance(is_correct, bool):
                    if is_correct:
                        num_correct += 1
                        fcorrect.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        num_incorrect += 1
                        finncorrect.write(json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    print(
                        f"[WARN] Non-boolean 'is_correct' in {src_path} line {line_idx}; skipping",
                        file=sys.stderr,
                    )
    except Exception as ex:
        print(f"[ERROR] Failed processing {src_path}: {ex}", file=sys.stderr)

    return (total, num_correct, num_incorrect)


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_root = (
        os.path.abspath(args.output_root) if args.output_root else results_dir
    )

    if not os.path.isdir(results_dir):
        print(f"[ERROR] results directory does not exist: {results_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover candidate files: include 'eval' and end with .jsonl
    try:
        entries = sorted(os.listdir(results_dir))
    except Exception as ex:
        print(f"[ERROR] Unable to list directory {results_dir}: {ex}", file=sys.stderr)
        sys.exit(1)

    candidates = [
        os.path.join(results_dir, name)
        for name in entries
        if name.endswith(".jsonl") and ("eval" in name)
    ]

    if not candidates:
        print(f"[INFO] No eval jsonl files found in {results_dir}")
        return

    print(f"[INFO] Found {len(candidates)} eval files")

    for src_path in candidates:
        model_name = derive_model_name(src_path)
        correct_path, incorrect_path = target_paths(output_root, model_name)

        if args.skip_existing and outputs_exist(correct_path, incorrect_path):
            print(
                f"[SKIP] Outputs exist for model '{model_name}':\n"
                f"       {correct_path}\n"
                f"       {incorrect_path}"
            )
            continue

        # Ensure directory structure
        ensure_dirs(os.path.dirname(correct_path), dry_run=args.dry_run)
        ensure_dirs(os.path.dirname(incorrect_path), dry_run=args.dry_run)

        total, n_ok, n_bad = split_file(
            src_path=src_path,
            correct_out_path=correct_path,
            incorrect_out_path=incorrect_path,
            dry_run=args.dry_run,
        )

        print(
            f"[DONE] {os.path.basename(src_path)} -> model='{model_name}': "
            f"total={total}, correct={n_ok}, incorrect={n_bad}"
        )


if __name__ == "__main__":
    main()


