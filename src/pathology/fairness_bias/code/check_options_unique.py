#!/usr/bin/env python3
"""
统计多个 JSONL 文件中 options 字段包含的所有唯一元素。
仅输出所有文件合并后的全局唯一 options，按出现次数从多到少排序。
"""

import json
from pathlib import Path
from collections import Counter


def collect_option_counts(files: list[Path]) -> Counter:
    """从多个 JSONL 文件中统计每个 option 的出现次数，返回 Counter。"""
    counter = Counter()
    for path in files:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                opts = obj.get("options")
                if opts is None:
                    continue
                for o in opts:
                    s = o.strip() if isinstance(o, str) else str(o)
                    if s:
                        counter[s] += 1
    return counter


def main():
    import sys
    out_path = None
    if len(sys.argv) > 1 and sys.argv[1] == "--out":
        out_path = sys.argv[2] if len(sys.argv) > 2 else "options_unique_report.txt"

    base = Path("/path/to/project_root/safety_explanation/fairness_bias/results")
    files = [
        base / "bbq_age_gemma-2-2b" / "correct.jsonl",
        base / "bbq_age_gemma-2-2b" / "incorrect.jsonl",
        base / "bbq_disability_status_gemma-2-2b" / "correct.jsonl",
        base / "bbq_disability_status_gemma-2-2b" / "incorrect.jsonl",
        base / "bbq_gender_identity_gemma-2-2b" / "correct.jsonl",
        base / "bbq_gender_identity_gemma-2-2b" / "incorrect.jsonl",
        base / "bbq_nationality_gemma-2-2b" / "correct.jsonl",
        base / "bbq_nationality_gemma-2-2b" / "incorrect.jsonl",
    ]

    counter = collect_option_counts(files)
    # 按出现次数从多到少排序，次数相同则按选项字符串排序
    sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    lines_out = [
        "全局唯一 options（所有文件合并），按出现次数从多到少排序",
        "=" * 60,
        f"唯一 option 数量: {len(counter)}",
        "",
    ]
    for h in lines_out:
        print(h)
    for opt, count in sorted_items:
        line = f"  {count:6d}  {opt!r}"
        lines_out.append(line)
        print(line)

    text = "\n".join(lines_out)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"\n完整结果已写入: {out_path}")


if __name__ == "__main__":
    main()
