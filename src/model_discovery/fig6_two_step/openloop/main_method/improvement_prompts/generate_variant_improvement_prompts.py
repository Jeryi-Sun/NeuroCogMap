#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成“改进模型实现”用的 prompt 文件：

- 读取模板 improvement_prompt.md
- 读取 *_improvement_suggestions.json
- 将模板中的占位符 [LLM-INFERRED BEHAVIORAL SUMMARY] 替换为各变体的 variant_strategy.explanation
- 输出为 improvement_prompt_{variant_name}.md 到 improvement_prompts 目录

注意：
- 支持 --skip_existing：如果目标文件已存在则跳过（满足批处理跳过需求）
- 如果占位符不存在，会直接报错（不做静默兜底）
"""

import argparse
import json
import os
import re
from pathlib import Path


PLACEHOLDER = "[LLM-INFERRED BEHAVIORAL SUMMARY]"


def apply_suffix_placeholders(template: str, model_suffix: str, record_suffix: str) -> str:
    """
    在模板中替换 [MODEL_SUFFIX] 和 [RECORD_SUFFIX] 占位符。
    这两个占位符允许我们为不同数据源生成不同后缀的模型代码和记录文件。
    """
    return (
        template.replace("[MODEL_SUFFIX]", model_suffix)
        .replace("[RECORD_SUFFIX]", record_suffix)
    )


def sanitize_variant_name_for_filename(name: str) -> str:
    """
    将 variant 名称变为安全文件名（保留可读性，避免路径分隔符等问题）。
    """
    name = name.strip()
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"错误: 读取文件失败: {path}: {e}")
        raise


def load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"错误: 读取 JSON 失败: {path}: {e}")
        raise


def write_text(path: str, content: str, skip_existing: bool):
    if skip_existing and os.path.exists(path):
        print(f"跳过: 输出文件已存在: {path}")
        return False
    try:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"错误: 写入文件失败: {path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从 *_improvement_suggestions.json 为每个变体生成 improvement_prompt_{variant}.md"
    )
    parser.add_argument(
        "--suggestions_json",
        type=str,
        default="/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_suggestion_results"
        "kool2016when_exp2_improvement_suggestions.json",
        help="包含各变体 variant_strategy.explanation 的 JSON 文件路径",
    )
    parser.add_argument(
        "--source_schema",
        type=str,
        choices=["variant_suggestions", "parcel_activation"],
        default="variant_suggestions",
        help=(
            "输入 JSON 的结构类型："
            "variant_suggestions（原先 *_improvement_suggestions.json 的结构，顶层是 variant_name -> ...），"
            "或 parcel_activation（parcel_activation_cogneuromap_improvement_suggestions*.json 的结构）"
        ),
    )
    parser.add_argument(
        "--template_md",
        type=str,
        default="/path/to/project_root/Human_LLM_align/"
        "Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_prompts/"
        "improvement_prompt.md",
        help="模板 Markdown 文件路径（包含占位符 [LLM-INFERRED BEHAVIORAL SUMMARY]）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/path/to/project_root/Human_LLM_align/"
        "Llama-3.1-Centaur-70B-main/openloop/main_method/improvement_prompts",
        help="输出目录（默认写到 improvement_prompts 目录）",
    )
    parser.add_argument(
        "--model_suffix",
        type=str,
        default="",
        help="写入到模板中 [MODEL_SUFFIX] 占位符的内容，用于区分不同改进模型代码文件后缀（例如 _cogneuromap_simple）",
    )
    parser.add_argument(
        "--record_suffix",
        type=str,
        default="",
        help="写入到模板中 [RECORD_SUFFIX] 占位符的内容，用于区分不同改进记录 markdown 文件后缀",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果目标文件已存在则跳过",
    )
    parser.add_argument(
        "--only_variants",
        type=str,
        nargs="+",
        default=None,
        help="只生成这些变体（默认生成 JSON 中所有包含 variant_strategy 的变体）",
    )

    args = parser.parse_args()

    if not os.path.exists(args.suggestions_json):
        raise FileNotFoundError(f"suggestions_json 不存在: {args.suggestions_json}")
    if not os.path.exists(args.template_md):
        raise FileNotFoundError(f"template_md 不存在: {args.template_md}")

    template = load_text(args.template_md)
    if PLACEHOLDER not in template:
        raise ValueError(
            f"模板中未找到占位符 {PLACEHOLDER}，请检查模板文件: {args.template_md}"
        )

    # 先在模板级别插入后缀占位符，保证每个生成的 prompt 都有正确的目标文件名约定
    template = apply_suffix_placeholders(
        template, args.model_suffix, args.record_suffix
    )

    data = load_json(args.suggestions_json)
    if not isinstance(data, dict):
        raise ValueError("suggestions_json 的顶层必须是 dict")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 根据 source_schema 决定如何解析变体
    if args.source_schema == "variant_suggestions":
        # 保持原有逻辑：顶层键是 variant_name
        variant_names = list(data.keys())
        if args.only_variants is not None:
            requested = set(args.only_variants)
            variant_names = [v for v in variant_names if v in requested]
            missing = sorted(list(requested - set(variant_names)))
            if missing:
                print(f"警告: 以下变体在 JSON 中不存在，将被忽略: {missing}")
    else:
        # parcel_activation 结构：data['conditions']['LLM>Baseline']['step1_step2_reward_combined']['explanation']
        conditions = data.get("conditions", {})
        if "LLM>Baseline" not in conditions:
            raise ValueError("parcel_activation JSON 中缺少条件 'LLM>Baseline'")
        cond = conditions["LLM>Baseline"]
        combined = cond.get("step1_step2_reward_combined")
        if not isinstance(combined, dict):
            raise ValueError(
                "parcel_activation JSON 中缺少 'step1_step2_reward_combined' 字段"
            )
        # 从文件名和 meta 信息推一个变体名，便于区分 simple / full
        json_basename = os.path.basename(args.suggestions_json)
        mode_tag = "simple" if "simple" in json_basename else "cogneuromap"
        explanation = combined.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("parcel_activation JSON 中 explanation 为空")
        variant_key = f"LLM>Baseline_{mode_tag}"
        # 统一走后面的写入逻辑，这里先构造一个“伪 data”字典
        data = {
            variant_key: {
                "variant_strategy": {
                    "explanation": explanation.strip(),
                }
            }
        }
        variant_names = [variant_key]

    n_written = 0
    n_skipped = 0
    n_failed = 0

    for variant_name in variant_names:
        try:
            entry = data.get(variant_name, {})
            variant_strategy = entry.get("variant_strategy")
            if not isinstance(variant_strategy, dict):
                print(f"警告: 变体 {variant_name} 缺少 variant_strategy，跳过")
                n_skipped += 1
                continue

            explanation = variant_strategy.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                print(f"警告: 变体 {variant_name} 缺少 explanation 或为空，跳过")
                n_skipped += 1
                continue

            filled = template.replace(PLACEHOLDER, explanation.strip())
            safe_name = sanitize_variant_name_for_filename(variant_name)
            out_path = str(output_dir / f"improvement_prompt_{safe_name}.md")

            wrote = write_text(out_path, filled, skip_existing=args.skip_existing)
            if wrote:
                print(f"已生成: {out_path}")
                n_written += 1
            else:
                n_skipped += 1

        except Exception as e:
            print(f"错误: 处理变体失败: {variant_name}: {e}")
            n_failed += 1
            continue

    print("\n完成。统计：")
    print(f"  写入: {n_written}")
    print(f"  跳过: {n_skipped}")
    print(f"  失败: {n_failed}")


if __name__ == "__main__":
    main()


