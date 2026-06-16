#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在各个 Yeo7 network 内，用 LLM 判断：
每个 Human parcel 的认知功能描述（Human parcel_descriptions.json 中的 function_description）
是否能被其 top-k LLM parcel 的认知功能（top_human_parcels_per_llm.csv 中的 llm_function）所“覆盖/匹配”。

输入：
  1) --input-csv: top_human_parcels_per_llm.csv
     期望包含列：
       human_parcel, human_parcel_name, human_function,
       llm_parcel, llm_function, rank_by_acc, selection_type,
       prediction_accuracy, semantic_similarity
  2) --human-parcel-descriptions: parcel_descriptions.json（人脑 parcel 的 function_description 来源）

输出（写到 --output-dir）：
  - judgements.jsonl: 每个 human parcel 一条 judge 结果（含 prompt 与原始返回，便于审计）
  - network_summary.csv: 按 network 汇总的匹配率统计
  - parcel_summary.csv: 每个 human parcel 的匹配结论与最匹配 LLM 条目

LLM 调用：
  兼容 OpenAI / vLLM 的 /v1/chat/completions 接口（参考 openloop/main_method 的实现）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def extract_network_from_human_parcel_name(parcel_name: str) -> Optional[str]:
    # e.g. 7Networks_LH_Vis_1 -> Vis
    parts = (parcel_name or "").split("_")
    if len(parts) >= 3 and parts[0] == "7Networks":
        return parts[2]
    return None


def load_human_parcel_descriptions(json_path: Path) -> Dict[str, Dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"找不到 human parcel 描述文件: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} 顶层应为 list，但实际为 {type(data)}")
    out: Dict[str, Dict[str, Any]] = {}
    for item in data:
        name = (item or {}).get("parcel_name")
        if name:
            out[str(name)] = item
    if not out:
        raise ValueError(f"{json_path} 未加载到任何 parcel_name")
    return out


def extract_llm_parcel_id(llm_parcel_label: str) -> int:
    # e.g. "LLM_Parcel_221" -> 221
    parts = (llm_parcel_label or "").split("_")
    if len(parts) >= 3 and parts[0] == "LLM" and parts[1] == "Parcel":
        return int(parts[2])
    raise ValueError(f"无法解析 llm_parcel id: {llm_parcel_label}")


def load_llm_parcel_long_descriptions(llm_parcel_file: Path) -> Dict[int, Dict[str, str]]:
    """
    读取 LLM parcel 的长描述（用于 prompt）。

    兼容两类常见格式：
    1) dict 且包含 'parcel_summaries'（compute_parcel_similarity_llm.py 的默认文件格式）
    2) list 直接是 summaries

    返回：
      parcel_id -> {
        "function_name": str,
        "description_long": str
      }
    """
    if not llm_parcel_file.exists():
        raise FileNotFoundError(f"找不到 LLM parcel 描述文件: {llm_parcel_file}")

    with llm_parcel_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "parcel_summaries" in data:
        parcel_summaries = data["parcel_summaries"]
    elif isinstance(data, list):
        parcel_summaries = data
    else:
        raise ValueError(f"LLM parcel 文件格式不符合预期: {type(data)}")

    out: Dict[int, Dict[str, str]] = {}
    for item in parcel_summaries:
        if not isinstance(item, dict):
            continue
        parcel_id = item.get("parcel_id")
        if parcel_id is None:
            continue

        fn = str(item.get("function_name", "")).replace("**", "").strip()
        fn = " ".join(fn.split())

        # 历史字段名兼容：function_description / functionality_description / model_role 等
        desc = str(item.get("function_description", "")).strip()
        if not desc:
            desc = str(item.get("functionality_description", "")).strip()

        model_role = str(item.get("model_role", "")).strip()
        if not model_role:
            model_role = str(item.get("role_in_large_model", "")).strip()

        chunks = []
        if fn:
            chunks.append(f"Function name: {fn}")
        if desc:
            chunks.append(f"Function description: {desc}")
        if model_role:
            chunks.append(f"Role in model: {model_role}")

        description_long = "\n".join(chunks).strip()
        if not description_long:
            continue

        out[int(parcel_id)] = {"function_name": fn, "description_long": description_long}

    if not out:
        raise ValueError(f"{llm_parcel_file} 未加载到任何 LLM parcel 长描述（parcel_id/function_description 等字段）")
    return out


def call_vllm_api(
    vllm_url: str,
    api_key: str,
    prompt: str,
    model: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
    reasoning_effort: Optional[str] = None,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    timeout: int = 120,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a cognitive neuroscience expert. You MUST return STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Some OpenAI-compatible gateways support reasoning controls. If unsupported, it should be ignored.
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{vllm_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if not (data and isinstance(data, dict) and "choices" in data and data["choices"]):
                raise ValueError(f"响应格式不正确: {data}")
            message = data["choices"][0].get("message", {})
            content = (message or {}).get("content", "")
            content = (content or "").strip()
            if not content:
                import pdb; pdb.set_trace()
                raise ValueError(f"模型返回 content 为空，无法解析 JSON。完整响应: {data}")
            return content
        except Exception as exc:
            last_err = exc
            if attempt < max_retries - 1:
                print(f"[Warn] API 调用失败，{retry_delay}s 后重试 ({attempt + 1}/{max_retries}): {exc}")
                time.sleep(retry_delay)
            else:
                print(f"[Error] API 调用失败，已达最大重试次数: {exc}")
                raise
    raise RuntimeError(f"API 调用失败: {last_err}")


def _extract_json_block(text: str) -> str:
    """
    允许模型输出额外文本；尽量截取第一个 JSON object。
    不做“吞异常式”处理：解析失败会在上层报错并写入原始输出便于排查。
    """
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


@dataclass
class JudgeResult:
    overall_match_level: str
    overall_score: float
    best_match_llm_parcel: Optional[str]
    best_match_score: Optional[float]
    matched_llm_parcels: List[Dict[str, Any]]
    raw_json: Dict[str, Any]


def parse_judge_json(raw_text: str) -> JudgeResult:
    json_text = _extract_json_block(raw_text)
    try:
        obj = json.loads(json_text)
    except Exception as exc:
        import pdb; pdb.set_trace()
        raise ValueError(f"LLM 返回不是 JSON object: {json_text}") from exc
    if not isinstance(obj, dict):
        raise ValueError("LLM 返回不是 JSON object")

    overall_match_level = obj.get("overall_match_level")
    if not isinstance(overall_match_level, str):
        raise ValueError(f"overall_match_level 字段缺失或非 str: {obj.get('overall_match_level')}")
    overall_match_level = overall_match_level.strip().lower()
    if overall_match_level not in {"full", "partial", "none"}:
        raise ValueError(f"overall_match_level 必须是 full/partial/none，当前: {overall_match_level}")

    overall_score = obj.get("overall_score")
    try:
        overall_score = float(overall_score)
    except Exception as exc:
        raise ValueError(f"overall_score 字段缺失或非数值: {obj.get('overall_score')}") from exc
    if not (0.0 <= float(overall_score) <= 1.0):
        raise ValueError(f"overall_score 必须在[0,1]，当前: {overall_score}")

    matched = obj.get("matched_llm_parcels", [])
    if matched is None:
        matched = []
    if not isinstance(matched, list):
        raise ValueError("matched_llm_parcels 必须是 list")

    expected_aspect_keys = {
        "perception_attention",
        "memory_knowledge",
        "language_concepts",
        "reasoning_executive",
        "learning_reward_adaptation",
        "emotion_social_action",
    }
    for i, item in enumerate(matched):
        if not isinstance(item, dict):
            raise ValueError(f"matched_llm_parcels[{i}] 不是 object")
        match_level = item.get("match_level")
        if not isinstance(match_level, str):
            raise ValueError(f"matched_llm_parcels[{i}].match_level 缺失或非 str: {item.get('match_level')}")
        match_level = match_level.strip().lower()
        if match_level not in {"full", "partial", "none"}:
            raise ValueError(f"matched_llm_parcels[{i}].match_level 必须是 full/partial/none，当前: {match_level}")
        aspect_scores = item.get("aspect_scores")
        if aspect_scores is None:
            continue
        if not isinstance(aspect_scores, dict):
            raise ValueError(f"matched_llm_parcels[{i}].aspect_scores 必须是 object")
        missing = expected_aspect_keys - set(aspect_scores.keys())
        if missing:
            raise ValueError(f"matched_llm_parcels[{i}].aspect_scores 缺少字段: {sorted(missing)}")
        # 数值校验（出现非数值直接报错，便于及时发现 prompt 不稳定）
        for k in expected_aspect_keys:
            try:
                v = float(aspect_scores.get(k))
            except Exception as exc:
                raise ValueError(f"matched_llm_parcels[{i}].aspect_scores.{k} 非数值: {aspect_scores.get(k)}") from exc
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"matched_llm_parcels[{i}].aspect_scores.{k} 必须在[0,1]，当前: {v}")

    best_parcel = obj.get("best_match_llm_parcel")
    best_score = obj.get("best_match_score")
    if best_score is not None:
        try:
            best_score = float(best_score)
        except Exception as exc:
            raise ValueError(f"best_match_score 无法转 float: {best_score}") from exc
        if not (0.0 <= float(best_score) <= 1.0):
            raise ValueError(f"best_match_score 必须在[0,1]，当前: {best_score}")

    return JudgeResult(
        overall_match_level=overall_match_level,
        overall_score=float(overall_score),
        best_match_llm_parcel=str(best_parcel) if best_parcel is not None else None,
        best_match_score=best_score,
        matched_llm_parcels=matched,
        raw_json=obj,
    )


def build_prompt(
    network: str,
    human_parcel_name: str,
    human_function: str,
    human_function_description: str,
    llm_topk_rows: List[Dict[str, Any]],
    top_k: int,
) -> str:
    items_lines = []
    for r in llm_topk_rows:
        items_lines.append(
            (
                f"- llm_parcel: {r['llm_parcel']}\n"
                f"  llm_function_short: {r.get('llm_function','')}\n"
                f"  llm_function_long: {r.get('llm_function_long','')}\n"
                f"  rank_by_acc: {r.get('rank_by_acc')}\n"
                f"  prediction_accuracy: {r.get('prediction_accuracy')}\n"
                f"  semantic_similarity: {r.get('semantic_similarity')}\n"
            )
        )
    llm_list_text = "\n".join(items_lines)

    return f"""You are a strict evaluator for cognitive function alignment.

Task: Judge whether the HUMAN parcel's long cognitive function description is matched/covered by ANY of the top-{top_k} LLM parcels' long cognitive function descriptions.

Important:
- Compare from the following core cognitive-function perspectives, then aggregate:
  (1) Perception & Attentional Selection:
      obtaining visual/auditory/language/face/spatial information; attentional selection; distraction suppression.
  (2) Memory & Knowledge Representation:
      encoding/maintenance/consolidation/retrieval; working/episodic/semantic/autobiographical memory.
  (3) Language & Conceptual Understanding:
      comprehension, lexical access, sentence processing, naming, reading, expression, concept formation, semantic processing.
  (4) Reasoning, Decision-Making & Executive Control:
      reasoning, judgment, planning, goal maintenance, rule use, response selection, inhibition, monitoring.
  (5) Learning, Reward & Adaptation:
      feedback/reward/risk/uncertainty-driven adjustment; reinforcement learning; skill formation; strategy updating.
  (6) Emotion, Social Cognition & Action Regulation:
      emotion/motivation/stress/self-other understanding; translating cognition into action, navigation, communication, and goal-directed behavior.
- Score each aspect in [0,1] and provide an overall_score in [0,1].
- You MUST output a 3-class coverage label:
  - full: the top-k set fully covers the human parcel function (core process and key subcomponents are covered).
  - partial: some core aspects are covered but important components are missing or only weakly covered.
  - none: essentially no meaningful coverage.
- Decide based on meaning, not word overlap.
- Output MUST be valid JSON (no markdown, no extra text).

Human parcel info:
- network: {network}
- human_parcel_name: {human_parcel_name}
- human_function (short): {human_function}
- human_function_description (long): {human_function_description}

Top-{top_k} candidate LLM parcels:
{llm_list_text}

Return JSON with this schema:
{{
  "overall_match_level": "full",
  "overall_score": number between 0 and 1,
  "best_match_llm_parcel": "LLM_Parcel_###" or null,
  "best_match_score": number between 0 and 1 or null,
  "matched_llm_parcels": [
    {{
      "llm_parcel": "LLM_Parcel_###",
      "match_level": "full",
      "overall_score": number between 0 and 1,
      "aspect_scores": {{
        "perception_attention": number,
        "memory_knowledge": number,
        "language_concepts": number,
        "reasoning_executive": number,
        "learning_reward_adaptation": number,
        "emotion_social_action": number
      }},
      "reason": "one short sentence"
    }}
  ],
  "overall_reason": "one short sentence"
}}

Allowed values:
- overall_match_level: one of ["full","partial","none"]
- match_level: one of ["full","partial","none"]
"""


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def load_existing_judgements(jsonl_path: Path) -> Tuple[Dict[str, Dict[str, Any]], set]:
    """
    断点续跑：从 judgements.jsonl 读取已完成的 human_parcel_name。

    返回：
      - existing_records: human_parcel_name -> record（仅保留最后一次出现的记录）
      - completed_names: set(human_parcel_name)
    """
    existing_records: Dict[str, Dict[str, Any]] = {}
    completed_names: set = set()
    if not jsonl_path.exists():
        return existing_records, completed_names

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 允许历史坏行存在（例如中途写入/外部编辑），不让它阻塞续跑
                print(f"[Warn] 无法解析现有 jsonl 第 {line_no} 行，跳过")
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("human_parcel_name")
            # parse_error 记录说明该条未成功完成，不纳入 completed
            if not name or obj.get("parse_error"):
                continue
            existing_records[str(name)] = obj
            completed_names.add(str(name))

    return existing_records, completed_names


def should_skip_file(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"[Skip] 输出已存在: {path}（使用 --overwrite 可重新生成）")
        return True
    return False


def _normalize_topk_rows(df: pd.DataFrame, top_k: int, selection_type: str) -> pd.DataFrame:
    required_cols = {
        "human_parcel",
        "human_function",
        "human_parcel_name",
        "llm_parcel",
        "llm_function",
        "rank_by_acc",
        "selection_type",
        "prediction_accuracy",
        "semantic_similarity",
    }
    missing = [c for c in sorted(required_cols) if c not in df.columns]
    if missing:
        raise ValueError(f"输入 CSV 缺少列: {missing}")

    df = df.copy()
    df["rank_by_acc"] = pd.to_numeric(df["rank_by_acc"], errors="raise")
    df = df[df["selection_type"].astype(str) == str(selection_type)]
    df = df[df["rank_by_acc"] <= top_k]
    if df.empty:
        raise ValueError(f"过滤后数据为空：selection_type={selection_type}, top_k={top_k}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM judge: Human parcel function description vs top-k LLM parcel functions (by network)")
    parser.add_argument("--input-csv", type=Path, required=True, help="top_human_parcels_per_llm.csv 路径")
    parser.add_argument(
        "--human-parcel-descriptions",
        type=Path,
        default=Path(
            "/path/to/project_root/Human_LLM_align/"
            "litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
        ),
        help="human parcel 描述 JSON（含 function_description）路径",
    )
    parser.add_argument(
        "--llm-parcel-descriptions",
        type=Path,
        default=Path(
            "/path/to/project_root/neural_area/divide_area_by_sae_act/"
            "cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/"
            "latent_parcel_topsamples_functionality_summary.json"
        ),
        help="LLM parcel 长描述 JSON（含 function_description/functionality_description 等）路径",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--top-k", type=int, default=10, help="取每个 human parcel 的 top-k LLM parcels（默认 10）")
    parser.add_argument("--selection-type", type=str, default="top", help="selection_type 过滤（默认 top）")

    parser.add_argument("--vllm-url", type=str, default="http://0.0.0.0:8000/v1", help="兼容 OpenAI 的 API base url（末尾不需要 /）")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key（不提供则读取 OPENAI_API_KEY 或 VLLM_API_KEY；仍为空则报错）",
    )
    parser.add_argument("--model", type=str, required=True, help="模型名（如 gpt-4o / gpt-5.2-... / 本地 vLLM 模型名）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=10000, help="completion max_tokens（你可按需要调大/调小）")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="可选：若网关支持，用于控制隐藏推理开销（不支持时会被忽略）",
    )
    parser.add_argument("--sleep", type=float, default=0.2, help="每次调用后 sleep 秒数，避免限流")
    parser.add_argument("--human-desc-max-chars", type=int, default=0, help="human 长描述截断字符数（0表示不截断）")
    parser.add_argument("--llm-desc-max-chars", type=int, default=0, help="每个候选 LLM 长描述截断字符数（0表示不截断）")

    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出文件（默认跳过）")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 judgements.jsonl 已存在则跳过整个任务（默认不跳过；更细粒度跳过由 overwrite 控制）",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError(f"--top-k 必须 > 0，当前为 {args.top_k}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY")
    if not api_key:
        raise ValueError("未提供 --api-key，且环境变量 OPENAI_API_KEY / VLLM_API_KEY 也不存在")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "judgements.jsonl"
    out_network_csv = args.output_dir / "network_summary.csv"
    out_parcel_csv = args.output_dir / "parcel_summary.csv"

    if args.skip_existing and out_jsonl.exists() and not args.overwrite:
        print(f"[Skip] {out_jsonl} 已存在（--skip-existing 生效，且未指定 --overwrite）")
        return

    if args.overwrite:
        # 覆盖意味着重跑：删除旧输出，避免重复追加。
        for p in [out_jsonl, out_network_csv, out_parcel_csv]:
            if p.exists():
                p.unlink()

    # 断点续跑：读取已完成记录，避免重复调用 LLM
    existing_records, completed_names = load_existing_judgements(out_jsonl)
    if completed_names and not args.overwrite:
        print(f"[Resume] 检测到已有结果 {len(completed_names)} 条，将跳过这些 human_parcel_name")

    print(f"读取输入 CSV: {args.input_csv}")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"找不到输入 CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df = _normalize_topk_rows(df, top_k=args.top_k, selection_type=args.selection_type)

    print(f"读取 human parcel 描述: {args.human_parcel_descriptions}")
    human_desc = load_human_parcel_descriptions(args.human_parcel_descriptions)

    print(f"读取 LLM parcel 长描述: {args.llm_parcel_descriptions}")
    llm_desc = load_llm_parcel_long_descriptions(args.llm_parcel_descriptions)

    # 以 human_parcel_name 为主键（更稳定）；同时保留 human_parcel 与 human_function。
    grouped = df.sort_values(["human_parcel_name", "rank_by_acc"]).groupby("human_parcel_name", sort=False)

    # parcel_summary 需要覆盖“已有 + 新增”的全量结果
    parcel_rows_out: List[Dict[str, Any]] = []
    for name, rec in existing_records.items():
        # 兼容旧字段名：overall_match(旧 bool) -> overall_match_level(新)
        match_level = rec.get("overall_match_level")
        if not match_level and "overall_match" in rec:
            match_level = "full" if rec.get("overall_match") is True else "none"
        parcel_rows_out.append(
            {
                "human_parcel_name": name,
                "human_parcel": rec.get("human_parcel", ""),
                "human_function": rec.get("human_function", ""),
                "network": rec.get("network", ""),
                "overall_match_level": match_level,
                "best_match_llm_parcel": rec.get("best_match_llm_parcel"),
                "overall_score": rec.get("overall_score"),
                "best_match_score": rec.get("best_match_score"),
            }
        )

    jsonl_f = out_jsonl.open("a", encoding="utf-8")

    processed = 0
    for human_parcel_name, g in grouped:
        processed += 1
        if (str(human_parcel_name) in completed_names) and not args.overwrite:
            print(f"[Skip] 已有结果，跳过: {human_parcel_name}")
            continue
        network = extract_network_from_human_parcel_name(str(human_parcel_name))
        if not network:
            raise ValueError(f"无法从 human_parcel_name 解析 network: {human_parcel_name}")

        human_info = human_desc.get(str(human_parcel_name))
        if not human_info:
            raise KeyError(f"在 parcel_descriptions.json 中找不到 {human_parcel_name}")

        human_function = str(g["human_function"].iloc[0])
        human_parcel = str(g["human_parcel"].iloc[0])
        human_function_description = str(human_info.get("function_description", "")).strip()
        if not human_function_description:
            raise ValueError(f"{human_parcel_name} 的 function_description 为空")
        human_function_description = truncate_text(human_function_description, args.human_desc_max_chars)

        llm_rows = g.sort_values("rank_by_acc").to_dict(orient="records")
        # 为每个候选 LLM parcel 补充长描述（用于更严格的语义对齐判断）
        for r in llm_rows:
            llm_pid = extract_llm_parcel_id(str(r.get("llm_parcel", "")))
            if llm_pid not in llm_desc:
                raise KeyError(f"LLM parcel 描述缺失：{r.get('llm_parcel')} (id={llm_pid}) 不在 {args.llm_parcel_descriptions}")
            r["llm_function_long"] = llm_desc[llm_pid]["description_long"]
        prompt = build_prompt(
            network=network,
            human_parcel_name=str(human_parcel_name),
            human_function=human_function,
            human_function_description=human_function_description,
            llm_topk_rows=llm_rows,
            top_k=args.top_k,
        )
        print(f"[{processed}/{len(grouped)}] Judge {human_parcel_name} ({network}) ...")
        raw_response = call_vllm_api(
            vllm_url=args.vllm_url,
            api_key=api_key,
            prompt=prompt,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )

        try:
            parsed = parse_judge_json(raw_response)
        except Exception as exc:
            # 不吞异常：把 raw_response 写入 jsonl 以便定位，然后抛出错误终止。
            jsonl_f.write(
                json.dumps(
                    {
                        "human_parcel_name": str(human_parcel_name),
                        "human_parcel": human_parcel,
                        "human_function": human_function,
                        "network": network,
                        "top_k": args.top_k,
                        "selection_type": args.selection_type,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parse_error": str(exc),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            jsonl_f.flush()
            raise

        record = {
            "human_parcel_name": str(human_parcel_name),
            "human_parcel": human_parcel,
            "human_function": human_function,
            "network": network,
            "top_k": args.top_k,
            "selection_type": args.selection_type,
            "best_match_llm_parcel": parsed.best_match_llm_parcel,
            "overall_score": parsed.overall_score,
            "best_match_score": parsed.best_match_score,
            "overall_match_level": parsed.overall_match_level,
            "matched_llm_parcels": parsed.matched_llm_parcels,
            "prompt": prompt,
            "raw_response": raw_response,
        }
        jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_f.flush()

        parcel_rows_out.append(
            {
                "human_parcel_name": str(human_parcel_name),
                "human_parcel": human_parcel,
                "human_function": human_function,
                "network": network,
                "overall_match_level": parsed.overall_match_level,
                "best_match_llm_parcel": parsed.best_match_llm_parcel,
                "overall_score": parsed.overall_score,
                "best_match_score": parsed.best_match_score,
            }
        )
        completed_names.add(str(human_parcel_name))

        time.sleep(max(args.sleep, 0.0))

    jsonl_f.close()

    parcel_summary = pd.DataFrame(parcel_rows_out)
    parcel_summary.to_csv(out_parcel_csv, index=False)

    # network 汇总
    # network 汇总：三分类占比 + 均分
    tmp = parcel_summary.copy()
    tmp["parcel_count"] = 1
    counts = (
        tmp.pivot_table(
            index="network",
            columns="overall_match_level",
            values="parcel_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    for col in ["full", "partial", "none"]:
        if col not in counts.columns:
            counts[col] = 0

    totals = tmp.groupby("network", as_index=False)["parcel_count"].sum().rename(columns={"parcel_count": "parcel_count"})
    score_mean = tmp.groupby("network", as_index=False)["overall_score"].mean().rename(columns={"overall_score": "overall_score_mean"})

    network_summary = counts.merge(totals, on="network", how="left").merge(score_mean, on="network", how="left")
    network_summary["full_rate"] = network_summary["full"] / network_summary["parcel_count"]
    network_summary["partial_rate"] = network_summary["partial"] / network_summary["parcel_count"]
    network_summary["none_rate"] = network_summary["none"] / network_summary["parcel_count"]
    network_summary = network_summary.reindex(columns=[
        "network",
        "parcel_count",
        "full", "partial", "none",
        "full_rate", "partial_rate", "none_rate",
        "overall_score_mean",
    ])
    network_summary = network_summary.set_index("network").reindex([n for n in NETWORK_ORDER if n in tmp["network"].unique()]).reset_index()
    network_summary.to_csv(out_network_csv, index=False)

    print("完成。输出：")
    print(f"- {out_jsonl}")
    print(f"- {out_parcel_csv}")
    print(f"- {out_network_csv}")


if __name__ == "__main__":
    main()

