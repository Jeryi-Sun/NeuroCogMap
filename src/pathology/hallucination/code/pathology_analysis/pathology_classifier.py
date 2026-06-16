#!/usr/bin/env python3
"""
Pathology Classifier for Capabilities and Parcels

使用本地 vLLM 模型，将 capability 描述（基于 definition_refined）
和 SAE parcel 功能描述（基于 function_description）分类为：
- Belief-related
- Control-related
- Mixed
- Neutral
- Unknown (如果明显不属于上述任何类别)

并给出简短的认知神经科学视角的解释，以及置信度评分（1-10分）。
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/path/to/project_root"
LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "safety_explanation",
    "hallucination",
    "code",
    "pathology_analysis",
    "logs",
)
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = f"pathology_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_filepath = os.path.join(LOG_DIR, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class PathologyClassifier:
    """
    使用 LLM 对能力 / parcel 进行信念相关 vs 控制相关病理分类。
    """

    SYSTEM_PROMPT = """You are an expert in cognitive neuroscience and large language model interpretability.

We have defined two types of cognitive pathologies in foundation models:

1. Belief Representation Pathologies:
   - Arise from distortions or drift in semantic representation.
   - Concern factual grounding, knowledge integration, belief updating, and internal representational coherence.
   - Related behaviors: hallucination, representational bias.

2. Behavioral Control Pathologies:
   - Arise from dysfunction in executive control, inhibition, and strategic arbitration.
   - Concern regulation of competing drives (e.g., preference vs safety, helpfulness vs honesty).
   - Related behaviors: refusal failure, sycophancy.

For each item you receive, classify it into:
- Belief-related
- Control-related
- Mixed (if it clearly spans both)
- Neutral (if not strongly related to either pathology)
- Unknown (if it clearly does not belong to any of the above categories)

IMPORTANT:
- Focus on the primary cognitive function.
- If the capability primarily concerns semantic representation, grounding, or knowledge integration → Belief-related.
- If it primarily concerns inhibition, arbitration, strategic regulation, or goal control → Control-related.
- If it clearly does not relate to either pathology type → Unknown.
- Provide a short 1-2 sentence justification grounded in cognitive neuroscience.
- Provide a confidence score from 1-10, where:
  * 1-3: Low confidence, uncertain classification
  * 4-6: Moderate confidence
  * 7-8: High confidence
  * 9-10: Very high confidence, very certain

Your output for EACH item MUST be a single JSON object, without any extra commentary, in the form:
{
  "category": "...",
  "justification": "...",
  "confidence": [integer from 1 to 10]
}
"""

    def __init__(self, vllm_url: str, api_key: str) -> None:
        self.vllm_url = vllm_url.rstrip("/")
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        logger.info("Initialized PathologyClassifier with vLLM URL: %s", self.vllm_url)

    # ------------------------------------------------------------------
    # LLM 调用
    # ------------------------------------------------------------------
    def call_vllm_api(
        self,
        user_prompt: str,
        max_tokens: int = 1024,  # 增加以容纳置信度字段
        temperature: float = 0.3,
    ) -> str:
        payload = {
            "model": "gpt-5.2-2025-12-11", #"/path/to/local_models/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            response = requests.post(
                f"{self.vllm_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()

            if (
                result
                and "choices" in result
                and len(result["choices"]) > 0
                and "message" in result["choices"][0]
                and "content" in result["choices"][0]["message"]
            ):
                content = result["choices"][0]["message"]["content"]
                if content is None:
                    error_msg = "API 返回的 content 为 None"
                    logger.error(error_msg)
                    return f"ERROR: {error_msg}"
                return content.strip()

            error_msg = f"API 返回结果格式异常: {result}"
            logger.error(error_msg)
            return f"ERROR: {error_msg}"
        except Exception as exc:
            error_msg = f"调用 vLLM API 失败: {exc}"
            logger.exception(error_msg)
            return f"ERROR: {error_msg}"

    # ------------------------------------------------------------------
    # Prompt 构造
    # ------------------------------------------------------------------
    @staticmethod
    def build_capability_prompt(
        item_name: str,
        definition_refined: str,
    ) -> str:
        return (
            "Below is a capability description.\n\n"
            f"Capability name: {item_name}\n\n"
            "Definition (refined):\n"
            f"{definition_refined}\n\n"
            "Classify this single capability according to the instructions."
        )

    @staticmethod
    def build_parcel_prompt(
        parcel_id: int,
        function_name: str,
        function_description: str,
    ) -> str:
        return (
            "Below is a functional parcel extracted from an SAE-based analysis "
            "of a large language model.\n\n"
            f"Parcel ID: {parcel_id}\n"
            f"Function name: {function_name}\n\n"
            "Function description:\n"
            f"{function_description}\n\n"
            "Classify this single parcel according to the instructions."
        )

    # ------------------------------------------------------------------
    # 解析与回退
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json_block(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError("未在模型输出中找到 JSON 块")
        return text[start:end]

    def parse_classification(self, raw_response: str) -> Dict[str, Any]:
        """
        解析 LLM 输出为 {category, justification, confidence}。
        如解析失败，则抛出异常，由上层捕获并记录。
        """
        if raw_response.startswith("ERROR:"):
            raise RuntimeError(raw_response)

        try:
            json_str = self._extract_json_block(raw_response)
            data = json.loads(json_str)
        except Exception as exc:
            raise ValueError(f"解析 JSON 失败: {exc}; 原始响应: {raw_response}") from exc

        # 兼容两种可能结构：
        # 1) {"category": "...", "justification": "...", "confidence": ...}
        # 2) {"Item name": {"category": "...", "justification": "...", "confidence": ...}}
        if "category" in data and "justification" in data:
            confidence = data.get("confidence")
            if confidence is None:
                # 如果没有提供置信度，默认为中等置信度 5
                logger.warning("模型输出缺少 confidence 字段，使用默认值 5")
                confidence = 5
            else:
                # 确保置信度在 1-10 范围内
                try:
                    confidence = int(confidence)
                    confidence = max(1, min(10, confidence))
                except (ValueError, TypeError):
                    logger.warning("confidence 字段无法转换为整数，使用默认值 5")
                    confidence = 5
            
            return {
                "category": data["category"],
                "justification": data["justification"],
                "confidence": confidence,
            }

        if len(data) == 1:
            (_, inner) = next(iter(data.items()))
            if isinstance(inner, dict) and "category" in inner and "justification" in inner:
                confidence = inner.get("confidence")
                if confidence is None:
                    logger.warning("模型输出缺少 confidence 字段，使用默认值 5")
                    confidence = 5
                else:
                    try:
                        confidence = int(confidence)
                        confidence = max(1, min(10, confidence))
                    except (ValueError, TypeError):
                        logger.warning("confidence 字段无法转换为整数，使用默认值 5")
                        confidence = 5
                
                return {
                    "category": inner["category"],
                    "justification": inner["justification"],
                    "confidence": confidence,
                }

        raise ValueError(f"JSON 结构不符合预期: {data}")

    def classify_item(self, prompt: str, item_label: str) -> Dict[str, Any]:
        """
        对单个 item 进行分类，返回结构：
        {
          "category": ...,
          "justification": ...,
          "confidence": ...,
          "parsing_error": bool,
          "raw_response": "..."
        }
        """
        logger.info("开始分类: %s", item_label)
        raw = self.call_vllm_api(prompt)

        try:
            parsed = self.parse_classification(raw)
            logger.info(
                "分类成功: %s -> %s (置信度: %d/10)",
                item_label,
                parsed.get("category"),
                parsed.get("confidence", 5),
            )
            return {
                "category": parsed.get("category"),
                "justification": parsed.get("justification"),
                "confidence": parsed.get("confidence", 5),
                "parsing_error": False,
                "raw_response": raw,
            }
        except Exception as exc:
            # 不静默失败，记录错误和原始输出
            logger.error("分类解析失败 [%s]: %s", item_label, exc)
            return {
                "category": "ERROR",
                "justification": f"Failed to parse model output: {exc}",
                "confidence": 0,  # 错误时置信度为 0
                "parsing_error": True,
                "raw_response": raw,
            }


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_capabilities(path: str) -> List[Tuple[str, str]]:
    """
    从 capability 描述文件中读取 (item_name, definition_refined) 列表。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Tuple[str, str]] = []
    for key, value in data.items():
        # value 中通常有 capability_name 和 definition_refined
        if not isinstance(value, dict):
            logger.warning("capability %s 的值不是字典，跳过", key)
            continue
        name = value.get("capability_name", key)
        definition = value.get("definition_refined")
        if not definition:
            logger.warning("capability %s 缺少 definition_refined，跳过", name)
            continue
        items.append((name, definition))
    logger.info("从 %s 加载到 %d 个 capability", path, len(items))
    return items


def load_parcels(path: str) -> List[Tuple[str, str]]:
    """
    从 parcel summary 文件中读取 (label, description) 列表。
    label 将形如 "parcel_{id}: <function_name>"。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summaries = data.get("parcel_summaries", [])
    items: List[Tuple[str, str]] = []
    for s in summaries:
        if not isinstance(s, dict):
            continue
        parcel_id = s.get("parcel_id")
        function_name = s.get("function_name", "").strip()
        function_description = s.get("function_description")
        if parcel_id is None or not function_description:
            logger.warning(
                "parcel 条目缺少 parcel_id 或 function_description，跳过: %s",
                s,
            )
            continue

        # 去掉可能的 markdown 装饰
        function_name_clean = function_name.replace("*", "").strip()
        label = f"parcel_{parcel_id}: {function_name_clean}"
        items.append((label, function_description))

    logger.info("从 %s 加载到 %d 个 parcels", path, len(items))
    return items


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def classify_items(
    mode: str,
    input_file: str,
    output_file: str,
    vllm_url: str,
    api_key: str,
    delay: float,
) -> Dict[str, Any]:
    classifier = PathologyClassifier(vllm_url=vllm_url, api_key=api_key)

    if mode == "capability":
        raw_items = load_capabilities(input_file)
    else:
        raw_items = load_parcels(input_file)

    results: Dict[str, Any] = {}
    logger.info("总共待分类条目数: %d", len(raw_items))

    for idx, (name_or_label, text) in enumerate(raw_items):
        if mode == "capability":
            prompt = classifier.build_capability_prompt(name_or_label, text)
        else:
            # 从 label 中解析 parcel_id 和 function_name 以便构造 prompt
            if name_or_label.startswith("parcel_") and ": " in name_or_label:
                id_part, func_name = name_or_label.split(": ", 1)
                try:
                    parcel_id = int(id_part.replace("parcel_", ""))
                except ValueError:
                    parcel_id = -1
                    func_name = name_or_label
            else:
                parcel_id = -1
                func_name = name_or_label
            prompt = classifier.build_parcel_prompt(parcel_id, func_name, text)

        result = classifier.classify_item(prompt, name_or_label)
        results[name_or_label] = result

        # 立即保存中间结果，防止中断丢失
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(
            "已完成 %d/%d 条 (%s)",
            idx + 1,
            len(raw_items),
            name_or_label,
        )
        if delay > 0:
            time.sleep(delay)

    logger.info("所有条目分类完成，结果写入: %s", output_file)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 LLM 对 capability 或 SAE parcels 进行病理类型分类",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["capability", "parcel"],
        required=True,
        help="输入类型: capability 或 parcel",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入 JSON 文件路径",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="分类结果输出 JSON 文件路径",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://0.0.0.0:8001/v1",
        help="vLLM 服务地址 (含 /v1)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="abcabc",
        help="API 密钥",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="两次调用之间的等待时间（秒）",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="如果输出文件已存在，则跳过当前任务（用于批量运行时避免重复计算）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if args.skip_existing and output_path.exists():
        logger.info("输出文件已存在且启用 --skip_existing，跳过: %s", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Pathology Classifier 启动 ===")
    logger.info("模式: %s", args.mode)
    logger.info("输入文件: %s", input_path)
    logger.info("输出文件: %s", output_path)
    logger.info("vLLM URL: %s", args.vllm_url)
    logger.info("delay: %.2f 秒", args.delay)
    logger.info("log file: %s", log_filepath)

    classify_items(
        mode=args.mode,
        input_file=str(input_path),
        output_file=str(output_path),
        vllm_url=args.vllm_url,
        api_key=args.api_key,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

