#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sycophancy 数据加载与（可选）评估模块

适配 sycophancy-eval / detection 产生的 JSONL 结果：
- 一般包含字段：
  - question / text / prompt 等用户输入
  - first_comment / second_comment / model_answer 等模型回答
  - 其它元信息 original_record, prompt_template_type 等

本模块目前主要负责：
- 从 JSONL 加载数据，提取用于干预实验的 prompt（问题）；
- 评估器接口暂不实现，如需评估需单独实现 SycophancyEvaluator。
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any


class SycophancyDataLoader:
    """Sycophancy 数据加载器（JSONL 格式）"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def load_jsonl(path: str) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")
        with p.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as ex:
                    raise ValueError(f"{path} 第{line_num}行 JSON 解析失败: {ex}")
        if not records:
            raise ValueError(f"JSONL 文件为空: {path}")
        return records

    @staticmethod
    def _extract_prompt(record: Dict[str, Any]) -> str:
        """
        从 sycophancy-eval 风格记录中尽量提取用户输入（含态度提示）的完整 prompt。
        优先级：
          1) record['record']['original_record']['prompt'][0]['content']
          2) record['record']['prompt'] 或 record['prompt']
          3) record['question']
          4) record['text']
        """
        # 1) original_record.prompt
        try:
            orig = record.get("original_record") or record.get("record", {}).get("original_record")
            if isinstance(orig, dict):
                prompt_list = orig.get("prompt", [])
                if isinstance(prompt_list, list) and prompt_list:
                    first = prompt_list[0]
                    if isinstance(first, dict):
                        content = (first.get("content") or "").strip()
                        if content:
                            return content
        except Exception as e:
            print(f"[WARN] 从 original_record.prompt 提取 prompt 失败: {e}")

        # 2) record['record']['prompt'] 或 record['prompt']
        rec = record.get("record", {})
        if isinstance(rec, dict):
            p2 = (rec.get("prompt") or "").strip()
            if p2:
                return p2
        p3 = (record.get("prompt") or "").strip()
        if p3:
            return p3

        # 3) question
        q = (record.get("question") or "").strip()
        if q:
            return q

        # 4) text
        t = (record.get("text") or "").strip()
        if t:
            return t

        raise KeyError("无法从记录中提取 prompt / question / text 字段")

    def load_dataset(self, dataset_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
        """
        加载 sycophancy JSONL 数据集，返回统一格式：
        {
          "question": <完整 prompt>,
          "context": [],
          "meta": {... 原始字段 ...}
        }
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

        records = self.load_jsonl(dataset_path)
        processed: List[Dict[str, Any]] = []

        for idx, rec in enumerate(records):
            if max_samples > 0 and idx >= max_samples:
                break
            try:
                prompt = self._extract_prompt(rec)
                processed.append(
                    {
                        "question": prompt,
                        "context": [],
                        "meta": rec,
                    }
                )
            except Exception as e:
                self.logger.warning(f"跳过第 {idx+1} 行: {e}")

        self.logger.info(f"成功加载 {len(processed)} 条数据 (原始 {len(records)} 条)")
        return processed


class SycophancyEvaluator:
    """
    占位评估器：目前未实现具体 sycophancy 评估逻辑。
    如需评估干预前后“谄媚率”的变化，可以在这里接入 sycophancy_eval.py 或 vLLM 评估。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "SycophancyEvaluator 尚未实现。如需评估干预效果，请在 data_loader.py 中实现 "
            "SycophancyEvaluator.evaluate_batch，并在 run_intervention.py 中启用。"
        )

