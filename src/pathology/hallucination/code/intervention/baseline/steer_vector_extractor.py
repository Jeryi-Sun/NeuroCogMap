#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steer Vector 提取工具

从指定的 correct / incorrect 数据中计算 steer vector，并保存到文件中。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import logging
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from sae_lens import SAE, HookedSAETransformer


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件"""
    records: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as exc:
                print(f"⚠️ 跳过无效 JSON 行: {exc}")
                continue
    if not records:
        raise ValueError(f"{file_path} 未读取到有效数据")
    return records


class SteerVectorExtractor:
    """Steer Vector 生成器"""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        sae_release: str = "gemma-scope-2b-pt-res",
        sae_local_base_dir: str = "/path/to/local_models/gemma-scope-2b-pt-res",
        results_dir: str = "/path/to/project_root/safety_explanation/hallucination/results/intervention/baseline",
        is_instruct: bool = False,
        layers: Optional[List[int]] = None,
        num_samples: int = 10,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_local_base_dir = sae_local_base_dir
        self.results_dir = results_dir
        self.is_instruct = is_instruct
        self.layers = layers
        self.num_samples = num_samples
        self.seed = seed

        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.results_dir, exist_ok=True)

        self.model: Optional[HookedSAETransformer] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.layer_to_sae: Dict[int, SAE] = {}
        self.layer_to_hook_name: Dict[int, str] = {}
        self.logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger("steer_vector_extractor")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            log_dir = os.path.join(self.results_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, f"steer_vector_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            fh.setFormatter(fmt)
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(fh)
            logger.addHandler(ch)
        return logger

    def load_model_and_tokenizer(self) -> None:
        """加载模型和分词器"""
        if self.model is not None and self.tokenizer is not None:
            return

        self.logger.info("加载模型和分词器...")
        torch._dynamo.config.disable = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        if "9b" in self.model_name:
            self.model = HookedSAETransformer.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=torch.bfloat16,
                n_devices=1,
                trust_remote_code=True,
            )
        else:
            self.model = HookedSAETransformer.from_pretrained(
                self.model_name,
                device=self.device,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        self.model.eval()
        self.logger.info("模型加载完成")

    def load_saes(self, sae_paths: List[str], max_saes: int = 100) -> None:
        """加载所需的 SAE"""
        self.logger.info("加载 SAE...")
        import re

        def extract_layer_num(path_str: str) -> Optional[int]:
            match = re.search(r"layer_(\d+)/", path_str)
            return int(match.group(1)) if match else None

        layer_to_sae_path: Dict[int, str] = {}
        for path in sae_paths:
            layer_num = extract_layer_num(path)
            if layer_num is not None:
                layer_to_sae_path[layer_num] = path

        if self.layers is not None:
            layer_to_sae_path = {k: v for k, v in layer_to_sae_path.items() if k in self.layers}

        loaded = 0
        for layer_id, sae_path in layer_to_sae_path.items():
            if loaded >= max_saes:
                self.logger.info("达到 SAE 加载上限，停止加载")
                break
            try:
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_path,
                    device=self.device,
                    local_path=os.path.join(self.sae_local_base_dir, sae_path, "params.npz"),
                )
                sae.to(self.device)
                sae.use_error_term = True
                self.layer_to_sae[layer_id] = sae
                hook_name = sae.cfg.metadata.hook_name if hasattr(sae.cfg, "metadata") else sae.cfg.hook_name
                self.layer_to_hook_name[layer_id] = hook_name
                loaded += 1
                self.logger.info("层 %d SAE 加载成功", layer_id)
            except Exception as exc:
                self.logger.error("层 %d SAE 加载失败: %s", layer_id, exc)

        if self.layers is None:
            self.layers = sorted(self.layer_to_sae.keys())

    def _build_prompt(self, question: str, context: List[str]) -> str:
        if context:
            context_text = " ".join(context) if isinstance(context, list) else str(context)
            return f"Context: {context_text}\nQ: {question}\nA:"
        return f"Q: {question}\nA:"

    def extract_activations(self, prompt: str, layer_id: int) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型尚未加载")
        if layer_id not in self.layer_to_sae:
            raise ValueError(f"层 {layer_id} 的 SAE 未加载")
        hook_name = self.layer_to_hook_name[layer_id]
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                input_ids,
                stop_at_layer=layer_id + 1,
                names_filter=[hook_name],
            )
        if hook_name not in cache:
            raise RuntimeError(f"未能获取层 {layer_id} 的激活值")
        return cache[hook_name][0]

    def compute(self, correct_data: List[Dict[str, Any]], incorrect_data: List[Dict[str, Any]]) -> Dict[int, torch.Tensor]:
        if not self.layers:
            raise ValueError("未指定需要计算的层，且无法从 SAE 加载列表中推断")

        self.logger.info("计算 Steer Vector，correct 样本 %d 条，incorrect 样本 %d 条", len(correct_data), len(incorrect_data))
        steer_vectors: Dict[int, torch.Tensor] = {}

        for layer_id in tqdm(self.layers, desc="计算各层 steer vector"):
            correct_acts: List[torch.Tensor] = []
            incorrect_acts: List[torch.Tensor] = []

            for sample in correct_data:
                try:
                    prompt = self._build_prompt(sample.get("question", ""), sample.get("context", []))
                    acts = self.extract_activations(prompt, layer_id)
                    correct_acts.append(acts.mean(dim=0).cpu())
                except Exception as exc:
                    self.logger.warning("处理 correct 样本失败: %s", exc)
                    continue

            for sample in incorrect_data:
                try:
                    prompt = self._build_prompt(sample.get("question", ""), sample.get("context", []))
                    acts = self.extract_activations(prompt, layer_id)
                    incorrect_acts.append(acts.mean(dim=0).cpu())
                except Exception as exc:
                    self.logger.warning("处理 incorrect 样本失败: %s", exc)
                    continue

            if not correct_acts or not incorrect_acts:
                self.logger.warning("层 %d 缺少激活数据，跳过", layer_id)
                continue

            correct_mean = torch.stack(correct_acts).mean(dim=0)
            incorrect_mean = torch.stack(incorrect_acts).mean(dim=0)
            steer_vector = correct_mean - incorrect_mean
            norm = torch.norm(steer_vector)
            if norm > 0:
                steer_vector = steer_vector / norm
            steer_vectors[layer_id] = steer_vector
            self.logger.info("层 %d steer vector 完成 (范数 %.4f)", layer_id, float(torch.norm(steer_vector)))

        if not steer_vectors:
            raise RuntimeError("未计算到任何 steer vector")
        return steer_vectors

    def save(self, steer_vectors: Dict[int, torch.Tensor], output_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        steer_vectors_cpu = {str(layer_id): tensor.detach().cpu() for layer_id, tensor in steer_vectors.items()}
        payload = {
            "metadata": metadata or {},
            "steer_vectors": steer_vectors_cpu,
        }
        torch.save(payload, output_path)
        self.logger.info("Steer vector 已保存至: %s", output_path)

    def run(self,
            correct_jsonl_path: str,
            incorrect_jsonl_path: str,
            sae_paths: List[str],
            output_path: str) -> str:
        random.seed(self.seed)

        correct_data = load_jsonl(correct_jsonl_path)
        incorrect_data = load_jsonl(incorrect_jsonl_path)

        if self.num_samples > 0:
            if len(correct_data) > self.num_samples:
                correct_data = random.sample(correct_data, self.num_samples)
            if len(incorrect_data) > self.num_samples:
                incorrect_data = random.sample(incorrect_data, self.num_samples)

        self.load_model_and_tokenizer()
        self.load_saes(sae_paths)

        steer_vectors = self.compute(correct_data, incorrect_data)
        metadata = {
            "model_name": self.model_name,
            "sae_release": self.sae_release,
            "num_correct": len(correct_data),
            "num_incorrect": len(incorrect_data),
            "layers": sorted(steer_vectors.keys()),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self.save(steer_vectors, output_path, metadata)
        return output_path

