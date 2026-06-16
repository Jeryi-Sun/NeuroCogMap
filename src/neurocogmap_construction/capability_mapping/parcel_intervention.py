#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel Intervention System
第三步：干预每个数据集对应的top parcels，计算激活状态变化

实现功能：
1. 从 data_driven_results.json 获取每个数据集对应的 top parcel id
2. 从 latent_parcel_assignments.json 获取 parcel id 对应的 latent id
3. 干预指定的 parcels，计算原始和干预后的 logprob 差异
4. 使用 steering 方式干预，干预强度：[-0.5, -1, -2, -4]

性能优化建议 (PERFORMANCE OPTIMIZATION SUGGESTIONS):
=======================================================

🚀 主要加速点：

1. **批量推理优化 (Batch Inference)**
   - 当前：逐个样本计算logprob，每次都要重新forward
   - 优化：批量处理多个样本，减少GPU利用率空闲时间
   - 预期加速：2-5倍

2. **SAE动态加载优化 (SAE Loading)**  
   - 当前：每次干预都可能重新加载SAE权重
   - 优化：预加载所有需要的SAE到内存，使用LRU缓存
   - 预期加速：避免重复I/O，节省10-30%时间

3. **计算图优化 (Computation Graph)**
   - 当前：每个logprob计算都创建新的计算图
   - 优化：复用计算图，减少tensor操作开销
   - 预期加速：10-20%

4. **内存访问优化 (Memory Access)**
   - 当前：频繁的CPU-GPU数据传输
   - 优化：预先将数据移到GPU，减少.to(device)调用
   - 预期加速：5-15%

5. **并行化优化 (Parallelization)**
   - 当前：串行处理不同parcel和强度
   - 优化：使用多进程/多线程并行处理独立的干预实验
   - 预期加速：2-4倍（取决于CPU核心数）

6. **数值计算优化 (Numerical Computation)**
   - 当前：使用torch.softmax + log，数值不稳定且慢
   - 优化：直接使用log_softmax，更稳定更快
   - 预期加速：5-10%

7. **缓存优化 (Caching)**
   - 当前：重复计算相同输入的结果
   - 优化：缓存tokenization结果和基础logits
   - 预期加速：避免重复计算，节省20-40%时间

🔧 具体实现方案见下面的优化版本函数
"""

import os
# 设置环境变量，使用HF镜像
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import json
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import re
import logging
import traceback
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from sae_lens import SAE, HookedSAETransformer
count = 0
# 设备选择
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

class ParcelIntervention:
    """Parcel 干预系统主类"""
    
    def __init__(self, 
                 data_driven_results_path: str,
                 latent_parcel_assignments_path: str,
                 max_activation_dir: str,
                 results_dir: str,
                 capability_data_dir: str = "/path/to/project_root/neural_area/capability_data",
                 model_name: str = "google/gemma-2-2b",
                 sae_release: str = "gemma-scope-2b-pt-res",
                 sae_local_base_dir: str = "/path/to/local_models/gemma-scope-2b-pt-res",
                 is_instruct: bool = False):
        """
        初始化干预系统
        
        Args:
            data_driven_results_path: 数据驱动结果文件路径
            latent_parcel_assignments_path: latent-parcel 分配文件路径
            max_activation_dir: 最大激活文件目录
            results_dir: 结果存储目录
            model_name: 模型名称
            sae_release: SAE 发布名
            sae_local_base_dir: 本地SAE权重目录
        """
        self.data_driven_results_path = data_driven_results_path
        self.latent_parcel_assignments_path = latent_parcel_assignments_path
        self.max_activation_dir = max_activation_dir
        self.results_dir = results_dir
        self.capability_data_dir = capability_data_dir
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_local_base_dir = sae_local_base_dir
        self.is_instruct = is_instruct
        
        # 创建结果目录
        # 如果存在就不用创建了
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        
        # 数据存储
        self.data_driven_results = None
        self.latent_parcel_assignments = None
        self.max_activations = {}
        self.model = None
        self.sae_list = []
        self.hook_names = []
        self.layer_to_sae_path: Dict[int, str] = {}
        self.layer_to_sae: Dict[int, SAE] = {}
        self.layer_to_hook_name: Dict[int, str] = {}
        
        # 干预强度
        self.intervention_strengths = [-1.0] #-2.0, -4.0, 1.0, 2.0, 4.0
        
        # 🚀 性能优化：添加缓存
        self._tokenization_cache = {}
        self._logits_cache = {}
        self._sae_cache = {}
        
        # 默认批量大小，可在main中通过参数覆盖
        self.batch_size = 8
        
        # 🔒 多线程并行处理相关
        self.temp_file_suffix = "_temp"
        self.optimized_suffix = "_optimized"
        
        print("🔧 Parcel Intervention System 初始化完成")
        
        # 日志设置：将运行期数值异常等记录到本地文件
        try:
            self.logs_dir = "/path/to/project_root/neural_area/connect_cap_parcel/results/intervention/logs"
            os.makedirs(self.logs_dir, exist_ok=True)
            log_file = os.path.join(self.logs_dir, f"parcel_intervention_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log")
            self.logger = logging.getLogger("parcel_intervention")
            if not self.logger.handlers:
                self.logger.setLevel(logging.INFO)
                fh = logging.FileHandler(log_file, encoding='utf-8')
                fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
                fh.setFormatter(fmt)
                self.logger.addHandler(fh)
            self.logger.info("Logger 初始化完成，日志文件: %s", log_file)
        except Exception as e:
            print(f"⚠️ 日志系统初始化失败: {e}")
    
    def load_data(self):
        """加载所有必要的数据"""
        print("📊 加载数据...")
        
        # 1. 加载数据驱动结果
        print("  加载数据驱动结果...")
        with open(self.data_driven_results_path, 'r', encoding='utf-8') as f:
            self.data_driven_results = json.load(f)
        print(f"  ✅ 加载了 {len(self.data_driven_results)} 个数据集的 top parcels")
        
        # 2. 加载 latent-parcel 分配
        print("  加载 latent-parcel 分配...")
        with open(self.latent_parcel_assignments_path, 'r', encoding='utf-8') as f:
            self.latent_parcel_assignments = json.load(f)
        n_parcels = len(self.latent_parcel_assignments.get('parcel_to_latents', {}))
        n_latents = len(self.latent_parcel_assignments.get('latent_to_parcel', {}))
        print(f"  ✅ 加载了 {n_parcels} 个 parcels, {n_latents} 个 latents")
        
        # 3. 加载合并后的最大激活数据
        print("  加载合并后的最大激活数据...")
        model_lower = str(self.model_name).lower()
        if "9b" in model_lower:
            merged_activation_path = os.path.join(self.max_activation_dir, 'merged_max_activations_9b.pkl')
        elif "llama" in model_lower and "8b" in model_lower:
            # Llama-3.1-8B Base + LXR-8x 使用单独的 8b 合并文件
            merged_activation_path = os.path.join(self.max_activation_dir, 'merged_max_activations_8b.pkl')
        else:
            merged_activation_path = os.path.join(self.max_activation_dir, 'merged_max_activations.pkl')
        print(f"  加载合并后的最大激活数据: {merged_activation_path}")
        if not os.path.exists(merged_activation_path):
            raise FileNotFoundError(f"未找到合并后的最大激活文件: {merged_activation_path}")
        
        with open(merged_activation_path, 'rb') as f:
            merged_activations = pickle.load(f)
        
        # 将合并后的激活数据存储为统一格式，所有数据集都使用同一份数据
        self.max_activations['merged'] = merged_activations
        print(f"    ✅ 合并激活数据: {len(merged_activations)} 个特征")
        
        print(f"  ✅ 加载了合并后的最大激活数据")
    
    def load_model_and_saes(self, sae_paths: List[str], max_saes: int = 10):
        """
        加载模型和SAE
        
        Args:
            sae_paths: SAE路径列表
            max_saes: 最大加载SAE数量，避免内存不足
        """
        print("🤖 加载模型和SAE...")
        
        # 加载模型
        print(f"  加载模型: {self.model_name}")
        try:
            if "9b" in self.model_name:
                self.model = HookedSAETransformer.from_pretrained(
                    self.model_name, device=device, dtype=torch.bfloat16, n_devices=1, trust_remote_code=True
                )
            else:
                self.model = HookedSAETransformer.from_pretrained(
                    self.model_name, device=device, dtype=torch.bfloat16, trust_remote_code=True
                )
            print(f"    ✅ 模型加载成功")
        except Exception as e:
            print(f"    ❌ 模型加载失败: {e}")
            # 尝试使用本地缓存路径
            local_model_path = os.path.expanduser("~/.cache/huggingface/hub")
            if os.path.exists(local_model_path):
                print(f"    尝试从本地缓存加载: {local_model_path}")
                try:
                    if "9b" in self.model_name:
                        self.model = HookedSAETransformer.from_pretrained(
                            self.model_name, device=device, dtype=torch.bfloat16, n_devices=1,
                            cache_dir=local_model_path, trust_remote_code=True
                        )
                    else:
                        self.model = HookedSAETransformer.from_pretrained(
                            self.model_name, device=device, dtype=torch.bfloat16,
                            cache_dir=local_model_path, trust_remote_code=True
                        )
                    print(f"    ✅ 从本地缓存加载模型成功")
                except Exception as e2:
                    print(f"    ❌ 从本地缓存加载也失败: {e2}")
                    raise e2
            else:
                raise e
        
        # 数值稳定性：推理模式，禁用dropout等
        if self.model is not None:
            try:
                self.model.eval()
                print("  🔒 模型已切换为 eval 模式")
            except Exception:
                pass
        
        # 构建 layer -> sae_path 的映射，便于分层按需加载
        def extract_layer_num(path_str: str) -> Optional[int]:
            """
            根据 sae_release 类型解析 SAE 路径中的层号：
            - Gemma 系列: 路径形如 layer_9/width_16k/...
            - Llama LXR-8x 系列: 路径形如 l0r_8x,l1r_8x,...
            """
            try:
                release_lower = self.sae_release.lower() if getattr(self, "sae_release", None) else ""
            except Exception:
                release_lower = ""
            if "gemma" in release_lower:
                m = re.search(r"layer_(\d+)/", path_str)
                return int(m.group(1)) if m else None
            else:
                m = re.search(r"l(\d+)r_8x", path_str)
                return int(m.group(1)) if m else None
        for p in sae_paths:
            ln = extract_layer_num(p)
            if ln is not None:
                self.layer_to_sae_path[ln] = p
        print(f"  可用SAE层数: {len(self.layer_to_sae_path)}，层索引: {sorted(self.layer_to_sae_path.keys())[:5]}...")
        print(f"  ✅ 模型和SAE路径索引准备完成（按需分层加载）")

    def _load_single_sae(self, sae_id: str) -> SAE:
        """
        加载单个 SAE，兼容 Gemma 系列与 Llama-3.1-8B LXR-8x 本地文件布局。
        
        优先级：
        1. 若匹配 LXR-8x 命名，则尝试使用 Llama Scope 本地 safetensors
        2. 否则若存在 Gemma 默认 params.npz，则用该路径
        3. 否则走 SAE 自带的 HuggingFace 下载 / 本地缓存逻辑
        """
        llama_scope_local_path = None
        m = re.search(r"l(\d+)r", sae_id)
        if m is not None:
            try:
                layer_num = int(m.group(1))
                real_path = f"Llama3_1-8B-Base-L{layer_num}R-8x"
                llama_scope_local_path = os.path.join(
                    self.sae_local_base_dir, real_path, "checkpoints", "final.safetensors"
                )
            except Exception as e:
                print(f"⚠️ 解析 Llama LXR-8x SAE 路径失败: sae_id={sae_id}, error={e}")
                llama_scope_local_path = None

        gemma_local_path = os.path.join(self.sae_local_base_dir, sae_id, "params.npz")

        try:
            if llama_scope_local_path is not None and os.path.exists(llama_scope_local_path):
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_id,
                    device=device,
                    local_path=llama_scope_local_path,
                )
            elif os.path.exists(gemma_local_path):
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_id,
                    device=device,
                    local_path=gemma_local_path,
                )
            else:
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_id,
                    device=device,
                )
        except Exception as e:
            print(f"❌ SAE 加载失败: sae_id={sae_id}, error={e}")
            raise

        try:
            sae.to(device)
            sae.use_error_term = True
        except Exception as e:
            print(f"⚠️ SAE 迁移到设备或设置 use_error_term 失败: sae_id={sae_id}, error={e}")
            raise

        return sae
    
    # 🚀 优化版本：批量计算logprob
    def calculate_logprob_batch(self, prompts: List[str], answers: List[str], batch_size: int = 8) -> List[float]:
        """
        批量计算logprob，显著提升GPU利用率
        
        Args:
            prompts: 提示列表
            answers: 答案列表  
            batch_size: 批量大小
            
        Returns:
            logprob列表
        """
        if len(prompts) != len(answers):
            raise ValueError("prompts和answers长度不匹配")
        
        all_logprobs = []
        
        # 分批处理
        for i in tqdm(range(0, len(prompts), batch_size), desc="批量计算logprob"):
            batch_prompts = prompts[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]
            # 构建批量输入
            if self.is_instruct:
                batch_texts = []
                for prompt, answer in zip(batch_prompts, batch_answers):
                    try:
                        messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}
                        ]
                        chat_text = self.model.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        batch_texts.append(chat_text)
                    except Exception as e:
                        print(f"Warning: Failed to apply chat template in batch: {e}")
                        batch_texts.append(f"{prompt} {answer}")
            else:
                batch_texts = [
                    f"Question: {prompt}\nAnswer: {answer}"
                    for prompt, answer in zip(batch_prompts, batch_answers)
                ]
            
            # 批量tokenization
            tokens = self.model.tokenizer(
                batch_texts,
                return_tensors="pt",
                add_special_tokens=True,
                return_offsets_mapping=True,
                padding=True,  # 启用padding进行批量处理
                truncation=True,
                max_length=1024,
            )
            input_ids = tokens["input_ids"].to(device)
            offsets = tokens["offset_mapping"]
            special_ids = set(self.model.tokenizer.all_special_ids) if hasattr(self.model.tokenizer, 'all_special_ids') else set()
            
            # 批量前向计算
            with torch.no_grad():
                logits = self.model(input_ids)  # [batch_size, seq_len, vocab_size]
                # 数值净化，避免后续 log_softmax 产生 NaN 传播
                if isinstance(logits, torch.Tensor):
                    try:
                        nonfinite = (~torch.isfinite(logits)).sum().item()
                        if nonfinite > 0:
                            getattr(self, 'logger', logging).warning("batch logits 含非有限值: %d", nonfinite)
                    except Exception:
                        pass
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
                
                # 为每个样本计算logprob
                batch_logprobs = []
                for j, (prompt, answer, text) in enumerate(zip(batch_prompts, batch_answers, batch_texts)):
                    # 计算answer的字符级起止位置
                    try:
                        answer_start = text.rfind(answer)
                    except Exception:
                        answer_start = text.find(answer)
                    answer_end = answer_start + len(answer)
                    
                    # 找到answer对应的token索引
                    sample_offsets = offsets[j]
                    answer_token_indices = [
                        idx for idx, (ofs_s, ofs_e) in enumerate(sample_offsets.tolist())
                        if (ofs_e > answer_start and ofs_e <= answer_end) or (ofs_s < answer_start and ofs_e > answer_start)
                    ]
                    
                    # 过滤特殊token
                    ids = input_ids[j].tolist()
                    valid_indices = [idx for idx in answer_token_indices if ids[idx] not in special_ids]
                    
                    if not valid_indices:
                        msg = f"样本{j}的answer未能映射到有效token，prompt_len={len(prompt)}, answer_len={len(answer)}"
                        count += 1
                        getattr(self, 'logger', logging).error(msg)
                        continue
                        #raise ValueError(msg)
                    
                    # 🚀 优化：使用log_softmax替代softmax + log
                    sample_logprobs = []
                    for idx in valid_indices:
                        if idx - 1 < 0 or idx - 1 >= logits.shape[1]:
                            continue
                        token_id = input_ids[j, idx].item()
                        if token_id >= logits.shape[2]:
                            continue
                        # 使用log_softmax，数值更稳定；过滤非有限值
                        log_probs = torch.log_softmax(logits[j, idx - 1, :], dim=-1)
                        val = log_probs[token_id].item()
                        if np.isfinite(val):
                            sample_logprobs.append(val)
                    
                    if sample_logprobs:
                        batch_logprobs.append(float(np.mean(sample_logprobs)))
                    else:
                        # 避免 NaN 扰动：回退为极小值并继续
                        getattr(self, 'logger', logging).warning(
                            "样本 j=%d, 全局索引=%d 所有有效 token 的 logprob 均不可用，回退极小值", j, i + j
                        )
                        batch_logprobs.append(float(-1e9))
                
                all_logprobs.extend(batch_logprobs)
        
        return all_logprobs
    
    # 🚀 优化版本：预加载SAE
    def preload_required_saes(self, parcel_names: List[str]):
        """
        预加载所有需要的SAE，避免运行时重复加载
        
        Args:
            parcel_names: 需要干预的parcel名称列表
        """
        print("🚀 预加载SAE权重...")
        required_layers = set()
        
        # 收集所有需要的层
        for parcel_name in parcel_names:
            try:
                parcel_latent_mapping = self.get_parcel_latent_mapping(parcel_name)
                for layer_id, _ in parcel_latent_mapping:
                    required_layers.add(layer_id)
            except KeyError:
                continue
        
        print(f"  需要加载的层: {sorted(required_layers)}")
        
        # 批量加载SAE
        for layer_id in tqdm(required_layers, desc="预加载SAE"):
            if layer_id in self.layer_to_sae:
                continue  # 已经加载过了
                
            if layer_id in self.layer_to_sae_path:
                sae_path = self.layer_to_sae_path[layer_id]
                try:
                    sae = self._load_single_sae(sae_path)
                    self.layer_to_sae[layer_id] = sae
                    self.layer_to_hook_name[layer_id] = (
                        sae.cfg.metadata.hook_name if hasattr(sae.cfg, 'metadata') else sae.cfg.hook_name
                    )
                    print(f"    ✅ 层 {layer_id} SAE加载成功（{sae_path}）")
                except Exception as e:
                    print(f"    ❌ 层 {layer_id} SAE加载失败: sae_path={sae_path}, error={e}")
        
        print(f"  ✅ 预加载完成，共加载 {len(self.layer_to_sae)} 个SAE")

    def _get_latents_per_layer(self) -> int:
        """
        根据模型 / SAE 配置返回每层 latent 数：
        - Gemma 2B / 9B: 16384
        - Llama-3.1-8B + LXR-8x: 32768
        其他情况默认 16384，并打印告警方便排查。
        """
        release_lower = str(getattr(self, "sae_release", "")).lower()
        model_lower = str(getattr(self, "model_name", "")).lower()

        if "gemma" in release_lower or "gemma" in model_lower:
            return 16384
        if "llama" in release_lower or "llama" in model_lower or "lxr" in release_lower:
            return 32768

        try:
            getattr(self, "logger", logging).warning(
                "未能根据 sae_release/model_name 识别每层 latent 数，使用默认 16384。"
            )
        except Exception:
            print("⚠️ 未能根据 sae_release/model_name 识别每层 latent 数，使用默认 16384。")
        return 16384

    def get_parcel_latent_mapping(self, parcel_input: Union[str, List[str]]) -> List[Tuple[int, int]]:
        """
        获取指定parcel对应的所有latent id及其层信息
        
        Args:
            parcel_input: parcel名称（字符串）或parcel名称列表，如 "parcel_61" 或 ["parcel_61", "parcel_62"]
        Returns:
            [(actual_layer_id, latent_id_in_layer), ...]
        """
        # 统一处理输入，确保是列表格式
        if isinstance(parcel_input, str):
            parcel_names = [parcel_input]
        elif isinstance(parcel_input, list):
            parcel_names = parcel_input
        else:
            raise TypeError(f"parcel_input 必须是字符串或列表，当前类型: {type(parcel_input)}")
        
        mapping = []
        
        # 获取可用的SAE层号列表，按顺序排序
        available_layers = sorted(self.layer_to_sae_path.keys())
        if not available_layers:
            raise RuntimeError("没有可用的SAE层，请先调用 load_model_and_saes 方法")
        latents_per_layer = self._get_latents_per_layer()

        print(f"  可用SAE层: {available_layers}")
        
        for parcel_name in parcel_names:
            if parcel_name not in self.latent_parcel_assignments['parcel_to_latents']:
                raise KeyError(f"未找到 parcel: {parcel_name} 于 parcel_to_latents 映射中")
            
            latent_ids = self.latent_parcel_assignments['parcel_to_latents'][parcel_name]
            print(f"  Parcel {parcel_name} 包含 {len(latent_ids)} 个 latent IDs")
            
            for latent_id in latent_ids:
                # 先通过latent_id计算出原始的层索引
                original_layer_idx = latent_id // latents_per_layer
                
                # 检查原始层索引是否在可用层范围内
                if original_layer_idx < len(available_layers):
                    actual_layer_id = available_layers[original_layer_idx]
                else:
                    # 如果超出范围，使用模运算回退到可用层
                    layer_index = original_layer_idx % len(available_layers)
                    actual_layer_id = available_layers[layer_index]
                    print(f"    ⚠️  Latent {latent_id} 的原始层索引 {original_layer_idx} 超出可用层范围，回退到层 {actual_layer_id}")
                
                # 计算该层内的latent id
                latent_in_layer = latent_id % latents_per_layer
                
                mapping.append((actual_layer_id, latent_in_layer))
                print(f"    Latent {latent_id} -> Layer {actual_layer_id}, Latent_in_layer {latent_in_layer}")
        
        return mapping
    
    def steering_hook(self, activations, hook, steering_strength: float, 
                      steering_vector: torch.Tensor, max_act: float):
        """
        Steering hook函数，用于干预激活
        
        Args:
            activations: 原始激活
            hook: hook对象
            steering_strength: 干预强度
            steering_vector: 干预向量
            max_act: 最大激活值
        """
        # 数值防护：净化 steering 向量与 max_act，避免 NaN/Inf 放大
        sv_before = steering_vector
        sv = torch.nan_to_num(steering_vector, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            ma = float(max_act)
            if not np.isfinite(ma):
                getattr(self, 'logger', logging).warning("max_act 非有限值，置为 0.0")
                ma = 0.0
            # 适度裁剪 max_act，避免极端放大
            ma = float(np.clip(ma, -1e3, 1e3))
        except Exception:
            ma = 0.0
        try:
            if torch.any(~torch.isfinite(sv_before)):
                getattr(self, 'logger', logging).warning("steering_vector 含非有限值，已净化并裁剪")
        except Exception:
            pass
        return activations + ma * steering_strength * sv
    
    def calculate_logprob(self, prompt: str, answer: str) -> float:
        """
        仅对 answer 段的条件对数概率取平均（给定 prompt）。
        使用 tokenizer 的 offsets 来精确选取 answer 对应的 token 范围，并过滤特殊 token。
        """
        # 构建完整文本
        if self.is_instruct:
            try:
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ]
                full_text = self.model.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
                full_text = f"{prompt} {answer}"
        else:
            full_text = f"Question: {prompt}\nAnswer: {answer}"
        
        # 使用 tokenizer 获取 input_ids 与 offsets
        tokens = self.model.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            truncation=True,
            max_length=1024,
        )
        input_ids = tokens["input_ids"].to(device)
        offsets = tokens["offset_mapping"][0]
        special_ids = set(self.model.tokenizer.all_special_ids) if hasattr(self.model.tokenizer, 'all_special_ids') else set()
        
        # 计算 answer 的字符级起止
        answer_start = full_text.rfind(answer)
        answer_end = answer_start + len(answer)
        
        # 选择落入 answer 范围内的 token 索引（与 get_sae_act.py 同逻辑的区间相交判定）
        answer_token_indices = [
            i for i, (ofs_s, ofs_e) in enumerate(offsets.tolist())
            if (ofs_e > answer_start and ofs_e <= answer_end) or (ofs_s < answer_start and ofs_e > answer_start)
        ]
        # 过滤特殊 token
        ids = input_ids[0].tolist()
        valid_indices = [i for i in answer_token_indices if ids[i] not in special_ids]
        if not valid_indices:
            raise ValueError("answer 对应的有效 token 为空，无法计算 logprob")
        
        # 前向计算 logits
        with torch.no_grad():
            logits = self.model(input_ids)
            if not (hasattr(logits, 'shape') and len(logits.shape) == 3):
                raise RuntimeError(f"logits 形状异常: {getattr(logits, 'shape', None)}，期望 [1, seq_len, vocab]")
            # 数值净化，避免后续 log_softmax 产生 NaN 传播
            if isinstance(logits, torch.Tensor):
                try:
                    nonfinite = (~torch.isfinite(logits)).sum().item()
                    if nonfinite > 0:
                        getattr(self, 'logger', logging).warning("single logits 含非有限值: %d", nonfinite)
                except Exception:
                    pass
                logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
            _, seq_len, vocab_size = logits.shape
            
            # 对每个 answer token i，取 logits 在位置 i-1 预测 token[i] 的 log prob
            logprobs = []
            for idx in valid_indices:
                if idx - 1 < 0 or idx - 1 >= seq_len:
                    continue
                token_id = input_ids[0, idx].item()
                logit_vec = logits[0, idx - 1, :]
                if token_id >= vocab_size:
                    continue
                lp = torch.log_softmax(logit_vec, dim=-1)[token_id].item()
                if np.isfinite(lp):
                    logprobs.append(lp)
        
        if not logprobs:
            # 避免向上传播 NaN：回退极小值
            getattr(self, 'logger', logging).warning(
                "单样本计算未得到任何有效 logprob，回退极小值。prompt_len=%d, answer_len=%d",
                len(prompt), len(answer)
            )
            return float(-1e9)
        return float(np.mean(logprobs))
    
    # 🚀 优化版本：快速干预parcel
    def intervene_parcel_optimized(self, dataset_name: str, parcel_name: Union[str, List[str]], 
                                  intervention_strength: float, test_prompts: List[str], test_answers: List[str]) -> Dict[str, Any]:
        """
        优化版本的parcel干预，使用批量计算和预加载的SAE
        
        Args:
            dataset_name: 数据集名称
            parcel_name: parcel名称（字符串）或parcel名称列表
            intervention_strength: 干预强度
            test_prompts: 测试提示列表
            test_answers: 测试答案列表
        Returns:
            干预结果字典
        """
        # 获取parcel对应的latent映射
        parcel_latent_mapping = self.get_parcel_latent_mapping(parcel_name)
        if not parcel_latent_mapping:
            parcel_str = parcel_name if isinstance(parcel_name, str) else ', '.join(parcel_name)
            return {"error": f"未找到parcel {parcel_str} 的latent映射"}
        
        # 获取最大激活值（从合并后的数据中获取）
        max_acts = {}
        merged_activations = self.max_activations['merged']
        for layer_id, latent_in_layer in parcel_latent_mapping:
            key = (layer_id, latent_in_layer)
            if key not in merged_activations:
                raise KeyError(f"合并激活数据中缺少层 {layer_id} latent {latent_in_layer} 的最大激活值")
            max_acts[key] = merged_activations[key]
        # 🚀 优化：批量计算原始logprob
        print(f"    🚀 批量计算原始logprob...")
        original_logprobs = self.calculate_logprob_batch(test_prompts, test_answers, batch_size=self.batch_size)
        original_avg_logprob = np.mean(original_logprobs)
        print(f"    平均原始logprob: {original_avg_logprob:.6f}")
        
        # 分层干预：逐层对该parcel的latents进行steer
        layer_to_latents: Dict[int, List[int]] = {}
        for layer_id, latent_in_layer in parcel_latent_mapping:
            layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
        
        per_layer_intervened_avg: Dict[int, float] = {}
        per_layer_intervened_logprobs: Dict[int, List[float]] = {}
        
        for layer_id, latents in tqdm(layer_to_latents.items(), desc="分层干预"):
            # 🚀 优化：使用预加载的SAE
            if layer_id not in self.layer_to_sae:
                raise RuntimeError(f"层 {layer_id} 的SAE未预加载，无法继续。请先调用 preload_required_saes 或检查 --sae_paths。")
                
            sae = self.layer_to_sae[layer_id]
            hook_name = self.layer_to_hook_name[layer_id]
            
            # 预先在GPU上合并该层的steering向量，避免hook内反复to(device)
            steering_vectors = torch.stack([sae.W_dec[li].to(device) for li in latents], dim=0)
            max_act_tensor = torch.tensor([max_acts.get((layer_id, li), 1.0) for li in latents], device=device, dtype=steering_vectors.dtype)

            def layer_steer_hook(activations, hook):
                acts = activations
                for k in range(steering_vectors.shape[0]):
                    acts = self.steering_hook(acts, hook, intervention_strength, steering_vectors[k], max_act_tensor[k].item())
                return acts
            
            # 🚀 优化：批量计算干预后logprob
            with self.model.hooks(fwd_hooks=[(hook_name, layer_steer_hook)]):
                layer_intervened_logprobs = self.calculate_logprob_batch(test_prompts, test_answers, batch_size=self.batch_size)
                if layer_intervened_logprobs:
                    per_layer_intervened_avg[layer_id] = float(np.mean(layer_intervened_logprobs))
                    per_layer_intervened_logprobs[layer_id] = [float(x) for x in layer_intervened_logprobs]
        
        if per_layer_intervened_avg:
            intervened_avg_logprob = float(np.mean(list(per_layer_intervened_avg.values())))
        else:
            intervened_avg_logprob = original_avg_logprob
        print(f"    平均干预后logprob(层均值): {intervened_avg_logprob:.6f}")
        
        # 计算差异
        logprob_diff = intervened_avg_logprob - original_avg_logprob
        
        # 计算每个样本的差异
        original_logprobs_list = [float(x) for x in original_logprobs]
        
        # 计算每层每个样本的差异
        per_layer_sample_diffs = {}
        for layer_id, layer_intervened_logprobs_list in per_layer_intervened_logprobs.items():
            sample_diffs = []
            for i in range(len(original_logprobs_list)):
                diff = layer_intervened_logprobs_list[i] - original_logprobs_list[i]
                sample_diffs.append(diff)
            per_layer_sample_diffs[str(layer_id)] = sample_diffs
        
        # 计算总体每个样本的差异（所有层的平均）
        overall_sample_diffs = []
        if per_layer_intervened_logprobs:
            for i in range(len(original_logprobs_list)):
                layer_avg_diff = 0
                for layer_id in per_layer_intervened_logprobs.keys():
                    layer_avg_diff += per_layer_intervened_logprobs[layer_id][i] - original_logprobs_list[i]
                layer_avg_diff /= len(per_layer_intervened_logprobs)
                overall_sample_diffs.append(layer_avg_diff)
        else:
            overall_sample_diffs = [0.0] * len(original_logprobs_list)
        
        # 计算总体每个样本的干预后logprob（所有层的平均）
        overall_intervened_logprobs = []
        if per_layer_intervened_logprobs:
            for i in range(len(original_logprobs_list)):
                layer_avg_logprob = 0
                for layer_id in per_layer_intervened_logprobs.keys():
                    layer_avg_logprob += per_layer_intervened_logprobs[layer_id][i]
                layer_avg_logprob /= len(per_layer_intervened_logprobs)
                overall_intervened_logprobs.append(layer_avg_logprob)
        else:
            overall_intervened_logprobs = [float(x) for x in original_logprobs_list]
        
        # 对 overall_original_logprobs 和 overall_intervened_logprobs 做 min-max 归一化（使用同一对 min/max，来自干预前数据）
        ref_min, ref_max = (min(original_logprobs_list), max(original_logprobs_list)) if original_logprobs_list else (0.0, 0.0)
        if ref_max > ref_min:
            denom = (ref_max - ref_min)
            overall_original_norm = [(v - ref_min) / denom for v in original_logprobs_list]
            overall_intervened_norm = [(v - ref_min) / denom for v in overall_intervened_logprobs]
        else:
            overall_original_norm = [0.0] * len(original_logprobs_list)
            overall_intervened_norm = [0.0] * len(overall_intervened_logprobs)
        
        # 归一化后的逐样本差值与均值
        sample_diffs_norm = [overall_intervened_norm[i] - overall_original_norm[i] for i in range(len(overall_original_norm))]
        logprob_diff_norm_avg = float(np.mean(sample_diffs_norm)) if sample_diffs_norm else 0.0
        
        return {
            "parcel_name": parcel_name,
            "intervention_strength": intervention_strength,
            "num_samples": len(original_logprobs_list),
            
            # 总体统计
            "original_logprob_avg": original_avg_logprob,
            "intervened_logprob_avg": intervened_avg_logprob,
            "logprob_diff_avg": logprob_diff,
            
            # 每个样本的详细数据
            "overall_original_logprobs": original_logprobs_list,
            "overall_intervened_logprobs": overall_intervened_logprobs,
            "overall_sample_diffs": overall_sample_diffs,
            "sample_diffs_norm": sample_diffs_norm,
            "logprob_diff_norm_avg": logprob_diff_norm_avg,
            
            # 每层的平均结果
            "per_layer_intervened_avg": {str(k): v for k, v in per_layer_intervened_avg.items()},
            
            # 每层每个样本的详细数据
            "per_layer_intervened_logprobs": {str(k): v for k, v in per_layer_intervened_logprobs.items()},
            "per_layer_sample_diffs": per_layer_sample_diffs,
            
            # "parcel_latent_mapping": [f"layer_{layer}_latent_{latent}" for layer, latent in parcel_latent_mapping],
            # "max_acts": {f"layer_{layer}_latent_{latent}": value for (layer, latent), value in max_acts.items()}
        }
    
    def intervene_parcel(self, dataset_name: str, parcel_name: str, 
                        intervention_strength: float, test_prompts: List[str], test_answers: List[str]) -> Dict[str, Any]:
        """
        干预指定的parcel
        
        Args:
            dataset_name: 数据集名称
            parcel_name: parcel名称
            intervention_strength: 干预强度
            test_prompts: 测试提示列表
            
        Returns:
            干预结果字典
        """
        # 获取parcel对应的latent映射
        parcel_latent_mapping = self.get_parcel_latent_mapping(parcel_name)
        if not parcel_latent_mapping:
            return {"error": f"未找到parcel {parcel_name} 的latent映射"}
        
        # 获取最大激活值（从合并后的数据中获取）
        max_acts = {}
        merged_activations = self.max_activations['merged']
        for layer_id, latent_in_layer in tqdm(parcel_latent_mapping, desc="获取最大激活值"):
            key = (layer_id, latent_in_layer)
            if key not in merged_activations:
                raise KeyError(f"合并激活数据中缺少层 {layer_id} latent {latent_in_layer} 的最大激活值")
            max_acts[key] = merged_activations[key]
        
        # 计算原始logprob
        original_logprobs = []
        
        for prompt, answer in tqdm(zip(test_prompts, test_answers), desc="计算原始logprob", total=len(test_prompts)):
            logprob = self.calculate_logprob(prompt, answer)
            original_logprobs.append(logprob)
            print(f"    原始logprob: '{prompt}{answer}' -> {logprob:.6f}")
        
        original_avg_logprob = np.mean(original_logprobs)
        print(f"    平均原始logprob: {original_avg_logprob:.6f}")
        
        # 分层干预：逐层对该parcel的latents进行steer并记录每层的logprob均值
        layer_to_latents: Dict[int, List[int]] = {}
        for layer_id, latent_in_layer in parcel_latent_mapping:
            layer_to_latents.setdefault(layer_id, []).append(latent_in_layer)
        
        per_layer_intervened_avg: Dict[int, float] = {}
        for layer_id, latents in tqdm(layer_to_latents.items(), desc="分层干预"):
            # 确保该层的SAE与hook名可用：优先使用按需加载索引
            sae = None
            hook_name = None
            if layer_id in getattr(self, 'layer_to_sae', {}):
                sae = self.layer_to_sae[layer_id]
                hook_name = self.layer_to_hook_name[layer_id]
            elif layer_id < len(self.sae_list):
                sae = self.sae_list[layer_id]
                hook_name = self.hook_names[layer_id]
            elif hasattr(self, 'layer_to_sae_path') and layer_id in self.layer_to_sae_path:
                # 动态加载该层
                sae_path = self.layer_to_sae_path[layer_id]
                try:
                    sae = self._load_single_sae(sae_path)
                    if hasattr(self, 'layer_to_sae'):
                        self.layer_to_sae[layer_id] = sae
                        self.layer_to_hook_name[layer_id] = sae.cfg.metadata.hook_name if hasattr(sae.cfg, 'metadata') else sae.cfg.hook_name
                    hook_name = self.layer_to_hook_name[layer_id]
                except Exception as e:
                    raise RuntimeError(f"加载层 {layer_id} SAE 失败: sae_path={sae_path}, error={e}")
            else:
                raise RuntimeError(f"无可用SAE的层: {layer_id}。请检查 layer_to_sae_path 或 --sae_paths 配置。")
            
            # 预先在GPU上合并该层的steering向量，避免hook内反复to(device)
            steering_vectors = torch.stack([sae.W_dec[li].to(device) for li in latents], dim=0)
            max_act_tensor = torch.tensor([max_acts.get((layer_id, li), 1.0) for li in latents], device=device, dtype=steering_vectors.dtype)

            def layer_steer_hook(activations, hook):
                acts = activations
                for k in range(steering_vectors.shape[0]):
                    acts = self.steering_hook(acts, hook, intervention_strength, steering_vectors[k], max_act_tensor[k].item())
                return acts
            
            layer_intervened = []
            with self.model.hooks(fwd_hooks=[(hook_name, layer_steer_hook)]):
                for prompt, answer in tqdm(zip(test_prompts, test_answers), desc="计算干预后logprob", total=len(test_prompts)):
                    logprob = self.calculate_logprob(prompt, answer)
                    layer_intervened.append(logprob)
            if layer_intervened:
                per_layer_intervened_avg[layer_id] = float(np.mean(layer_intervened))
        
        if per_layer_intervened_avg:
            intervened_avg_logprob = float(np.mean(list(per_layer_intervened_avg.values())))
        else:
            intervened_avg_logprob = original_avg_logprob
        print(f"    平均干预后logprob(层均值): {intervened_avg_logprob:.6f}")
        
        # 计算差异
        logprob_diff = intervened_avg_logprob - original_avg_logprob
        
        return {
            "parcel_name": parcel_name,
            "intervention_strength": intervention_strength,
            "original_logprob": original_avg_logprob,
            "intervened_logprob": intervened_avg_logprob,
            "logprob_diff": logprob_diff,
            "per_layer_intervened_avg": {str(k): v for k, v in per_layer_intervened_avg.items()},
            # "parcel_latent_mapping": [f"layer_{layer}_latent_{latent}" for layer, latent in parcel_latent_mapping],
            # "max_acts": {f"layer_{layer}_latent_{latent}": value for (layer, latent), value in max_acts.items()}
        }
    
    def _load_qa_dataset(self, dataset_name: str) -> List[Dict[str, str]]:
        """按 get_sae_act.py 的方式读取 capability_data 中的 {dataset}_qa.json"""
        import glob
        qa_path = os.path.join(self.capability_data_dir, f"{dataset_name}_qa.json")
        if not os.path.exists(qa_path):
            # 兼容可能的命名差异：尝试扫描
            candidates = glob.glob(os.path.join(self.capability_data_dir, f"*{dataset_name}*_test.json"))
            if not candidates:
                raise FileNotFoundError(f"未找到 {dataset_name} 的 QA 数据: {qa_path}")
            qa_path = candidates[0]
        with open(qa_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 只保留包含 question, answer 的样本
        qa_list = []
        for item in data:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                qa_list.append({"question": q, "answer": a})
        if not qa_list:
            raise ValueError(f"{dataset_name} QA 数据为空或不含有效条目: {qa_path}")
        return qa_list

    def _format_qa_prompt(self, question: str) -> str:
        # 仅返回原始问题文本；模板统一在 logprob 计算处拼接
        return question

    def run_intervention_experiment(self, dataset_name: str, top_k: int = 5, top_m: int = -1) -> Dict[str, Any]:
        """
        对指定数据集运行干预实验
        
        Args:
            dataset_name: 数据集名称
            top_k: 取前k个parcel进行干预
            
        Returns:
            干预实验结果
        """
        print(f"🧪 对数据集 {dataset_name} 运行干预实验...")
        try:
            getattr(self, 'logger', logging).info("开始标准实验: dataset=%s", dataset_name)
        except Exception:
            pass
        
        if dataset_name not in self.data_driven_results:
            raise KeyError(f"数据集 {dataset_name} 不在 data_driven_results 的 top_parcels_by_dataset 中")
        
        # 获取top parcels
        top_parcels = self.data_driven_results[dataset_name][:top_k]
        parcel_names = [parcel for parcel, _ in top_parcels]
        
        print(f"  干预前 {top_k} 个parcels: {parcel_names}")
        try:
            getattr(self, 'logger', logging).info("标准实验: dataset=%s, parcels=%s", dataset_name, ",".join(parcel_names))
        except Exception:
            pass
        
        # 读取真实 QA 数据，并构造 prompts 与答案
        qa_items = self._load_qa_dataset(dataset_name)
        # 采样前N条以控制计算量（与先前5条对齐）
        sample_items = qa_items[:top_m]
        # prompt 仅包含问题与 Answer: 前缀；答案单独传入计算
        test_prompts = [self._format_qa_prompt(it["question"]) for it in sample_items]
        test_answers = [it["answer"] for it in sample_items]
        try:
            getattr(self, 'logger', logging).info("标准实验: dataset=%s, 样本数=%d", dataset_name, len(sample_items))
        except Exception:
            pass
        
        # 对每个parcel和每个干预强度进行实验
        results = {
            "dataset_name": dataset_name,
            "top_parcels": parcel_names,
            "intervention_results": {}
        }
        
        for parcel_name in tqdm(parcel_names, desc=f"干预 {dataset_name} 的parcels"):
            results["intervention_results"][parcel_name] = {}
            
            for strength in self.intervention_strengths:
                print(f"    干预 {parcel_name}，强度: {strength}")
                try:
                    getattr(self, 'logger', logging).info("标准实验: dataset=%s, parcel=%s, strength=%s", dataset_name, parcel_name, str(strength))
                except Exception:
                    pass
                intervention_result = self.intervene_parcel(
                    dataset_name, parcel_name, strength, test_prompts, test_answers
                )
                results["intervention_results"][parcel_name][str(strength)] = intervention_result
        
        return results
    
    # 🚀 优化版本：快速运行干预实验
    def run_intervention_experiment_optimized(self, dataset_name: str, top_k: int = 5, top_m: int = -1, use_multiprocessing: bool = False) -> Dict[str, Any]:
        """
        优化版本的干预实验，使用批量处理和预加载
        
        Args:
            dataset_name: 数据集名称
            top_k: 取前k个parcel进行干预
            top_m: 取前m个样本，-1表示使用所有样本
            use_multiprocessing: 是否使用多进程并行
            
        Returns:
            干预实验结果
        """
        print(f"🚀 对数据集 {dataset_name} 运行优化干预实验...")
        try:
            getattr(self, 'logger', logging).info("开始优化实验: dataset=%s", dataset_name)
        except Exception:
            pass
        
        if dataset_name not in self.data_driven_results:
            raise KeyError(f"数据集 {dataset_name} 不在 data_driven_results 的 top_parcels_by_dataset 中")
        
        # 获取top parcels
        # 如果 top_k > 10, 那么就从 0-top_k 个parcel进行干预 计入从0 到 top_k 个parcel

        top_parcels = self.data_driven_results[dataset_name][:top_k] 
        parcel_names = [parcel for parcel, _ in top_parcels]
        
        print(f"  干预前 {top_k} 个parcels: {parcel_names}")
        try:
            getattr(self, 'logger', logging).info("优化实验: dataset=%s, parcels=%s", dataset_name, ",".join(parcel_names))
        except Exception:
            pass
        
        # 🚀 优化：预加载所有需要的SAE
        self.preload_required_saes(parcel_names)
        
        # 读取真实 QA 数据
        qa_items = self._load_qa_dataset(dataset_name)
        sample_items = qa_items[:top_m] if top_m > 0 else qa_items
        test_prompts = [self._format_qa_prompt(it["question"]) for it in sample_items]
        test_answers = [it["answer"] for it in sample_items]
        
        print(f"  使用 {len(sample_items)} 个测试样本")
        try:
            getattr(self, 'logger', logging).info("优化实验: dataset=%s, 样本数=%d", dataset_name, len(sample_items))
        except Exception:
            pass
        
        # 对每个parcel和每个干预强度进行实验
        results = {
            "dataset_name": dataset_name,
            "top_parcels": parcel_names,
            "intervention_results": {}
        }
        
            
        # 串行优化处理
        for parcel_name in tqdm(parcel_names, desc=f"干预 {dataset_name} 的parcels"):
            results["intervention_results"][parcel_name] = {}
            
            for strength in self.intervention_strengths:
                print(f"    🚀 优化干预 {parcel_name}，强度: {strength}")
                try:
                    getattr(self, 'logger', logging).info("优化实验: dataset=%s, parcel=%s, strength=%s", dataset_name, parcel_name, str(strength))
                except Exception:
                    pass
                intervention_result = self.intervene_parcel_optimized(
                    dataset_name, parcel_name, strength, test_prompts, test_answers
                )
                results["intervention_results"][parcel_name][str(strength)] = intervention_result
        
        return results
    
    def run_all_interventions(self, top_k: int = 5, top_m: int = -1) -> Dict[str, Any]:
        """
        对所有数据集运行干预实验
        
        Args:
            top_k: 每个数据集取前k个parcel
            
        Returns:
            所有干预实验结果
        """
        print("🚀 开始对所有数据集运行干预实验...")
        
        all_results = {}
        datasets = list(self.data_driven_results.keys())
        
        for dataset_name in tqdm(datasets, desc="处理数据集"):
            print(f"\n处理数据集: {dataset_name}")
            
            result = self.run_intervention_experiment(dataset_name, top_k, top_m)
            all_results[dataset_name] = result
            
            # 保存单个数据集的结果
            result_file = os.path.join(self.results_dir, f"{dataset_name}_intervention_results.json")
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  ✅ {dataset_name} 结果已保存到: {result_file}")
        
        # 保存所有结果
        all_results_file = os.path.join(self.results_dir, "all_intervention_results.json")
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 所有干预实验完成！结果已保存到: {all_results_file}")
        return all_results
    
    def analyze_intervention_effects(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        分析干预效果，生成汇总报告
        
        Args:
            results: 干预实验结果
            
        Returns:
            汇总报告DataFrame
        """
        print("📊 分析干预效果...")
        
        analysis_data = []
        
        for dataset_name, dataset_result in results.items():
            if "error" in dataset_result:
                continue
                
            for parcel_name, parcel_results in dataset_result.get("intervention_results", {}).items():
                for strength_str, intervention_result in parcel_results.items():
                    if "error" in intervention_result:
                        continue
                        
                    analysis_data.append({
                        "dataset": dataset_name,
                        "parcel": parcel_name,
                        "strength": float(strength_str),
                        "original_logprob": intervention_result["original_logprob"],
                        "intervened_logprob": intervention_result["intervened_logprob"],
                        "logprob_diff": intervention_result["logprob_diff"],
                        "relative_change": (intervention_result["logprob_diff"] / 
                                         abs(intervention_result["original_logprob"])) * 100
                    })
        
        # 创建DataFrame
        df = pd.DataFrame(analysis_data)
        
        # 保存分析结果
        analysis_file = os.path.join(self.results_dir, "intervention_analysis.csv")
        df.to_csv(analysis_file, index=False, encoding='utf-8')
        print(f"  ✅ 分析结果已保存到: {analysis_file}")
        
        # 生成统计摘要
        summary = {
            "total_interventions": len(df),
            "datasets_count": df["dataset"].nunique(),
            "parcels_count": df["parcel"].nunique(),
            "strengths_count": df["strength"].nunique(),
            "avg_logprob_diff": df["logprob_diff"].mean(),
            "std_logprob_diff": df["logprob_diff"].std(),
            "max_positive_effect": df["logprob_diff"].max(),
            "max_negative_effect": df["logprob_diff"].min()
        }
        
        summary_file = os.path.join(self.results_dir, "intervention_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 统计摘要已保存到: {summary_file}")
        
        return df

    def _get_result_file_paths(self, dataset_name: str, use_optimized: bool = False) -> Tuple[str, str, str]:
        """
        获取数据集的结果文件路径
        
        Args:
            dataset_name: 数据集名称
            use_optimized: 是否使用优化版本
            
        Returns:
            (正式文件路径, 临时文件路径, 带优化后缀的正式文件路径)
        """
        suffix = self.optimized_suffix if use_optimized else ""
        temp_suffix = f"{suffix}{self.temp_file_suffix}"
        
        formal_file = os.path.join(self.results_dir, f"{dataset_name}_intervention_results{suffix}.json")
        temp_file = os.path.join(self.results_dir, f"{dataset_name}_intervention_results{temp_suffix}.json")
        optimized_formal_file = os.path.join(self.results_dir, f"{dataset_name}_intervention_results{self.optimized_suffix}.json")
        
        return formal_file, temp_file, optimized_formal_file
    
    def _is_dataset_already_processed(self, dataset_name: str, use_optimized: bool = False) -> bool:
        """
        检查数据集是否已经被处理过（包括正式文件和临时文件）
        
        Args:
            dataset_name: 数据集名称
            use_optimized: 是否使用优化版本
            
        Returns:
            True如果已经处理过，False否则
        """
        formal_file, temp_file, optimized_formal_file = self._get_result_file_paths(dataset_name, use_optimized)
        
        # 检查正式文件是否存在
        if os.path.exists(formal_file):
            print(f"  ⏭️  数据集 {dataset_name} 的正式结果文件已存在: {formal_file}")
            return True
        
        # 检查优化版本的正式文件是否存在
        if use_optimized and os.path.exists(optimized_formal_file):
            print(f"  ⏭️  数据集 {dataset_name} 的优化版本正式结果文件已存在: {optimized_formal_file}")
            return True
        
        # 检查临时文件是否存在
        if os.path.exists(temp_file):
            print(f"  ⏭️  数据集 {dataset_name} 的临时文件已存在，其他进程正在处理: {temp_file}")
            return True
        
        return False
    
    def _create_temp_file(self, dataset_name: str, use_optimized: bool = False) -> str:
        """
        创建临时文件，标记当前进程正在处理该数据集
        
        Args:
            dataset_name: 数据集名称
            use_optimized: 是否使用优化版本
            
        Returns:
            临时文件路径
        """
        _, temp_file, _ = self._get_result_file_paths(dataset_name, use_optimized)
        
        # 创建临时文件，包含进程信息
        temp_info = {
            "dataset_name": dataset_name,
            "process_id": os.getpid(),
            "start_time": datetime.datetime.now().isoformat(),
            "use_optimized": use_optimized,
            "status": "processing"
        }
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_info, f, ensure_ascii=False, indent=2)
        
        print(f"   创建临时文件: {temp_file}")
        return temp_file
    
    def _cleanup_temp_file(self, temp_file: str):
        """
        清理临时文件
        
        Args:
            temp_file: 临时文件路径
        """
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"   清理临时文件: {temp_file}")
        except Exception as e:
            print(f"  ⚠️  清理临时文件失败: {e}")
    
    def _save_final_result(self, dataset_name: str, result: Dict[str, Any], use_optimized: bool = False):
        """
        保存最终结果文件
        
        Args:
            dataset_name: 数据集名称
            result: 结果数据
            use_optimized: 是否使用优化版本
        """
        formal_file, _, _ = self._get_result_file_paths(dataset_name, use_optimized)
        
        with open(formal_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ {dataset_name} 最终结果已保存到: {formal_file}")

    # 🚀 优化版本：快速运行干预实验（带文件锁）
    def run_intervention_experiment_optimized_with_lock(self, dataset_name: str, top_k: int = 5, top_m: int = -1, use_multiprocessing: bool = False) -> Optional[Dict[str, Any]]:
        """
        带文件锁的优化版本干预实验，防止多进程重复处理
        
        Args:
            dataset_name: 数据集名称
            top_k: 取前k个parcel进行干预
            top_m: 取前m个样本，-1表示使用所有样本
            use_multiprocessing: 是否使用多进程并行
            
        Returns:
            干预实验结果，如果已处理过则返回None
        """
        print(f"🚀 对数据集 {dataset_name} 运行优化干预实验（带文件锁）...")
        
        # 🔒 检查是否已经处理过
        if self._is_dataset_already_processed(dataset_name, use_optimized=True):
            return None
        
        # 🔒 创建临时文件
        temp_file = self._create_temp_file(dataset_name, use_optimized=True)
        
        try:
            # 执行实际的干预实验
            result = self.run_intervention_experiment_optimized(dataset_name, top_k, top_m, use_multiprocessing)
            
            # 保存最终结果
            self._save_final_result(dataset_name, result, use_optimized=True)
            
            return result
            
        except Exception as e:
            err_type = type(e).__name__
            print(f"  ❌ 处理数据集 {dataset_name} 时发生错误 [{err_type}]: {e}")
            print(traceback.format_exc())
            try:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.exception("处理数据集 %s 时发生错误", dataset_name)
            except Exception:
                pass
            raise
        finally:
            # 清理临时文件
            self._cleanup_temp_file(temp_file)

    def run_intervention_experiment_with_lock(self, dataset_name: str, top_k: int = 5, top_m: int = -1) -> Optional[Dict[str, Any]]:
        """
        带文件锁的标准版本干预实验，防止多进程重复处理
        
        Args:
            dataset_name: 数据集名称
            top_k: 取前k个parcel进行干预
            top_m: 取前m个样本，-1表示使用所有样本
            
        Returns:
            干预实验结果，如果已处理过则返回None
        """
        print(f"📊 对数据集 {dataset_name} 运行标准干预实验（带文件锁）...")
        
        # 🔒 检查是否已经处理过
        if self._is_dataset_already_processed(dataset_name, use_optimized=False):
            return None
        
        # 🔒 创建临时文件
        temp_file = self._create_temp_file(dataset_name, use_optimized=False)
        
        try:
            # 执行实际的干预实验
            result = self.run_intervention_experiment(dataset_name, top_k)
            
            # 保存最终结果
            self._save_final_result(dataset_name, result, use_optimized=False)
            
            return result
            
        except Exception as e:
            err_type = type(e).__name__
            print(f"  ❌ 处理数据集 {dataset_name} 时发生错误 [{err_type}]: {e}")
            print(traceback.format_exc())
            try:
                if hasattr(self, 'logger') and self.logger:
                    self.logger.exception("处理数据集 %s 时发生错误", dataset_name)
            except Exception:
                pass
            raise
        finally:
            # 清理临时文件
            self._cleanup_temp_file(temp_file)

    def run_all_interventions_with_lock(self, top_k: int = 5, top_m: int = -1, use_optimized: bool = False) -> Dict[str, Any]:
        """
        带文件锁的对所有数据集运行干预实验
        
        Args:
            top_k: 每个数据集取前k个parcel
            use_optimized: 是否使用优化版本
            
        Returns:
            所有干预实验结果
        """
        print("🚀 开始对所有数据集运行干预实验（带文件锁）...")
        
        # 添加日志记录
        try:
            getattr(self, 'logger', logging).info("开始批量干预实验（带文件锁）: top_k=%d, top_m=%d, use_optimized=%s", top_k, top_m, use_optimized)
        except Exception:
            pass
        
        all_results = {}
        datasets = list(self.data_driven_results.keys())
        
        try:
            getattr(self, 'logger', logging).info("批量干预实验: 总数据集数=%d, 数据集列表=%s", len(datasets), ",".join(datasets))
        except Exception:
            pass
        
        for dataset_name in tqdm(datasets, desc="处理数据集"):
            print(f"\n处理数据集: {dataset_name}")
            
            try:
                getattr(self, 'logger', logging).info("批量干预实验: 开始处理数据集=%s", dataset_name)
            except Exception:
                pass
            
            try:
                if use_optimized:
                    result = self.run_intervention_experiment_optimized_with_lock(dataset_name, top_k, top_m)
                else:
                    result = self.run_intervention_experiment_with_lock(dataset_name, top_k, top_m)
                
                if result is not None:
                    all_results[dataset_name] = result
                    try:
                        getattr(self, 'logger', logging).info("批量干预实验: 数据集=%s 处理完成", dataset_name)
                    except Exception:
                        pass
                else:
                    print(f"  ⏭️  数据集 {dataset_name} 已处理过，跳过")
                    try:
                        getattr(self, 'logger', logging).info("批量干预实验: 数据集=%s 已处理过，跳过", dataset_name)
                    except Exception:
                        pass
                    
            except Exception as e:
                print(f"  ❌ 处理数据集 {dataset_name} 失败: {e}")
                try:
                    getattr(self, 'logger', logging).error("批量干预实验: 数据集=%s 处理失败: %s", dataset_name, str(e))
                except Exception:
                    pass
                continue
        
        # 保存所有结果
        suffix = "_optimized" if use_optimized else ""
        all_results_file = os.path.join(self.results_dir, f"all_intervention_results{suffix}.json")
        with open(all_results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 所有干预实验完成！结果已保存到: {all_results_file}")
        
        # 添加完成日志记录
        try:
            getattr(self, 'logger', logging).info("批量干预实验完成: 成功处理数据集数=%d, 结果文件=%s", len(all_results), all_results_file)
        except Exception:
            pass
        
        return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parcel Intervention System')
    parser.add_argument('--data_driven_results', type=str, 
                       default='/path/to/project_root/neural_area/connect_cap_parcel/results/rank_activation/data_driven_results.json',
                       help='数据驱动结果文件路径')
    parser.add_argument('--latent_parcel_assignments', type=str,
                       default='/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json',
                       help='latent-parcel分配文件路径')
    parser.add_argument('--max_activation_dir', type=str,
                       default='/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation',
                       help='最大激活文件目录')
    parser.add_argument('--results_dir', type=str,
                       default='/path/to/project_root/neural_area/connect_cap_parcel/results/intervention',
                       help='结果存储目录')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-2b',
                       help='模型名称')
    parser.add_argument('--sae_release', type=str, default='gemma-scope-2b-pt-res',
                       help='SAE发布名')
    parser.add_argument('--sae_local_base_dir', type=str,
                       default='/path/to/local_models/gemma-scope-2b-pt-res',
                       help='本地SAE权重目录')
    parser.add_argument('--sae_paths', type=str, default='',
                       help='逗号分隔的SAE路径列表，为空则使用默认列表')
    parser.add_argument('--top_k', type=int, default=5,
                       help='每个数据集取前k个parcel进行干预')
    parser.add_argument('--dataset', type=str, default='',
                       help='指定单个数据集进行干预，为空则处理所有数据集')
    parser.add_argument('--capability_data_dir', type=str,
                       default='/path/to/project_root/neural_area/capability_data_v2',
                       help='capability QA 数据目录路径')
    parser.add_argument('--use_optimized', action='store_true',
                       help='使用优化版本的干预函数（批量处理、预加载SAE等）')
    parser.add_argument('--is_instruct', action='store_true',
                       help='是否使用 instruct 模型（chat 模板）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批量处理的batch size')
    parser.add_argument('--sample_limit', type=int, default=-1,
                       help='每个数据集使用的样本数量限制，-1表示使用全部')
    parser.add_argument('--use_file_lock', action='store_true',
                       help='使用文件锁机制防止多进程重复处理同一数据集')
    
    args = parser.parse_args()
    
    # 创建干预系统
    intervention_system = ParcelIntervention(
        data_driven_results_path=args.data_driven_results,
        latent_parcel_assignments_path=args.latent_parcel_assignments,
        max_activation_dir=args.max_activation_dir,
        results_dir=args.results_dir,
        model_name=args.model_name,
        sae_release=args.sae_release,
        sae_local_base_dir=args.sae_local_base_dir,
        capability_data_dir=args.capability_data_dir,
        is_instruct=args.is_instruct
    )
    # 覆盖默认batch_size
    intervention_system.batch_size = args.batch_size
    
    # 加载数据
    intervention_system.load_data()
    
    # 设置SAE路径
    if args.sae_paths.strip():
        sae_paths = [p.strip() for p in args.sae_paths.split(',') if p.strip()]
    else:
        # 兼容旧默认：2B-PT 的全层（与get_sae_act.py保持一致）
        sae_paths = [
            "layer_0/width_16k/average_l0_105",
            "layer_1/width_16k/average_l0_102",
            "layer_2/width_16k/average_l0_141",
            "layer_3/width_16k/average_l0_59",
            "layer_4/width_16k/average_l0_124",
            "layer_5/width_16k/average_l0_68",
            "layer_6/width_16k/average_l0_70",
            "layer_7/width_16k/average_l0_69",
            "layer_8/width_16k/average_l0_71",
            "layer_9/width_16k/average_l0_73",
            "layer_10/width_16k/average_l0_77",
            "layer_11/width_16k/average_l0_80",
            "layer_12/width_16k/average_l0_82",
            "layer_13/width_16k/average_l0_84",
            "layer_14/width_16k/average_l0_84",
            "layer_15/width_16k/average_l0_78",
            "layer_16/width_16k/average_l0_78",
            "layer_17/width_16k/average_l0_77",
            "layer_18/width_16k/average_l0_74",
            "layer_19/width_16k/average_l0_73",
            "layer_20/width_16k/average_l0_71",
            "layer_21/width_16k/average_l0_70",
            "layer_22/width_16k/average_l0_72",
            "layer_23/width_16k/average_l0_75",
            "layer_24/width_16k/average_l0_73",
            "layer_25/width_16k/average_l0_116",
        ]
        print(f"未提供 --sae_paths，使用默认的2B-PT SAE路径，共 {len(sae_paths)} 个")
    
    # 加载模型和SAE（限制SAE数量以避免内存不足）
    intervention_system.load_model_and_saes(sae_paths, max_saes=10)
    
    # 运行干预实验
    if args.dataset:
        # 处理单个数据集
        print(f"🎯 处理指定数据集: {args.dataset}")
        
        try:
            if args.use_optimized:
                print("🚀 使用优化版本进行干预实验")
                if args.use_file_lock:
                    result = intervention_system.run_intervention_experiment_optimized_with_lock(
                        args.dataset, args.top_k, args.sample_limit
                    )
                else:
                    result = intervention_system.run_intervention_experiment_optimized(
                        args.dataset, args.top_k, args.sample_limit
                    )
            else:
                print("📊 使用标准版本进行干预实验")
                if args.use_file_lock:
                    result = intervention_system.run_intervention_experiment_with_lock(
                        args.dataset, args.top_k, args.sample_limit
                    )
                else:
                    result = intervention_system.run_intervention_experiment(args.dataset, args.top_k, args.sample_limit)
            
            if result is not None:
                # 分析效果
                all_results = {args.dataset: result}
                intervention_system.analyze_intervention_effects(all_results)
            else:
                print(f"⏭️  数据集 {args.dataset} 已处理过，跳过分析")
                
        except Exception as e:
            print(f"❌ 处理数据集 {args.dataset} 失败: {e}")
        
    else:
        # 处理所有数据集
        try:
            if args.use_optimized:
                print("🚀 使用优化版本处理所有数据集")
                if args.use_file_lock:
                    all_results = intervention_system.run_all_interventions_with_lock(
                        args.top_k, args.sample_limit, use_optimized=True
                    )
                else:
                    # 原有的优化版本处理逻辑
                    all_results = {}
                    datasets = list(intervention_system.data_driven_results.keys())
                    
                    for dataset_name in tqdm(datasets, desc="处理数据集"):
                        print(f"\n🚀 优化处理数据集: {dataset_name}")
                        result = intervention_system.run_intervention_experiment_optimized(
                            dataset_name, args.top_k, args.sample_limit
                        )
                        all_results[dataset_name] = result
                        
                        # 保存单个数据集的结果
                        result_file = os.path.join(args.results_dir, f"{dataset_name}_intervention_results_optimized.json")
                        with open(result_file, 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        print(f"  ✅ {dataset_name} 结果已保存到: {result_file}")
                    
                    # 保存所有结果
                    all_results_file = os.path.join(args.results_dir, "all_intervention_results_optimized.json")
                    with open(all_results_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=2)
                    print(f"\n✅ 所有优化干预实验完成！结果已保存到: {all_results_file}")
            else:
                print("📊 使用标准版本处理所有数据集")
                if args.use_file_lock:
                    all_results = intervention_system.run_all_interventions_with_lock(
                        args.top_k, args.sample_limit, use_optimized=False
                    )
                else:
                    all_results = intervention_system.run_all_interventions(args.top_k, args.sample_limit)
            
            # 分析效果
            if all_results:
                intervention_system.analyze_intervention_effects(all_results)
            else:
                print("⏭️  所有数据集都已处理过，跳过分析")
                
        except Exception as e:
            print(f"❌ 处理所有数据集失败: {e}")
    
    print("🎉 干预实验完成！")


if __name__ == "__main__":
    main()