#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幻觉干预系统主类

基于Parcel激活调节机制，对模型进行幻觉缓解干预。
整合模型加载、SAE处理、激活调节和生成评估功能。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import numpy as np
import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from tqdm import tqdm
import traceback

# 设置环境变量
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer
from parcel_scaler import ParcelScaler
import re


def get_latents_per_layer(model_name: str, sae_release: str) -> int:
    """
    根据模型名称和 SAE release 自动选择每层 latent 数。
    支持 Gemma (16384) 和 Llama-3.1-8B + LXR-8x (32768)。
    """
    ml = model_name.lower()
    rl = sae_release.lower()
    if "gemma" in ml or "gemma" in rl:
        return 16384
    if "llama" in ml or "llama" in rl or "lxr" in rl:
        return 32768
    # 默认回退 16384，并打印 warning
    import warnings
    warnings.warn(f"未知模型/SAE 组合 (model={model_name}, release={sae_release})，使用默认 latents_per_layer=16384")
    return 16384


def extract_layer_num(path_str: str, sae_release: str) -> Optional[int]:
    """
    从 SAE 路径中解析层号，根据 sae_release 自动区分 Gemma 和 Llama 格式。
    """
    if "gemma" in sae_release.lower():
        # Gemma: layer_9/width_16k/...
        m = re.search(r"layer_(\d+)/", path_str)
        return int(m.group(1)) if m else None
    else:
        # Llama LXR-8x: l0r_8x,l1r_8x
        m = re.search(r"l(\d+)r_8x", path_str)
        return int(m.group(1)) if m else None


class HallucinationIntervention:
    """
    幻觉干预系统主类
    
    功能：
    1. 加载模型和SAE
    2. 应用Parcel激活调节
    3. 生成文本并评估幻觉缓解效果
    """
    
    def __init__(self,
                 model_name: str = "google/gemma-2-2b",
                 sae_release: str = "gemma-scope-2b-pt-res",
                 sae_local_base_dir: str = "/path/to/local_models/gemma-scope-2b-pt-res",
                 parcel_json_path: str = "/path/to/project_root/safety_explanation/hallucination/results/analysis_output/truthfulqa_gemma-2-2b/parcel_level/top_anomalous_parcels.json",
                 latent_parcel_assignments_path: str = "/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json",
                 max_activation_dir: str = "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation",
                 results_dir: str = "/path/to/project_root/safety_explanation/hallucination/results/intervention",
                 is_instruct: bool = False,
                 lambda_scale: float = 0.3,
                 smooth: float = 80.0,
                 min_scale: float = -1.0,
                 max_scale: float = 1.0,
                 strength: float = 0.0):
        """
        初始化幻觉干预系统
        
        Args:
            model_name: 模型名称
            sae_release: SAE发布名
            sae_local_base_dir: 本地SAE权重目录
            parcel_json_path: 异常parcel分析结果文件路径
            latent_parcel_assignments_path: latent-parcel分配文件路径
            results_dir: 结果存储目录
            is_instruct: 是否使用instruct模型
            lambda_scale: 调节幅度
            smooth: 平滑系数
            min_scale: 最小缩放系数
            max_scale: 最大缩放系数
            strength: 干预强度
        """
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_local_base_dir = sae_local_base_dir
        self.parcel_json_path = parcel_json_path
        self.latent_parcel_assignments_path = latent_parcel_assignments_path
        self.max_activation_dir = max_activation_dir
        self.results_dir = results_dir
        self.is_instruct = is_instruct
        self.strength = strength
        
        # 设备选择
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.tokenizer = None
        self.sae_list = []
        self.layer_to_sae = {}
        self.layer_to_hook_name = {}
        self.latent_parcel_assignments = None
        self.max_activations = {}
        
        # 计算每层 latent 数
        self.latents_per_layer = get_latents_per_layer(model_name, sae_release)
        print(f"📊 每层 latent 数: {self.latents_per_layer} (model={model_name}, release={sae_release})")
        
        # 初始化logger
        self.logger = self._get_logger()
        
        # 初始化Parcel调节器
        self.parcel_scaler = ParcelScaler(
            parcel_json_path=parcel_json_path,
            lambda_scale=lambda_scale,
            smooth=smooth,
            min_scale=min_scale,
            max_scale=max_scale,
            logger=self.logger
        )
        
        # 干预强度配置
        self.intervention_strengths = [self.strength]
        
        print("🧠 Hallucination Intervention System 初始化完成")
    
    def _get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        logger = logging.getLogger("hallucination_intervention")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # 创建日志目录
            log_dir = os.path.join(self.results_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 文件处理器
            log_file = os.path.join(log_dir, f"intervention_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            
            # 控制台处理器
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        
        return logger
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("🤖 加载模型和分词器...")
        
        try:
            # 禁用torch.compile以提高兼容性
            torch._dynamo.config.disable = True
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # 加载模型
            if "9b" in self.model_name:
                self.model = HookedSAETransformer.from_pretrained(
                    self.model_name,
                    device=self.device,
                    dtype=torch.bfloat16,
                    n_devices=1,
                    trust_remote_code=True
                )
            else:
                self.model = HookedSAETransformer.from_pretrained(
                    self.model_name,
                    device=self.device,
                    dtype=torch.bfloat16,
                    trust_remote_code=True
                )
            
            # 设置为评估模式
            self.model.eval()
            
            print(f"  ✅ 模型加载成功: {self.model_name}")
            print(f"  ✅ 设备: {self.device}")
            
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            raise
    
    def load_saes(self, sae_paths: List[str], max_saes: int = 100):
        """
        加载SAE模型
        兼容 Gemma 与 Llama Scope LXR-8x:
        1. 优先使用本地 Llama Scope final.safetensors
        2. 其次尝试 Gemma 本地 params.npz
        3. 最后退回到 SAE.from_pretrained 默认下载/缓存
        """
        print("🔧 加载SAE模型...")
        
        # 加载latent-parcel分配
        with open(self.latent_parcel_assignments_path, 'r', encoding='utf-8') as f:
            self.latent_parcel_assignments = json.load(f)
        
        # 加载最大激活数据
        self.load_max_activations()
        
        # 构建layer到SAE路径的映射
        layer_to_sae_path = {}
        for path in sae_paths:
            layer_num = extract_layer_num(path, self.sae_release)
            if layer_num is not None:
                layer_to_sae_path[layer_num] = path
        
        print(f"  可用SAE层数: {len(layer_to_sae_path)}")
        
        # 加载SAE（限制数量以避免内存不足）
        loaded_count = 0
        for layer_id, sae_path in layer_to_sae_path.items():
            if loaded_count >= max_saes:
                print(f"  达到最大SAE加载数量限制: {max_saes}")
                break
                
            try:
                # 检查是否是 Llama Scope LXR-8x 格式 (l0r_8x, l1r_8x, ...)
                m = re.search(r"l(\d+)r", sae_path)
                if m:
                    sae_id_num = m.group(1)
                    real_path = f"Llama3_1-8B-Base-L{sae_id_num}R-8x"
                    llama_scope_local_path = os.path.join(
                        self.sae_local_base_dir, real_path, "checkpoints", "final.safetensors"
                    )
                else:
                    real_path = sae_path
                    llama_scope_local_path = None
                
                gemma_local_path = os.path.join(
                    self.sae_local_base_dir, sae_path, "params.npz"
                )
                
                # 按优先级尝试加载
                if llama_scope_local_path and os.path.exists(llama_scope_local_path):
                    # 优先使用本地 Llama Scope final.safetensors
                    sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release=self.sae_release,
                        sae_id=sae_path,
                        device=self.device,
                        local_path=llama_scope_local_path,
                    )
                elif os.path.exists(gemma_local_path):
                    # 其次尝试 Gemma 本地 params.npz
                    sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release=self.sae_release,
                        sae_id=sae_path,
                        device=self.device,
                        local_path=gemma_local_path,
                    )
                else:
                    # 最后退回到 SAE.from_pretrained 默认下载/缓存
                    sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release=self.sae_release,
                        sae_id=sae_path,
                        device=self.device,
                    )
                
                sae.to(self.device)
                sae.use_error_term = True
                
                self.layer_to_sae[layer_id] = sae
                self.layer_to_hook_name[layer_id] = sae.cfg.metadata.hook_name if hasattr(sae.cfg, 'metadata') else sae.cfg.hook_name
                
                loaded_count += 1
                print(f"    ✅ 层 {layer_id} SAE加载成功")
                
            except Exception as e:
                print(f"    ❌ 层 {layer_id} SAE加载失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"  ✅ 共加载 {loaded_count} 个SAE")
    
    def load_max_activations(self):
        """加载最大激活数据"""
        print("📊 加载最大激活数据...")
        
        # 根据模型类型选择文件（先匹配 9b，再 8b，避免误用默认 2b 文件）
        model_name_lower = self.model_name.lower()
        if "9b" in model_name_lower:
            merged_activation_path = os.path.join(self.max_activation_dir, 'merged_max_activations_9b.pkl')
        elif "8b" in model_name_lower:
            merged_activation_path = os.path.join(self.max_activation_dir, 'merged_max_activations_8b.pkl')
        else:
            merged_activation_path = os.path.join(self.max_activation_dir, 'merged_max_activations.pkl')
        
        print(f"  加载最大激活文件: {merged_activation_path}")
        
        if not os.path.exists(merged_activation_path):
            raise FileNotFoundError(f"未找到最大激活文件: {merged_activation_path}")
        
        import pickle
        with open(merged_activation_path, 'rb') as f:
            merged_activations = pickle.load(f)
        
        # 存储最大激活数据
        self.max_activations['merged'] = merged_activations
        print(f"  ✅ 加载了 {len(merged_activations)} 个特征的最大激活数据")
    
    def get_parcel_latent_mapping(self, parcel_id: int) -> List[Tuple[int, int]]:
        """
        获取指定parcel对应的所有latent id及其层信息
        
        Args:
            parcel_id: parcel ID
            
        Returns:
            [(layer_id, latent_id_in_layer), ...]
        """
        parcel_name = f"parcel_{parcel_id}"
        
        if parcel_name not in self.latent_parcel_assignments['parcel_to_latents']:
            raise KeyError(f"未找到 parcel: {parcel_name}")
        
        latent_ids = self.latent_parcel_assignments['parcel_to_latents'][parcel_name]
        available_layers = sorted(self.layer_to_sae.keys())
        
        # 如果没有加载SAE，使用默认层号
        if not available_layers:
            available_layers = list(range(26))  # 假设有26层
            print(f"⚠️ 没有加载SAE，使用默认层号: {available_layers[:5]}...")
        
        mapping = []
        for latent_id in latent_ids:
            # 计算原始层索引（使用动态的 latents_per_layer）
            original_layer_idx = latent_id // self.latents_per_layer
            
            # 检查是否在可用层范围内
            if original_layer_idx < len(available_layers):
                actual_layer_id = available_layers[original_layer_idx]
            else:
                # 回退到可用层
                layer_index = original_layer_idx % len(available_layers)
                actual_layer_id = available_layers[layer_index]
            
            # 计算该层内的latent id（使用动态的 latents_per_layer）
            latent_in_layer = latent_id % self.latents_per_layer
            mapping.append((actual_layer_id, latent_in_layer))
        
        return mapping
    
    def create_steering_hook(self, parcel_id: int, layer_id: int, intervention_strength: float):
        """
        创建针对特定parcel和特定层的steering hook
        
        Args:
            parcel_id: 要干预的parcel ID
            layer_id: 要干预的层ID
            intervention_strength: 干预强度
            
        Returns:
            hook函数
        """
        # 获取parcel对应的latent映射
        parcel_latent_mapping = self.get_parcel_latent_mapping(parcel_id)
        
        # 只获取该parcel在该层的latents
        layer_latents = [latent for l_id, latent in parcel_latent_mapping if l_id == layer_id]
        
        if not layer_latents:
            return None  # 该parcel在该层没有latents
        
        # 获取scaling系数
        scaling_factor = self.parcel_scaler.get_scaling_factor(parcel_id)
        # 获取最大激活值（只针对当前层）
        max_acts = {}
        merged_activations = self.max_activations['merged']
        for latent_in_layer in layer_latents:
            key = (layer_id, latent_in_layer)
            if key not in merged_activations:
                self.logger.warning(f"缺少层 {layer_id} latent {latent_in_layer} 的最大激活值，使用默认值1.0")
                print(f"缺少层 {layer_id} latent {latent_in_layer} 的最大激活值，使用默认值1.0")
                max_acts[key] = 1.0
            else:
                max_acts[key] = merged_activations[key]
        
        def steering_hook(activations, hook):
            """Steering hook函数 - 只处理当前层"""
            if layer_id not in self.layer_to_sae:
                return activations
            
            sae = self.layer_to_sae[layer_id]
            
            # 获取该层的steering向量和最大激活值
            steering_vectors = torch.stack([sae.W_dec[li] for li in layer_latents], dim=0)
            max_act_values = torch.tensor([max_acts[(layer_id, li)] for li in layer_latents], 
                                        device=activations.device, dtype=activations.dtype)
            
            # 应用scaling和intervention strength
            total_strength = intervention_strength * scaling_factor
            
            # 对每个latent应用steering
            for i, latent_idx in enumerate(layer_latents):
                steering_vector = steering_vectors[i]
                max_act = max_act_values[i]
                
                # 数值防护：净化steering向量和max_act
                sv = torch.nan_to_num(steering_vector, nan=0.0, posinf=1e6, neginf=-1e6)
                ma = torch.clamp(max_act, -1e3, 1e3)
                # 应用steering: activations + max_act * strength * steering_vector
                activations = activations + ma * total_strength * sv
            
            return activations
        
        return steering_hook
    
    def _build_incontext_prompt(self, current_question: str, examples: List[Dict[str, Any]]) -> str:
        """
        构建in-context learning prompt（仅用于PT模型）
        
        Args:
            current_question: 当前要回答的问题
            examples: 前两个样本作为示例
            
        Returns:
            构建好的prompt
        """
        # 构建示例部分
        example_text = ""
        for idx, ex in enumerate(examples):
            ctx_text = ""
            if ex.get("context"):
                # context是列表，合并为文本
                ctx_text = " ".join(ex["context"])
                example_text += f"Context: {ctx_text}\n"
            example_text += f"Q: {ex['question']}\n"
            # 使用第一个answer作为示例
            answer_true = ex.get("answer_true", [])
            if isinstance(answer_true, list) and answer_true:
                example_text += f"A: {answer_true[0]}\n\n"
            else:
                example_text += f"A: {answer_true}\n\n"
        
        # 构建当前问题部分
        # 检测 current_question 是否已经包含 Context: 前缀
        if current_question.startswith("Context:"):
            # 如果已经包含 Context:，则直接添加 \nQ: 和 \nA:
            # 从文本中提取真正的 question 部分
            # 格式应该是: "Context: ...\n问题内容"
            parts = current_question.split("\n", 1)
            if len(parts) == 2:
                # 第一部分是 Context，第二部分是 question
                context_part = parts[0]
                question_part = parts[1]
                current_text = f"{context_part}\nQ: {question_part}\nA:"
            else:
                # 如果格式不对，直接使用原文本
                current_text = f"{current_question}\nQ: \nA:"
        else:
            # 如果只是单纯的问题，添加 Q: 前缀
            current_text = f"Q: {current_question}\nA:"
        
        # PT模型直接使用拼接的文本
        return f"{example_text}{current_text}"
    
    def generate_with_intervention(self, 
                                 prompt: str, 
                                 parcel_ids: List[int], 
                                 intervention_strength: float = 1.0,
                                 max_new_tokens: int = 256,
                                 temperature: float = 0.0,
                                 examples: List[Dict[str, Any]] = None) -> str:
        """
        使用干预生成文本
        
        Args:
            prompt: 输入提示
            parcel_ids: 要干预的parcel ID列表
            intervention_strength: 干预强度
            max_new_tokens: 最大新token数
            temperature: 生成温度
            examples: in-context learning示例（仅用于PT模型）
            
        Returns:
            生成的文本
        """
        # 构建输入
        # PT模型：如果有examples则使用in-context learning
        if examples:
            full_text = self._build_incontext_prompt(prompt, examples)
        else:
            full_text = prompt
        if self.is_instruct:
            messages = [{"role": "user", "content": f"Please answer the question shortly and concisely without any additional explanation. {full_text}"}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                print(f"IT模型prompt: {full_text}")
            else:
                full_text = f"User: {full_text}\nAssistant:"
        # 添加max_length限制
        tokenized = self.tokenizer(full_text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        
        # 创建hooks
        hooks = []
        for parcel_id in parcel_ids:
            if parcel_id in self.parcel_scaler.get_scaling_table():
                # 获取该parcel涉及的所有层（去重）
                parcel_latent_mapping = self.get_parcel_latent_mapping(parcel_id)
                unique_layers = {layer_id for layer_id, _ in parcel_latent_mapping}
                
                # 为每个层创建专门的steering_hook
                for layer_id in unique_layers:
                    if layer_id in self.layer_to_hook_name:
                        hook_name = self.layer_to_hook_name[layer_id]
                        hook_func = self.create_steering_hook(parcel_id, layer_id, intervention_strength)
                        if hook_func is not None:  # 该parcel在该层有latents
                            hooks.append((hook_name, hook_func))
        
        # 生成文本
        with torch.no_grad():
            # 基础生成配置
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True if temperature > 0 else False,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # IT模型添加采样参数
            if self.is_instruct:
                gen_kwargs.update({
                    "temperature": temperature,
                    "top_p": 0.9,
                })
            
            # PT模型停止字符串（生成后处理）
            stop_strings = []
            if not self.is_instruct and "2b" in self.model_name:
                stop_strings = ["\nQ:", "\nContext:"]
            if hooks:
                with self.model.hooks(fwd_hooks=hooks):
                    outputs = self.model.generate(input_ids, **gen_kwargs)
            else:
                outputs = self.model.generate(input_ids, **gen_kwargs)
        
        # 解码输出
        generated_text = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        # 移除停止字符串（仅PT模型）
        if stop_strings:
            for stop_string in stop_strings:
                if stop_string in generated_text:
                    generated_text = generated_text.split(stop_string)[0]
        
        return generated_text.strip()
    
    def generate_baseline(self, 
                         prompt: str, 
                         max_new_tokens: int = 256,
                         temperature: float = 0.0,
                         examples: List[Dict[str, Any]] = None) -> str:
        """生成基线文本（无干预）"""
        return self.generate_with_intervention(
            prompt=prompt,
            parcel_ids=[],
            intervention_strength=0.0,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            examples=examples
        )
    
    def evaluate_intervention(self, 
                             test_data: List[Dict[str, Any]], 
                             parcel_ids: List[int],
                             intervention_strength: float = 1.0,
                             use_incontext: bool = False) -> Dict[str, Any]:
        """
        评估干预效果
        
        Args:
            test_data: 测试数据列表
            parcel_ids: 要干预的parcel ID列表
            intervention_strength: 干预强度
            use_incontext: 是否使用in-context learning（仅PT模型）
            
        Returns:
            评估结果字典
        """
        print(f"🧪 开始评估干预效果，parcel_ids={parcel_ids}, strength={intervention_strength}")
        
        results = {
            "intervention_strength": intervention_strength,
            "parcel_ids": parcel_ids,
            "num_samples": len(test_data),
            "baseline_results": [],
            "intervention_results": [],
            "comparison": {}
        }
        
        # 准备in-context示例（仅用于PT模型）
        examples = []
        start_idx = 1  # 默认从第1条开始
        if use_incontext and not self.is_instruct and len(test_data) >= 2:
            try:
                # 使用前两个样本作为示例（仅PT模型）
                for i in range(2):
                    ex_sample = test_data[i]
                    ex_fields = {
                        "question": ex_sample.get("question", ""),
                        "context": ex_sample.get("context", []),
                        "answer_true": ex_sample.get("answer_true", [])
                    }
                    examples.append(ex_fields)
                start_idx = 3  # 从第3条开始生成
                print(f"  📚 使用in-context learning，前2条作为示例，从第3条开始生成")
            except Exception as e:
                print(f"  ⚠️ 无法构建in-context示例: {e}，将不使用in-context learning")
                examples = []
                start_idx = 1
        
        # 处理每个测试样本
        for i, sample in tqdm(enumerate(test_data), desc="评估样本"):
            # 跳过示例
            if i < start_idx:
                continue
            prompt = sample.get("question", "")
            context = sample.get("context", [])
            answer_true = sample.get("answer_true", [])
            
            if not prompt:
                continue
            
            # 构建完整prompt
            # 如果使用 in-context learning，不需要添加 Q: A: 标签（_build_incontext_prompt 会处理）
            if examples:
                # 只传入问题和context，不带 Q: A: 标签
                if context:
                    context_text = " ".join(context) if isinstance(context, list) else str(context)
                    current_question = f"Context: {context_text}\n{prompt}"
                else:
                    current_question = prompt
            else:
                # 不使用 in-context learning，需要添加 Q: A: 标签
                if context:
                    context_text = " ".join(context) if isinstance(context, list) else str(context)
                    current_question = f"Context: {context_text}\nQ: {prompt}\nA:"
                else:
                    current_question = f"Q: {prompt}\nA:"
            
            try:
                # 生成基线文本
                baseline_text = self.generate_baseline(current_question, examples=examples, max_new_tokens=128)
                
                # 生成干预文本
                intervention_text = self.generate_with_intervention(
                    current_question, parcel_ids, intervention_strength, examples=examples, max_new_tokens=128
                )
                
                # 记录结果
                sample_result = {
                    "index": i,
                    "question": current_question,
                    "context": context,
                    "answer_true": answer_true,
                    "baseline_text": baseline_text,
                    "intervention_text": intervention_text
                }
                # 若有 meta（如 sycophancy 的 gen_eval 格式），保留用于评估（group_id、original_record 等）
                meta = sample.get("meta", {})
                if meta:
                    for key in ("original_record", "group_id", "group_position", "base", "source_file"):
                        if key in meta:
                            sample_result[key] = meta[key]

                results["baseline_results"].append(sample_result)
                results["intervention_results"].append(sample_result)
                
                print(f"\n样本 {i+1}:")
                print(f"  问题: {prompt}")
                print(f"  基线回答: {baseline_text[:100]}...")
                print(f"  干预回答: {intervention_text[:100]}...")
                print(f"  正确答案: {answer_true}")
                print(f"  干预效应: {baseline_text != intervention_text}")
                
            except Exception as e:
                print(f"  ❌ 样本 {i} 处理失败: {e}")
        
        # 计算比较统计
        if results["baseline_results"] and results["intervention_results"]:
            baseline_lengths = [len(r["baseline_text"]) for r in results["baseline_results"]]
            intervention_lengths = [len(r["intervention_text"]) for r in results["intervention_results"]]
            
            results["comparison"] = {
                "avg_baseline_length": np.mean(baseline_lengths),
                "avg_intervention_length": np.mean(intervention_lengths),
                "length_change": np.mean(intervention_lengths) - np.mean(baseline_lengths)
            }
        
        return results
    
    def run_intervention_experiment(self, 
                                   test_data: List[Dict[str, Any]], 
                                   parcel_ids: List[int],
                                   save_results: bool = True,
                                   use_incontext: bool = False) -> Dict[str, Any]:
        """
        运行完整的干预实验
        
        Args:
            test_data: 测试数据
            parcel_ids: 要干预的parcel ID列表
            save_results: 是否保存结果
            use_incontext: 是否使用in-context learning（仅PT模型）
            
        Returns:
            实验结果
        """
        print(f"🚀 开始干预实验，parcel_ids={parcel_ids}")
        
        all_results = {
            "experiment_info": {
                "model_name": self.model_name,
                "parcel_ids": parcel_ids,
                "num_samples": len(test_data),
                "timestamp": datetime.datetime.now().isoformat()
            },
            "parcel_scaler_info": self.parcel_scaler.summary(),
            "intervention_results": {}
        }
        
        # 测试不同干预强度
        for strength in self.intervention_strengths:
            print(f"\n测试干预强度: {strength}")
            
            try:
                result = self.evaluate_intervention(test_data, parcel_ids, strength, use_incontext)
                all_results["intervention_results"][str(strength)] = result
                
            except Exception as e:
                print(f"  ❌ 强度 {strength} 测试失败: {e}")
                all_results["intervention_results"][str(strength)] = {"error": str(e)}
        
        # 保存结果
        if save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                self.results_dir, 
                f"intervention_experiment_{timestamp}.json"
            )
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ 实验结果已保存到: {result_file}")
        
        return all_results


def main():
    """主函数示例"""
    # 创建干预系统
    intervention = HallucinationIntervention(
        model_name="google/gemma-2-2b",
        is_instruct=False
    )
    
    # 加载模型
    intervention.load_model_and_tokenizer()
    
    # 加载SAE（使用默认路径）
    default_sae_paths = [
        "layer_0/width_16k/average_l0_105",
        "layer_1/width_16k/average_l0_102",
        "layer_2/width_16k/average_l0_141",
        "layer_3/width_16k/average_l0_59",
        "layer_4/width_16k/average_l0_124",
    ]
    intervention.load_saes(default_sae_paths, max_saes=100)
    
    # 测试数据
    test_data = [
        {
            "question": "What is the capital of France?",
            "context": [],
            "answer_true": ["Paris"]
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "context": [],
            "answer_true": ["William Shakespeare"]
        }
    ]
    
    # 获取需要干预的parcel IDs
    scaling_table = intervention.parcel_scaler.get_scaling_table()
    parcel_ids = list(scaling_table.keys())[:5]  # 取前5个
    
    print(f"干预parcel IDs: {parcel_ids}")
    print(intervention.parcel_scaler.visualize_scaling())
    
    # 运行实验
    results = intervention.run_intervention_experiment(test_data, parcel_ids)
    
    print("🎉 干预实验完成！")


if __name__ == "__main__":
    main()
