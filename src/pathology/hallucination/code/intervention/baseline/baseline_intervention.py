#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Steer Vector 的 Baseline 干预系统

使用 10 条 correct 和 incorrect 数据对比得到 steer vector，然后进行干预。
评测方案与上级目录的 intervention 保持一致。
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

from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE, HookedSAETransformer


class BaselineIntervention:
    """
    基于 Steer Vector 的 Baseline 干预系统
    
    功能：
    1. 从 correct 和 incorrect 数据中提取激活值
    2. 计算 steer vector（correct - incorrect 的均值差）
    3. 在生成时应用 steer vector 进行干预
    4. 评估干预效果
    """
    
    def __init__(self,
                 model_name: str = "google/gemma-2-2b",
                 sae_release: str = "gemma-scope-2b-pt-res",
                 sae_local_base_dir: str = "/path/to/local_models/gemma-scope-2b-pt-res",
                 results_dir: str = "/path/to/project_root/safety_explanation/hallucination/results/intervention/baseline",
                 is_instruct: bool = False,
                 num_samples: int = 10,
                 layers: List[int] = None):
        """
        初始化 Baseline 干预系统
        
        Args:
            model_name: 模型名称
            sae_release: SAE发布名
            sae_local_base_dir: 本地SAE权重目录
            results_dir: 结果存储目录
            is_instruct: 是否使用instruct模型
            num_samples: 用于计算 steer vector 的样本数量（correct 和 incorrect 各取 num_samples 条）
            layers: 要干预的层列表，None 表示使用所有层
        """
        self.model_name = model_name
        self.sae_release = sae_release
        self.sae_local_base_dir = sae_local_base_dir
        self.results_dir = results_dir
        self.is_instruct = is_instruct
        self.num_samples = num_samples
        
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
        
        # Steer vector 存储
        self.steer_vectors = {}  # {layer_id: steer_vector_tensor}
        self.layers = layers  # 要干预的层列表
        self.loaded_steer_vector_path: Optional[str] = None
        
        print("🧠 Baseline Intervention System 初始化完成")
    
    def _get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        logger = logging.getLogger("baseline_intervention")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # 创建日志目录
            log_dir = os.path.join(self.results_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # 文件处理器
            log_file = os.path.join(log_dir, f"baseline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        """加载SAE模型"""
        print("🔧 加载SAE模型...")
        
        # 构建layer到SAE路径的映射
        import re
        def extract_layer_num(path_str: str) -> Optional[int]:
            m = re.search(r"layer_(\d+)/", path_str)
            return int(m.group(1)) if m else None
        
        layer_to_sae_path = {}
        for path in sae_paths:
            layer_num = extract_layer_num(path)
            if layer_num is not None:
                layer_to_sae_path[layer_num] = path
        
        print(f"  可用SAE层数: {len(layer_to_sae_path)}")
        
        # 如果指定了层列表，只加载这些层
        if self.layers is not None:
            layer_to_sae_path = {k: v for k, v in layer_to_sae_path.items() if k in self.layers}
            print(f"  筛选后SAE层数: {len(layer_to_sae_path)}")
        
        # 加载SAE（限制数量以避免内存不足）
        loaded_count = 0
        for layer_id, sae_path in layer_to_sae_path.items():
            if loaded_count >= max_saes:
                print(f"  达到最大SAE加载数量限制: {max_saes}")
                break
                
            try:
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_path,
                    device=self.device,
                    local_path=os.path.join(self.sae_local_base_dir, sae_path, "params.npz")
                )
                sae.to(self.device)
                sae.use_error_term = True
                
                self.layer_to_sae[layer_id] = sae
                self.layer_to_hook_name[layer_id] = sae.cfg.metadata.hook_name if hasattr(sae.cfg, 'metadata') else sae.cfg.hook_name
                
                loaded_count += 1
                print(f"    ✅ 层 {layer_id} SAE加载成功")
                
            except Exception as e:
                print(f"    ❌ 层 {layer_id} SAE加载失败: {e}")
        
        print(f"  ✅ 共加载 {loaded_count} 个SAE")
        
        # 如果没有指定层列表，使用所有加载的层
        if self.layers is None:
            self.layers = sorted(self.layer_to_sae.keys())
    
    def _build_prompt(self, question: str, context: List[str] = None) -> str:
        """构建prompt"""
        if context:
            context_text = " ".join(context) if isinstance(context, list) else str(context)
            return f"Context: {context_text}\nQ: {question}\nA:"
        else:
            return f"Q: {question}\nA:"
    
    def extract_activations(self, prompt: str, layer_id: int) -> torch.Tensor:
        """
        提取指定层的激活值
        
        Args:
            prompt: 输入提示
            layer_id: 层ID
            
        Returns:
            激活值张量 [seq_len, hidden_dim]
        """
        if layer_id not in self.layer_to_sae:
            raise ValueError(f"层 {layer_id} 的SAE未加载")
        
        hook_name = self.layer_to_hook_name[layer_id]
        
        # Tokenize
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        
        # 获取激活值
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                input_ids,
                stop_at_layer=layer_id + 1,
                names_filter=[hook_name]
            )
        
        if hook_name not in cache:
            raise RuntimeError(f"未能获取层 {layer_id} 的激活值")
        
        activations = cache[hook_name]  # [batch, seq_len, hidden_dim]
        return activations[0]  # [seq_len, hidden_dim]
    
    def load_steer_vectors(self, steer_vector_path: str) -> None:
        """
        从文件加载预计算的 steer vectors
        
        Args:
            steer_vector_path: 保存 steer vectors 的文件路径
        """
        if not os.path.exists(steer_vector_path):
            raise FileNotFoundError(f"未找到 steer vector 文件: {steer_vector_path}")
        
        print(f"📥 加载 steer vector 文件: {steer_vector_path}")
        try:
            data = torch.load(steer_vector_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"加载 steer vector 文件失败: {e}")
        
        if isinstance(data, dict):
            if "steer_vectors" in data:
                steer_vectors_raw = data["steer_vectors"]
                metadata = data.get("metadata", {})
            else:
                steer_vectors_raw = data
                metadata = {}
        else:
            raise ValueError("steer vector 文件格式不正确，期望为包含 'steer_vectors' 字段的字典")
        
        loaded_vectors: Dict[int, torch.Tensor] = {}
        for key, value in steer_vectors_raw.items():
            try:
                layer_id = int(key)
            except Exception:
                raise ValueError(f"steer vector 文件中的层编号无法转换为整数: {key}")
            
            tensor = torch.tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value.to(torch.float32)
            loaded_vectors[layer_id] = tensor.to(self.device)
        
        if not loaded_vectors:
            raise ValueError("steer vector 文件为空，未加载到任何向量")
        
        self.steer_vectors = loaded_vectors
        if self.layers is None or not self.layers:
            self.layers = sorted(loaded_vectors.keys())
        else:
            missing_layers = [layer for layer in self.layers if layer not in loaded_vectors]
            if missing_layers:
                print(f"⚠️ 指定的层 {missing_layers} 在 steer vector 文件中缺失，将忽略这些层")
                self.layers = [layer for layer in self.layers if layer in loaded_vectors]
        
        self.loaded_steer_vector_path = steer_vector_path
        print(f"  ✅ 成功加载 {len(self.steer_vectors)} 个层的 steer vector")
    
    def create_steering_hook(self, layer_id: int, intervention_strength: float):
        """
        创建针对特定层的 steering hook
        
        Args:
            layer_id: 要干预的层ID
            intervention_strength: 干预强度
            
        Returns:
            hook函数
        """
        if layer_id not in self.steer_vectors:
            return None
        
        steer_vector = self.steer_vectors[layer_id]
        
        def steering_hook(activations, hook):
            """Steering hook函数"""
            # 应用steering: activations + strength * steer_vector
            # steer_vector 需要广播到 [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activations.shape
            sv = steer_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            sv = sv.expand(batch_size, seq_len, hidden_dim)  # [batch, seq_len, hidden_dim]
            
            # 数值防护
            sv = torch.nan_to_num(sv, nan=0.0, posinf=1e6, neginf=-1e6)
            
            activations = activations + intervention_strength * sv
            return activations
        
        return steering_hook
    
    def generate_with_intervention(self, 
                                 prompt: str, 
                                 intervention_strength: float = 1.0,
                                 max_new_tokens: int = 256,
                                 temperature: float = 0.0,
                                 examples: List[Dict[str, Any]] = None) -> str:
        """
        使用干预生成文本
        
        Args:
            prompt: 输入提示
            intervention_strength: 干预强度
            max_new_tokens: 最大新token数
            temperature: 生成温度
            examples: in-context learning示例（仅用于PT模型）
            
        Returns:
            生成的文本
        """
        if not self.steer_vectors:
            raise RuntimeError("尚未加载任何 steer vector，请先调用 load_steer_vectors()")
        
        if not self.layers:
            self.layers = sorted(self.steer_vectors.keys())
        
        # 构建输入
        if self.is_instruct:
            messages = [{"role": "user", "content": f"Please answer the question shortly and concisely without any additional explanation. {prompt}"}]
            if hasattr(self.tokenizer, "apply_chat_template"):
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                full_text = f"User: {prompt}\nAssistant:"
        else:
            full_text = prompt
        
        # Tokenize
        tokenized = self.tokenizer(full_text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        
        # 创建hooks
        hooks = []
        for layer_id in self.layers:
            if layer_id in self.layer_to_hook_name:
                hook_name = self.layer_to_hook_name[layer_id]
                hook_func = self.create_steering_hook(layer_id, intervention_strength)
                if hook_func is not None:
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
            intervention_strength=0.0,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            examples=examples
        )
    
    def evaluate_intervention(self, 
                             test_data: List[Dict[str, Any]], 
                             intervention_strength: float = 1.0,
                             use_incontext: bool = False) -> Dict[str, Any]:
        """
        评估干预效果
        
        Args:
            test_data: 测试数据列表
            intervention_strength: 干预强度
            use_incontext: 是否使用in-context learning（仅PT模型）
            
        Returns:
            评估结果字典
        """
        print(f"🧪 开始评估干预效果，strength={intervention_strength}")
        
        results = {
            "intervention_strength": intervention_strength,
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
                    current_question, intervention_strength, examples=examples, max_new_tokens=128
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
                traceback.print_exc()
        
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
                                   save_results: bool = True,
                                   use_incontext: bool = False,
                                   intervention_strength: float = 1.0) -> Dict[str, Any]:
        """
        运行完整的干预实验
        
        Args:
            test_data: 测试数据
            save_results: 是否保存结果
            use_incontext: 是否使用in-context learning（仅PT模型）
            intervention_strength: 干预强度
            
        Returns:
            实验结果
        """
        print(f"🚀 开始干预实验，strength={intervention_strength}")
        
        all_results = {
            "experiment_info": {
                "model_name": self.model_name,
                "num_samples": len(test_data),
                "intervention_strength": intervention_strength,
                "layers": self.layers,
                "timestamp": datetime.datetime.now().isoformat()
            },
            "intervention_results": {}
        }
        
        try:
            result = self.evaluate_intervention(test_data, intervention_strength, use_incontext)
            all_results["intervention_results"][str(intervention_strength)] = result
            
        except Exception as e:
            print(f"  ❌ 强度 {intervention_strength} 测试失败: {e}")
            traceback.print_exc()
            all_results["intervention_results"][str(intervention_strength)] = {"error": str(e)}
        
        # 保存结果
        if save_results:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                self.results_dir, 
                f"baseline_intervention_{timestamp}.json"
            )
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ 实验结果已保存到: {result_file}")
        
        return all_results

