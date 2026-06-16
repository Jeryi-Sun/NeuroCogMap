#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幻觉数据读取和评估模块

支持多种幻觉数据集的读取、处理和评估功能。
"""

import os
import json
import csv
import ast
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from tqdm import tqdm


class HallucinationDataLoader:
    """
    幻觉数据加载器
    
    支持CSV格式的幻觉数据集读取和预处理
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def parse_list_string(self, value: str) -> List[str]:
        """尝试将字符串解析为列表，如果失败则返回单元素列表"""
        if not value or not isinstance(value, str):
            return []
        
        value = value.strip()
        if not value:
            return []
        
        # 尝试解析为Python列表
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            else:
                return [str(parsed).strip()]
        except:
            # 如果解析失败，返回原字符串作为单元素列表
            return [value]
    
    def load_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """读取CSV文件并返回字典列表"""
        rows: List[Dict[str, Any]] = []
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
            if not rows:
                raise ValueError(f"CSV文件为空或无法读取任何行: {csv_path}")
            return rows
        except Exception as e:
            raise Exception(f"读取CSV文件失败 {csv_path}: {e}")
    
    def resolve_row_fields(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """解析行中的关键字段：context, question, answer_true"""
        # 获取question
        question = row.get("question", "").strip()
        if not question:
            raise KeyError(f"缺少question字段: {json.dumps(row, ensure_ascii=False)[:200]}")
        
        # 获取context（可选）
        context_raw = row.get("context", "").strip()
        context_list = []
        if context_raw:
            context_list = self.parse_list_string(context_raw)
        
        # 获取answer_true
        answer_raw = row.get("answer_true", "").strip()
        if not answer_raw:
            raise KeyError(f"缺少answer_true字段: {json.dumps(row, ensure_ascii=False)[:200]}")
        
        answer_list = self.parse_list_string(answer_raw)
        if not answer_list:
            raise ValueError(f"answer_true为空: {json.dumps(row, ensure_ascii=False)[:200]}")
        
        return {
            "question": question,
            "context": context_list,
            "answer_true": answer_list
        }
    
    def load_dataset(self, dataset_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集路径（CSV文件或目录）
            max_samples: 最大样本数，0表示全部
            
        Returns:
            处理后的数据列表
        """
        if os.path.isfile(dataset_path):
            # 单个文件
            return self._load_single_file(dataset_path, max_samples)
        elif os.path.isdir(dataset_path):
            # 目录，处理所有CSV文件
            return self._load_directory(dataset_path, max_samples)
        else:
            raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    
    def _load_single_file(self, file_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
        """加载单个CSV文件"""
        if not file_path.endswith('.csv'):
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        self.logger.info(f"加载数据集文件: {file_path}")
        
        # 读取CSV
        rows = self.load_csv(file_path)
        
        # 处理数据
        processed_data = []
        for i, row in enumerate(rows):
            if max_samples > 0 and i >= max_samples:
                break
                
            try:
                fields = self.resolve_row_fields(row)
                processed_data.append(fields)
            except Exception as e:
                self.logger.warning(f"跳过第{i+1}行数据: {e}")
                continue
        
        self.logger.info(f"成功加载 {len(processed_data)} 条数据")
        return processed_data
    
    def _load_directory(self, dir_path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
        """加载目录下所有CSV文件"""
        csv_files = list(Path(dir_path).glob("*.csv"))
        
        if not csv_files:
            raise ValueError(f"目录下没有找到CSV文件: {dir_path}")
        
        self.logger.info(f"找到 {len(csv_files)} 个CSV文件")
        
        all_data = []
        for csv_file in csv_files:
            try:
                file_data = self._load_single_file(str(csv_file), max_samples)
                all_data.extend(file_data)
                
                if max_samples > 0 and len(all_data) >= max_samples:
                    all_data = all_data[:max_samples]
                    break
                    
            except Exception as e:
                self.logger.error(f"加载文件 {csv_file} 失败: {e}")
                continue
        
        self.logger.info(f"总共加载 {len(all_data)} 条数据")
        return all_data


class HallucinationEvaluator:
    """
    幻觉评估器
    
    使用外部模型（如GPT）评估生成文本的幻觉程度
    """
    
    def __init__(self, 
                 vllm_url: str = "http://127.0.0.1:8001/v1",
                 api_key: str = "abcabc",
                 logger: Optional[logging.Logger] = None):
        self.vllm_url = vllm_url
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        
        # 检查vLLM服务是否可用
        self.logger.info(f"初始化HallucinationEvaluator，vLLM URL: {vllm_url}")
        try:
            import requests
            # 尝试几种常见的健康检查端点
            health_urls = [
                vllm_url.replace("/v1", "/health"),
                vllm_url.replace("/chat/completions", "/health"),
                f"{vllm_url}/health"
            ]
            
            health_check_passed = False
            for health_url in health_urls:
                try:
                    test_resp = requests.get(health_url, timeout=2)
                    if test_resp.status_code == 200:
                        self.logger.info(f"vLLM服务健康检查通过: {health_url}")
                        health_check_passed = True
                        break
                except:
                    continue
            
            if not health_check_passed:
                self.logger.warning("vLLM服务健康检查失败，评估时可能会出错")
                print("⚠️  警告: vLLM服务可能未启动，评估功能可能无法使用")
                
        except requests.exceptions.ConnectionError:
            self.logger.warning("vLLM服务连接测试失败，评估时可能会出错")
            print("⚠️  警告: vLLM服务可能未启动，评估功能可能无法使用")
        except Exception as e:
            self.logger.warning(f"vLLM健康检查失败: {e}")
    
    def call_vllm_api(self, prompt: str, max_tokens: int = 10000, 
                     temperature: float = 0.0, timeout: int = 60) -> str:
        """调用vLLM API获取响应"""
        payload = {
            "model": "/path/to/local_models/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Return STRICT JSON when asked."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            self.logger.info(f"调用vLLM API: {self.vllm_url}/chat/completions")
            resp = requests.post(
                f"{self.vllm_url}/chat/completions", 
                headers=headers, 
                json=payload, 
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            if data and "choices" in data and data["choices"] and "message" in data["choices"][0]:
                content = data["choices"][0]["message"].get("content", "")
                return (content or "").strip()
            else:
                raise Exception(f"响应格式不正确: {data}")
                
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"vLLM服务连接失败 - 请检查服务是否运行: {e}")
        except requests.exceptions.Timeout as e:
            raise Exception(f"vLLM调用超时（>{timeout}s）- 服务响应过慢: {e}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"vLLM调用HTTP错误 (状态码: {e.response.status_code}): {e}")
        except Exception as e:
            raise Exception(f"vLLM调用失败: {e}")
    
    def build_eval_prompt(self, question: str, model_answer: str, 
                         answer_true_list: List[str], context: List[str] = None) -> str:
        """构建评估prompt"""
        context_text = ""
        if context:
            context_text = f"Context: {' '.join(context)}\n"
        
        answers_text = "\n".join([f"- {ans}" for ans in answer_true_list])
        
        return f"""
You are a strict evaluator. Given a question, model answer, and reference correct answers, please judge whether the model answer is correct.

Please consider:
1) The model answer should match at least ONE of the reference correct answers in semantic meaning and factual accuracy;
2) Do not be overly lenient, avoid judging uncertain or off-topic answers as correct;
3) Consider synonyms and paraphrasing as valid matches.

Please return only strict JSON:
{{
  "is_correct": true/false,
  "matched_answer": "which reference answer was matched (if any)",
  "reason": "Brief explanation of the judgment",
  "score": 0.0-1.0  // confidence level
}}

{context_text}Question: {question}
Model Answer: {model_answer}

Reference Correct Answers:
{answers_text}
""".strip()
    
    def parse_json_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """尝试从响应中提取JSON"""
        if not raw:
            raise Exception("响应为空")
        
        try:
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                raise Exception(f"无法从响应中提取JSON: {raw[:200]}")
            return json.loads(m.group(0))
        except Exception as e:
            raise Exception(f"解析JSON失败: {e}, 原始响应: {raw[:400]}")
    
    def extract_match(self, model_answer: str, answer_true: List[str]) -> Dict[str, Any]:
        """简单的文本匹配作为备用评估方法"""
        if not model_answer or not answer_true:
            return {
                "is_correct": False,
                "matched_answer": "",
                "reason": "模型答案或正确答案为空"
            }
        
        model_answer_lower = model_answer.lower().strip()
        
        # 尝试精确匹配
        for true_answer in answer_true:
            if not true_answer:
                continue
            true_answer_lower = true_answer.lower().strip()
            
            # 精确匹配
            if model_answer_lower == true_answer_lower:
                return {
                    "is_correct": True,
                    "matched_answer": true_answer,
                    "reason": "精确匹配"
                }
            
            # 包含匹配（模型答案包含正确答案）
            if true_answer_lower in model_answer_lower:
                return {
                    "is_correct": True,
                    "matched_answer": true_answer,
                    "reason": "包含匹配"
                }
            
            # 反向包含匹配（正确答案包含模型答案）
            if model_answer_lower in true_answer_lower:
                return {
                    "is_correct": True,
                    "matched_answer": true_answer,
                    "reason": "反向包含匹配"
                }
        
        return {
            "is_correct": False,
            "matched_answer": "",
            "reason": "无匹配"
        }

    def evaluate_single(self, question: str, model_answer: str, 
                       answer_true: List[str], context: List[str] = None) -> Dict[str, Any]:
        """评估单个样本"""
        try:
            # 构建评估prompt
            prompt = self.build_eval_prompt(question, model_answer, answer_true, context)
            
            # 调用评估模型
            raw_response = self.call_vllm_api(prompt, max_tokens=10000, temperature=0.7)
            # 解析响应
            parsed = self.parse_json_response(raw_response)
            print(parsed)
            return {
                "question": question,
                "model_answer": model_answer,
                "answer_true": answer_true,
                "context": context,
                "is_correct": parsed.get("is_correct", False),
                "matched_answer": parsed.get("matched_answer", ""),
                "reason": parsed.get("reason", ""),
                "score": parsed.get("score", 0.0),
                "raw_response": raw_response
            }
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            print(f"评估异常: {e}")  # 按照用户要求打印异常
            
            # 使用备用匹配方法
            self.logger.info("使用备用匹配方法进行评估")
            match_result = self.extract_match(model_answer, answer_true)
            
            return {
                "is_correct": match_result["is_correct"],
                "matched_answer": match_result["matched_answer"],
                "reason": f"评估错误，使用备用匹配: {match_result['reason']}",
                "score": 1.0 if match_result["is_correct"] else 0.0,
                "raw_response": "",
                "error": str(e),
                "fallback_used": True
            }
    
    def evaluate_batch(self, test_data: List[Dict[str, Any]], 
                      baseline_results: List[Dict[str, Any]], 
                      intervention_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        批量评估干预效果
        
        Args:
            test_data: 原始测试数据（可选，用于后备）
            baseline_results: 基线结果
            intervention_results: 干预结果
            
        Returns:
            评估结果
        """
        # 从 baseline_results 和 intervention_results 中提取信息
        num_samples = min(len(baseline_results), len(intervention_results))
        self.logger.info(f"开始批量评估，样本数: {num_samples}")
        
        baseline_evaluations = []
        intervention_evaluations = []
        
        for i in tqdm(range(num_samples)):
            baseline_sample = baseline_results[i]
            intervention_sample = intervention_results[i]
            
            # 从结果中获取问题、上下文和正确答案
            question = baseline_sample.get("question", "")
            answer_true = baseline_sample.get("answer_true", [])
            context = baseline_sample.get("context", [])
            
            # 如果结果中没有这些字段，尝试从 test_data 获取（向后兼容）
            if not question or not answer_true:
                if test_data and i < len(test_data):
                    question = test_data[i].get("question", question)
                    answer_true = test_data[i].get("answer_true", answer_true)
                    context = test_data[i].get("context", context)
            
            if not question or not answer_true:
                self.logger.warning(f"样本 {i} 跳过：缺少question或answer_true")
                continue
            
            try:
                # 评估基线结果
                baseline_text = baseline_sample.get("baseline_text", "")
                if baseline_text:
                    baseline_eval = self.evaluate_single(question, baseline_text, answer_true, context)
                    baseline_evaluations.append(baseline_eval)
                
                # 评估干预结果
                intervention_text = intervention_sample.get("intervention_text", "")
                if intervention_text:
                    intervention_eval = self.evaluate_single(question, intervention_text, answer_true, context)
                    intervention_evaluations.append(intervention_eval)
                    
            except Exception as e:
                self.logger.error(f"评估样本 {i} 时出错: {e}")
                print(f"评估样本 {i} 时出错: {e}")
                # 继续处理下一个样本，不中断整个批次
                continue
        
        # 计算统计信息
        baseline_stats = self._compute_stats(baseline_evaluations)
        intervention_stats = self._compute_stats(intervention_evaluations)
        
        # 计算改善情况
        improvement = {
            "accuracy_change": intervention_stats["accuracy"] - baseline_stats["accuracy"],
            "score_change": intervention_stats["avg_score"] - baseline_stats["avg_score"],
            "relative_improvement": (intervention_stats["accuracy"] - baseline_stats["accuracy"]) / max(baseline_stats["accuracy"], 0.01) * 100
        }
        
        return {
            "baseline_stats": baseline_stats,
            "intervention_stats": intervention_stats,
            "improvement": improvement,
            "baseline_evaluations": baseline_evaluations,
            "intervention_evaluations": intervention_evaluations
        }
    
    def _compute_stats(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算评估统计信息"""
        if not evaluations:
            return {
                "total_samples": 0,
                "accuracy": 0.0,
                "avg_score": 0.0,
                "correct_count": 0,
                "error_count": 0
            }
        
        correct_count = sum(1 for eval_result in evaluations if eval_result.get("is_correct", False))
        error_count = sum(1 for eval_result in evaluations if "error" in eval_result)
        scores = [eval_result.get("score", 0.0) for eval_result in evaluations if "error" not in eval_result]
        
        return {
            "total_samples": len(evaluations),
            "accuracy": correct_count / len(evaluations) if evaluations else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "correct_count": correct_count,
            "error_count": error_count,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0
        }


def test_data_loader():
    """测试数据加载器"""
    # 创建测试数据
    test_data = [
        {
            "question": "What is the capital of France?",
            "context": "",
            "answer_true": "Paris"
        },
        {
            "question": "Who wrote 'Romeo and Juliet'?",
            "context": "",
            "answer_true": "William Shakespeare"
        }
    ]
    
    # 保存测试CSV
    test_file = "/tmp/test_hallucination.csv"
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "context", "answer_true"])
        writer.writeheader()
        writer.writerows(test_data)
    
    # 测试加载器
    loader = HallucinationDataLoader()
    loaded_data = loader.load_dataset(test_file)
    
    print("=== 数据加载器测试 ===")
    print(f"加载数据: {len(loaded_data)} 条")
    for i, item in enumerate(loaded_data):
        print(f"样本 {i+1}: {item}")
    
    # 测试评估器（需要vLLM服务）
    evaluator = HallucinationEvaluator()
    print("\n=== 评估器测试 ===")
    print("评估器初始化完成（需要vLLM服务才能实际评估）")


if __name__ == "__main__":
    test_data_loader()
