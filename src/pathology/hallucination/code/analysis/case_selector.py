#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Case Selector - 案例选择工具

基于分析结果和问答数据，智能选择最有价值的案例进行深度分析。
支持多种选择策略：异常强度、错误类型、语义相似性等。

作者: Jeryi
日期: 2025
"""

import json
import numpy as np
import argparse
import os
import sys
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_VLLM_URL = "http://0.0.0.0:8001/v1"
DEFAULT_API_KEY = "abcabc"

class CaseSelector:
    """案例选择器"""
    
    def __init__(self, correct_jsonl_path: str, incorrect_jsonl_path: str,
                 parcel_analysis_path: str, capability_analysis_path: str,
                 vllm_url: str = DEFAULT_VLLM_URL, api_key: str = DEFAULT_API_KEY):
        """
        初始化案例选择器
        
        Args:
            correct_jsonl_path: 正确样本数据路径
            incorrect_jsonl_path: 幻觉样本数据路径
            parcel_analysis_path: Parcel分析结果路径
            capability_analysis_path: Capability分析结果路径
            vllm_url: vLLM API地址
            api_key: API密钥
        """
        self.correct_jsonl_path = correct_jsonl_path
        self.incorrect_jsonl_path = incorrect_jsonl_path
        self.parcel_analysis_path = parcel_analysis_path
        self.capability_analysis_path = capability_analysis_path
        self.vllm_url = vllm_url
        self.api_key = api_key
        
        # 数据存储
        self.correct_data = []
        self.incorrect_data = []
        self.parcel_analysis = {}
        self.capability_analysis = {}
        
        # 错误类型分类器
        self.error_types = {
            'factual_error': ['fact', 'truth', 'real', 'actual', 'correct', 'wrong', 'false', 'true'],
            'logical_error': ['logic', 'reason', 'because', 'therefore', 'thus', 'hence', 'conclusion'],
            'causal_error': ['cause', 'effect', 'why', 'how', 'due to', 'result', 'consequence'],
            'definition_error': ['what is', 'define', 'definition', 'meaning', 'means'],
            'comparison_error': ['compare', 'difference', 'vs', 'versus', 'better', 'worse', 'more', 'less'],
            'temporal_error': ['when', 'time', 'before', 'after', 'during', 'while', 'since'],
            'spatial_error': ['where', 'location', 'place', 'position', 'here', 'there'],
            'quantitative_error': ['how many', 'how much', 'number', 'count', 'percentage', 'amount']
        }
    
    def load_data(self) -> None:
        """加载所有数据"""
        logger.info("加载数据...")
        
        # 加载问答数据
        self._load_jsonl_data(self.correct_jsonl_path, self.correct_data)
        self._load_jsonl_data(self.incorrect_jsonl_path, self.incorrect_data)
        
        # 加载分析结果
        self._load_analysis_results()
        
        logger.info(f"数据加载完成: {len(self.correct_data)} 正确样本, {len(self.incorrect_data)} 幻觉样本")
    
    def call_vllm_api(self, prompt: str, max_tokens: int = 2048, 
                     temperature: float = 0.0, timeout: int = 120) -> str:
        """调用vLLM API获取响应"""
        payload = {
            "model": "/path/to/local_models/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "你是一个专业的神经科学研究助手，擅长分析大语言模型的内部神经活动模式。请用科学、严谨的语言进行分析。"},
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
            resp = requests.post(f"{self.vllm_url}/chat/completions", 
                               headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            if data and "choices" in data and data["choices"] and "message" in data["choices"][0]:
                content = data["choices"][0]["message"].get("content", "")
                return (content or "").strip()
            else:
                raise Exception(f"响应格式不正确: {data}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"vLLM API调用失败: {e}")
        except Exception as e:
            raise Exception(f"vLLM调用失败: {e}")
    
    def _load_jsonl_data(self, file_path: str, data_list: List) -> None:
        """加载JSONL数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data_list.append(json.loads(line.strip()))
        except Exception as e:
            logger.error(f"加载数据失败 {file_path}: {e}")
            raise
    
    def _load_analysis_results(self) -> None:
        """加载分析结果"""
        # 加载Parcel分析结果
        try:
            with open(self.parcel_analysis_path, 'r', encoding='utf-8') as f:
                self.parcel_analysis = json.load(f)
        except Exception as e:
            logger.warning(f"加载Parcel分析结果失败: {e}")
            self.parcel_analysis = {}
        
        # 加载Capability分析结果
        try:
            with open(self.capability_analysis_path, 'r', encoding='utf-8') as f:
                self.capability_analysis = json.load(f)
        except Exception as e:
            logger.warning(f"加载Capability分析结果失败: {e}")
            self.capability_analysis = {}
    
    def classify_error_type(self, question: str, model_answer: str, reason: str) -> str:
        """分类错误类型"""
        text = f"{question} {model_answer} {reason}".lower()
        
        # 计算每种错误类型的匹配分数
        type_scores = {}
        for error_type, keywords in self.error_types.items():
            score = sum(1 for keyword in keywords if keyword in text)
            type_scores[error_type] = score
        
        # 返回得分最高的错误类型
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return 'unknown'
    
    def calculate_anomaly_scores(self) -> Dict[int, float]:
        """计算每个样本的异常分数"""
        anomaly_scores = {}
        
        # 从Parcel分析结果中提取异常分数
        if 'top_anomalous_parcels' in self.parcel_analysis:
            parcel_anomalies = self.parcel_analysis['top_anomalous_parcels']
            for item in parcel_anomalies:
                sample_id = item.get('parcel_id', 0)  # 这里需要根据实际数据结构调整
                score = item.get('anomaly_score', 0)
                anomaly_scores[sample_id] = anomaly_scores.get(sample_id, 0) + score
        
        # 从Capability分析结果中提取异常分数
        if 'top_anomalous_capabilities' in self.capability_analysis:
            cap_anomalies = self.capability_analysis['top_anomalous_capabilities']
            for item in cap_anomalies:
                sample_id = item.get('capability_id', 0)  # 这里需要根据实际数据结构调整
                score = item.get('anomaly_score', 0)
                anomaly_scores[sample_id] = anomaly_scores.get(sample_id, 0) + score
        
        return anomaly_scores
    
    def find_high_confidence_hallucinations(self, num_cases: int = 10) -> List[Dict]:
        """寻找高置信度的幻觉案例"""
        logger.info(f"寻找 {num_cases} 个高置信度幻觉案例...")
        
        # 按置信度分数排序
        sorted_incorrect = sorted(self.incorrect_data, key=lambda x: x.get('score', 0), reverse=True)
        
        cases = []
        for i, case in enumerate(sorted_incorrect[:num_cases]):
            error_type = self.classify_error_type(
                case.get('question', ''),
                case.get('model_answer', ''),
                case.get('reason', '')
            )
            
            # 计算答案长度和复杂度，用于细粒度分析
            answer_length = len(case.get('model_answer', '').split())
            question_length = len(case.get('question', '').split())
            complexity_score = answer_length + question_length
            
            cases.append({
                'sample_id': case['index'],
                'question': case['question'],
                'model_answer': case['model_answer'],
                'score': case.get('score', 0),
                'reason': case.get('reason', ''),
                'error_type': error_type,
                'priority': 'high_confidence_hallucination',
                'is_correct': False,
                'answer_length': answer_length,
                'question_length': question_length,
                'complexity_score': complexity_score
            })
        
        return cases
    
    def find_diverse_error_types(self, num_per_type: int = 2) -> List[Dict]:
        """寻找不同错误类型的案例"""
        logger.info(f"寻找每种错误类型 {num_per_type} 个案例...")
        
        # 按错误类型分组
        error_groups = defaultdict(list)
        
        for case in self.incorrect_data:
            error_type = self.classify_error_type(
                case.get('question', ''),
                case.get('model_answer', ''),
                case.get('reason', '')
            )
            error_groups[error_type].append(case)
        
        cases = []
        for error_type, group_cases in error_groups.items():
            # 按置信度排序
            sorted_cases = sorted(group_cases, key=lambda x: x.get('score', 0), reverse=True)
            
            for case in sorted_cases[:num_per_type]:
                cases.append({
                    'sample_id': case['index'],
                    'question': case['question'],
                    'model_answer': case['model_answer'],
                    'score': case.get('score', 0),
                    'reason': case.get('reason', ''),
                    'error_type': error_type,
                    'priority': 'diverse_error_type',
                    'is_correct': False
                })
        
        return cases
    
    def find_token_level_analysis_cases(self, num_cases: int = 10) -> List[Dict]:
        """寻找适合Token级别细粒度分析的案例"""
        logger.info(f"寻找 {num_cases} 个适合Token级别分析的案例...")
        
        # 筛选条件：答案长度适中，问题复杂度适中
        suitable_cases = []
        
        for case in self.incorrect_data:
            answer_length = len(case.get('model_answer', '').split())
            question_length = len(case.get('question', '').split())
            
            # 筛选条件：答案长度在10-100个token之间，问题长度在5-50个token之间
            if (10 <= answer_length <= 100 and 
                5 <= question_length <= 50 and 
                case.get('score', 0) > 0.5):  # 置信度适中
                
                error_type = self.classify_error_type(
                    case.get('question', ''),
                    case.get('model_answer', ''),
                    case.get('reason', '')
                )
                
                # 计算细粒度分析适合度分数
                token_diversity = len(set(case.get('model_answer', '').split()))  # 词汇多样性
                complexity_score = answer_length * question_length / 100  # 复杂度分数
                analysis_suitability = token_diversity + complexity_score + case.get('score', 0) * 10
                
                suitable_cases.append({
                    'sample_id': case['index'],
                    'question': case['question'],
                    'model_answer': case['model_answer'],
                    'score': case.get('score', 0),
                    'reason': case.get('reason', ''),
                    'error_type': error_type,
                    'priority': 'token_level_analysis',
                    'is_correct': False,
                    'answer_length': answer_length,
                    'question_length': question_length,
                    'token_diversity': token_diversity,
                    'complexity_score': complexity_score,
                    'analysis_suitability': analysis_suitability
                })
        
        # 按分析适合度排序
        suitable_cases.sort(key=lambda x: x['analysis_suitability'], reverse=True)
        
        return suitable_cases[:num_cases]
    
    def find_contrastive_token_analysis_pairs(self, num_pairs: int = 5) -> List[Dict]:
        """寻找适合Token级别对比分析的配对案例"""
        logger.info(f"寻找 {num_pairs} 对适合Token级别对比分析的案例...")
        
        # 获取适合细粒度分析的幻觉案例
        hallucination_cases = self.find_token_level_analysis_cases(num_pairs * 2)
        
        # 获取适合细粒度分析的正确案例
        correct_cases = []
        for case in self.correct_data:
            answer_length = len(case.get('model_answer', '').split())
            question_length = len(case.get('question', '').split())
            
            if (10 <= answer_length <= 100 and 
                5 <= question_length <= 50 and 
                case.get('score', 0) > 0.5):
                
                correct_cases.append({
                    'sample_id': case['index'],
                    'question': case['question'],
                    'model_answer': case['model_answer'],
                    'score': case.get('score', 0),
                    'answer_length': answer_length,
                    'question_length': question_length,
                    'is_correct': True
                })
        
        pairs = []
        used_correct_indices = set()
        
        for hall_case in hallucination_cases[:num_pairs * 2]:
            best_pair = None
            best_similarity = 0
            
            # 寻找最相似的正确案例
            for i, correct_case in enumerate(correct_cases):
                if i in used_correct_indices:
                    continue
                
                # 计算语义相似性（简单的词汇重叠）
                hall_words = set(hall_case['question'].lower().split())
                correct_words = set(correct_case['question'].lower().split())
                similarity = len(hall_words.intersection(correct_words)) / len(hall_words.union(correct_words))
                
                # 长度相似性
                length_similarity = 1 - abs(hall_case['answer_length'] - correct_case['answer_length']) / max(hall_case['answer_length'], correct_case['answer_length'])
                
                # 综合相似性
                combined_similarity = similarity * 0.7 + length_similarity * 0.3
                
                if combined_similarity > best_similarity and combined_similarity > 0.2:
                    best_similarity = combined_similarity
                    best_pair = correct_case
                    best_pair_index = i
            
            if best_pair and best_similarity > 0.2:
                # 创建配对案例
                pair_case = hall_case.copy()
                pair_case['pair_sample_id'] = best_pair['sample_id']
                pair_case['pair_question'] = best_pair['question']
                pair_case['pair_answer'] = best_pair['model_answer']
                pair_case['pair_score'] = best_pair['score']
                pair_case['similarity_score'] = best_similarity
                pair_case['priority'] = 'contrastive_token_analysis'
                
                pairs.append(pair_case)
                used_correct_indices.add(best_pair_index)
                
                if len(pairs) >= num_pairs:
                    break
        
        return pairs
    
    def find_semantic_pairs(self, num_pairs: int = 5) -> List[Dict]:
        """寻找语义相似的配对案例"""
        logger.info(f"寻找 {num_pairs} 对语义相似案例...")
        
        # 提取所有问题文本
        all_questions = []
        all_cases = []
        
        for case in self.correct_data + self.incorrect_data:
            all_questions.append(case['question'])
            all_cases.append(case)
        
        # 使用TF-IDF计算语义相似性
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        question_vectors = vectorizer.fit_transform(all_questions)
        similarity_matrix = cosine_similarity(question_vectors)
        
        pairs = []
        used_indices = set()
        
        # 寻找最相似的配对
        for i in range(len(all_cases)):
            if i in used_indices:
                continue
            
            # 找到最相似的案例
            similarities = similarity_matrix[i]
            similarities[i] = -1  # 排除自己
            
            best_match_idx = np.argmax(similarities)
            similarity_score = similarities[best_match_idx]
            
            # 如果相似度足够高且类型不同
            if (similarity_score > 0.3 and 
                best_match_idx not in used_indices and
                all_cases[i]['is_correct'] != all_cases[best_match_idx]['is_correct']):
                
                case1 = all_cases[i]
                case2 = all_cases[best_match_idx]
                
                pairs.append({
                    'sample_id': case1['index'],
                    'question': case1['question'],
                    'model_answer': case1['model_answer'],
                    'score': case1.get('score', 0),
                    'reason': case1.get('reason', ''),
                    'error_type': self.classify_error_type(
                        case1['question'], case1['model_answer'], case1.get('reason', '')
                    ),
                    'priority': 'semantic_pair',
                    'is_correct': case1['is_correct'],
                    'pair_sample_id': case2['index'],
                    'pair_question': case2['question'],
                    'pair_answer': case2['model_answer'],
                    'similarity_score': similarity_score
                })
                
                used_indices.add(i)
                used_indices.add(best_match_idx)
                
                if len(pairs) >= num_pairs:
                    break
        
        return pairs
    
    def find_extreme_cases(self, num_cases: int = 5) -> List[Dict]:
        """寻找极端案例（最长/最短答案，最高/最低置信度等）"""
        logger.info(f"寻找 {num_cases} 个极端案例...")
        
        cases = []
        
        # 最长答案的幻觉案例
        longest_incorrect = max(self.incorrect_data, 
                              key=lambda x: len(x.get('model_answer', '')))
        cases.append({
            'sample_id': longest_incorrect['index'],
            'question': longest_incorrect['question'],
            'model_answer': longest_incorrect['model_answer'],
            'score': longest_incorrect.get('score', 0),
            'reason': longest_incorrect.get('reason', ''),
            'error_type': 'longest_answer',
            'priority': 'extreme_case',
            'is_correct': False,
            'extreme_type': 'longest_answer'
        })
        
        # 最短答案的幻觉案例
        shortest_incorrect = min(self.incorrect_data, 
                               key=lambda x: len(x.get('model_answer', '')))
        cases.append({
            'sample_id': shortest_incorrect['index'],
            'question': shortest_incorrect['question'],
            'model_answer': shortest_incorrect['model_answer'],
            'score': shortest_incorrect.get('score', 0),
            'reason': shortest_incorrect.get('reason', ''),
            'error_type': 'shortest_answer',
            'priority': 'extreme_case',
            'is_correct': False,
            'extreme_type': 'shortest_answer'
        })
        
        # 最高置信度的正确案例
        highest_correct = max(self.correct_data, 
                            key=lambda x: x.get('score', 0))
        cases.append({
            'sample_id': highest_correct['index'],
            'question': highest_correct['question'],
            'model_answer': highest_correct['model_answer'],
            'score': highest_correct.get('score', 0),
            'reason': highest_correct.get('reason', ''),
            'error_type': 'high_confidence_correct',
            'priority': 'extreme_case',
            'is_correct': True,
            'extreme_type': 'high_confidence_correct'
        })
        
        # 最低置信度的正确案例
        lowest_correct = min(self.correct_data, 
                           key=lambda x: x.get('score', 0))
        cases.append({
            'sample_id': lowest_correct['index'],
            'question': lowest_correct['question'],
            'model_answer': lowest_correct['model_answer'],
            'score': lowest_correct.get('score', 0),
            'reason': lowest_correct.get('reason', ''),
            'error_type': 'low_confidence_correct',
            'priority': 'extreme_case',
            'is_correct': True,
            'extreme_type': 'low_confidence_correct'
        })
        
        return cases[:num_cases]
    
    def llm_evaluate_case_insight(self, case: Dict, analysis_context: str = "") -> Dict:
        """使用LLM评估案例的洞察价值"""
        logger.info(f"使用LLM评估案例 {case['sample_id']} 的洞察价值...")
        
        # 构建评估提示
        prompt = f"""
作为神经科学研究专家，请评估以下幻觉案例的科学研究价值，特别是对理解大语言模型内部机制和幻觉产生原因的洞察价值。

案例信息：
- 样本ID: {case['sample_id']}
- 问题: {case['question']}
- 模型答案: {case['model_answer']}
- 置信度分数: {case['score']}
- 错误原因: {case.get('reason', 'N/A')}
- 错误类型: {case.get('error_type', 'unknown')}

分析上下文：
{analysis_context}

请从以下维度评估该案例的洞察价值（1-10分，10分最高）：

1. **科学新颖性** (Novelty): 该案例是否揭示了新的幻觉模式或机制？
2. **机制清晰性** (Mechanism Clarity): 该案例是否清晰地展示了幻觉产生的内部过程？
3. **可解释性** (Interpretability): 该案例是否有助于理解模型的认知架构？
4. **对比价值** (Comparative Value): 该案例是否提供了与正确回答的鲜明对比？
5. **理论意义** (Theoretical Significance): 该案例是否对认知科学或AI理论有重要贡献？
6. **实验价值** (Experimental Value): 该案例是否适合进行深入的神经活动分析？
7. **发表潜力** (Publication Potential): 该案例是否具有Nature/Science级别期刊的发表价值？

请按以下JSON格式返回评估结果：
{{
    "overall_score": 8.5,
    "novelty": 8,
    "mechanism_clarity": 9,
    "interpretability": 8,
    "comparative_value": 9,
    "theoretical_significance": 8,
    "experimental_value": 9,
    "publication_potential": 8,
    "insight_summary": "该案例展示了模型在事实检索失败时的典型激活模式...",
    "research_questions": ["模型为何在特定领域出现系统性错误？", "哪些神经模块参与了错误信息的生成？"],
    "recommendation": "强烈推荐进行深度分析"
}}

请确保返回有效的JSON格式。
"""
        
        try:
            response = self.call_vllm_api(prompt, max_tokens=1024, temperature=0.1)
            
            # 尝试解析JSON响应
            try:
                # 提取JSON部分
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    evaluation = json.loads(json_str)
                    
                    # 添加案例基本信息
                    evaluation['case_id'] = case['sample_id']
                    evaluation['question'] = case['question']
                    evaluation['model_answer'] = case['model_answer']
                    
                    return evaluation
                else:
                    raise ValueError("未找到有效的JSON响应")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"LLM响应解析失败: {e}")
                # 返回默认评估
                return {
                    'case_id': case['sample_id'],
                    'overall_score': 5.0,
                    'novelty': 5,
                    'mechanism_clarity': 5,
                    'interpretability': 5,
                    'comparative_value': 5,
                    'theoretical_significance': 5,
                    'experimental_value': 5,
                    'publication_potential': 5,
                    'insight_summary': 'LLM评估失败，使用默认评分',
                    'research_questions': [],
                    'recommendation': '需要人工评估',
                    'question': case['question'],
                    'model_answer': case['model_answer']
                }
                
        except Exception as e:
            logger.error(f"LLM评估失败: {e}")
            # 返回默认评估
            return {
                'case_id': case['sample_id'],
                'overall_score': 5.0,
                'novelty': 5,
                'mechanism_clarity': 5,
                'interpretability': 5,
                'comparative_value': 5,
                'theoretical_significance': 5,
                'experimental_value': 5,
                'publication_potential': 5,
                'insight_summary': 'LLM评估失败，使用默认评分',
                'research_questions': [],
                'recommendation': '需要人工评估',
                'question': case['question'],
                'model_answer': case['model_answer']
            }
    
    def llm_select_high_insight_cases(self, num_cases: int = 10, 
                                    min_score: float = 7.0) -> List[Dict]:
        """使用LLM选择高洞察价值的案例"""
        logger.info(f"使用LLM选择 {num_cases} 个高洞察价值案例...")
        
        # 准备分析上下文
        analysis_context = self._prepare_analysis_context()
        
        # 获取候选案例（高置信度幻觉案例）
        candidate_cases = self.find_high_confidence_hallucinations(num_cases * 3)  # 获取更多候选案例
        
        logger.info(f"评估 {len(candidate_cases)} 个候选案例...")
        
        # 使用LLM评估每个案例
        evaluated_cases = []
        for i, case in enumerate(candidate_cases):
            logger.info(f"评估案例 {i+1}/{len(candidate_cases)}: {case['sample_id']}")
            evaluation = self.llm_evaluate_case_insight(case, analysis_context)
            evaluated_cases.append(evaluation)
        
        # 按综合分数排序
        evaluated_cases.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # 筛选高分案例
        high_insight_cases = [case for case in evaluated_cases if case['overall_score'] >= min_score]
        
        # 如果高分案例不够，降低阈值
        if len(high_insight_cases) < num_cases:
            logger.warning(f"高分案例不足，降低阈值到 {min_score - 1.0}")
            high_insight_cases = [case for case in evaluated_cases if case['overall_score'] >= min_score - 1.0]
        
        # 返回前N个案例
        selected_cases = high_insight_cases[:num_cases]
        
        logger.info(f"选择了 {len(selected_cases)} 个高洞察价值案例")
        for case in selected_cases:
            logger.info(f"案例 {case['case_id']}: 综合分数 {case['overall_score']:.2f} - {case['insight_summary'][:100]}...")
        
        return selected_cases
    
    def llm_find_contrastive_pairs(self, num_pairs: int = 5) -> List[Dict]:
        """使用LLM寻找具有对比价值的配对案例"""
        logger.info(f"使用LLM寻找 {num_pairs} 对对比案例...")
        
        # 获取候选案例
        hallucination_cases = self.find_high_confidence_hallucinations(num_pairs * 2)
        correct_cases = sorted(self.correct_data, key=lambda x: x.get('score', 0), reverse=True)[:num_pairs * 2]
        
        pairs = []
        
        for hall_case in hallucination_cases[:num_pairs * 2]:
            # 为每个幻觉案例寻找最佳配对
            best_pair = None
            best_score = 0
            
            for correct_case in correct_cases:
                # 构建配对评估提示
                prompt = f"""
作为神经科学研究专家，请评估以下配对案例的对比研究价值。

幻觉案例：
- 样本ID: {hall_case['sample_id']}
- 问题: {hall_case['question']}
- 模型答案: {hall_case['model_answer']}
- 置信度: {hall_case['score']}

正确案例：
- 样本ID: {correct_case['index']}
- 问题: {correct_case['question']}
- 模型答案: {correct_case['model_answer']}
- 置信度: {correct_case['score']}

请评估这对案例的对比价值（1-10分）：
1. **语义相似性**: 两个问题是否在语义上相似？
2. **认知对比性**: 两个答案是否展示了不同的认知过程？
3. **机制对比性**: 是否有助于理解正确与错误推理的差异？
4. **实验价值**: 是否适合进行神经活动的对比分析？

请按以下JSON格式返回评估：
{{
    "overall_score": 8.5,
    "semantic_similarity": 8,
    "cognitive_contrast": 9,
    "mechanism_contrast": 8,
    "experimental_value": 9,
    "contrast_summary": "这对案例展示了模型在处理相似问题时正确与错误推理的差异...",
    "research_value": "高"
}}

请确保返回有效的JSON格式。
"""
                
                try:
                    response = self.call_vllm_api(prompt, max_tokens=512, temperature=0.1)
                    
                    # 解析JSON响应
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        evaluation = json.loads(json_str)
                        
                        if evaluation['overall_score'] > best_score:
                            best_score = evaluation['overall_score']
                            best_pair = {
                                'hallucination_case': hall_case,
                                'correct_case': correct_case,
                                'evaluation': evaluation
                            }
                            
                except Exception as e:
                    logger.warning(f"配对评估失败: {e}")
                    continue
            
            if best_pair and best_score >= 6.0:  # 只选择评分较高的配对
                pairs.append(best_pair)
                logger.info(f"找到配对: 幻觉案例 {hall_case['sample_id']} + 正确案例 {best_pair['correct_case']['index']} (分数: {best_score:.2f})")
        
        # 按评分排序并返回前N对
        pairs.sort(key=lambda x: x['evaluation']['overall_score'], reverse=True)
        return pairs[:num_pairs]
    
    def _prepare_analysis_context(self) -> str:
        """准备分析上下文信息"""
        context = "分析上下文信息：\n\n"
        
        # 添加Parcel分析结果摘要
        if 'top_anomalous_parcels' in self.parcel_analysis:
            context += "Parcel级别异常摘要：\n"
            for i, parcel in enumerate(self.parcel_analysis['top_anomalous_parcels'][:5]):
                context += f"- Parcel {parcel.get('parcel_id', 'N/A')}: 激活差异={parcel.get('activation_diff', 0):.4f}\n"
        
        # 添加Capability分析结果摘要
        if 'top_anomalous_capabilities' in self.capability_analysis:
            context += "\nCapability级别异常摘要：\n"
            for i, cap in enumerate(self.capability_analysis['top_anomalous_capabilities'][:5]):
                context += f"- {cap.get('capability_name', 'Unknown')}: 激活差异={cap.get('activation_diff', 0):.4f}\n"
        
        # 添加数据统计信息
        context += f"\n数据统计：\n"
        context += f"- 正确样本数量: {len(self.correct_data)}\n"
        context += f"- 幻觉样本数量: {len(self.incorrect_data)}\n"
        
        return context
    
    def find_clustered_cases(self, num_clusters: int = 3, cases_per_cluster: int = 2) -> List[Dict]:
        """基于聚类寻找代表性案例"""
        logger.info(f"基于聚类寻找 {num_clusters} 个簇的代表性案例...")
        
        # 提取特征
        features = []
        case_indices = []
        
        for case in self.incorrect_data:
            # 特征：问题长度、答案长度、置信度分数
            question_len = len(case.get('question', ''))
            answer_len = len(case.get('model_answer', ''))
            score = case.get('score', 0)
            
            features.append([question_len, answer_len, score])
            case_indices.append(case['index'])
        
        if len(features) < num_clusters:
            logger.warning("样本数量不足以进行聚类")
            return []
        
        # 标准化特征
        features = np.array(features)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        # 为每个簇选择代表性案例
        cases = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_cases = [self.incorrect_data[i] for i in cluster_indices]
            
            # 选择距离簇中心最近的案例
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(features[cluster_indices] - cluster_center, axis=1)
            closest_indices = np.argsort(distances)[:cases_per_cluster]
            
            for idx in closest_indices:
                case = cluster_cases[idx]
                cases.append({
                    'sample_id': case['index'],
                    'question': case['question'],
                    'model_answer': case['model_answer'],
                    'score': case.get('score', 0),
                    'reason': case.get('reason', ''),
                    'error_type': f'cluster_{cluster_id}',
                    'priority': 'clustered_case',
                    'is_correct': False,
                    'cluster_id': cluster_id
                })
        
        return cases
    
    def select_comprehensive_cases(self, total_cases: int = 20, use_llm: bool = True, 
                                 focus_token_analysis: bool = False) -> List[Dict]:
        """综合选择案例（集成LLM智能选择）"""
        logger.info(f"综合选择 {total_cases} 个案例（LLM增强: {use_llm}, 专注Token分析: {focus_token_analysis}）...")
        
        all_cases = []
        
        if focus_token_analysis:
            # 专注Token级别细粒度分析的策略
            # Token级别分析案例 (50%)
            token_cases = self.find_token_level_analysis_cases(int(total_cases * 0.5))
            all_cases.extend(token_cases)
            
            # Token级别对比分析案例 (30%)
            token_pairs = self.find_contrastive_token_analysis_pairs(int(total_cases * 0.3))
            all_cases.extend(token_pairs)
            
            # 高置信度案例补充 (20%)
            high_conf_cases = self.find_high_confidence_hallucinations(int(total_cases * 0.2))
            all_cases.extend(high_conf_cases)
            
        elif use_llm:
            # LLM增强选择策略
            try:
                # LLM高洞察价值案例 (40%)
                llm_high_insight = self.llm_select_high_insight_cases(int(total_cases * 0.4))
                all_cases.extend(llm_high_insight)
                
                # LLM对比配对案例 (30%)
                llm_pairs = self.llm_find_contrastive_pairs(int(total_cases * 0.3))
                for pair in llm_pairs:
                    # 添加幻觉案例
                    hall_case = pair['hallucination_case'].copy()
                    hall_case['pair_sample_id'] = pair['correct_case']['index']
                    hall_case['pair_question'] = pair['correct_case']['question']
                    hall_case['pair_answer'] = pair['correct_case']['model_answer']
                    hall_case['priority'] = 'llm_contrastive_pair'
                    hall_case['llm_evaluation'] = pair['evaluation']
                    all_cases.append(hall_case)
                
                # 传统方法补充 (30%)
                diverse_cases = self.find_diverse_error_types(int(total_cases * 0.15 / 8))
                all_cases.extend(diverse_cases)
                
                extreme_cases = self.find_extreme_cases(int(total_cases * 0.15))
                all_cases.extend(extreme_cases)
                
            except Exception as e:
                logger.error(f"LLM选择失败，回退到传统方法: {e}")
                use_llm = False
        
        if not use_llm and not focus_token_analysis:
            # 传统选择策略
            # 高置信度幻觉案例 (30%)
            high_conf_cases = self.find_high_confidence_hallucinations(int(total_cases * 0.3))
            all_cases.extend(high_conf_cases)
            
            # 不同错误类型案例 (25%)
            diverse_cases = self.find_diverse_error_types(int(total_cases * 0.25 / 8))
            all_cases.extend(diverse_cases)
            
            # 语义配对案例 (20%)
            pair_cases = self.find_semantic_pairs(int(total_cases * 0.2))
            all_cases.extend(pair_cases)
            
            # 极端案例 (15%)
            extreme_cases = self.find_extreme_cases(int(total_cases * 0.15))
            all_cases.extend(extreme_cases)
            
            # 聚类案例 (10%)
            cluster_cases = self.find_clustered_cases(int(total_cases * 0.1 / 2), 2)
            all_cases.extend(cluster_cases)
        
        # 去重并限制数量
        seen_ids = set()
        unique_cases = []
        for case in all_cases:
            case_id = case.get('sample_id') or case.get('case_id')
            if case_id not in seen_ids:
                unique_cases.append(case)
                seen_ids.add(case_id)
                if len(unique_cases) >= total_cases:
                    break
        
        return unique_cases
    
    def save_selected_cases(self, cases: List[Dict], output_path: str) -> None:
        """保存选中的案例"""
        logger.info(f"保存 {len(cases)} 个选中案例到 {output_path}")
        
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        
        # 生成Markdown报告
        self._generate_case_report(cases, output_path.replace('.json', '.md'))
        
        logger.info("案例选择完成！")
    
    def _generate_case_report(self, cases: List[Dict], output_path: str) -> None:
        """生成案例报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 选中案例报告（LLM增强版）\n\n")
            f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**总案例数**: {len(cases)}\n\n")
            
            # 统计LLM评估的案例
            llm_evaluated = [case for case in cases if 'overall_score' in case]
            f.write(f"**LLM评估案例数**: {len(llm_evaluated)}\n")
            if llm_evaluated:
                avg_score = sum(case['overall_score'] for case in llm_evaluated) / len(llm_evaluated)
                f.write(f"**平均洞察分数**: {avg_score:.2f}\n")
            f.write("\n")
            
            # 按优先级分组
            priority_groups = defaultdict(list)
            for case in cases:
                priority = case.get('priority', 'unknown')
                priority_groups[priority].append(case)
            
            for priority, group_cases in priority_groups.items():
                f.write(f"## {priority.replace('_', ' ').title()}\n\n")
                f.write(f"**案例数量**: {len(group_cases)}\n\n")
                
                for i, case in enumerate(group_cases, 1):
                    case_id = case.get('sample_id') or case.get('case_id')
                    f.write(f"### 案例 {i}: 样本 {case_id}\n\n")
                    f.write(f"- **问题**: {case['question']}\n")
                    f.write(f"- **模型答案**: {case['model_answer'][:200]}{'...' if len(case['model_answer']) > 200 else ''}\n")
                    f.write(f"- **置信度分数**: {case.get('score', 'N/A')}\n")
                    f.write(f"- **错误类型**: {case.get('error_type', 'unknown')}\n")
                    f.write(f"- **正确性**: {'正确' if case.get('is_correct', False) else '幻觉'}\n")
                    
                    # LLM评估信息
                    if 'overall_score' in case:
                        f.write(f"- **LLM洞察分数**: {case['overall_score']:.2f}/10\n")
                        f.write(f"  - 新颖性: {case.get('novelty', 'N/A')}/10\n")
                        f.write(f"  - 机制清晰性: {case.get('mechanism_clarity', 'N/A')}/10\n")
                        f.write(f"  - 可解释性: {case.get('interpretability', 'N/A')}/10\n")
                        f.write(f"  - 对比价值: {case.get('comparative_value', 'N/A')}/10\n")
                        f.write(f"  - 理论意义: {case.get('theoretical_significance', 'N/A')}/10\n")
                        f.write(f"  - 实验价值: {case.get('experimental_value', 'N/A')}/10\n")
                        f.write(f"  - 发表潜力: {case.get('publication_potential', 'N/A')}/10\n")
                        f.write(f"- **洞察摘要**: {case.get('insight_summary', 'N/A')}\n")
                        f.write(f"- **研究问题**: {', '.join(case.get('research_questions', []))}\n")
                        f.write(f"- **推荐等级**: {case.get('recommendation', 'N/A')}\n")
                    
                    if 'pair_sample_id' in case:
                        f.write(f"- **配对样本**: {case['pair_sample_id']}\n")
                        f.write(f"- **配对问题**: {case['pair_question']}\n")
                    
                    f.write(f"- **选择原因**: {case.get('reason', 'N/A')[:100]}{'...' if len(case.get('reason', '')) > 100 else ''}\n\n")
                
                f.write("---\n\n")
            
            # 添加LLM评估总结
            if llm_evaluated:
                f.write("## LLM评估总结\n\n")
                f.write("### 高分案例洞察\n")
                high_score_cases = [case for case in llm_evaluated if case['overall_score'] >= 8.0]
                for case in high_score_cases[:5]:
                    f.write(f"- **案例 {case.get('case_id', 'N/A')}**: {case.get('insight_summary', 'N/A')[:150]}...\n")
                
                f.write("\n### 研究问题汇总\n")
                all_questions = []
                for case in llm_evaluated:
                    all_questions.extend(case.get('research_questions', []))
                unique_questions = list(set(all_questions))
                for question in unique_questions[:10]:
                    f.write(f"- {question}\n")
                
                f.write("\n### 推荐等级分布\n")
                recommendations = defaultdict(int)
                for case in llm_evaluated:
                    rec = case.get('recommendation', '未知')
                    recommendations[rec] += 1
                for rec, count in recommendations.items():
                    f.write(f"- {rec}: {count} 个案例\n")
    
    def run_case_selection(self, strategy: str = 'comprehensive', num_cases: int = 20, 
                          output_path: str = 'selected_cases.json', use_llm: bool = True) -> List[Dict]:
        """运行案例选择"""
        logger.info(f"开始案例选择: 策略={strategy}, 数量={num_cases}, LLM增强={use_llm}")
        
        # 加载数据
        self.load_data()
        
        # 根据策略选择案例
        if strategy == 'high_confidence':
            cases = self.find_high_confidence_hallucinations(num_cases)
        elif strategy == 'diverse':
            cases = self.find_diverse_error_types(num_cases // 8)  # 假设8种错误类型
        elif strategy == 'pairs':
            cases = self.find_semantic_pairs(num_cases)
        elif strategy == 'extreme':
            cases = self.find_extreme_cases(num_cases)
        elif strategy == 'clustered':
            cases = self.find_clustered_cases(num_cases // 2, 2)
        elif strategy == 'llm_high_insight':
            cases = self.llm_select_high_insight_cases(num_cases)
        elif strategy == 'llm_contrastive':
            pairs = self.llm_find_contrastive_pairs(num_cases)
            cases = []
            for pair in pairs:
                hall_case = pair['hallucination_case'].copy()
                hall_case['pair_sample_id'] = pair['correct_case']['index']
                hall_case['pair_question'] = pair['correct_case']['question']
                hall_case['pair_answer'] = pair['correct_case']['model_answer']
                hall_case['priority'] = 'llm_contrastive_pair'
                hall_case['llm_evaluation'] = pair['evaluation']
                cases.append(hall_case)
        elif strategy == 'comprehensive':
            cases = self.select_comprehensive_cases(num_cases, use_llm=use_llm)
        else:
            raise ValueError(f"未知的选择策略: {strategy}")
        
        # 保存结果
        self.save_selected_cases(cases, output_path)
        
        return cases


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Case Selector - LLM增强案例选择工具')
    parser.add_argument('--correct_jsonl', type=str, required=True,
                       help='正确样本数据路径')
    parser.add_argument('--incorrect_jsonl', type=str, required=True,
                       help='幻觉样本数据路径')
    parser.add_argument('--parcel_analysis', type=str, required=True,
                       help='Parcel分析结果路径')
    parser.add_argument('--capability_analysis', type=str, required=True,
                       help='Capability分析结果路径')
    parser.add_argument('--strategy', type=str, default='comprehensive',
                       choices=['high_confidence', 'diverse', 'pairs', 'extreme', 'clustered', 
                               'llm_high_insight', 'llm_contrastive', 'comprehensive'],
                       help='选择策略')
    parser.add_argument('--num_cases', type=int, default=20,
                       help='选择案例数量')
    parser.add_argument('--output', type=str, default='selected_cases.json',
                       help='输出文件路径')
    parser.add_argument('--use_llm', action='store_true', default=True,
                       help='使用LLM增强选择')
    parser.add_argument('--vllm_url', type=str, default=DEFAULT_VLLM_URL,
                       help=f'vLLM API地址 (默认: {DEFAULT_VLLM_URL})')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY,
                       help=f'API密钥 (默认: {DEFAULT_API_KEY})')
    parser.add_argument('--min_score', type=float, default=7.0,
                       help='LLM评估最低分数阈值')
    
    args = parser.parse_args()
    
    # 创建选择器
    selector = CaseSelector(
        correct_jsonl_path=args.correct_jsonl,
        incorrect_jsonl_path=args.incorrect_jsonl,
        parcel_analysis_path=args.parcel_analysis,
        capability_analysis_path=args.capability_analysis,
        vllm_url=args.vllm_url,
        api_key=args.api_key
    )
    
    # 运行案例选择
    cases = selector.run_case_selection(
        strategy=args.strategy,
        num_cases=args.num_cases,
        output_path=args.output,
        use_llm=args.use_llm
    )
    
    print(f"\n选择了 {len(cases)} 个案例:")
    for i, case in enumerate(cases, 1):
        case_id = case.get('sample_id') or case.get('case_id')
        is_correct = case.get('is_correct', False)
        priority = case.get('priority', 'unknown')
        score = case.get('score', 'N/A')
        
        print(f"{i}. 样本 {case_id} ({'正确' if is_correct else '幻觉'}) - {priority}")
        print(f"   问题: {case['question'][:80]}...")
        print(f"   分数: {score}")
        
        # 显示LLM评估信息
        if 'overall_score' in case:
            print(f"   LLM洞察分数: {case['overall_score']:.2f}/10")
            print(f"   洞察摘要: {case.get('insight_summary', 'N/A')[:100]}...")
        
        print()


if __name__ == "__main__":
    main()
