#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel功能分析器
根据聚类结果中的激活样本，分析每个parcel的功能并生成类似大脑分区的功能描述
"""

import os
import json
import argparse
import requests
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ParcelFunctionalityAnalyzer:
    def __init__(self, vllm_url: str = "http://0.0.0.0:8000/v1", api_key: str = "abcabc"):
        """
        初始化分析器
        Args:
            vllm_url: vLLM服务地址
            api_key: API密钥
        """
        self.vllm_url = vllm_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 初始化NLP工具
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # 加载数据集介绍
        self.dataset_introductions = self._load_dataset_introductions()
        
        # 扩展停用词列表，包含常见的无意义词汇
        self.stop_words.update({
            'question', 'answer', 'based', 'please', 'what', 'who', 'which', 'where',
            'when', 'why', 'how', 'name', 'title', 'page', 'wiki', 'wikipedia',
            'would', 'could', 'should', 'might', 'may', 'will', 'can', 'must',
            'also', 'even', 'still', 'just', 'only', 'very', 'really', 'quite',
            'much', 'many', 'few', 'several', 'some', 'any', 'all', 'every',
            'each', 'both', 'either', 'neither', 'none', 'no', 'yes', 'okay',
            'well', 'good', 'bad', 'great', 'small', 'large', 'big', 'little',
            'new', 'old', 'young', 'first', 'last', 'next', 'previous', 'current',
            'former', 'latter', 'early', 'late', 'recent', 'ancient', 'modern',
            'traditional', 'classical', 'contemporary', 'historical', 'future',
            'past', 'present', 'today', 'yesterday', 'tomorrow', 'now', 'then',
            'here', 'there', 'everywhere', 'nowhere', 'somewhere', 'anywhere',
            'above', 'below', 'under', 'over', 'inside', 'outside', 'within',
            'without', 'between', 'among', 'around', 'through', 'across', 'along',
            'toward', 'towards', 'against', 'for', 'from', 'to', 'of', 'in', 'on',
            'at', 'by', 'with', 'without', 'about', 'against', 'between', 'into',
            'during', 'before', 'after', 'since', 'until', 'while', 'during',
            'because', 'although', 'though', 'unless', 'if', 'whether', 'while',
            'whereas', 'however', 'nevertheless', 'nonetheless', 'therefore',
            'thus', 'hence', 'consequently', 'accordingly', 'moreover', 'furthermore',
            'additionally', 'besides', 'also', 'too', 'as', 'like', 'such', 'so',
            'than', 'rather', 'quite', 'very', 'extremely', 'highly', 'completely',
            'entirely', 'totally', 'absolutely', 'definitely', 'certainly', 'surely',
            'probably', 'possibly', 'maybe', 'perhaps', 'likely', 'unlikely',
            'impossible', 'necessary', 'essential', 'important', 'significant',
            'major', 'minor', 'main', 'primary', 'secondary', 'tertiary',
            'original', 'copy', 'version', 'edition', 'series', 'collection',
            'group', 'set', 'pair', 'couple', 'dozen', 'hundred', 'thousand',
            'million', 'billion', 'trillion', 'zero', 'one', 'two', 'three',
            'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
            'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
            'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
            'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand'
        })
        
    def call_vllm_api(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """
        调用vLLM API
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度参数
        Returns:
            生成的文本
        """
        payload = {
            "model": "/path/to/local_models/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Reasoning: Medium"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.vllm_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # 安全地提取内容，处理可能的None值
            if (result and "choices" in result and 
                len(result["choices"]) > 0 and 
                "message" in result["choices"][0] and 
                "content" in result["choices"][0]["message"]):
                
                content = result["choices"][0]["message"]["content"]
                if content is not None:
                    return content.strip()
                else:
                    error_msg = "API返回的内容为None"
                    print(f"❌ {error_msg}")
                    print(f"完整API响应: {result}")
                    return f"ERROR: {error_msg}"
            else:
                error_msg = f"API返回结果格式异常: {result}"
                print(f"❌ {error_msg}")
                return f"ERROR: {error_msg}"
                
        except Exception as e:
            error_msg = f"API调用失败: {e}"
            print(f"❌ {error_msg}")
            print(f"请求payload: {payload}")
            return f"ERROR: {error_msg}"
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        Args:
            text: 原始文本
        Returns:
            预处理后的文本
        """
        try:
            # 安全处理None值和空字符串
            if text is None:
                return ""
            
            # 转换为小写
            text = text.lower()
            
            # 移除特殊字符，保留字母、数字和空格
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # 移除多余空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            error_msg = f"文本预处理失败: {e}"
            print(f"❌ {error_msg}")
            print(f"原始文本: {text}")
            return f"ERROR: {error_msg}"
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        分词和词形还原
        Args:
            text: 预处理后的文本
        Returns:
            词形还原后的词列表
        """
        try:
            # 分词
            tokens = word_tokenize(text)
            
            # 词形还原和过滤
            lemmatized_tokens = []
            for token in tokens:
                # 跳过停用词和短词
                if token in self.stop_words or len(token) < 3:
                    continue
                
                # 词形还原
                lemmatized = self.lemmatizer.lemmatize(token)
                if len(lemmatized) >= 3:
                    lemmatized_tokens.append(lemmatized)
            
            return lemmatized_tokens
        except Exception as e:
            error_msg = f"分词和词形还原失败: {e}"
            print(f"❌ {error_msg}")
            print(f"输入文本: {text}")
            return [f"ERROR: {error_msg}"]
    
    def extract_content_from_sample(self, sample: Dict) -> Tuple[str, str, str]:
        """
        从样本中提取问题、答案和激活句子
        Args:
            sample: 样本字典
        Returns:
            (question, answer, activated_sentence)
        """
        content = sample.get("content", "")
        
        # 安全处理None值
        if content is None:
            content = ""
        
        question = ""
        answer = ""
        activated_sentence = ""
        
        try:
            if "Question:" in content and "Answer:" in content:
                # 提取问题部分
                question_match = re.search(r"Question:(.*?)(?=Answer:|$)", content, re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip()
                
                # 提取答案部分
                answer_match = re.search(r"Answer:(.*?)(?=Activated Sentence:|$)", content, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                
                # 提取激活句子
                sentence_match = re.search(r"Activated Sentence:(.*?)$", content, re.DOTALL)
                if sentence_match:
                    activated_sentence = sentence_match.group(1).strip()
            elif "Q:" in content and "A:" in content:
                # 处理 "Q: ...\nA: ..." 格式
                # 提取问题部分
                question_match = re.search(r"Q:(.*?)(?=\nA:|$)", content, re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip()
                
                # 提取答案部分
                answer_match = re.search(r"A:(.*?)$", content, re.DOTALL)
                if answer_match:
                    answer = answer_match.group(1).strip()
                
                # 对于这种格式，激活句子通常是答案的一部分或整个答案
                activated_sentence = answer
            else:
                # 如果没有标准格式，将整个内容作为激活句子
                activated_sentence = content
        except Exception as e:
            error_msg = f"内容提取出错: {e}"
            print(f"❌ {error_msg}")
            print(f"问题样本内容: {sample}")
            # 出错时返回错误信息而不是空值
            question = f"ERROR: {error_msg}"
            answer = f"ERROR: {error_msg}"
            activated_sentence = f"ERROR: {error_msg}"
        
        return question, answer, activated_sentence
    
    def extract_keywords_tfidf_activation_weighted(self, samples: List[Dict], top_k: int = 20) -> List[str]:
        """
        使用TF-IDF + 激活强度加权提取关键词（第一阶段改进）
        Args:
            samples: 样本列表
            top_k: 返回前k个关键词
        Returns:
            关键词列表
        """
        print(f"使用TF-IDF + 激活强度加权提取关键词...")
        
        # 准备文档和激活强度
        documents = []
        activation_weights = []
        
        for sample in samples:
            question, answer, activated_sentence = self.extract_content_from_sample(sample)
            
            # 合并所有文本内容
            combined_text = f"{question} {answer} {activated_sentence}".strip()
            if combined_text:
                # 预处理文本
                processed_text = self.preprocess_text(combined_text)
                documents.append(processed_text)
                
                # 获取激活强度作为权重
                activation = sample.get("avg_activation", 0.0)
                activation_weights.append(activation)
        
        if not documents:
            print("警告：没有有效的文档内容")
            return []
        
        # 创建TF-IDF向量器
        tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # 限制特征数量
            min_df=1,  # 最小文档频率
            max_df=0.95,  # 最大文档频率（过滤过于常见的词）
            ngram_range=(1, 2),  # 支持单词和双词组合
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # 只保留3个字符以上的单词
        )
        
        try:
            # 计算TF-IDF矩阵
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # 计算激活强度加权的TF-IDF分数
            weighted_scores = np.zeros(len(feature_names))
            
            for i, doc_tfidf in enumerate(tfidf_matrix.toarray()):
                # 使用激活强度作为权重
                weight = activation_weights[i] if i < len(activation_weights) else 1.0
                weighted_scores += doc_tfidf * weight
            
            # 归一化加权分数
            if weighted_scores.sum() > 0:
                weighted_scores = weighted_scores / weighted_scores.sum()
            
            # 获取top-k关键词
            top_indices = np.argsort(weighted_scores)[::-1][:top_k]
            keywords = [feature_names[i] for i in top_indices if weighted_scores[i] > 0]
            
            print(f"提取到 {len(keywords)} 个关键词")
            return keywords
            
        except Exception as e:
            print(f"TF-IDF计算失败，回退到简单方法: {e}")
            return self.extract_keywords_simple(samples, top_k)
    
    def extract_keywords_simple(self, samples: List[Dict], top_k: int = 20) -> List[str]:
        """
        简单关键词提取（回退方法）
        Args:
            samples: 样本列表
            top_k: 返回前k个关键词
        Returns:
            关键词列表
        """
        print("使用简单关键词提取方法...")
        
        # 提取所有文本内容
        all_text = []
        for sample in samples:
            question, answer, activated_sentence = self.extract_content_from_sample(sample)
            all_text.extend([question, answer, activated_sentence])
        
        # 合并所有文本
        combined_text = " ".join(all_text)
        
        # 预处理和分词
        processed_text = self.preprocess_text(combined_text)
        tokens = self.tokenize_and_lemmatize(processed_text)
        
        # 统计词频
        word_counts = Counter(tokens)
        
        # 返回前k个高频词
        return [word for word, count in word_counts.most_common(top_k)]
    
    def extract_keywords_from_samples(self, samples: List[Dict], top_k: int = 20) -> List[str]:
        """
        从样本中提取关键词（主入口函数）
        Args:
            samples: 样本列表
            top_k: 返回前k个关键词
        Returns:
            关键词列表
        """
        # 使用改进的TF-IDF + 激活强度加权方法
        return self.extract_keywords_tfidf_activation_weighted(samples, top_k)
    
    def analyze_parcel_functionality(self, parcel_id: int, samples: List[Dict], 
                                   n_top_samples: int = 10) -> Dict[str, Any]:
        """
        分析单个parcel的功能
        Args:
            parcel_id: parcel ID
            samples: 该parcel的样本列表
            n_top_samples: 用于分析的前N个样本
        Returns:
            功能分析结果
        """
        print(f"分析 Parcel {parcel_id} 的功能...")
        
        # 选择激活强度最高的样本
        top_samples = sorted(samples, key=lambda x: x.get("avg_activation", 0), reverse=True)[:n_top_samples]
        
        # 提取关键词
        keywords = self.extract_keywords_from_samples(top_samples)
        
        # 统计数据集分布
        dataset_counts = Counter([sample.get("dataset", "unknown") for sample in top_samples])
        
        # 构建分析提示
        prompt = self._build_analysis_prompt(parcel_id, top_samples, keywords, dataset_counts)
        
        # 调用API进行分析
        functionality_description = self.call_vllm_api(prompt,max_tokens=10000)
        
        return {
            "parcel_id": parcel_id,
            "functionality_description": functionality_description,
            "keywords": keywords,
            "dataset_distribution": dict(dataset_counts),
            "top_samples_count": len(top_samples),
            "avg_activation_range": {
                "min": min(sample.get("avg_activation", 0) for sample in top_samples),
                "max": max(sample.get("avg_activation", 0) for sample in top_samples),
                "mean": np.mean([sample.get("avg_activation", 0) for sample in top_samples])
            }
        }
    
    def _build_analysis_prompt(self, parcel_id: int, samples: List[Dict], 
                             keywords: List[str], dataset_counts: Counter, top_k_samples: int = 50) -> str:
        """
        构建分析提示
        Args:
            parcel_id: parcel ID
            samples: 样本列表
            keywords: 关键词列表
            dataset_counts: 数据集分布
            top_k_samples: 用于展示的样本数量
        Returns:
            分析提示
        """
        # 构建样本示例
        sample_examples = []
        for i, sample in enumerate(samples[:top_k_samples]):
            question, answer, activated_sentence = self.extract_content_from_sample(sample)
            
            # 如果问题过长（超过200个token），进行总结
            if self._count_tokens(question) > 200:
                question = self._summarize_question(question)
            
            # 构建格式化的样本展示
            sample_text = f"""Sample {i+1}:
Question: {question}
Answer: {answer}
Activated Sentence: {activated_sentence}
Activation Strength: {sample.get('avg_activation', 0):.4f}
Dataset: {sample.get('dataset', 'unknown')}"""
            
            sample_examples.append(sample_text)
        
        # 构建数据集分布信息（包含介绍）
        dataset_info = []
        # 确保dataset_counts是Counter对象
        if isinstance(dataset_counts, dict):
            dataset_counts = Counter(dataset_counts)
        
        for dataset, count in dataset_counts.most_common(5):
            description = self._get_dataset_description(dataset)
            dataset_info.append(f"{dataset} ({count} samples): {description}")
        
        prompt = f"""You are a neuroscientist analyzing the functional specialization of different regions in artificial neural networks. Please analyze the functionality of Parcel {parcel_id} based on the following information:

**Parcel {parcel_id} Activation Sample Information:**

**Keywords (ranked by importance):**
{', '.join(keywords[:15])}

**Dataset Distribution:**
{chr(10).join(dataset_info)}

**High Activation Sample Examples:**
Each sample consists of three key components:
- **Question**: The input query that triggers the neural activation
- **Answer**: The expected or generated response to the question
- **Activated Sentence**: The specific text segment that shows the highest activation in this parcel, indicating what this neural region is most responsive to

{chr(10).join(sample_examples)}

**Analysis Requirements:**
1. Based on the sample content, analyze what type of information or tasks this parcel primarily processes
2. Use terminology similar to human brain region descriptions (e.g., visual processing, language comprehension, reasoning, etc.)
3. Provide a concise function name (2-4 words)
4. Provide a detailed function description (100-200 words)
5. Analyze the potential role of this parcel in the overall functionality of the large language model

Please respond in the following format:

**Function Name:** [concise function name]

**Function Description:** [detailed function description]

**Role in Large Model:** [analysis of the parcel's role in overall functionality]

Please respond in English."""

        return prompt
    
    def analyze_all_parcels(self, topsamples_file: str, output_file: str, 
                           n_top_samples: int = 10, delay: float = 1.0) -> Dict[str, Any]:
        """
        分析所有parcel的功能
        Args:
            topsamples_file: topsamples文件路径
            output_file: 输出文件路径
            n_top_samples: 每个parcel用于分析的前N个样本
            delay: API调用间隔（秒）
        Returns:
            所有parcel的分析结果
        """
        print(f"加载topsamples文件: {topsamples_file}")
        
        # 加载topsamples数据
        with open(topsamples_file, 'r', encoding='utf-8') as f:
            topsamples_data = json.load(f)
        
        print(f"发现 {len(topsamples_data)} 个parcel")
        
        # 分析每个parcel
        all_analyses = {}
        for parcel_id_str, samples in topsamples_data.items():
            parcel_id = int(parcel_id_str)
            print(f"\n处理 Parcel {parcel_id}/{len(topsamples_data)}")
            
            # 分析该parcel
            analysis_result = self.analyze_parcel_functionality(parcel_id, samples, n_top_samples)
            all_analyses[parcel_id_str] = analysis_result
            
            # 保存中间结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_analyses, f, ensure_ascii=False, indent=2)
            
            print(f"Parcel {parcel_id} 分析完成")
            
            # 延迟避免API限制
            if delay > 0:
                time.sleep(delay)
        
        print(f"\n所有parcel分析完成，结果保存到: {output_file}")
        return all_analyses
    
    def generate_summary_report(self, analyses: Dict[str, Any], output_file: str):
        """
        生成汇总报告
        Args:
            analyses: 所有parcel的分析结果
            output_file: 输出文件路径
        """
        print("生成汇总报告...")
        
        # 提取功能名称和描述
        parcel_summaries = []
        for parcel_id, analysis in analyses.items():
            functionality_desc = analysis.get("functionality_description", "")
            
            # 提取功能名称
            name_match = re.search(r"Function Name:\s*(.+)", functionality_desc)
            function_name = name_match.group(1).strip() if name_match else f"Parcel_{parcel_id}"
            
            # 提取功能描述
            desc_match = re.search(r"Function Description:\s*(.+?)(?=\n\n|\n\*\*|$)", functionality_desc, re.DOTALL)
            function_desc = desc_match.group(1).strip() if desc_match else "No description available"
            
            # 提取在大模型中的作用
            role_match = re.search(r"Role in Large Model:\s*(.+?)(?=\n\n|\n\*\*|$)", functionality_desc, re.DOTALL)
            model_role = role_match.group(1).strip() if role_match else "No analysis available"
            
            parcel_summaries.append({
                "parcel_id": int(parcel_id),
                "function_name": function_name,
                "function_description": function_desc,
                "model_role": model_role,
                "keywords": analysis.get("keywords", []),
                "dataset_distribution": analysis.get("dataset_distribution", {}),
                "avg_activation": analysis.get("avg_activation_range", {})
            })
        
        # 按parcel_id排序
        parcel_summaries.sort(key=lambda x: x["parcel_id"])
        
        # 生成汇总报告
        summary = {
            "total_parcels": len(analyses),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parcel_summaries": parcel_summaries,
            "function_categories": self._categorize_functions(parcel_summaries)
        }
        
        # 保存汇总报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"汇总报告已保存到: {output_file}")
        
        # 打印简要汇总
        print("\n=== Parcel功能汇总 ===")
        for summary_item in parcel_summaries:
            print(f"Parcel {summary_item['parcel_id']:2d}: {summary_item['function_name']}")
    
    def _categorize_functions(self, parcel_summaries: List[Dict]) -> Dict[str, List[int]]:
        """
        对功能进行分类
        Args:
            parcel_summaries: parcel汇总列表
        Returns:
            功能分类结果
        """
        categories = {
            "语言处理": [],
            "知识检索": [],
            "推理分析": [],
            "情感理解": [],
            "代码处理": [],
            "数学计算": [],
            "其他": []
        }
        
        # 关键词映射
        category_keywords = {
            "语言处理": ["语言", "翻译", "语法", "词汇", "句子", "文本"],
            "知识检索": ["知识", "检索", "查询", "信息", "事实", "百科"],
            "推理分析": ["推理", "分析", "逻辑", "因果", "推理", "判断"],
            "情感理解": ["情感", "情绪", "感受", "态度", "观点", "评价"],
            "代码处理": ["代码", "编程", "函数", "算法", "程序", "技术"],
            "数学计算": ["数学", "计算", "数字", "公式", "统计", "数值"]
        }
        
        for summary in parcel_summaries:
            function_name = summary["function_name"].lower()
            function_desc = summary["function_description"].lower()
            keywords = [kw.lower() for kw in summary["keywords"]]
            
            # 检查属于哪个类别
            categorized = False
            for category, keywords_list in category_keywords.items():
                for keyword in keywords_list:
                    if (keyword in function_name or keyword in function_desc or 
                        any(kw in keyword or keyword in kw for kw in keywords)):
                        categories[category].append(summary["parcel_id"])
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                categories["其他"].append(summary["parcel_id"])
        
        return categories
    
    def _load_dataset_introductions(self) -> Dict[str, List[str]]:
        """
        加载数据集介绍信息
        Returns:
            数据集介绍字典
        """
        try:
            dataset_intro_path = "/path/to/project_root/neural_area/capability_data_v2/data_stastic/capability_coverage_summary.json"
            with open(dataset_intro_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告：无法加载数据集介绍文件: {e}")
            return {}
    
    def _get_dataset_description(self, dataset_name: str) -> str:
        """
        获取数据集描述
        Args:
            dataset_name: 数据集名称
        Returns:
            数据集描述
        """
        if dataset_name in self.dataset_introductions:
            intro = self.dataset_introductions[dataset_name]
            if len(intro) >= 2:
                return f"{intro[0]} {intro[1]}"
            elif len(intro) >= 1:
                return intro[0]
        return f"Dataset: {dataset_name}"
    
    def _summarize_question(self, question: str) -> str:
        """
        使用GPT-OSS总结过长的问题
        Args:
            question: 原始问题
        Returns:
            总结后的问题
        """
        # 使用GPT tokenizer计算token数量
        token_count = self._count_tokens(question)
        if token_count <= 200:
            return question
        
        try:
            summary_prompt = f"""Please provide a concise summary of the following question in a few sentences, keeping the key information and context. The summary should be within 200 tokens:

Question: {question}

Summary:"""
            
            summary = self.call_vllm_api(summary_prompt, max_tokens=10000)
            
            # 安全地清理总结结果
            if summary and isinstance(summary, str):
                summary = summary.strip()
                if summary.startswith("Summary:"):
                    summary = summary[8:].strip()
                
                if summary:
                    return summary
            
            # 如果总结失败或为空，使用截断
            return self._truncate_to_tokens(question, 200)
            
        except Exception as e:
            error_msg = f"问题总结失败: {e}"
            print(f"❌ {error_msg}")
            print(f"原始问题: {question}")
            return f"ERROR: {error_msg}"
    
    def _count_tokens(self, text: str) -> int:
        """
        使用GPT tokenizer计算token数量
        Args:
            text: 输入文本
        Returns:
            token数量
        """
        try:
            # 尝试使用tiktoken（OpenAI的tokenizer）
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4使用的编码
            return len(encoding.encode(text))
        except ImportError:
            try:
                # 尝试使用transformers的tokenizer
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                return len(tokenizer.encode(text))
            except ImportError:
                # 如果都没有，使用简单的字符估算（1 token ≈ 4字符）
                return len(text) // 4
        except Exception as e:
            error_msg = f"Token计数失败: {e}"
            print(f"❌ {error_msg}")
            print(f"输入文本: {text}")
            # 返回一个默认值，避免程序崩溃
            return 0
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        将文本截断到指定token数量
        Args:
            text: 输入文本
            max_tokens: 最大token数量
        Returns:
            截断后的文本
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens) + "..."
        except ImportError:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                tokens = tokenizer.encode(text)
                if len(tokens) <= max_tokens:
                    return text
                truncated_tokens = tokens[:max_tokens]
                return tokenizer.decode(truncated_tokens) + "..."
            except ImportError:
                # 简单字符截断
                return text[:max_tokens * 4] + "..."
        except Exception as e:
            error_msg = f"文本截断失败: {e}"
            print(f"❌ {error_msg}")
            print(f"输入文本: {text}")
            print(f"最大token数: {max_tokens}")
            # 返回原始文本，避免程序崩溃
            return text

def main():
    parser = argparse.ArgumentParser(description='分析parcel功能并生成类似大脑分区的功能描述')
    parser.add_argument('--topsamples_file', type=str, required=True,
                       help='topsamples文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='./parcel_functionality_analysis',
                       help='输出目录')
    parser.add_argument('--vllm_url', type=str, 
                       default='http://0.0.0.0:8000/v1',
                       help='vLLM服务地址')
    parser.add_argument('--api_key', type=str, default='abcabc',
                       help='API密钥')
    parser.add_argument('--n_top_samples', type=int, default=10,
                       help='每个parcel用于分析的前N个样本')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='API调用间隔（秒）')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='跳过分析，只生成汇总报告（如果已有分析结果）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构建输出文件路径
    base_name = Path(args.topsamples_file).stem
    analysis_file = os.path.join(args.output_dir, f"{base_name}_functionality_analysis.json")
    summary_file = os.path.join(args.output_dir, f"{base_name}_functionality_summary.json")
    
    # 初始化分析器
    analyzer = ParcelFunctionalityAnalyzer(args.vllm_url, args.api_key)
    
    if not args.skip_analysis:
        # 分析所有parcel
        analyses = analyzer.analyze_all_parcels(
            args.topsamples_file, 
            analysis_file, 
            args.n_top_samples, 
            args.delay
        )
    else:
        # 加载已有分析结果
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analyses = json.load(f)
            print(f"加载已有分析结果: {analysis_file}")
        else:
            print(f"错误：未找到分析结果文件 {analysis_file}")
            return
    
    # 生成汇总报告
    analyzer.generate_summary_report(analyses, summary_file)
    
    print(f"\n分析完成！")
    print(f"详细分析结果: {analysis_file}")
    print(f"汇总报告: {summary_file}")

if __name__ == "__main__":
    main() 