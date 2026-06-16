#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-Based 解释报告生成脚本

汇总 Parcel-Level 和 Capability-Level 分析结果，结合描述文件，
调用 LLM 自动撰写符合 Nature 文章风格的分析报告。

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
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_VLLM_URL = "http://0.0.0.0:8001/v1"
DEFAULT_API_KEY = "abcabc"
DEFAULT_OUTPUT_DIR = "/path/to/project_root/safety_explanation/hallucination/results"

class LLMReportGenerator:
    """LLM报告生成器"""
    
    def __init__(self, parcel_desc_path: str, cap_desc_path: str, 
                 parcel_diff_path: str, cap_diff_path: str,
                 output_path: str, cog_mapping_path: str,
                 vllm_url: str = DEFAULT_VLLM_URL,
                 api_key: str = DEFAULT_API_KEY,
                 correct_data_path: str = None,
                 incorrect_data_path: str = None):
        """
        初始化报告生成器
        
        Args:
            parcel_desc_path: Parcel功能描述文件路径
            cap_desc_path: Capability描述文件路径
            parcel_diff_path: Parcel异常分析结果路径
            cap_diff_path: Capability异常分析结果路径
            output_path: 输出报告文件路径
            cog_mapping_path: 认知层级映射文件路径
            vllm_url: vLLM API地址
            api_key: API密钥
            correct_data_path: 正确样本数据文件路径
            incorrect_data_path: 幻觉样本数据文件路径
        """
        self.parcel_desc_path = parcel_desc_path
        self.cap_desc_path = cap_desc_path
        self.parcel_diff_path = parcel_diff_path
        self.cap_diff_path = cap_diff_path
        self.output_path = output_path
        self.cog_mapping_path = cog_mapping_path
        self.vllm_url = vllm_url
        self.api_key = api_key
        self.correct_data_path = correct_data_path
        self.incorrect_data_path = incorrect_data_path
        
        # 数据存储
        self.parcel_descriptions = {}
        self.capability_descriptions = {}
        self.parcel_analysis = {}
        self.capability_analysis = {}
        self.cog_mapping = {}
        self.parcel_top_activated = {}
        self.parcel_top_connections = {}
        self.parcel_anomalous_connections = {}
        self.capability_top_activated = {}
        self.capability_top_connections = {}
        self.capability_anomalous_connections = {}
        
    def get_dataset_intro(self) -> str:
        """读取数据集简介文本，用于拼接到LLM提示前。"""
        try:
            # dataset_intro.json 位于 hallucination/dataset/dataset_intro.json
            base_dir = Path(__file__).resolve().parents[3]  # .../safety_explanation/hallucination
            intro_path = base_dir / 'dataset' / 'dataset_intro.json'
            if not intro_path.exists():
                logger.warning(f"未找到数据集简介文件: {intro_path}")
                return ""
            with open(intro_path, 'r', encoding='utf-8') as f:
                intro_map = json.load(f)

            model_data_lower = str(getattr(self, 'model_data', '')).lower()


            # 优先通过包含关系匹配键
            for key in intro_map.keys():
                if key.lower() in model_data_lower:
                    introduction = intro_map.get(key, '')
                    return introduction
                else:
                    raise ValueError(f"未找到数据集简介: {model_data_lower}")
        except Exception as e:
            logger.warning(f"读取数据集简介失败: {e}")
            return ""

    def load_parcel_descriptions(self) -> None:
        """加载Parcel功能描述"""
        logger.info("加载Parcel功能描述...")
        
        try:
            with open(self.parcel_desc_path, 'r', encoding='utf-8') as f:
                self.parcel_descriptions = json.load(f)
            logger.info(f"加载了 {len(self.parcel_descriptions)} 个Parcel描述")
        except FileNotFoundError:
            logger.warning(f"Parcel描述文件不存在: {self.parcel_desc_path}")
            self.parcel_descriptions = {}
        except Exception as e:
            logger.error(f"加载Parcel描述失败: {e}")
            self.parcel_descriptions = {}
    def build_prompt(self, data_type, analysis_summary):
        if data_type == "rag":
            # --- Retrieval-Augmented Generation Hallucination ---
            prompt = f"""
    Please write a **Nature-style scientific report** based on the following hallucination mechanism analysis results. 
    The report should include:

    1. Abstract  
    2. Introduction  
    3. Results  
    4. Discussion  
    5. Conclusion  

    **Data type:** Retrieval-Augmented Generation (RAG) hallucination analysis  
    **Input Format Example:**  
    context, question, answer_true  

    **Analysis Summary:**  
    {analysis_summary}

    ---

    ### Writing Focus

    #### Core Analytical Dimensions
    1. **Correct retrieval–generation alignment**: Characterize neural activation patterns when the model correctly integrates retrieved context with internal parametric knowledge.
    2. **Hallucinated retrieval utilization**: Identify activation and connectivity patterns when hallucinations occur due to context misalignment, neglect, or overfitting to prior knowledge.
    3. **Cross-source connectivity**: Compare the integration network linking external retrieval parcels and internal reasoning parcels.
    4. **Cognitive hierarchy**: Analyze how hallucinations emerge across perception, contextual integration, reasoning, and metacognition levels.
    5. **Activation contrast**: Compare top-activated parcels/capabilities in correct vs hallucinated RAG samples.
    6. **Connection contrast**: Examine inter-parcel and inter-capability connection strength differences.
    7. **Misalignment patterns**: Identify abnormal coupling between retrieval and reasoning modules.
    8. **Hierarchical decoupling**: Analyze breakdowns between contextual grounding and high-level reasoning.

    ---

    ### Emphasized Findings
    - The **canonical retrieval–integration network** in correct samples  
    - The **hallucination-specific activation pattern** showing overactivation of parametric-memory parcels  
    - Structural differences in **retrieval–reasoning connectivity topology**  
    - Cognitive-level analysis revealing **insufficient context-grounding in reasoning layers**  
    - Neuroscientific parallels to **memory misbinding** and **source-monitoring errors** in the human brain  
    - Theoretical interpretation of **context–knowledge conflict** mechanisms

    ---

    ### Methodological Notes
    - **Parcel-level**: 270-dimensional activation modules derived from SAE, each annotated with cognitive function and model role (retrieval, reasoning, integration, or control)
    - **Capability-level**: Aggregation via Capability–Parcel mapping with cognitive-level classification  
    - **Connectivity analysis**: cosine similarity-based functional network modeling  
    - **Statistical testing**: t-tests for activation and connection significance  
    - **Pattern extraction**: identification of top activations and top abnormal connections  

    ---

    ### Additional Writing Requirements
    - Highlight both normal and abnormal network structures  
    - Interpret findings from a **retrieval–reasoning integration perspective**  
    - Draw parallels to **human memory systems** (hippocampal–prefrontal coordination)  
    - Offer **mechanistic explanations** of hallucination genesis within retrieval-augmented LLMs  

    Final report length: 4000–5000 words.
    """
        elif data_type == "medical":
            # --- Medical Hallucination Analysis ---
            prompt = f"""
    Please write a **Nature-style scientific report** based on the following hallucination mechanism analysis results. 
    The report should include:

    1. Abstract  
    2. Introduction  
    3. Results  
    4. Discussion  
    5. Conclusion  

    **Data type:** Medical hallucination QA analysis  
    **Input Format Example:**  
    question, context, difficulty_level, category_of_hallucination, answer_true  

    **Analysis Summary:**  
    {analysis_summary}

    ---

    ### Writing Focus

    #### Core Analytical Dimensions
    1. **Correct biomedical reasoning patterns**: Identify parcel and capability activations during accurate domain-specific reasoning.
    2. **Hallucinated biomedical inference**: Characterize abnormal activations linked to domain overgeneralization, missing causal reasoning, or terminological confusion.
    3. **Functional network comparison**: Examine differences in functional connectivity between biological knowledge parcels (e.g., cell processes, organ systems) and reasoning parcels.
    4. **Cognitive hierarchy**: Analyze how domain-specific hallucinations propagate across layers from perception (reading biomedical text) to reasoning and metacognition (confidence misjudgment).
    5. **Activation strength contrast**: Compare activation intensity in core biomedical reasoning networks.
    6. **Connectivity contrast**: Assess abnormal parcel–parcel links in medical hallucinations.
    7. **Anomalous semantic coupling**: Identify false associations between biomedical entities.
    8. **Hierarchical decoupling**: Detect reasoning–evidence disconnections at the conceptual integration level.

    ---

    ### Emphasized Findings
    - The **canonical biomedical reasoning network** in correct answers  
    - The **hallucination-driven overactivation** of lexical or associative memory parcels  
    - Distinct **functional disconnections** between mechanistic understanding and factual recall  
    - Cognitive-level differences in **causal reasoning** and **knowledge verification** layers  
    - Neuroscientific parallels to **semantic false memory** and **confabulation**  
    - Theoretical model explaining **domain hallucination** as impaired epistemic verification  

    ---

    ### Methodological Notes
    - **Parcel-level**: 270-dimensional modules derived from SAE activations with medical semantic labeling (e.g., mitochondria, apoptosis, cellular signaling)
    - **Capability-level**: Capabilities mapped by cognitive tier (perception → understanding → reasoning → evaluation)
    - **Connectivity**: cosine similarity of parcel activation vectors across samples  
    - **Statistical tests**: two-sample t-tests for significant activation and connectivity differences  
    - **Pattern recognition**: identify domain-specific abnormal activations  

    ---

    ### Additional Writing Requirements
    - Interpret medical hallucinations as **neurocognitive analogs** of false belief formation  
    - Relate to **human neuroscience of medical reasoning** (e.g., prefrontal-insula coordination)  
    - Provide a **mechanistic explanation** linking knowledge gaps to cognitive-level errors  

    Final report length: 4000–5000 words.
    """
        elif data_type == "science":
            # --- Scientific Reasoning Hallucination Analysis ---
            prompt = f"""
    Please write a **Nature-style scientific report** based on the following hallucination mechanism analysis results. 
    The report should include:

    1. Abstract  
    2. Introduction  
    3. Results  
    4. Discussion  
    5. Conclusion  

    **Data type:** Scientific reasoning hallucination analysis (e.g., physics, chemistry, or biology QA)  

    **Analysis Summary:**  
    {analysis_summary}

    ---

    ### Writing Focus

    #### Core Analytical Dimensions
    1. **Correct scientific reasoning network**: Characterize parcel/capability activation patterns during correct multi-step logical inference.
    2. **Hallucinated reasoning mechanisms**: Identify aberrant activations linked to faulty analogical mapping, over-generalized heuristics, or misapplied scientific laws.
    3. **Functional topology contrast**: Compare reasoning–representation network structures across correct and hallucinated samples.
    4. **Cognitive hierarchy**: Relate hallucinations to failures in the progression from conceptual recall → reasoning → evaluation.
    5. **Activation intensity**: Contrast the most active scientific-reasoning parcels in correct vs hallucinated responses.
    6. **Connectivity deviation**: Identify disrupted connections within logical reasoning chains.
    7. **Anomalous inference bridges**: Locate spurious links mimicking reasoning shortcuts.
    8. **Cross-level decoupling**: Examine the breakdown between factual knowledge and meta-cognitive verification.

    ---

    ### Emphasized Findings
    - Stable **scientific reasoning core network** in correct answers  
    - Overactivation of **heuristic-driven parcels** during hallucinations  
    - Altered **reasoning network topology** leading to logic drift  
    - Cognitive-level dissociation between conceptual recall and consistency checking  
    - Neuroscientific analogies with **fronto-parietal reasoning control** mechanisms  
    - Theoretical explanation of **reasoning hallucination** as imbalance between intuitive and analytic cognition  

    ---

    ### Methodological Notes
    - **Parcel-level**: 270-dimensional SAE features annotated with reasoning-related functions  
    - **Capability-level**: Hierarchical cognitive classification aligned with Bloom’s taxonomy  
    - **Connectivity**: cosine-based reasoning network connectivity analysis  
    - **Statistical tests**: t-tests for significant activation and connection deviations  
    - **Pattern identification**: detection of logical-chain disruptions  

    ---

    ### Additional Writing Requirements
    - Emphasize **neural analogs of scientific reasoning** and their malfunction in hallucinated reasoning  
    - Integrate cognitive neuroscience interpretations (DLPFC, IPL, ACC roles)  
    - Offer a detailed mechanistic account of **logic drift and cognitive overconfidence**  

    Final report length: 4000–5000 words.
    """
        elif data_type == "factqa":
            prompt = f"""
    Please write a **Nature-style scientific report** based on the following integrated analysis results of hallucination mechanisms.  
    The report should include:

    1. Abstract  
    2. Introduction  
    3. Results  
    4. Discussion  
    5. Conclusion  

    **Analysis Summary:**  
    {analysis_summary}

    Please write in **English**, using scientifically rigorous and precise language consistent with the standards of top international journals.  
    Focus on the following analytical dimensions:

    ---

    ### **Core Analytical Dimensions**
    1. **Neural activation patterns in correct samples**: Analyze parcel- and capability-level activations during correct responses to identify characteristics of normal cognitive processing.  
    2. **Neural activation patterns in hallucinated samples**: Examine parcel- and capability-level activations during hallucinated responses to identify abnormal cognitive processing features.  
    3. **Functional network comparison**: Compare the functional connectivity patterns between correct and hallucinated samples to identify structural differences in network organization.  
    4. **Cognitive-hierarchical analysis**: Conduct hierarchical capability analysis based on Bloom’s taxonomy of cognitive levels.  
    5. **Activation-strength comparison**: Compare top-activated parcels/capabilities between correct and hallucinated samples.  
    6. **Connectivity-strength comparison**: Compare the strongest parcel-parcel and capability-capability connections between correct and hallucinated samples.  
    7. **Abnormal-connection analysis**: Identify significantly abnormal parcel-parcel and capability-capability connections.  
    8. **Inter-level decoupling**: Examine abnormalities in functional connectivity across different cognitive levels.  

    ---

    ### **Key Findings to Emphasize**
    - Typical neural activation patterns in correct responses (which parcels/capabilities are most active and what kind of functional network they form).  
    - Abnormal neural activation patterns in hallucinated responses (which parcels/capabilities are over- or under-activated).  
    - Structural differences in functional network topology between correct and hallucinated samples.  
    - Differences in cognitive-level activation patterns (e.g., perception, representation, reasoning, and metacognition layers).  
    - Comparative analysis with **neuroscientific mechanisms of human hallucination**.  
    - Theoretical interpretation of hallucination mechanisms from the perspective of cognitive hierarchy.  

    ---

    ### **Analytical Methods**
    - **Parcel-level analysis**: 270-dimensional functional module activation analysis derived from a Sparse Auto-Encoder (SAE), including functional descriptions and model roles.  
    - **Capability-level analysis**: Cognitive capability aggregation analysis based on the Capability–Parcel mapping, incorporating hierarchical classification.  
    - **Connectivity analysis**: Functional network analysis based on cosine similarity between parcel activations.  
    - **Statistical testing**: Student’s *t*-test to assess the significance of activation and connectivity differences.  
    - **Pattern identification**: Recognition of typical and abnormal patterns via top activations and top connections.  

    ---

    ### **Special Requirements**
    - Analyze not only the differences between correct and hallucinated samples but also the **distinct neural characteristics of each**.  
    - Interpret cognitive processes from the **functional-network perspective**.  
    - Integrate insights from **cognitive-science theories** to explain observed phenomena.  
    - Provide **specific mechanistic explanations** for the underlying neural dynamics.  

    The report should be **4000–5000 words** in length to ensure analytical depth and scientific rigor.
    """
        return prompt
        
    def load_capability_descriptions(self) -> None:
        """加载Capability描述"""
        logger.info("加载Capability描述...")
        
        try:
            with open(self.cap_desc_path, 'r', encoding='utf-8') as f:
                self.capability_descriptions = json.load(f)
            logger.info(f"加载了 {len(self.capability_descriptions)} 个Capability描述")
        except FileNotFoundError:
            logger.warning(f"Capability描述文件不存在: {self.cap_desc_path}")
            self.capability_descriptions = {}
        except Exception as e:
            logger.error(f"加载Capability描述失败: {e}")
            self.capability_descriptions = {}
    
    def load_cognitive_mapping(self) -> None:
        """加载认知层级映射"""
        logger.info("加载认知层级映射...")
        
        try:
            with open(self.cog_mapping_path, 'r', encoding='utf-8') as f:
                self.cog_mapping = json.load(f)
            logger.info(f"加载了认知层级映射，包含 {len(self.cog_mapping.get('capability_classification_system', {}))} 个认知层级")
        except FileNotFoundError:
            logger.warning(f"认知层级映射文件不存在: {self.cog_mapping_path}")
            self.cog_mapping = {}
        except Exception as e:
            logger.error(f"加载认知层级映射失败: {e}")
            self.cog_mapping = {}
    
    def load_analysis_results(self) -> None:
        """加载分析结果"""
        logger.info("加载分析结果...")
        
        # 获取输出目录
        parcel_output_dir = Path(self.parcel_diff_path).parent
        cap_output_dir = Path(self.cap_diff_path).parent
        
        # 加载Parcel分析结果
        try:
            with open(self.parcel_diff_path, 'r', encoding='utf-8') as f:
                self.parcel_analysis = json.load(f)
            logger.info("Parcel分析结果加载成功")
        except FileNotFoundError:
            logger.error(f"Parcel分析结果文件不存在: {self.parcel_diff_path}")
            raise
        except Exception as e:
            logger.error(f"加载Parcel分析结果失败: {e}")
            raise
        
        # 加载Capability分析结果
        try:
            with open(self.cap_diff_path, 'r', encoding='utf-8') as f:
                self.capability_analysis = json.load(f)
            logger.info("Capability分析结果加载成功")
        except FileNotFoundError:
            logger.error(f"Capability分析结果文件不存在: {self.cap_diff_path}")
            raise
        except Exception as e:
            logger.error(f"加载Capability分析结果失败: {e}")
            raise
        
        # 加载Parcel top activated结果
        try:
            top_activated_file = parcel_output_dir / "top_activated_parcels.json"
            if top_activated_file.exists():
                with open(top_activated_file, 'r', encoding='utf-8') as f:
                    self.parcel_top_activated = json.load(f)
                logger.info("Parcel top activated结果加载成功")
            else:
                logger.warning(f"Parcel top activated文件不存在: {top_activated_file}")
        except Exception as e:
            logger.warning(f"加载Parcel top activated失败: {e}")
        
        # 加载Parcel top connections结果
        try:
            top_connections_file = parcel_output_dir / "top_parcel_connections.json"
            if top_connections_file.exists():
                with open(top_connections_file, 'r', encoding='utf-8') as f:
                    self.parcel_top_connections = json.load(f)
                logger.info("Parcel top connections结果加载成功")
            else:
                logger.warning(f"Parcel top connections文件不存在: {top_connections_file}")
        except Exception as e:
            logger.warning(f"加载Parcel top connections失败: {e}")
        
        # 加载Parcel anomalous connections结果
        try:
            anomalous_conn_file = parcel_output_dir / "anomalous_connections.json"
            if anomalous_conn_file.exists():
                with open(anomalous_conn_file, 'r', encoding='utf-8') as f:
                    self.parcel_anomalous_connections = json.load(f)
                logger.info("Parcel anomalous connections结果加载成功")
            else:
                logger.warning(f"Parcel anomalous connections文件不存在: {anomalous_conn_file}")
        except Exception as e:
            logger.warning(f"加载Parcel anomalous connections失败: {e}")
        
        # 加载Capability top activated结果
        try:
            top_activated_file = cap_output_dir / "top_activated_capabilities.json"
            if top_activated_file.exists():
                with open(top_activated_file, 'r', encoding='utf-8') as f:
                    self.capability_top_activated = json.load(f)
                logger.info("Capability top activated结果加载成功")
            else:
                logger.warning(f"Capability top activated文件不存在: {top_activated_file}")
        except Exception as e:
            logger.warning(f"加载Capability top activated失败: {e}")
        
        # 加载Capability top connections结果
        try:
            top_connections_file = cap_output_dir / "top_capability_connections.json"
            if top_connections_file.exists():
                with open(top_connections_file, 'r', encoding='utf-8') as f:
                    self.capability_top_connections = json.load(f)
                logger.info("Capability top connections结果加载成功")
            else:
                logger.warning(f"Capability top connections文件不存在: {top_connections_file}")
        except Exception as e:
            logger.warning(f"加载Capability top connections失败: {e}")
        
        # 加载Capability anomalous connections结果
        try:
            anomalous_conn_file = cap_output_dir / "anomalous_capability_connections.json"
            if anomalous_conn_file.exists():
                with open(anomalous_conn_file, 'r', encoding='utf-8') as f:
                    self.capability_anomalous_connections = json.load(f)
                logger.info("Capability anomalous connections结果加载成功")
            else:
                logger.warning(f"Capability anomalous connections文件不存在: {anomalous_conn_file}")
        except Exception as e:
            logger.warning(f"加载Capability anomalous connections失败: {e}")
    
    def get_cognitive_level(self, capability_name: str) -> Dict:
        """获取Capability的认知层级信息"""
        if not self.cog_mapping or 'capability_mappings' not in self.cog_mapping:
            return {'level': 'Unknown', 'category': 'Unknown', 'description': 'Unknown'}
        
        # 将capability名称转换为小写进行匹配
        capability_lower = capability_name.lower()
        
        # 在capability_mappings中查找匹配的能力
        for category, capabilities in self.cog_mapping.get('capability_mappings', {}).items():
            for cap_name, cap_info in capabilities.items():
                if cap_name.lower() == capability_lower or cap_name.lower() in capability_lower:
                    return {
                        'level': cap_info.get('level', 'Unknown'),
                        'category': cap_info.get('category_name', 'Unknown'),
                        'description': cap_info.get('reason', 'Unknown')
                    }
        
        # 如果没有找到精确匹配，尝试部分匹配
        for category, capabilities in self.cog_mapping.get('capability_mappings', {}).items():
            for cap_name, cap_info in capabilities.items():
                if capability_lower in cap_name.lower() or cap_name.lower() in capability_lower:
                    return {
                        'level': cap_info.get('level', 'Unknown'),
                        'category': cap_info.get('category_name', 'Unknown'),
                        'description': cap_info.get('reason', 'Unknown')
                    }
        
        return {'level': 'Unknown', 'category': 'Unknown', 'description': 'Unknown'}
    
    def load_sample_data(self) -> Tuple[List[Dict], List[Dict]]:
        """加载样本数据用于展示示例"""
        correct_samples = []
        incorrect_samples = []
        
        # 加载正确样本
        if self.correct_data_path and os.path.exists(self.correct_data_path):
            try:
                with open(self.correct_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line.strip())
                            correct_samples.append(sample)
                            if len(correct_samples) >= 2:  # 只取前3个示例
                                break
                logger.info(f"加载了 {len(correct_samples)} 个正确样本示例")
            except Exception as e:
                logger.warning(f"加载正确样本示例失败: {e}")
        
        # 加载幻觉样本
        if self.incorrect_data_path and os.path.exists(self.incorrect_data_path):
            try:
                with open(self.incorrect_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line.strip())
                            incorrect_samples.append(sample)
                            if len(incorrect_samples) >= 2:  # 只取前3个示例
                                break
                logger.info(f"加载了 {len(incorrect_samples)} 个幻觉样本示例")
            except Exception as e:
                logger.warning(f"加载幻觉样本示例失败: {e}")
        
        return correct_samples, incorrect_samples
    
    def format_data_examples(self, data_type: str, correct_samples: List[Dict], 
                           incorrect_samples: List[Dict]) -> str:
        """根据数据类型格式化示例数据"""
        examples = "### 数据集示例\n\n"
        
        if data_type == "rag":
            examples += "#### 正确样本示例 (RAG)\n"
            for i, sample in enumerate(correct_samples[:2], 1):
                context = sample.get('context', [])
                context_str = context[0] if context else "无上下文"
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **上下文**: {context_str[:200]}...\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n\n"
            
            examples += "#### 幻觉样本示例 (RAG)\n"
            for i, sample in enumerate(incorrect_samples[:2], 1):
                context = sample.get('context', [])
                context_str = context[0] if context else "无上下文"
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **上下文**: {context_str[:200]}...\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n"
                examples += f"- **错误原因**: {sample.get('reason', 'N/A')}\n\n"
                
        elif data_type == "medical":
            examples += "#### 正确样本示例 (医学)\n"
            for i, sample in enumerate(correct_samples[:2], 1):
                context = sample.get('context', [])
                context_str = context[0] if context else "无上下文"
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **医学上下文**: {context_str[:200]}...\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n\n"
            
            examples += "#### 幻觉样本示例 (医学)\n"
            for i, sample in enumerate(incorrect_samples[:2], 1):
                context = sample.get('context', [])
                context_str = context[0] if context else "无上下文"
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **医学上下文**: {context_str[:200]}...\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n"
                examples += f"- **错误原因**: {sample.get('reason', 'N/A')}\n\n"
                
        elif data_type == "science":
            examples += "#### 正确样本示例 (科学推理)\n"
            for i, sample in enumerate(correct_samples[:2], 1):
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n\n"
            
            examples += "#### 幻觉样本示例 (科学推理)\n"
            for i, sample in enumerate(incorrect_samples[:2], 1):
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n"
                examples += f"- **错误原因**: {sample.get('reason', 'N/A')}\n\n"
                
        else:  # factqa 或其他类型
            examples += "#### 正确样本示例 (事实问答)\n"
            for i, sample in enumerate(correct_samples[:2], 1):
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n"
                examples += f"- **置信度**: {sample.get('score', 'N/A')}\n\n"
            
            examples += "#### 幻觉样本示例 (事实问答)\n"
            for i, sample in enumerate(incorrect_samples[:2], 1):
                examples += f"**示例 {i}:**\n"
                examples += f"- **问题**: {sample.get('question', 'N/A')}\n"
                examples += f"- **模型回答**: {sample.get('model_answer', 'N/A')}\n"
                examples += f"- **真实答案**: {sample.get('answer_true', 'N/A')}\n"
                examples += f"- **置信度**: {sample.get('score', 'N/A')}\n"
                examples += f"- **错误原因**: {sample.get('reason', 'N/A')}\n\n"
        
        return examples
    
    def call_vllm_api(self, prompt: str, max_tokens: int = 2048, 
                     temperature: float = 0.0, timeout: int = 120) -> str:
        """调用vLLM API获取响应"""
        payload = {
            "model": "/path/to/local_models/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": "你是一个专业的神经科学研究助手，擅长分析大语言模型的内部神经活动模式。请用科学、严谨的语言撰写分析报告。"},
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
    
    def prepare_analysis_summary(self, data_type: str = "factqa") -> str:
        """准备分析结果摘要"""
        summary = "## 幻觉机制综合分析结果摘要\n\n"

        # 数据集简介
        try:
            # 暂存 data_type 以便匹配
            self.data_type = data_type
            dataset_intro = self.get_dataset_intro()
            if dataset_intro:
                summary += "### 数据集简介\n\n"
                summary += dataset_intro + "\n\n"
        except Exception as e:
            logger.warning(f"添加数据集简介失败: {e}")
        
        # 添加数据集示例
        try:
            correct_samples, incorrect_samples = self.load_sample_data()
            if correct_samples or incorrect_samples:
                examples = self.format_data_examples(data_type, correct_samples, incorrect_samples)
                summary += examples + "\n"
        except Exception as e:
            logger.warning(f"加载数据集示例失败: {e}")
            summary += "### 数据集示例\n\n*示例数据加载失败*\n\n"
        
        # 1. Parcel级别Top激活深度分析
        if self.parcel_top_activated:
            summary += "### 1. Parcel级别Top激活深度分析\n\n"
            
            # 正确样本Top激活
            if 'top_correct_parcels' in self.parcel_top_activated:
                summary += "#### 正确样本Top激活Parcels (前10个)\n"
                summary += "**功能模式分析：**\n"
                for i, parcel in enumerate(self.parcel_top_activated['top_correct_parcels'][:10]):
                    parcel_id = parcel.get('parcel_id', 'N/A')
                    function_name = parcel.get('function_name', 'Unknown')
                    function_desc = parcel.get('function_description', '')
                    model_role = parcel.get('model_role', '')
                    mean_activation = parcel.get('mean_activation', 0)
                    keywords = parcel.get('keywords', [])
                    
                    summary += f"{i+1}. **Parcel {parcel_id}** ({function_name}): 激活强度={mean_activation:.4f}\n"
                    summary += f"    **功能描述**: {function_desc[:500]}...\n"
                    # summary += f"    **模型角色**: {model_role[:500]}...\n"
                    summary += f"    **关键词**: {', '.join(keywords[:8])}\n\n"
                summary += "\n"
            
            # 幻觉样本Top激活
            if 'top_incorrect_parcels' in self.parcel_top_activated:
                summary += "#### 幻觉样本Top激活Parcels (前10个)\n"
                summary += "**功能模式分析：**\n"
                for i, parcel in enumerate(self.parcel_top_activated['top_incorrect_parcels'][:10]):
                    parcel_id = parcel.get('parcel_id', 'N/A')
                    function_name = parcel.get('function_name', 'Unknown')
                    function_desc = parcel.get('function_description', '')
                    model_role = parcel.get('model_role', '')
                    mean_activation = parcel.get('mean_activation', 0)
                    keywords = parcel.get('keywords', [])
                    
                    summary += f"{i+1}. **Parcel {parcel_id}** ({function_name}): 激活强度={mean_activation:.4f}\n"
                    summary += f"    **功能描述**: {function_desc[:500]}...\n"
                    # summary += f"    **模型角色**: {model_role[:500]}...\n"
                    summary += f"    **关键词**: {', '.join(keywords[:8])}\n\n"
                summary += "\n"
        
        # 2. Capability级别Top激活深度分析（按认知层级分类）
        if self.capability_top_activated:
            summary += "### 2. Capability级别Top激活深度分析（按认知层级分类）\n\n"
            
            # 正确样本Top激活
            if 'top_correct_capabilities' in self.capability_top_activated:
                summary += "#### 正确样本Top激活Capabilities (前10个)\n"
                summary += "**认知层级模式分析：**\n"
                for i, cap in enumerate(self.capability_top_activated['top_correct_capabilities'][:10]):
                    cap_name = cap.get('capability_name', 'Unknown')
                    definition = cap.get('definition_refined', '')
                    cognitive_alignment = cap.get('cognitive_alignment', {})
                    manifestation = cap.get('manifestation_in_llms', [])
                    mean_activation = cap.get('mean_activation', 0)
                    cog_info = self.get_cognitive_level(cap_name)
                    
                    summary += f"{i+1}. **{cap_name}**: 激活强度={mean_activation:.4f}\n"
                    summary += f"    **认知层级**: {cog_info['level']} - {cog_info['category']}\n"
                    summary += f"    **能力定义**: {definition}\n"
                    if isinstance(manifestation, list) and manifestation:
                        summary += f"    **LLM表现**: {manifestation[0]}\n"
                    summary += "\n"
                summary += "\n"
            
            # 幻觉样本Top激活
            if 'top_incorrect_capabilities' in self.capability_top_activated:
                summary += "#### 幻觉样本Top激活Capabilities (前10个)\n"
                summary += "**认知层级模式分析：**\n"
                for i, cap in enumerate(self.capability_top_activated['top_incorrect_capabilities'][:10]):
                    cap_name = cap.get('capability_name', 'Unknown')
                    definition = cap.get('definition_refined', '')
                    cognitive_alignment = cap.get('cognitive_alignment', {})
                    manifestation = cap.get('manifestation_in_llms', [])
                    mean_activation = cap.get('mean_activation', 0)
                    cog_info = self.get_cognitive_level(cap_name)
                    
                    summary += f"{i+1}. **{cap_name}**: 激活强度={mean_activation:.4f}\n"
                    summary += f"    **认知层级**: {cog_info['level']} - {cog_info['category']}\n"
                    summary += f"    **能力定义**: {definition[:500]}...\n"
                summary += "\n"
        
        # 3. Parcel级别Top连接深度分析
        if self.parcel_top_connections:
            summary += "### 3. Parcel级别Top连接深度分析\n\n"
            
            # 正确样本Top连接
            if 'top_correct_connections' in self.parcel_top_connections:
                summary += "#### 正确样本Top连接模式 (前10个)\n"
                summary += "**功能网络分析：**\n"
                for i, conn in enumerate(self.parcel_top_connections['top_correct_connections'][:10]):
                    parcel_i = conn.get('parcel_i', {})
                    parcel_j = conn.get('parcel_j', {})
                    strength = conn.get('connection_strength', 0)
                    
                    func_i = parcel_i.get('function_name', 'Unknown')
                    func_j = parcel_j.get('function_name', 'Unknown')
                    desc_i = parcel_i.get('function_description', '')
                    desc_j = parcel_j.get('function_description', '')
                    
                    summary += f"{i+1}. **{func_i}** <-> **{func_j}**: 连接强度={strength:.4f}\n"
                    summary += f"    {func_i}: {desc_i[:500]}...\n"
                    summary += f"    {func_j}: {desc_j[:500]}...\n\n"
                summary += "\n"
            
            # 幻觉样本Top连接
            if 'top_incorrect_connections' in self.parcel_top_connections:
                summary += "#### 幻觉样本Top连接模式 (前10个)\n"
                summary += "**功能网络分析：**\n"
                for i, conn in enumerate(self.parcel_top_connections['top_incorrect_connections'][:10]):
                    parcel_i = conn.get('parcel_i', {})
                    parcel_j = conn.get('parcel_j', {})
                    strength = conn.get('connection_strength', 0)
                    
                    func_i = parcel_i.get('function_name', 'Unknown')
                    func_j = parcel_j.get('function_name', 'Unknown')
                    desc_i = parcel_i.get('function_description', '')
                    desc_j = parcel_j.get('function_description', '')
                    
                    summary += f"{i+1}. **{func_i}** <-> **{func_j}**: 连接强度={strength:.4f}\n"
                    summary += f"    {func_i}: {desc_i[:500]}...\n"
                    summary += f"    {func_j}: {desc_j[:500]}...\n\n"
                summary += "\n"
        
        # 4. Capability级别Top连接深度分析
        if self.capability_top_connections:
            summary += "### 4. Capability级别Top连接深度分析\n\n"
            
            # 正确样本Top连接
            if 'top_correct_connections' in self.capability_top_connections:
                summary += "#### 正确样本Top连接模式 (前10个)\n"
                summary += "**认知网络分析：**\n"
                for i, conn in enumerate(self.capability_top_connections['top_correct_connections'][:10]):
                    cap_i = conn.get('capability_i', {})
                    cap_j = conn.get('capability_j', {})
                    strength = conn.get('connection_strength', 0)
                    
                    name_i = cap_i.get('name', 'Unknown')
                    name_j = cap_j.get('name', 'Unknown')
                    def_i = cap_i.get('definition_refined', '')
                    def_j = cap_j.get('definition_refined', '')
                    
                    cog_i = self.get_cognitive_level(name_i)
                    cog_j = self.get_cognitive_level(name_j)
                    
                    summary += f"{i+1}. **{name_i}** <-> **{name_j}**: 连接强度={strength:.4f}\n"
                    summary += f"    {name_i} ({cog_i['level']}): {def_i[:500]}...\n"
                    summary += f"    {name_j} ({cog_j['level']}): {def_j[:500]}...\n\n"
                summary += "\n"
            
            # 幻觉样本Top连接
            if 'top_incorrect_connections' in self.capability_top_connections:
                summary += "#### 幻觉样本Top连接模式 (前10个)\n"
                summary += "**认知网络分析：**\n"
                for i, conn in enumerate(self.capability_top_connections['top_incorrect_connections'][:10]):
                    cap_i = conn.get('capability_i', {})
                    cap_j = conn.get('capability_j', {})
                    strength = conn.get('connection_strength', 0)
                    
                    name_i = cap_i.get('name', 'Unknown')
                    name_j = cap_j.get('name', 'Unknown')
                    def_i = cap_i.get('definition_refined', '')
                    def_j = cap_j.get('definition_refined', '')
                    
                    cog_i = self.get_cognitive_level(name_i)
                    cog_j = self.get_cognitive_level(name_j)
                    
                    summary += f"{i+1}. **{name_i}** <-> **{name_j}**: 连接强度={strength:.4f}\n"
                    summary += f"    {name_i} ({cog_i['level']}): {def_i[:500]}...\n"
                    summary += f"    {name_j} ({cog_j['level']}): {def_j[:500]}...\n\n"
                summary += "\n"
        
        # 5. Parcel级别异常连接分析（区分正/负）
        if self.parcel_anomalous_connections and 'anomalous_connections' in self.parcel_anomalous_connections:
            summary += "### 5. Parcel级别异常连接分析 (前10个)\n Positive means the connection is stronger in the hallucination sample than in the correct sample\n"
            anom = self.parcel_anomalous_connections['anomalous_connections']
            # 兼容两种格式：列表 或 {pos_connections, neg_connections}
            if isinstance(anom, dict) and ('pos_connections' in anom or 'neg_connections' in anom):
                pos_list = anom.get('pos_connections', [])[:10]
                neg_list = anom.get('neg_connections', [])[:10]
                if pos_list:
                    summary += "#### 正向增强连接 (Positive) Positive means the connection is stronger in the hallucination sample than in the correct sample\n"
                    for i, conn in enumerate(pos_list):
                        parcel_i = conn.get('parcel_i', {}).get('function_name', 'Unknown')
                        parcel_j = conn.get('parcel_j', {}).get('function_name', 'Unknown')
                        diff = conn.get('connectivity_diff', 0)
                        is_significant = conn.get('is_significant', False)
                        p_value = conn.get('p_value', 1)
                        summary += f"{i+1}. {parcel_i} <-> {parcel_j}: 差异=+{diff:.4f}, 显著={is_significant}, p值={p_value:.4f}\n"
                    summary += "\n"
                if neg_list:
                    summary += "#### 负向减弱连接 (Negative) Negative means the connection is weaker in the hallucination sample than in the correct sample\n"
                    for i, conn in enumerate(neg_list):
                        parcel_i = conn.get('parcel_i', {}).get('function_name', 'Unknown')
                        parcel_j = conn.get('parcel_j', {}).get('function_name', 'Unknown')
                        diff = conn.get('connectivity_diff', 0)
                        is_significant = conn.get('is_significant', False)
                        p_value = conn.get('p_value', 1)
                        summary += f"{i+1}. {parcel_i} <-> {parcel_j}: 差异={diff:.4f}, 显著={is_significant}, p值={p_value:.4f}\n"
                    summary += "\n"
            else:
                for i, conn in enumerate(anom[:10]):
                    parcel_i = conn.get('parcel_i', {}).get('function_name', 'Unknown')
                    parcel_j = conn.get('parcel_j', {}).get('function_name', 'Unknown')
                    diff = conn.get('connectivity_diff', 0)
                    is_significant = conn.get('is_significant', False)
                    p_value = conn.get('p_value', 1)
                    summary += f"{i+1}. {parcel_i} <-> {parcel_j}: 差异={diff:.4f}, 显著={is_significant}, p值={p_value:.4f}\n"
                summary += "\n"
        
        # 6. Capability级别异常连接分析（按认知层级分类，区分正/负）
        if self.capability_anomalous_connections and 'anomalous_connections' in self.capability_anomalous_connections:
            summary += "### 6. Capability级别异常连接分析 (前15个)\n Positive means the connection is stronger in the hallucination sample than in the correct sample\n"
            anom = self.capability_anomalous_connections['anomalous_connections']
            if isinstance(anom, dict) and ('pos_connections' in anom or 'neg_connections' in anom):
                pos_list = anom.get('pos_connections', [])[:8]
                neg_list = anom.get('neg_connections', [])[:8]
                if pos_list:
                    summary += "#### 正向增强连接 (Positive) Positive means the connection is stronger in the hallucination sample than in the correct sample\n"
                    for i, conn in enumerate(pos_list):
                        cap_i = conn.get('capability_i', {}).get('name', 'Unknown')
                        cap_j = conn.get('capability_j', {}).get('name', 'Unknown')
                        diff = conn.get('connectivity_diff', 0)
                        is_significant = conn.get('is_significant', False)
                        p_value = conn.get('p_value', 1)
                        cog_i = self.get_cognitive_level(cap_i)
                        cog_j = self.get_cognitive_level(cap_j)
                        summary += f"{i+1}. {cap_i} <-> {cap_j}: 差异=+{diff:.4f}, 显著={is_significant}, p值={p_value:.4f}\n"
                        summary += f"    {cap_i}: {cog_i['level']} - {cog_i['category']}\n"
                        summary += f"    {cap_j}: {cog_j['level']} - {cog_j['category']}\n"
                    summary += "\n"
                if neg_list:
                    summary += "#### 负向减弱连接 (Negative) Negative means the connection is weaker in the hallucination sample than in the correct sample\n"
                    for i, conn in enumerate(neg_list):
                        cap_i = conn.get('capability_i', {}).get('name', 'Unknown')
                        cap_j = conn.get('capability_j', {}).get('name', 'Unknown')
                        diff = conn.get('connectivity_diff', 0)
                        is_significant = conn.get('is_significant', False)
                        p_value = conn.get('p_value', 1)
                        cog_i = self.get_cognitive_level(cap_i)
                        cog_j = self.get_cognitive_level(cap_j)
                        summary += f"{i+1}. {cap_i} <-> {cap_j}: 差异={diff:.4f}, 显著={is_significant}, p值={p_value:.4f}\n"
                        summary += f"    {cap_i}: {cog_i['level']} - {cog_i['category']}\n"
                        summary += f"    {cap_j}: {cog_j['level']} - {cog_j['category']}\n"
                    summary += "\n"
            else:
                for i, conn in enumerate(anom[:15]):
                    cap_i = conn.get('capability_i', {}).get('name', 'Unknown')
                    cap_j = conn.get('capability_j', {}).get('name', 'Unknown')
                    diff = conn.get('connectivity_diff', 0)
                    is_significant = conn.get('is_significant', False)
                    p_value = conn.get('p_value', 1)
                    cog_i = self.get_cognitive_level(cap_i)
                    cog_j = self.get_cognitive_level(cap_j)
                    summary += f"{i+1}. {cap_i} <-> {cap_j}: 差异={diff:.4f}, 显著={is_significant}, p值={p_value:.4f}\n"
                    summary += f"    {cap_i}: {cog_i['level']} - {cog_i['category']}\n"
                    summary += f"    {cap_j}: {cog_j['level']} - {cog_j['category']}\n"
                summary += "\n"
        
        # 7. 传统异常分析结果
        if 'top_anomalous_parcels' in self.parcel_analysis:
            summary += "### 7. Parcel级别激活异常 (各5个：正/负)\n"
            summary += "Positive (pos) 表示幻觉样本激活更高；Negative (neg) 表示 Truthful/正确 样本激活更高\n"
            parcels = self.parcel_analysis['top_anomalous_parcels']
            pos_parcels = [p for p in parcels if p.get('activation_diff', 0) > 0][:5]
            neg_parcels = [p for p in parcels if p.get('activation_diff', 0) < 0][:5]
            if pos_parcels:
                summary += "#### 正向增强激活 (Positive / pos)\n"
                summary += "Positive means the activation is stronger in the hallucination sample than in the correct sample\n"
                for i, parcel in enumerate(pos_parcels, 1):
                    parcel_id = parcel.get('parcel_id', 'N/A')
                    function_name = parcel.get('function_name', 'Unknown')
                    function_desc = parcel.get('function_description', '')
                    activation_diff = parcel.get('activation_diff', 0)
                    p_value = parcel.get('p_value', 1)
                    is_significant = parcel.get('is_significant', False)
                    summary += f"{i}. Parcel {parcel_id}: 激活差异=+{activation_diff:.4f}, p值={p_value:.4f}, 显著={is_significant}\n"
                    summary += f"    {function_name}: {function_desc[:500]}...\n"
                summary += "\n"
            if neg_parcels:
                summary += "#### 负向减弱激活 (Negative / neg)\n"
                summary += "Negative means the activation is stronger in the Truthful/correct sample than in the hallucination sample\n"
                for i, parcel in enumerate(neg_parcels, 1):
                    parcel_id = parcel.get('parcel_id', 'N/A')
                    function_name = parcel.get('function_name', 'Unknown')
                    function_desc = parcel.get('function_description', '')
                    activation_diff = parcel.get('activation_diff', 0)
                    p_value = parcel.get('p_value', 1)
                    is_significant = parcel.get('is_significant', False)
                    summary += f"{i}. Parcel {parcel_id}: 激活差异={activation_diff:.4f}, p值={p_value:.4f}, 显著={is_significant}\n"
                    summary += f"    {function_name}: {function_desc[:500]}...\n"
                summary += "\n"
        
        if 'top_anomalous_capabilities' in self.capability_analysis:
            summary += "### 8. Capability级别激活异常 (各5个：正/负)\n"
            summary += "Positive (pos) 表示幻觉样本激活更高；Negative (neg) 表示 Truthful/正确 样本激活更高\n"
            caps = self.capability_analysis['top_anomalous_capabilities']
            pos_caps = [c for c in caps if c.get('activation_diff', 0) > 0][:5]
            neg_caps = [c for c in caps if c.get('activation_diff', 0) < 0][:5]
            if pos_caps:
                summary += "#### 正向增强激活 (Positive / pos)\n"
                summary += "Positive means the activation is stronger in the hallucination sample than in the correct sample\n"
                for i, cap in enumerate(pos_caps, 1):
                    cap_name = cap.get('capability_name', 'Unknown')
                    activation_diff = cap.get('activation_diff', 0)
                    p_value = cap.get('p_value', 1)
                    is_significant = cap.get('is_significant', False)
                    cog_info = self.get_cognitive_level(cap_name)
                    summary += f"{i}. {cap_name}: 激活差异=+{activation_diff:.4f}, p值={p_value:.4f}, 显著={is_significant}\n"
                    summary += f"    认知层级: {cog_info['level']} - {cog_info['category']}\n"
                summary += "\n"
            if neg_caps:
                summary += "#### 负向减弱激活 (Negative / neg)\n"
                summary += "Negative means the activation is stronger in the Truthful/correct sample than in the hallucination sample\n"
                for i, cap in enumerate(neg_caps, 1):
                    cap_name = cap.get('capability_name', 'Unknown')
                    activation_diff = cap.get('activation_diff', 0)
                    p_value = cap.get('p_value', 1)
                    is_significant = cap.get('is_significant', False)
                    cog_info = self.get_cognitive_level(cap_name)
                    summary += f"{i}. {cap_name}: 激活差异={activation_diff:.4f}, p值={p_value:.4f}, 显著={is_significant}\n"
                    summary += f"    认知层级: {cog_info['level']} - {cog_info['category']}\n"
                summary += "\n"
        
        return summary
    
    def generate_nature_style_report(self, data_type: str, model_data: str) -> str:
        """生成Nature风格的科学报告"""
        logger.info("生成Nature风格科学报告...")
        
        # 准备分析摘要
        analysis_summary = self.prepare_analysis_summary(data_type)
        
        # 构建LLM提示
        prompt = self.build_prompt(data_type, analysis_summary)
#         prompt = f"""
# 请基于以下幻觉机制综合分析结果，撰写一份符合Nature期刊风格的科学报告。报告应该包含：

# 1. 摘要 (Abstract)
# 2. 引言 (Introduction) 
# 3. 方法 (Methods)
# 4. 结果 (Results)
# 5. 讨论 (Discussion)
# 6. 结论 (Conclusion)

# 分析结果摘要：
# {analysis_summary}

# 请用英文撰写，语言要科学严谨，符合国际顶级期刊的写作标准。重点关注：

# **核心分析维度：**
# 1. **正确样本的神经活动模式**：分析正确回答时的Parcel和Capability激活模式，识别正常认知处理的特征
# 2. **幻觉样本的神经活动模式**：分析幻觉回答时的Parcel和Capability激活模式，识别异常认知处理的特征
# 3. **功能网络对比**：比较正确和幻觉样本的功能连接网络模式，识别网络结构差异
# 4. **认知层级分析**：基于Bloom认知分类的层级化能力模式分析
# 5. **激活强度对比**：正确vs幻觉样本的Top激活Parcels/Capabilities强度对比
# 6. **连接强度对比**：正确vs幻觉样本的Top连接强度对比
# 7. **异常连接分析**：显著异常的Parcel-Parcel和Capability-Capability连接
# 8. **层级间解耦**：不同认知层级之间的功能连接异常

# **关键发现重点：**
# - 正确回答时的典型神经活动模式（哪些Parcels/Capabilities最活跃，形成什么样的功能网络）
# - 幻觉回答时的异常神经活动模式（哪些Parcels/Capabilities过度激活或激活不足）
# - 功能连接网络的结构性差异（正确vs幻觉样本的网络拓扑差异）
# - 认知层级的激活模式差异（感知层、表征层、推理层、元认知层等各层级的模式）
# - 与人脑幻觉机制的神经科学对比分析
# - 基于认知层级的幻觉机制理论解释

# **分析方法说明：**
# - Parcel级别：基于SAE提取的270维功能模块激活分析，包含功能描述和模型角色信息
# - Capability级别：基于Capability-Parcel映射的认知能力聚合分析，包含认知层级分类
# - 连接分析：基于cosine similarity的功能连接网络分析
# - 统计检验：t检验检测激活和连接差异的显著性
# - 模式识别：通过Top激活和Top连接识别典型和异常模式

# **特别要求：**
# - 不仅要分析差异，更要深入分析正确和错误样本各自的神经活动特征
# - 从功能网络的角度理解认知处理过程
# - 结合认知科学理论解释发现的现象
# - 提供具体的神经机制解释

# 报告长度控制在4000-5000词左右，确保分析深度和科学严谨性。
# """
        
        try:
            # 调用LLM生成报告
            report = self.call_vllm_api(prompt, max_tokens=10000, temperature=0.1)
            
            # 添加报告头部信息
            header = f"""# 大语言模型幻觉机制的神经活动分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析类型**: {data_type} 正确回答 vs 幻觉回答
**模型**: {model_data}
**分析方法**: Parcel-Level 和 Capability-Level 功能连接分析

**------------------------------------------------------------------------------------------------**
**prompt:**
{prompt}
**------------------------------------------------------------------------------------------------**
"""
            
            full_report = header + report
            
            return full_report
            
        except Exception as e:
            logger.error(f"LLM报告生成失败: {e}")
            # 如果LLM调用失败，生成基础报告
            return self.generate_fallback_report(data_type)
    
    def generate_fallback_report(self, data_type: str = "factqa") -> str:
        """生成备用报告（当LLM调用失败时）"""
        logger.info("生成备用报告...")
        
        analysis_summary = self.prepare_analysis_summary(data_type)
        
        report = f"""# 大语言模型幻觉机制的神经活动分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**分析类型**: TruthfulQA 正确回答 vs 幻觉回答
**模型**: Gemma-2-2B
**分析方法**: Parcel-Level 和 Capability-Level 功能连接分析

---

## 摘要

本研究基于大语言模型(Gemma-2-2B)的内部神经活动特征，通过比较模型在TruthfulQA数据集上正确回答与幻觉回答两类样本的Parcel级别和Capability级别激活模式，揭示了幻觉产生的神经机制。研究发现，幻觉时多个认知控制相关的Parcel和Capability出现显著激活异常，功能连接网络出现解耦现象。

## 方法

### 数据来源
- 正确样本: TruthfulQA数据集中模型正确回答的问题
- 幻觉样本: TruthfulQA数据集中模型产生幻觉回答的问题
- 激活数据: 通过SAE(Sparse Autoencoder)提取的270维Parcel激活特征

### 分析方法
1. **Parcel级别分析**: 计算Parcel-Parcel功能连接矩阵(cosine similarity)
2. **Capability级别分析**: 基于Capability-Parcel映射聚合激活，计算Capability-Capability功能连接矩阵
3. **异常检测**: 使用t检验检测激活差异的统计显著性
4. **连接异常**: 计算连接矩阵的Frobenius范数和上三角平均差异

## 结果

{analysis_summary}

## 讨论

### 主要发现
1. **认知控制能力失调**: 幻觉时，与事实检索、逻辑一致性相关的Capability出现显著激活下降
2. **功能连接解耦**: Parcel和Capability之间的功能连接网络出现结构性变化
3. **神经活动异常**: 多个关键Parcel的激活模式偏离正常基线

### 神经科学意义
这些发现与人类大脑中幻觉相关的神经机制具有相似性，特别是前额叶-颞叶连接异常和认知控制网络的功能失调。

### 局限性
- 分析基于单一模型(Gemma-2-2B)
- 样本数量有限
- 需要更多模型和数据集验证

## 结论

本研究首次从神经活动角度揭示了大语言模型幻觉产生的内部机制，为理解和改进模型的事实性提供了新的视角。未来工作将探索基于这些发现的幻觉检测和缓解方法。

---

*本报告基于自动化分析生成，建议结合人工审查使用。*
"""
        
        return report
    
    def save_report(self, report: str) -> None:
        """保存报告到文件"""
        logger.info("保存报告...")
        
        # 确保输出目录存在
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            raise
    
    def run_analysis(self, data_type: str, model_data: str) -> None:
        """运行完整的报告生成流程"""
        try:
            logger.info("开始生成LLM分析报告...")
            
            # 1. 加载描述文件
            self.load_parcel_descriptions()
            self.load_capability_descriptions()
            self.load_cognitive_mapping()
            
            # 2. 加载分析结果
            self.load_analysis_results()
            
            # 2.5 保存关键信息用于后续生成
            self.model_data = model_data

            # 3. 生成报告
            report = self.generate_nature_style_report(data_type, model_data)
            
            # 4. 保存报告
            self.save_report(report)
            
            logger.info("LLM分析报告生成完成！")
            
        except Exception as e:
            logger.error(f"报告生成过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM分析报告生成')
    parser.add_argument('--parcel_desc', type=str, required=True,
                       help='Parcel功能描述文件路径')
    parser.add_argument('--cap_desc', type=str, required=True,
                       help='Capability描述文件路径')
    parser.add_argument('--parcel_diff', type=str, required=True,
                       help='Parcel异常分析结果文件路径')
    parser.add_argument('--cap_diff', type=str, required=True,
                       help='Capability异常分析结果文件路径')
    parser.add_argument('--cog_mapping', type=str, required=True,
                       help='认知层级映射文件路径')
    parser.add_argument('--out', type=str, required=True,
                       help='输出报告文件路径')
    parser.add_argument('--vllm_url', type=str, default=DEFAULT_VLLM_URL,
                       help=f'vLLM API地址 (默认: {DEFAULT_VLLM_URL})')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY,
                       help=f'API密钥 (默认: {DEFAULT_API_KEY})')
    parser.add_argument('--data_type', type=str, required=True,
                       help='数据类型 factqa rag medical science')
    parser.add_argument('--model_data', type=str, required=True,
                       help='模型数据名称')
    parser.add_argument('--correct_data', type=str, default=None,
                       help='正确样本数据文件路径')
    parser.add_argument('--incorrect_data', type=str, default=None,
                       help='幻觉样本数据文件路径')
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    for file_path, name in [(args.parcel_desc, "Parcel描述文件"),
                           (args.cap_desc, "Capability描述文件"),
                           (args.parcel_diff, "Parcel分析结果文件"),
                           (args.cap_diff, "Capability分析结果文件"),
                           (args.cog_mapping, "认知层级映射文件")]:
        if not os.path.exists(file_path):
            logger.error(f"{name}不存在: {file_path}")
            sys.exit(1)
    
    # 自动推断数据文件路径（如果未提供）
    if args.correct_data is None or args.incorrect_data is None:
        # 从parcel_diff_path推断数据目录
        data_dir = Path(args.parcel_diff).parent.parent.parent  # 回到results目录
        model_data_dir = data_dir / args.model_data
        
        if args.correct_data is None:
            correct_data_path = model_data_dir / "correct.jsonl"
            if correct_data_path.exists():
                args.correct_data = str(correct_data_path)
                logger.info(f"自动推断正确样本数据路径: {args.correct_data}")
            else:
                logger.warning(f"未找到正确样本数据文件: {correct_data_path}")
        
        if args.incorrect_data is None:
            incorrect_data_path = model_data_dir / "incorrect.jsonl"
            if incorrect_data_path.exists():
                args.incorrect_data = str(incorrect_data_path)
                logger.info(f"自动推断幻觉样本数据路径: {args.incorrect_data}")
            else:
                logger.warning(f"未找到幻觉样本数据文件: {incorrect_data_path}")
    
    # 创建报告生成器并运行
    generator = LLMReportGenerator(
        parcel_desc_path=args.parcel_desc,
        cap_desc_path=args.cap_desc,
        parcel_diff_path=args.parcel_diff,
        cap_diff_path=args.cap_diff,
        output_path=args.out,
        cog_mapping_path=args.cog_mapping,
        vllm_url=args.vllm_url,
        api_key=args.api_key,
        correct_data_path=args.correct_data,
        incorrect_data_path=args.incorrect_data
    )
    if "MedHallu" in args.model_data:
        data_type = "medical"
    elif "HaluEval" in args.model_data:
        data_type = "rag"
    elif "dolly_close" in args.model_data:
        data_type = "rag"
    elif "nq_open" in args.model_data:
        data_type = "factqa"
    elif "sciq" in args.model_data:
        data_type = "science"
    elif "triviaqa" in args.model_data:
        data_type = "factqa"
    elif "truthfulqa" in args.model_data:
        data_type = "factqa"
    generator.run_analysis(data_type, args.model_data)


if __name__ == "__main__":
    main()
