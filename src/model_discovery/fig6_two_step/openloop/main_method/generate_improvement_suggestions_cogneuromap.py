#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Parcel 和 Capability 激活信息生成认知模型改进建议

功能：
    使用 LLM 分析 parcel_activation_summary.json 中的神经激活模式，
    生成针对认知模型改进的建议。重点关注：
    1. Baseline_LLM_Inter: 两者都表现好的条件（已被 baseline 捕获的决策模式）
    2. LLM>Baseline: LLM 表现更好的条件（baseline 未捕获但 LLM 捕获的模式）
    3. 差值分析：LLM>Baseline - Baseline_LLM_Inter（需要重点关注的认知脑区）
    4. 反向差值：Baseline_LLM_Inter - LLM>Baseline（baseline 更强的区域）

参考 Model-guided scientific discovery 方法，使用神经激活映射来指导认知模型改进。
"""

import json
import os
import argparse
from pathlib import Path
import requests
import time
from typing import Dict, List, Any, Optional


def load_parcel_activation_summary(summary_file: str) -> Dict[str, Any]:
    """
    加载 Parcel 激活汇总文件
    Args:
        summary_file: parcel_activation_summary.json 文件路径
    Returns:
        dict: 激活汇总数据
    """
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"错误: 读取激活汇总文件失败 {summary_file}: {e}")
        raise


def load_baseline_model_description(intro_file: Optional[str] = None) -> str:
    """
    加载 Baseline 认知模型描述
    Args:
        intro_file: 模型介绍文件路径，如果为 None 则使用默认路径
    Returns:
        str: Baseline 模型描述
    """
    if intro_file is None:
        # 默认路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        intro_file = os.path.join(script_dir, 'model_introduction', 'introduction.json')
    
    try:
        if os.path.exists(intro_file):
            with open(intro_file, 'r', encoding='utf-8') as f:
                intro_data = json.load(f)
            baseline_desc = intro_data.get('Dual-Systems Model', '')
            if baseline_desc:
                return baseline_desc
    except Exception as e:
        print(f"警告: 读取模型介绍文件失败 {intro_file}: {e}")
    
    # 如果文件不存在或读取失败，返回默认描述
    return """### Dual-Systems Model

**Reference:** [7]

**Experiments:**
- Two-step task

**Log-likelihood:**

$$
 p(c_t = i \\mid s_t = s) \\propto
 \\begin{cases}
  \\exp\\left( \\beta [ \\sigma(\\tau) Q^{MB}_{s,i} + (1-\\sigma(\\tau)) Q^{MF}_{s,i} ] \\right), & s = 0 \\\\
  \\exp\\left( \\beta Q^{MF}_{s,i} \\right), & s > 0
 \\end{cases}
$$

$Q^{MB}$ and $Q^{MF}$ denote model-based and model-free value estimates. Free parameters: $\\beta, \\tau$. A first-stage stickiness term is included but omitted for brevity."""


def find_and_load_experiment_data(summary_file: str) -> Optional[Dict[str, Any]]:
    """
    查找并加载原始用户实验数据文件（with_data.json）
    Args:
        summary_file: parcel_activation_summary.json 文件路径
    Returns:
        dict: 原始实验数据，如果找不到则返回 None
    """
    summary_dir = os.path.dirname(summary_file)
    
    # 查找 with_data.json 文件
    possible_names = [
        'kool2016when_exp2_with_data.json',
        # 'kool2017cost_exp2_with_data.json',
    ]
    
    # 也尝试从目录中查找所有 with_data.json 文件
    for filename in os.listdir(summary_dir):
        if filename.endswith('_with_data.json'):
            possible_names.append(filename)
            break
    
    for filename in possible_names:
        filepath = os.path.join(summary_dir, filename)
        if os.path.exists(filepath):
            try:
                print(f"找到原始实验数据文件: {filepath}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                print(f"警告: 读取实验数据文件失败 {filepath}: {e}")
                continue
    
    print(f"警告: 在 {summary_dir} 中未找到原始实验数据文件（with_data.json）")
    return None


def format_participant_data(participant_data: Dict[str, Any], max_trials: int = 20) -> str:
    """
    格式化单个 participant 的实验数据
    Args:
        participant_data: participant 数据字典
        max_trials: 最多显示多少个 trial
    Returns:
        str: 格式化后的文本
    """
    if 'exp_list' in participant_data:
        # 如果有 instruction，先添加 instruction
        if 'instruction' in participant_data:
            transcript = participant_data['instruction'] + '\n\n'
        else:
            transcript = ''
        trials = participant_data['exp_list'][:max_trials]
        transcript += '\n'.join(trials)
        if len(participant_data['exp_list']) > max_trials:
            transcript += f'\n... (remaining {len(participant_data["exp_list"]) - max_trials} trials are omitted)'
        return transcript
    else:
        return "No data available"


def format_parcel_info(parcel: Dict[str, Any]) -> str:
    """格式化单个 Parcel 信息为文本"""
    parcel_id = parcel.get('parcel_id', 'N/A')
    function_name = parcel.get('function_name', 'N/A')
    function_description = parcel.get('function_description', 'N/A')
    activation = parcel.get('mean_activation', 0.0)
    
    return f"""Parcel {parcel_id}:
  Function: {function_name}
  Activation: {activation:.4f}
  Description: {function_description}"""


def format_capability_info(capability: Dict[str, Any]) -> str:
    """格式化单个 Capability 信息为文本"""
    capability_name = capability.get('capability_name', 'N/A')
    activation = capability.get('mean_activation', 0.0)
    
    return f"  - {capability_name}: {activation:.4f}"


def build_prompt_for_condition_combined(
    condition_name: str,
    condition_data: Dict[str, Any],
    experiment_data: Optional[Dict[str, Any]] = None,
    baseline_model_desc: Optional[str] = None
) -> str:
    """
    为特定条件构建包含 step1、step2 和 reward 的合并 prompt
    Args:
        condition_name: 条件名称（如 "Baseline_LLM_Inter" 或 "LLM>Baseline"）
        condition_data: 条件数据
        experiment_data: 原始用户实验数据（可选，主要用于 LLM>Baseline）
        baseline_model_desc: Baseline 模型描述（可选）
    Returns:
        str: 构建好的 prompt，如果数据不完整则返回 None
    """
    # 获取 step1、step2 和 reward 的数据
    step1_data = condition_data.get('steps', {}).get('step1', {})
    step2_data = condition_data.get('steps', {}).get('step2', {})
    reward_data = condition_data.get('steps', {}).get('reward', {})
    
    if not step1_data or not step2_data or not reward_data:
        return None
    
    # 提取 step1、step2 和 reward 的 Parcel 和 Capability 信息
    step1_parcels = step1_data.get('top_parcels', [])
    step1_capabilities = step1_data.get('top_capabilities', [])
    step2_parcels = step2_data.get('top_parcels', [])
    step2_capabilities = step2_data.get('top_capabilities', [])
    reward_parcels = reward_data.get('top_parcels', [])
    reward_capabilities = reward_data.get('top_capabilities', [])
    
    if not step1_parcels or not step2_parcels or not reward_parcels:
        return None
    
    # 构建实验数据文本（先展示）
    experiment_text = ""
    if experiment_data and condition_name == "LLM>Baseline":
        # 尝试从实验数据中提取 LLM 变体的数据（非 baseline）
        llm_variants = {k: v for k, v in experiment_data.items() if k != 'baseline'}
        if llm_variants:
            # 选择第一个 LLM 变体的前几个 participant
            variant_name = list(llm_variants.keys())[0]
            variant_data = llm_variants[variant_name]
            participant_samples = list(variant_data.items())[:3]  # 取前3个 participant
            
            experiment_text = "\n\n### Sample Participant Experimental Data ###\n"
            for participant_id, participant_data in participant_samples:
                experiment_text += f"\n--- Participant {participant_id} ---\n"
                experiment_text += format_participant_data(participant_data, max_trials=50)
                experiment_text += "\n"
    
    # 构建 step1、step2 和 reward 的 Parcel 和 Capability 文本
    step1_parcels_text = "\n\n".join([format_parcel_info(p) for p in step1_parcels])
    step1_capabilities_text = ""
    if step1_capabilities:
        step1_capabilities_text = "\n".join([format_capability_info(c) for c in step1_capabilities])
    
    step2_parcels_text = "\n\n".join([format_parcel_info(p) for p in step2_parcels])
    step2_capabilities_text = ""
    if step2_capabilities:
        step2_capabilities_text = "\n".join([format_capability_info(c) for c in step2_capabilities])
    
    reward_parcels_text = "\n\n".join([format_parcel_info(p) for p in reward_parcels])
    reward_capabilities_text = ""
    if reward_capabilities:
        reward_capabilities_text = "\n".join([format_capability_info(c) for c in reward_capabilities])
    
    # 根据条件类型构建不同的 prompt
    if condition_name == "Baseline_LLM_Inter":
        # Baseline_LLM_Inter: 两者都表现好的条件，看 Baseline 策略捕捉到了什么信号
        prompt = f"""You are an expert cognitive scientist analyzing neural activation patterns from a two-step decision-making task.

**Context**: The data presented here comes from cognitive behaviors where BOTH the Baseline cognitive model and LLM prediction perform well. These represent decision patterns that the Baseline strategy has successfully captured. We use LLM neural activations to understand what signals the Baseline strategy is detecting.

We have constructed a cognitive functional brain atlas for the LLM, mapping its internal representations to functional regions (Parcels) and cognitive capabilities. The following shows the top activated LLM cognitive regions (Parcels) and cognitive capabilities for these cases at **step1 (first decision)**, **step2 (second decision)**, and **reward** stages:
{experiment_text}
### Step 1 (First Decision) - Top Activated LLM Cognitive Regions (Parcels) ###
{step1_parcels_text}

### Step 1 (First Decision) - Top Activated LLM Cognitive Capabilities ###
{step1_capabilities_text if step1_capabilities_text else "N/A"}

### Step 2 (Second Decision) - Top Activated LLM Cognitive Regions (Parcels) ###
{step2_parcels_text}

### Step 2 (Second Decision) - Top Activated LLM Cognitive Capabilities ###
{step2_capabilities_text if step2_capabilities_text else "N/A"}

### Reward (After Receiving Reward) - Top Activated LLM Cognitive Regions (Parcels) ###
{reward_parcels_text}

### Reward (After Receiving Reward) - Top Activated LLM Cognitive Capabilities ###
{reward_capabilities_text if reward_capabilities_text else "N/A"}

Please analyze these LLM activation patterns and explain:
1. What cognitive processes and signals is the Baseline strategy capturing in these successful cases?
2. What strategies or mechanisms might be represented by these activated LLM regions at step1, step2, and reward stages?
3. How do these patterns relate to successful decision-making in a two-step task?

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 500 words):"""
    
    elif condition_name == "LLM>Baseline":
        # LLM>Baseline: 最关键的条件，需要包含原始实验数据
        # 添加 Baseline 模型描述
        baseline_info = ""
        if baseline_model_desc:
            baseline_info = f"\n\n**Baseline Cognitive Model (Dual-Systems Model):**\n\n{baseline_model_desc}\n"
        
        prompt = f"""You are an expert cognitive scientist analyzing neural activation patterns from a two-step decision-making task.

**Context**: The data presented here comes from cognitive behaviors where LLM prediction performs BETTER than the Baseline cognitive model. These represent decision patterns that the Baseline cognitive model fails to capture, but the LLM successfully captures. This is the MOST CRITICAL data for improving cognitive models, as it identifies what the Baseline model is missing.
{baseline_info}
We have constructed a cognitive functional brain atlas for the LLM, mapping its internal representations to functional regions (Parcels) and cognitive capabilities. The following shows:
1. Sample participant experimental data from these cases
2. The top activated LLM cognitive regions (Parcels) and cognitive capabilities for these cases at **step1 (first decision)**, **step2 (second decision)**, and **reward** stages
{experiment_text}
### Step 1 (First Decision) - Top Activated LLM Cognitive Regions (Parcels) ###
{step1_parcels_text}

### Step 1 (First Decision) - Top Activated LLM Cognitive Capabilities ###
{step1_capabilities_text if step1_capabilities_text else "N/A"}

### Step 2 (Second Decision) - Top Activated LLM Cognitive Regions (Parcels) ###
{step2_parcels_text}

### Step 2 (Second Decision) - Top Activated LLM Cognitive Capabilities ###
{step2_capabilities_text if step2_capabilities_text else "N/A"}

### Reward (After Receiving Reward) - Top Activated LLM Cognitive Regions (Parcels) ###
{reward_parcels_text}

### Reward (After Receiving Reward) - Top Activated LLM Cognitive Capabilities ###
{reward_capabilities_text if reward_capabilities_text else "N/A"}

Please analyze these LLM activation patterns and experimental data, and explain:
1. What cognitive processes are engaged in these cases where LLM outperforms Baseline, particularly at step1, step2, and reward stages?
2. What strategies or mechanisms might explain why LLM captures these patterns better than Baseline?
3. Based on the LLM neural activations and experimental data, what specific improvements should be made to the Baseline cognitive model (Dual-Systems Model)? Consider how the model's current structure (model-based vs model-free integration) might be enhanced.
4. How can these insights be formalized into computational cognitive model modifications? Provide concrete suggestions for modifying the Dual-Systems Model equations or adding new components.

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 700 words):"""
    
    else:
        # 其他条件使用通用 prompt
        prompt = f"""You are an expert cognitive scientist analyzing neural activation patterns from a two-step decision-making task.

We have constructed a cognitive functional brain atlas for the LLM, mapping its internal representations to functional regions (Parcels) and cognitive capabilities. The following shows the top activated LLM cognitive regions (Parcels) and cognitive capabilities for the condition: **{condition_name}** at **step1 (first decision)**, **step2 (second decision)**, and **reward** stages:
{experiment_text}
### Step 1 (First Decision) - Top Activated LLM Cognitive Regions (Parcels) ###
{step1_parcels_text}

### Step 1 (First Decision) - Top Activated LLM Cognitive Capabilities ###
{step1_capabilities_text if step1_capabilities_text else "N/A"}

### Step 2 (Second Decision) - Top Activated LLM Cognitive Regions (Parcels) ###
{step2_parcels_text}

### Step 2 (Second Decision) - Top Activated LLM Cognitive Capabilities ###
{step2_capabilities_text if step2_capabilities_text else "N/A"}

### Reward (After Receiving Reward) - Top Activated LLM Cognitive Regions (Parcels) ###
{reward_parcels_text}

### Reward (After Receiving Reward) - Top Activated LLM Cognitive Capabilities ###
{reward_capabilities_text if reward_capabilities_text else "N/A"}

Please analyze these LLM activation patterns and explain:
1. What cognitive processes are likely engaged in this condition at step1, step2, and reward stages?
2. What strategies or mechanisms might be represented by these activated LLM regions?
3. How do these patterns relate to decision-making in a two-step task?

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 500 words):"""
    
    return prompt


def build_prompt_for_condition_simple(
    condition_name: str,
    experiment_data: Optional[Dict[str, Any]] = None,
    baseline_model_desc: Optional[str] = None
) -> str:
    """
    为特定条件构建 simple 版本的 prompt（只包含实验数据，不包含 cogNeuromap 信息）
    Args:
        condition_name: 条件名称（如 "Baseline_LLM_Inter" 或 "LLM>Baseline"）
        experiment_data: 原始用户实验数据
        baseline_model_desc: Baseline 模型描述（可选）
    Returns:
        str: 构建好的 prompt，如果数据不完整则返回 None
    """
    # 构建实验数据文本
    experiment_text = ""
    if experiment_data:
        # 根据条件名称选择不同的数据
        if condition_name == "LLM>Baseline":
            # LLM>Baseline: 使用 LLM 变体的数据（非 baseline）
            llm_variants = {k: v for k, v in experiment_data.items() if k != 'baseline'}
            if llm_variants:
                variant_name = list(llm_variants.keys())[0]
                variant_data = llm_variants[variant_name]
                participant_samples = list(variant_data.items())[:3]  # 取前3个 participant（与非 simple 模式一致）
                
                experiment_text = "\n\n### Participant Experimental Data ###\n"
                for participant_id, participant_data in participant_samples:
                    experiment_text += f"\n--- Participant {participant_id} ---\n"
                    experiment_text += format_participant_data(participant_data, max_trials=50)  # 与非 simple 模式一致
                    experiment_text += "\n"
        elif condition_name == "Baseline_LLM_Inter":
            # Baseline_LLM_Inter: 使用 baseline 和 LLM 的数据
            if 'baseline' in experiment_data:
                baseline_data = experiment_data['baseline']
                participant_samples = list(baseline_data.items())[:3]  # 取前3个 participant（减少数据量）
                
                experiment_text = "\n\n### Participant Experimental Data (Baseline) ###\n"
                for participant_id, participant_data in participant_samples:
                    experiment_text += f"\n--- Participant {participant_id} ---\n"
                    experiment_text += format_participant_data(participant_data, max_trials=50)  # 减少数据量
                    experiment_text += "\n"
            
            # 也添加 LLM 变体的数据
            llm_variants = {k: v for k, v in experiment_data.items() if k != 'baseline'}
            if llm_variants:
                variant_name = list(llm_variants.keys())[0]
                variant_data = llm_variants[variant_name]
                participant_samples = list(variant_data.items())[:3]  # 取前3个 participant（减少数据量）
                
                experiment_text += "\n\n### Participant Experimental Data (LLM) ###\n"
                for participant_id, participant_data in participant_samples:
                    experiment_text += f"\n--- Participant {participant_id} ---\n"
                    experiment_text += format_participant_data(participant_data, max_trials=50)  # 减少数据量
                    experiment_text += "\n"
        else:
            # 其他条件：使用所有可用数据
            participant_samples = []
            for variant_name, variant_data in experiment_data.items():
                participant_samples.extend(list(variant_data.items())[:2])  # 每个变体取前2个（减少数据量）
                if len(participant_samples) >= 3:
                    break
            
            experiment_text = "\n\n### Participant Experimental Data ###\n"
            for participant_id, participant_data in participant_samples[:3]:  # 最多3个
                experiment_text += f"\n--- Participant {participant_id} ---\n"
                experiment_text += format_participant_data(participant_data, max_trials=30)  # 减少数据量
                experiment_text += "\n"
    
    if not experiment_text:
        print(f"警告: 条件 {condition_name} 没有可用的实验数据")
        return None
    
    # 根据条件类型构建不同的 prompt
    if condition_name == "Baseline_LLM_Inter":
        prompt = f"""You are an expert cognitive scientist analyzing behavioral data from a two-step decision-making task.

**Context**: The data presented here comes from cognitive behaviors where BOTH the Baseline cognitive model and LLM prediction perform well. These represent decision patterns that the Baseline strategy has successfully captured.

{experiment_text}

Please analyze these experimental data and explain:
1. What cognitive processes and signals is the Baseline strategy capturing in these successful cases?
2. What strategies or mechanisms might be represented by these behavioral patterns?
3. How do these patterns relate to successful decision-making in a two-step task?

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 500 words):"""
    
    elif condition_name == "LLM>Baseline":
        # LLM>Baseline: 最关键的条件，需要包含 Baseline 模型描述
        baseline_info = ""
        if baseline_model_desc:
            baseline_info = f"\n\n**Baseline Cognitive Model (Dual-Systems Model):**\n\n{baseline_model_desc}\n"
        
        prompt = f"""You are an expert cognitive scientist analyzing behavioral data from a two-step decision-making task.

**Context**: The data presented here comes from cognitive behaviors where LLM prediction performs BETTER than the Baseline cognitive model. These represent decision patterns that the Baseline cognitive model fails to capture, but the LLM successfully captures. This is the MOST CRITICAL data for improving cognitive models, as it identifies what the Baseline model is missing.
{baseline_info}
{experiment_text}

Please analyze these experimental data and explain:
1. What cognitive processes are engaged in these cases where LLM outperforms Baseline?
2. What strategies or mechanisms might explain why LLM captures these patterns better than Baseline?
3. Based on the experimental data, what specific improvements should be made to the Baseline cognitive model (Dual-Systems Model)? Consider how the model's current structure (model-based vs model-free integration) might be enhanced.
4. How can these insights be formalized into computational cognitive model modifications? Provide concrete suggestions for modifying the Dual-Systems Model equations or adding new components.

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 700 words):"""
    
    else:
        # 其他条件使用通用 prompt
        prompt = f"""You are an expert cognitive scientist analyzing behavioral data from a two-step decision-making task.

The following shows experimental data for the condition: **{condition_name}**:
{experiment_text}

Please analyze these experimental data and explain:
1. What cognitive processes are likely engaged in this condition?
2. What strategies or mechanisms might be represented by these behavioral patterns?
3. How do these patterns relate to decision-making in a two-step task?

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 500 words):"""
    
    return prompt


def build_prompt_for_diff(
    diff_data: Dict[str, Any],
    step_name: str = "step1",
    diff_type: str = "LLM>Baseline_minus_Baseline_LLM_Inter"
) -> str:
    """
    为差值条件构建 prompt
    Args:
        diff_data: 差值数据
        step_name: 步骤名称
        diff_type: 差值类型
    Returns:
        str: 构建好的 prompt
    """
    step_data = diff_data.get('steps', {}).get(step_name, {})
    if not step_data:
        return None
    
    top_parcels = step_data.get('top_parcels', [])
    top_capabilities = step_data.get('top_capabilities', [])
    
    if not top_parcels:
        return None
    
    parcels_text = "\n\n".join([format_parcel_info(p) for p in top_parcels])
    
    capabilities_text = ""
    if top_capabilities:
        capabilities_text = "\n".join([format_capability_info(c) for c in top_capabilities])
    
    # 根据差值类型确定描述
    if "LLM>Baseline_minus_Baseline_LLM_Inter" in diff_type:
        interpretation = "These LLM cognitive regions show stronger activation in cases where LLM outperforms Baseline compared to cases where both perform well, indicating cognitive processes that LLM captures but Baseline does not."
    else:
        interpretation = "These LLM cognitive regions show stronger activation in cases where both Baseline and LLM perform well compared to cases where LLM outperforms Baseline, indicating cognitive processes that Baseline captures more strongly."
    
    prompt = f"""You are an expert cognitive scientist analyzing neural activation differences from a two-step decision-making task.

We have constructed a cognitive functional brain atlas for the LLM, mapping its internal representations to functional regions (Parcels) and cognitive capabilities. The following shows the top LLM cognitive regions (Parcels) and cognitive capabilities with the highest **difference** in activation at the **{step_name}** stage.

{interpretation}

### Top LLM Cognitive Regions (Parcels) with Highest Difference ###
{parcels_text}

### Top LLM Cognitive Capabilities with Highest Difference ###
{capabilities_text if capabilities_text else "N/A"}

Please analyze these differential LLM activation patterns and explain:
1. What cognitive processes are uniquely or more strongly engaged in the condition with higher activation?
2. What strategies or mechanisms might explain why these LLM regions show differential activation?
3. How should these patterns inform improvements to cognitive models?
4. What specific modifications to computational cognitive models would better capture these processes?

Your explanation should be targeted at an expert cognitive scientist and suitable for formalizing into a computational cognitive model (less than 400 words):"""
    
    return prompt


def call_vllm_api(vllm_url: str, api_key: str, prompt: str, model: str, 
                  max_tokens: int = 20000, temperature: float = 0.7, 
                  max_retries: int = 3, retry_delay: int = 1, timeout: int = 120) -> str:
    """
    调用兼容 OpenAI API 格式的 LLM API 生成解释
    支持本地 vLLM 和 OpenAI API
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(f"{vllm_url}/chat/completions", 
                               headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            
            if data and "choices" in data and data["choices"] and "message" in data["choices"][0]:
                content = data["choices"][0]["message"].get("content", "")
                return (content or "").strip()
            else:
                raise Exception(f"响应格式不正确: {data}")
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"警告: API 调用失败，{retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                print(f"错误信息: {e}")
                time.sleep(retry_delay)
            else:
                print(f"错误: API 调用失败，已达到最大重试次数: {e}")
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"警告: API 调用失败，{retry_delay} 秒后重试... (尝试 {attempt + 1}/{max_retries})")
                print(f"错误信息: {e}")
                time.sleep(retry_delay)
            else:
                print(f"错误: API 调用失败，已达到最大重试次数: {e}")
                raise


def process_condition(
    summary_data: Dict[str, Any],
    condition_name: str,
    vllm_url: str,
    api_key: str,
    model_name: str,
    steps: List[str] = ["step1", "step2", "reward"],
    experiment_data: Optional[Dict[str, Any]] = None,
    baseline_model_desc: Optional[str] = None,
    simple_mode: bool = False
) -> Dict[str, Any]:
    """
    处理一个条件，为 step1、step2 和 reward 生成合并分析
    Args:
        summary_data: 激活汇总数据
        condition_name: 条件名称
        vllm_url: API URL
        api_key: API 密钥
        model_name: 模型名称
        steps: 步骤列表
        experiment_data: 原始实验数据（可选，主要用于 LLM>Baseline）
        baseline_model_desc: Baseline 模型描述（可选）
        simple_mode: 是否使用 simple 模式（只包含实验数据，不包含 cogNeuromap 信息）
    """
    if not simple_mode:
        if condition_name not in summary_data:
            print(f"警告: 条件 {condition_name} 不存在于数据中")
            return {}
        condition_data = summary_data[condition_name]
    else:
        # simple 模式不需要 summary_data 中的条件数据
        condition_data = None
    
    results = {}
    
    mode_str = "simple" if simple_mode else "cogNeuromap"
    print(f"\n处理条件: {condition_name} ({mode_str} 模式，合并 step1、step2 和 reward)")
    
    try:
        if simple_mode:
            # Simple 模式：只使用实验数据
            prompt = build_prompt_for_condition_simple(
                condition_name,
                experiment_data=experiment_data,
                baseline_model_desc=baseline_model_desc
            )
        else:
            # 正常模式：使用 cogNeuromap 信息
            prompt = build_prompt_for_condition_combined(
                condition_name, 
                condition_data,
                experiment_data=experiment_data if condition_name == "LLM>Baseline" else None,
                baseline_model_desc=baseline_model_desc if condition_name == "LLM>Baseline" else None
            )
        
        if not prompt:
            print(f"    警告: 无法构建 prompt，跳过")
            return {}
        
        explanation = call_vllm_api(vllm_url, api_key, prompt, model_name)
        
        # 保存合并结果
        if simple_mode:
            results['step1_step2_reward_combined'] = {
                'steps': ['step1', 'step2', 'reward'],
                'explanation': explanation,
                'prompt': prompt,
                'mode': 'simple'
            }
        else:
            results['step1_step2_reward_combined'] = {
                'steps': ['step1', 'step2', 'reward'],
                'explanation': explanation,
                'prompt': prompt,
                'step1': {
                    'top_parcels': condition_data.get('steps', {}).get('step1', {}).get('top_parcels', []),
                    'top_capabilities': condition_data.get('steps', {}).get('step1', {}).get('top_capabilities', [])
                },
                'step2': {
                    'top_parcels': condition_data.get('steps', {}).get('step2', {}).get('top_parcels', []),
                    'top_capabilities': condition_data.get('steps', {}).get('step2', {}).get('top_capabilities', [])
                },
                'reward': {
                    'top_parcels': condition_data.get('steps', {}).get('reward', {}).get('top_parcels', []),
                    'top_capabilities': condition_data.get('steps', {}).get('reward', {}).get('top_capabilities', [])
                }
            }
        
        time.sleep(0.5)  # 避免 API 限流
        
    except Exception as e:
        print(f"    错误: 处理条件失败: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def process_diff_condition(
    summary_data: Dict[str, Any],
    diff_name: str,
    vllm_url: str,
    api_key: str,
    model_name: str,
    steps: List[str] = ["step1", "step2", "reward"]
) -> Dict[str, Any]:
    """
    处理差值条件，为每个步骤生成分析
    """
    if diff_name not in summary_data:
        print(f"警告: 差值条件 {diff_name} 不存在于数据中")
        return {}
    
    diff_data = summary_data[diff_name]
    results = {}
    
    print(f"\n处理差值条件: {diff_name}")
    
    for step_name in steps:
        print(f"  处理步骤: {step_name}...")
        
        try:
            prompt = build_prompt_for_diff(diff_data, step_name, diff_name)
            if not prompt:
                print(f"    警告: 无法为 {step_name} 构建 prompt，跳过")
                continue
            
            explanation = call_vllm_api(vllm_url, api_key, prompt, model_name)
            
            results[step_name] = {
                'step_name': step_name,
                'explanation': explanation,
                'prompt': prompt,
                'top_parcels': diff_data.get('steps', {}).get(step_name, {}).get('top_parcels', []),
                'top_capabilities': diff_data.get('steps', {}).get(step_name, {}).get('top_capabilities', [])
            }
            
            time.sleep(0.5)  # 避免 API 限流
            
        except Exception as e:
            print(f"    错误: 处理 {step_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='基于 Parcel 和 Capability 激活信息生成认知模型改进建议'
    )
    parser.add_argument(
        '--summary_file',
        type=str,
        default='/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/results/comparison_results/parcel_activation_summary.json',
        help='Parcel 激活汇总 JSON 文件路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/path/to/project_root/Human_LLM_align/Llama-3.1-Centaur-70B-main/openloop/results/comparison_results',
        help='输出结果目录路径'
    )
    parser.add_argument(
        '--vllm_url',
        type=str,
        default='http://0.0.0.0:8000/v1',
        help='API 地址（默认: http://0.0.0.0:8001/v1，OpenAI 使用: https://api.openai.com/v1）'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API 密钥（如果不提供，将从环境变量 VLLM_API_KEY 或 OPENAI_API_KEY 读取，否则使用默认值 abcabc）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='/path/to/local_models/gpt-oss-20b',
        help='使用的模型名称（默认: /path/to/local_models/gpt-oss-20b，OpenAI 使用: gpt-4o）'
    )
    parser.add_argument(
        '--conditions',
        type=str,
        nargs='+',
        default=None,
        help='要处理的条件列表（如果不提供，将处理所有条件）'
    )
    parser.add_argument(
        '--steps',
        type=str,
        nargs='+',
        default=['step1', 'step2'],
        help='要处理的步骤列表（默认: step1 step2 reward）'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='如果输出文件已存在则跳过整个文件'
    )
    parser.add_argument(
        '--include_diff',
        action='store_true',
        help='是否包含差值条件分析（LLM>Baseline_minus_Baseline_LLM_Inter 等）'
    )
    parser.add_argument(
        '--include_baseline_inter',
        action='store_true',
        help='是否包含 Baseline_LLM_Inter 条件分析'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='使用 simple 模式：只包含用户实验数据，不包含 cogNeuromap 信息'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.summary_file):
        raise FileNotFoundError(f"激活汇总文件不存在: {args.summary_file}")
    
    # 从文件名提取数据集名称
    dataset_name = os.path.basename(args.summary_file)
    if dataset_name.endswith('_summary.json'):
        dataset_name = dataset_name[:-len('_summary.json')]
    elif dataset_name.endswith('.json'):
        dataset_name = dataset_name[:-5]
    
    # 根据模式确定输出文件名
    if args.simple:
        output_file = os.path.join(args.output_dir, f'{dataset_name}_cogneuromap_improvement_suggestions_simple.json')
    else:
        output_file = os.path.join(args.output_dir, f'{dataset_name}_cogneuromap_improvement_suggestions.json')
    
    # 检查输出文件是否已存在
    if args.skip_existing and os.path.exists(output_file):
        print(f"输出文件 {output_file} 已存在，跳过处理")
        return
    
    # 获取 API 配置
    api_key = args.api_key or os.getenv('OPENAI_API_KEY') or os.getenv('VLLM_API_KEY', 'abcabc')
    vllm_url = args.vllm_url
    
    # 加载激活汇总数据（simple 模式下不需要）
    if not args.simple:
        print(f"读取激活汇总文件: {args.summary_file}")
        summary_data = load_parcel_activation_summary(args.summary_file)
    else:
        summary_data = {}
    
    # 查找并加载原始实验数据（simple 模式下必需）
    experiment_data = find_and_load_experiment_data(args.summary_file)
    if args.simple and not experiment_data:
        raise FileNotFoundError(f"Simple 模式需要实验数据文件，但在 {os.path.dirname(args.summary_file)} 中未找到")
    
    # 加载 Baseline 模型描述
    baseline_model_desc = load_baseline_model_description()
    print("已加载 Baseline 认知模型（Dual-Systems Model）描述")
    
    # 确定要处理的条件
    # 默认主要处理 LLM>Baseline（最关键），其他条件可选
    available_conditions = [
        'LLM>Baseline',  # 最关键，放在第一位
        'Baseline_LLM_Inter',
        'LLM>Baseline_minus_Baseline_LLM_Inter',  # 差值，可选
        'Baseline_LLM_Inter_minus_LLM>Baseline'  # 反向差值，可选
    ]
    
    if args.conditions:
        if args.simple:
            # simple 模式下，直接使用用户指定的条件
            conditions_to_process = args.conditions
        else:
            conditions_to_process = [c for c in args.conditions if c in summary_data]
    else:
        # 默认只处理 LLM>Baseline（最关键）
        if args.simple:
            conditions_to_process = ['LLM>Baseline']
        else:
            conditions_to_process = ['LLM>Baseline'] if 'LLM>Baseline' in summary_data else []
        
        # 根据参数决定是否添加其他条件（simple 模式下不支持差值条件）
        if not args.simple:
            if args.include_baseline_inter and 'Baseline_LLM_Inter' in summary_data:
                conditions_to_process.append('Baseline_LLM_Inter')
            
            if args.include_diff:
                if 'LLM>Baseline_minus_Baseline_LLM_Inter' in summary_data:
                    conditions_to_process.append('LLM>Baseline_minus_Baseline_LLM_Inter')
                if 'Baseline_LLM_Inter_minus_LLM>Baseline' in summary_data:
                    conditions_to_process.append('Baseline_LLM_Inter_minus_LLM>Baseline')
        else:
            if args.include_baseline_inter:
                conditions_to_process.append('Baseline_LLM_Inter')
        
        if len(conditions_to_process) == 1 and not args.simple:
            print(f"提示: 默认只处理 LLM>Baseline 条件（最关键）。")
            print(f"     使用 --include_baseline_inter 可添加 Baseline_LLM_Inter 条件")
            print(f"     使用 --include_diff 可添加差值条件分析")
    
    print(f"找到 {len(conditions_to_process)} 个条件需要处理: {conditions_to_process}")
    
    # 构建结果字典
    mode_note = 'simple mode (experimental data only)' if args.simple else 'cogNeuromap mode'
    all_results = {
        'meta': {
            'summary_file': args.summary_file,
            'conditions_processed': conditions_to_process,
            'steps_processed': ['step1', 'step2', 'reward'],  # 处理 step1、step2 和 reward
            'mode': mode_note,
            'note': 'step1, step2, and reward are combined in a single analysis'
        },
        'conditions': {}
    }
    
    # 处理每个条件
    for condition_name in conditions_to_process:
        if condition_name in ['LLM>Baseline_minus_Baseline_LLM_Inter', 'Baseline_LLM_Inter_minus_LLM>Baseline']:
            # 差值条件（simple 模式下不支持）
            if args.simple:
                print(f"警告: simple 模式下不支持差值条件 {condition_name}，跳过")
                continue
            condition_result = process_diff_condition(
                summary_data,
                condition_name,
                vllm_url,
                api_key,
                args.model,
                steps=args.steps
            )
        else:
            # 普通条件（LLM>Baseline 会传入 experiment_data 和 baseline_model_desc）
            condition_result = process_condition(
                summary_data,
                condition_name,
                vllm_url,
                api_key,
                args.model,
                steps=args.steps,
                experiment_data=experiment_data,
                baseline_model_desc=baseline_model_desc,
                simple_mode=args.simple
            )
        
        if condition_result:
            all_results['conditions'][condition_name] = condition_result
        
        # 保存中间结果
        os.makedirs(args.output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"已保存中间结果到: {output_file}")
    
    # 保存最终结果
    print(f"\n保存最终结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 统计信息
    print("\n处理完成！")
    print(f"结果文件: {output_file}")
    print("\n统计信息:")
    for condition_name, condition_results in all_results['conditions'].items():
        if 'step1_step2_reward_combined' in condition_results:
            print(f"  {condition_name}: step1、step2 和 reward 合并分析")
        else:
            step_count = len(condition_results)
            print(f"  {condition_name}: {step_count} 个步骤的分析")


if __name__ == '__main__':
    main()

