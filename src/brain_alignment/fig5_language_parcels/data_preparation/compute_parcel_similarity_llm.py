#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 LLM 计算 Human Parcel 和 LLM Parcel 之间的功能相似度矩阵

基于 LLM_similarity_prompt.md 中的 prompt，调用 LLM API 评估每对 Parcel 的相似度（0-10分）
生成 H×L 相似度矩阵，其中 H 是 Human Parcels，L 是 LLM Parcels
"""

import json
import pandas as pd
import numpy as np
import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging

import requests

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_VLLM_URL = "http://0.0.0.0:8001/v1"
DEFAULT_API_KEY = "abcabc"
DEFAULT_HUMAN_PARCEL_FILE = "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
DEFAULT_LLM_PARCEL_FILE = "/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json"
DEFAULT_OUTPUT_FILE = "/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data_preparation/semantic_matrix_llm.csv"


def load_prompt_template(prompt_file: str) -> str:
    """加载 prompt 模板文件"""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt 文件不存在: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    return prompt_template


def call_vllm_api(prompt: str, vllm_url: str = DEFAULT_VLLM_URL, api_key: str = DEFAULT_API_KEY,
                  max_tokens: int = 1024, temperature: float = 0.6, timeout: int = 120) -> str:
    """调用vLLM chat接口获取响应"""
    payload = {
        "model": "/path/to/local_models/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "You are a cognitive neuroscience expert. Return STRICT JSON when asked."},
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
    try:
        resp = requests.post(f"{vllm_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data and "choices" in data and data["choices"] and "message" in data["choices"][0]:
            content = data["choices"][0]["message"].get("content", "")
            return (content or "").strip()
        raise Exception(f"响应格式不正确: {data}")
    except Exception as e:
        raise Exception(f"vLLM调用失败: {e}")


def parse_json_response(raw: str) -> Dict[str, Any]:
    """从模型响应中提取严格JSON"""
    if not raw:
        raise Exception("响应为空")
    try:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise Exception(f"无法从响应中提取JSON: {raw[:200]}")
        return json.loads(m.group(0))
    except Exception as e:
        raise Exception(f"解析JSON失败: {e}, 原始响应: {raw[:400]}")


def check_api_available(vllm_url: str = DEFAULT_VLLM_URL, api_key: str = DEFAULT_API_KEY) -> bool:
    """检查 API 是否可用"""
    try:
        test_prompt = "Test connection. Reply with: OK"
        response = call_vllm_api(test_prompt, vllm_url, api_key, max_tokens=10, timeout=10)
        logger.info(f"✓ API 连接成功: {vllm_url}")
        return True
    except Exception as e:
        logger.error(f"✗ API 连接失败: {e}")
        return False


def load_human_parcels(human_parcel_file: str) -> List[Dict[str, Any]]:
    """加载 Human Parcel descriptions"""
    logger.info(f"加载 Human Parcels: {human_parcel_file}")
    with open(human_parcel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Human parcel file should be a list, got {type(data)}")
    
    # 按 parcel_id 排序
    parcels = []
    for item in data:
        parcel_id = item.get('parcel_id')
        if parcel_id is None:
            logger.warning(f"跳过没有 parcel_id 的 Human Parcel: {item.get('parcel_name', 'unknown')}")
            continue
        
        parcels.append({
            'parcel_id': parcel_id,
            'parcel_name': item.get('parcel_name', ''),
            'function_name': item.get('function_name', ''),
            'function_description': item.get('function_description', ''),
            'role_in_human_brain': item.get('role_in_human_brain', ''),
            'reasoning': item.get('reasoning', '')
        })
    
    parcels.sort(key=lambda x: x['parcel_id'])
    logger.info(f"加载了 {len(parcels)} 个 Human Parcels")
    return parcels


def load_llm_parcels(llm_parcel_file: str) -> List[Dict[str, Any]]:
    """加载 LLM Parcel descriptions"""
    logger.info(f"加载 LLM Parcels: {llm_parcel_file}")
    with open(llm_parcel_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取 parcel_summaries
    if isinstance(data, dict) and 'parcel_summaries' in data:
        parcel_summaries = data['parcel_summaries']
    elif isinstance(data, list):
        parcel_summaries = data
    else:
        raise ValueError(f"Unexpected LLM parcel file format: {type(data)}")
    
    # 提取信息并按 parcel_id 排序
    parcels = []
    for item in parcel_summaries:
        parcel_id = item.get('parcel_id')
        if parcel_id is None:
            logger.warning(f"跳过没有 parcel_id 的 LLM Parcel")
            continue
        
        function_name = item.get('function_name', '')
        # 清理 function_name（去除可能的 ** 标记和多余空格）
        function_name = function_name.replace('**', '').strip()
        function_name = ' '.join(function_name.split())
        
        parcels.append({
            'parcel_id': parcel_id,
            'function_name': function_name,
            'function_description': item.get('function_description', ''),
            'role_in_large_model': item.get('model_role', '')  # 注意：LLM Parcel 中使用 model_role
        })
    
    parcels.sort(key=lambda x: x['parcel_id'])
    logger.info(f"加载了 {len(parcels)} 个 LLM Parcels")
    return parcels


def build_similarity_prompt(human_parcel: Dict[str, Any], llm_parcel: Dict[str, Any], 
                            prompt_template: str) -> str:
    """构建用于评估相似度的 prompt"""
    # 构建 Human Parcel 信息部分
    human_info = f"""### Human Parcel
- function_name: *{human_parcel['function_name']}*
- function_description: {human_parcel['function_description']}
- role_in_human_brain: {human_parcel['role_in_human_brain']}
- reasoning: {human_parcel['reasoning']}
- parcel_name: *{human_parcel['parcel_name']}*"""

    # 构建 LLM Parcel 信息部分
    llm_info = f"""### LLM Parcel
- function_name: *{llm_parcel['function_name']}*
- functionality_description: {llm_parcel['function_description']}
- role_in_large_model: {llm_parcel['role_in_large_model']}"""

    # 将 Human Parcel 和 LLM Parcel 信息插入到 prompt 模板的末尾
    prompt = prompt_template.rstrip()
    # 确保以 "# Evaluate the following pair:" 结尾，然后添加内容
    if "# Evaluate the following pair:" in prompt:
        # 如果已经包含这一行，在其后添加内容
        prompt += f"\n\n{human_info}\n\n{llm_info}"
    else:
        # 如果没有，添加这一行和内容
        prompt += "\n\n# Evaluate the following pair:\n\n"
        prompt += f"{human_info}\n\n{llm_info}"
    return prompt


def evaluate_similarity(human_parcel: Dict[str, Any], llm_parcel: Dict[str, Any],
                        prompt_template: str, vllm_url: str, api_key: str,
                        max_retries: int = 3) -> Tuple[int, str]:
    """
    评估一对 Parcel 的相似度
    
    Returns:
        (rating, justification): rating 是 0-10 的整数，justification 是字符串
    """
    prompt = build_similarity_prompt(human_parcel, llm_parcel, prompt_template)
    
    for attempt in range(max_retries):
        try:
            response = call_vllm_api(prompt, vllm_url, api_key, max_tokens=10000, timeout=120)
            result = parse_json_response(response)
            
            # 验证必需字段
            if 'rating' not in result:
                raise Exception(f"响应中缺少 'rating' 字段: {result}")
            if 'justification' not in result:
                raise Exception(f"响应中缺少 'justification' 字段: {result}")
            
            rating = result['rating']
            justification = result['justification']
            
            # 验证 rating 是 0-10 的整数
            try:
                rating = int(rating)
                if rating < 0 or rating > 10:
                    raise Exception(f"rating 必须在 0-10 之间，得到: {rating}")
            except (ValueError, TypeError):
                raise Exception(f"rating 必须是整数，得到: {rating} (类型: {type(rating)})")
            
            return rating, justification
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # 递增等待时间
                logger.warning(f"评估失败 (尝试 {attempt + 1}/{max_retries}): {e}，等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"评估失败，已重试 {max_retries} 次: {e}")
                raise
    
    raise Exception("评估失败，已达到最大重试次数")


def load_existing_results(output_file: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """加载已存在的结果文件"""
    existing_results = {}
    
    try:
        # 尝试加载 JSON 格式的结果文件（包含详细结果）
        json_file = output_file.replace('.csv', '_detailed_results.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'results' in data:
                    for result in data['results']:
                        h_id = result.get('human_parcel_id')
                        l_id = result.get('llm_parcel_id')
                        if h_id is not None and l_id is not None:
                            existing_results[(h_id, l_id)] = result
                    logger.info(f"从已有文件加载了 {len(existing_results)} 个相似度评估结果")
    except Exception as e:
        logger.warning(f"无法读取已有结果文件: {e}")
        import traceback
        traceback.print_exc()
    
    return existing_results


def save_detailed_results(output_file: str, all_results: List[Dict[str, Any]]):
    """保存详细结果到 JSON 文件"""
    json_file = output_file.replace('.csv', '_detailed_results.json')
    output_data = {
        'total_pairs': len(all_results),
        'results': all_results
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"详细结果已保存到: {json_file}")


def build_similarity_matrix(human_parcels: List[Dict[str, Any]], 
                           llm_parcels: List[Dict[str, Any]],
                           similarity_results: Dict[Tuple[int, int], Dict[str, Any]]) -> pd.DataFrame:
    """构建相似度矩阵 DataFrame"""
    # 创建矩阵
    matrix = np.zeros((len(human_parcels), len(llm_parcels)))
    
    # 填充矩阵
    for i, human_parcel in enumerate(human_parcels):
        for j, llm_parcel in enumerate(llm_parcels):
            key = (human_parcel['parcel_id'], llm_parcel['parcel_id'])
            if key in similarity_results:
                matrix[i, j] = similarity_results[key]['rating']
            else:
                # 如果结果不存在，使用 NaN
                matrix[i, j] = np.nan
    
    # 创建 DataFrame
    human_parcel_ids = [f"Human_Parcel_{p['parcel_id']}" for p in human_parcels]
    llm_parcel_ids = [f"LLM_Parcel_{p['parcel_id']}" for p in llm_parcels]
    
    df = pd.DataFrame(
        matrix,
        index=human_parcel_ids,
        columns=llm_parcel_ids
    )
    
    return df


def main():
    parser = argparse.ArgumentParser(description='使用 LLM 计算 Human 和 LLM Parcel 之间的功能相似度')
    parser.add_argument('--human_parcel_file',
                       default=DEFAULT_HUMAN_PARCEL_FILE,
                       help='Human Parcel descriptions JSON 文件路径')
    parser.add_argument('--llm_parcel_file',
                       default=DEFAULT_LLM_PARCEL_FILE,
                       help='LLM Parcel functionality summary JSON 文件路径')
    parser.add_argument('--prompt_file',
                       default='/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data_preparation/LLM_similarity_prompt.md',
                       help='Prompt 模板文件路径')
    parser.add_argument('--output_file',
                       default=DEFAULT_OUTPUT_FILE,
                       help='输出 CSV 文件路径')
    parser.add_argument('--vllm_url',
                       default=DEFAULT_VLLM_URL,
                       help='vLLM API URL')
    parser.add_argument('--api_key',
                       default=DEFAULT_API_KEY,
                       help='API Key')
    parser.add_argument('--skip_existing',
                       action='store_true',
                       default=True,
                       help='跳过已存在的评估结果')
    parser.add_argument('--no_skip_existing',
                       action='store_false',
                       dest='skip_existing',
                       help='不跳过已存在的评估结果，重新评估')
    parser.add_argument('--check_api',
                       action='store_true',
                       default=True,
                       help='检查 API 是否可用')
    parser.add_argument('--no_check_api',
                       action='store_false',
                       dest='check_api',
                       help='不检查 API 是否可用')
    parser.add_argument('--delay',
                       type=float,
                       default=0.5,
                       help='每次 API 调用之间的延迟（秒）')
    
    args = parser.parse_args()
    
    # 检查 API
    if args.check_api:
        if not check_api_available(args.vllm_url, args.api_key):
            logger.error("API 不可用，退出")
            return 1
    
    # 加载 prompt 模板
    logger.info(f"加载 prompt 模板: {args.prompt_file}")
    prompt_template = load_prompt_template(args.prompt_file)
    
    # 加载数据
    human_parcels = load_human_parcels(args.human_parcel_file)
    llm_parcels = load_llm_parcels(args.llm_parcel_file)
    
    logger.info(f"总共需要评估 {len(human_parcels)} × {len(llm_parcels)} = {len(human_parcels) * len(llm_parcels)} 对 Parcel")
    
    # 加载已有结果
    existing_results = {}
    if args.skip_existing:
        existing_results = load_existing_results(args.output_file)
        if existing_results:
            logger.info(f"找到 {len(existing_results)} 个已存在的评估结果")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # 存储所有结果
    all_results = []
    similarity_results = {}  # {(human_parcel_id, llm_parcel_id): {'rating': int, 'justification': str}}
    
    # 加载已有的详细结果
    json_file = args.output_file.replace('.csv', '_detailed_results.json')
    if args.skip_existing and os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'results' in data:
                    for result in data['results']:
                        h_id = result.get('human_parcel_id')
                        l_id = result.get('llm_parcel_id')
                        if h_id is not None and l_id is not None:
                            key = (h_id, l_id)
                            similarity_results[key] = {
                                'rating': result.get('rating'),
                                'justification': result.get('justification', '')
                            }
                            all_results.append(result)
        except Exception as e:
            logger.warning(f"无法读取已有详细结果文件: {e}")
            import traceback
            traceback.print_exc()
    
    # 评估每对 Parcel
    total_pairs = len(human_parcels) * len(llm_parcels)
    processed_count = 0
    skipped_count = 0
    success_count = 0
    fail_count = 0
    
    for i, human_parcel in enumerate(human_parcels):
        for j, llm_parcel in enumerate(llm_parcels):
            processed_count += 1
            key = (human_parcel['parcel_id'], llm_parcel['parcel_id'])
            
            # 检查是否已存在
            if args.skip_existing and key in similarity_results:
                skipped_count += 1
                logger.info(f"[{processed_count}/{total_pairs}] 跳过已存在的评估: Human Parcel {human_parcel['parcel_id']} vs LLM Parcel {llm_parcel['parcel_id']}")
                continue
            
            logger.info(f"[{processed_count}/{total_pairs}] 评估: Human Parcel {human_parcel['parcel_id']} ({human_parcel['function_name']}) vs LLM Parcel {llm_parcel['parcel_id']} ({llm_parcel['function_name']})")
            
            try:
                rating, justification = evaluate_similarity(
                    human_parcel, llm_parcel, prompt_template,
                    args.vllm_url, args.api_key
                )
                
                # 保存结果
                result = {
                    'human_parcel_id': human_parcel['parcel_id'],
                    'human_parcel_name': human_parcel['parcel_name'],
                    'human_function_name': human_parcel['function_name'],
                    'llm_parcel_id': llm_parcel['parcel_id'],
                    'llm_function_name': llm_parcel['function_name'],
                    'rating': rating,
                    'justification': justification
                }
                
                similarity_results[key] = {
                    'rating': rating,
                    'justification': justification
                }
                all_results.append(result)
                success_count += 1
                
                logger.info(f"✓ 评估完成: rating={rating}")
                
                # 增量保存
                save_detailed_results(args.output_file, all_results)
                
            except Exception as e:
                fail_count += 1
                logger.error(f"✗ 评估失败: {e}")
                import traceback
                traceback.print_exc()
            
            # 添加延迟避免 API 限流
            if processed_count < total_pairs:
                time.sleep(args.delay)
    
    # 构建相似度矩阵并保存
    logger.info("构建相似度矩阵...")
    similarity_df = build_similarity_matrix(human_parcels, llm_parcels, similarity_results)
    similarity_df.to_csv(args.output_file)
    logger.info(f"相似度矩阵已保存到: {args.output_file}")
    
    # 最终保存详细结果
    save_detailed_results(args.output_file, all_results)
    
    logger.info(f"\n处理完成:")
    logger.info(f"  总对数: {total_pairs}")
    logger.info(f"  跳过: {skipped_count}")
    logger.info(f"  成功: {success_count}")
    logger.info(f"  失败: {fail_count}")
    logger.info(f"  相似度矩阵形状: {similarity_df.shape}")
    logger.info(f"  结果已保存到: {args.output_file}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())

