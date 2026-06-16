#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Schaefer2018_100Parcels_7Networks 生成功能描述

读取 CSV 文件中的 cognitive term 关联强度（z-score），
调用 LLM 为每个 Parcel 生成功能描述，输出为 JSON 格式。
"""

import os
import sys
import csv
import json
import argparse
import time
import re
from typing import Dict, List, Tuple, Any
from pathlib import Path

import requests

# 默认配置
DEFAULT_VLLM_URL = "http://0.0.0.0:8001/v1"
DEFAULT_API_KEY = "abcabc"
DEFAULT_CSV_PATH = "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/ns_scale100.csv"
DEFAULT_OUTPUT_DIR = "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description"


def call_vllm_api(prompt: str, vllm_url: str = DEFAULT_VLLM_URL, api_key: str = DEFAULT_API_KEY,
                  max_tokens: int = 1024, temperature: float = 0.0, timeout: int = 120) -> str:
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
        print(f"✓ API 连接成功: {vllm_url}")
        return True
    except Exception as e:
        print(f"✗ API 连接失败: {e}")
        return False


def load_parcel_data(csv_path: str) -> List[Tuple[str, List[Tuple[str, float]]]]:
    """加载 CSV 文件，返回 [(parcel_name, [(term, score), ...]), ...]"""
    parcels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # 获取所有 cognitive terms（除了第一列）
        # 第一列列名是空的，用于存储 parcel name
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise Exception("CSV 文件没有列名")
        
        # 第一列是空的列名，用于 parcel name
        parcel_key = fieldnames[0]  # 通常是空字符串 ''
        terms = [col for col in fieldnames[1:] if col]  # 跳过第一列，获取所有 cognitive terms
        
        for row in reader:
            parcel_name = row.get(parcel_key, '').strip()
            if not parcel_name:
                continue
            
            # 收集所有 term 和对应的 score
            term_scores = []
            for term in terms:
                try:
                    score_str = row.get(term, '').strip()
                    if not score_str:
                        continue
                    score = float(score_str)
                    term_scores.append((term, score))
                except (ValueError, KeyError) as e:
                    print(f"警告: Parcel {parcel_name}, term {term} 无法解析: {e}")
                    continue
            
            # 按 score 绝对值降序排序
            term_scores.sort(key=lambda x: abs(x[1]), reverse=True)
            parcels.append((parcel_name, term_scores))
    
    return parcels


def safe_filename(parcel_name: str) -> str:
    """将 parcel name 转换为安全的文件名"""
    # 替换可能不安全的字符
    safe = parcel_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe = safe.replace('*', '_').replace('?', '_').replace('"', '_')
    safe = safe.replace('<', '_').replace('>', '_').replace('|', '_')
    return safe


def build_prompt(parcel_name: str, term_scores: List[Tuple[str, float]]) -> str:
    """构建 LLM prompt，输出格式与 LLM Parcel 分析对齐"""
    # 只使用前50个最重要的 term_scores，避免 prompt 过长
    top_terms = term_scores[:50]
    # 格式化 term_scores 为字符串
    term_scores_str = "\n".join([f"({term}, {score:.4f})" for term, score in top_terms])
    
    prompt = f"""You are a cognitive neuroscience expert analyzing the functional specialization of brain regions. Please analyze the functionality of Parcel {parcel_name} based on the following information:

**Parcel {parcel_name} Cognitive Term Association Information:**

**Cognitive terms and association strengths (z-scores):**
(term, score)
{term_scores_str}

**Analysis Requirements:**
1. Based on the cognitive terms and their association strengths (z-scores), analyze what type of information or tasks this brain parcel primarily processes
2. Use terminology similar to human brain region descriptions (e.g., visual processing, language comprehension, memory encoding, executive control, attention, motor control, etc.)
3. Provide a concise function name (2-4 words)
4. Provide a detailed function description (100-200 words) describing the primary functional roles of this parcel based on the cognitive terms and their association strengths
5. Analyze the potential role of this parcel in the overall functionality of the human brain
6. Provide reasoning that explains how you derived the function description from the cognitive terms and their z-scores

Please respond in STRICT JSON format with the following structure:

{{
  "function_name": "concise function name (2-4 words)",
  "function_description": "detailed function description (100-200 words)",
  "role_in_human_brain": "analysis of the parcel's role in overall brain functionality",
  "reasoning": "explanation of how the cognitive terms and their z-scores led to the function description, including which terms were most influential and how they were interpreted"
}}

Important:
- Do not invent functions that are not supported by the terms.
- Focus on terms with higher absolute z-scores (stronger associations).
- Write in academic tone, similar to how neuroscientists describe brain region functions.
- Return ONLY valid JSON, no additional text or markdown formatting."""
    
    return prompt


def generate_parcel_description(parcel_name: str, term_scores: List[Tuple[str, float]], 
                                vllm_url: str, api_key: str) -> Dict[str, Any]:
    """为单个 Parcel 生成描述，返回结果字典"""
    try:
        if not term_scores:
            raise Exception(f"{parcel_name} 没有有效的 term scores")
        
        prompt = build_prompt(parcel_name, term_scores)
        print(f"正在处理: {parcel_name} (共 {len(term_scores)} 个 terms)")
        response = call_vllm_api(prompt, vllm_url, api_key, max_tokens=10000, timeout=120)
        result = parse_json_response(response)
        
        # 使用 parcel_name 作为 parcel_id
        result['parcel_id'] = parcel_name
        
        # 验证必需的字段（与 LLM Parcel 分析格式对齐）
        required_fields = ['function_name', 'function_description', 'role_in_human_brain', 'reasoning']
        for field in required_fields:
            if field not in result:
                raise Exception(f"响应中缺少 '{field}' 字段")
            if not isinstance(result.get(field), str):
                raise Exception(f"'{field}' 必须是字符串类型")
        
        # 验证 function_name 长度（2-4个词）
        function_name = result['function_name'].strip()
        word_count = len(function_name.split())
        if word_count < 2 or word_count > 4:
            print(f"警告: function_name 应该包含 2-4 个词，当前为 {word_count} 个词: {function_name}")
        
        print(f"✓ 完成: {parcel_name}")
        return result
        
    except Exception as e:
        print(f"✗ 处理 {parcel_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="为 Parcel 生成功能描述")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH,
                        help="输入 CSV 文件路径")
    parser.add_argument("--output_file", type=str, 
                        default=os.path.join(DEFAULT_OUTPUT_DIR, "parcel_descriptions.json"),
                        help="输出 JSON 文件路径")
    parser.add_argument("--vllm_url", type=str, default=DEFAULT_VLLM_URL,
                        help="vLLM API URL")
    parser.add_argument("--api_key", type=str, default=DEFAULT_API_KEY,
                        help="API Key")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的 parcel（从输出文件中读取）")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                        help="不跳过已存在的 parcel，重新生成")
    parser.add_argument("--check_api", action="store_true", default=True,
                        help="检查 API 是否可用")
    
    args = parser.parse_args()
    
    # 检查 API
    if args.check_api:
        if not check_api_available(args.vllm_url, args.api_key):
            print("API 不可用，退出")
            sys.exit(1)
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"正在加载数据: {args.csv_path}")
    parcels = load_parcel_data(args.csv_path)
    print(f"加载了 {len(parcels)} 个 Parcels")
    
    # 加载已有的结果（如果存在且需要跳过）
    existing_results = {}
    if args.skip_existing and os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                existing_list = json.load(f)
                if isinstance(existing_list, list):
                    for item in existing_list:
                        parcel_id = item.get('parcel_id')
                        if parcel_id:
                            existing_results[parcel_id] = item
                    print(f"从已有文件加载了 {len(existing_results)} 个 Parcel 的结果")
        except Exception as e:
            print(f"警告: 无法读取已有文件 {args.output_file}: {e}")
    
    # 处理每个 Parcel
    all_results = []
    success_count = 0
    fail_count = 0
    
    for i, (parcel_name, term_scores) in enumerate(parcels, 1):
        print(f"\n[{i}/{len(parcels)}] ", end="")
        
        # 检查是否已存在
        if args.skip_existing and parcel_name in existing_results:
            print(f"跳过已存在的 Parcel: {parcel_name}")
            all_results.append(existing_results[parcel_name])
            success_count += 1
        else:
            try:
                result = generate_parcel_description(parcel_name, term_scores, args.vllm_url, args.api_key)
                all_results.append(result)
                success_count += 1
            except Exception as e:
                print(f"处理失败: {e}")
                fail_count += 1
        
        # 每处理一个就保存一次（增量保存）
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        # 添加延迟避免 API 限流
        if i < len(parcels):
            time.sleep(0.5)
    
    # 最终保存（确保所有结果都在）
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成: 成功 {success_count}, 失败 {fail_count}")
    print(f"结果已保存到: {args.output_file} (包含 {len(all_results)} 个 Parcels)")


if __name__ == "__main__":
    main()

