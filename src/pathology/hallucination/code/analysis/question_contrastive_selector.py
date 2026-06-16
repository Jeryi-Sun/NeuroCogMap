#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Question Contrastive Selector - 精简版对比样例选择脚本

功能：
1) 读取正确/幻觉样本的 JSONL（含 index, question, model_answer, score）
2) 调用 LLM 对全部 question 进行分组分类（5-10 类），得到 question->category 映射
3) 在每个类别中筛选 response 长度在 [10,100] 词、置信度较高的样本
4) 构造直接对比配对（幻觉 vs 正确）
5) 调用 LLM 对直接配对进行精细复筛，选出更优对比样例
6) 保存 JSON：categories, question_to_category, direct_pairs, llm_refined_pairs（均保留原始编号）

注意：
- 遵循用户要求：遇到异常需要报告（日志/异常），不做无声默认值吞掉
- 支持 skip_existing：若输出已存在且开启则跳过
"""

import json
import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import requests
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_VLLM_URL = "http://0.0.0.0:8001/v1"
DEFAULT_API_KEY = "abcabc"


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件。出现错误时记录并抛出。"""
    data: List[Dict[str, Any]] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except Exception as e:
                    logger.error(f"{file_path} 第{line_num}行解析失败: {e}")
                    raise
    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        raise
    except Exception as e:
        logger.error(f"加载JSONL失败: {file_path}, 错误: {e}")
        raise
    return data


def call_vllm_api(prompt: str, vllm_url: str, api_key: str,
                  max_tokens: int = 2048, temperature: float = 0.0,
                  timeout: int = 120) -> str:
    """调用 vLLM /chat/completions 接口，返回文本。出错抛异常。"""
    payload = {
        "model": "/path/to/local_models/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "你是一个严谨的研究助理，擅长对自然语言问题进行分类与比较。"},
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
        raise RuntimeError(f"响应格式不正确: {data}")
    except Exception as e:
        logger.error(f"vLLM API调用失败: {e}")
        raise


def count_words(text: str) -> int:
    return len([w for w in (text or '').strip().split() if w])


def embed_questions(questions: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """对问题进行句向量编码。"""
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"加载句向量模型失败: {e}")
        raise
    try:
        embeddings = model.encode(questions)
    except Exception as e:
        logger.error(f"问题编码失败: {e}")
        raise
    return np.array(embeddings)


def parse_json_response(text: str) -> Dict[str, Any]:
    # 提取最外层 JSON
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("未找到有效的JSON片段")
    json_str = text[start:end+1]
    return json.loads(json_str)


def make_direct_pairs_by_category(
    category_to_indices: Dict[str, List[int]],
    all_items: Dict[int, Dict[str, Any]],
    min_words: int,
    max_words: int,
    min_confidence: float,
    max_pairs_per_cat: int,
    index_to_embedding: Optional[Dict[int, np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """在每个类别内构造直接对比配对（幻觉 vs 正确）。"""
    pairs: List[Dict[str, Any]] = []
    for cat, idx_list in category_to_indices.items():
        correct_candidates: List[Dict[str, Any]] = []
        halluc_candidates: List[Dict[str, Any]] = []
        for idx in idx_list:
            item = all_items.get(idx)
            if not item:
                logger.warning(f"类别 {cat} 中的索引 {idx} 未在数据集中找到")
                continue
            ans = item.get('model_answer', '')
            score = float(item.get('score', 0.0) or 0.0)
            w = count_words(ans)
            if w < min_words or w > max_words:
                continue
            if score < min_confidence:
                continue
            if bool(item.get('is_correct', False)):
                correct_candidates.append(item)
            else:
                halluc_candidates.append(item)
        if not correct_candidates or not halluc_candidates:
            continue
        # 相似度：优先使用嵌入余弦相似度；可加长度相似度微调
        def compare(h: Dict[str, Any], c: Dict[str, Any]) -> float:
            if index_to_embedding is not None:
                hv = index_to_embedding.get(h['index'])
                cv = index_to_embedding.get(c['index'])
                if hv is not None and cv is not None:
                    cos = float(cosine_similarity(hv.reshape(1, -1), cv.reshape(1, -1))[0, 0])
                    h_len = count_words(h.get('model_answer', ''))
                    c_len = count_words(c.get('model_answer', ''))
                    len_sim = 1.0 - abs(h_len - c_len) / max(h_len, c_len)
                    return 0.9 * cos + 0.1 * len_sim
            # 回退到词袋 + 长度
            h_words = set((h.get('question') or '').lower().split())
            c_words = set((c.get('question') or '').lower().split())
            jacc = len(h_words & c_words) / max(1, len(h_words | c_words))
            h_len = count_words(h.get('model_answer', ''))
            c_len = count_words(c.get('model_answer', ''))
            len_sim = 1.0 - abs(h_len - c_len) / max(h_len, c_len)
            return 0.7 * jacc + 0.3 * len_sim
        # 为每个幻觉挑一个最佳正确
        used_c: set = set()
        local_pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for h in halluc_candidates:
            best_c = None
            best_s = 0.0
            best_c_key = None
            for c in correct_candidates:
                c_key = c['index']
                if c_key in used_c:
                    continue
                s = compare(h, c)
                if s > best_s:
                    best_s = s
                    best_c = c
                    best_c_key = c_key
            if best_c is not None and best_s >= 0.2:
                used_c.add(best_c_key)
                local_pairs.append((best_s, h, best_c))
        # 取前若干对
        local_pairs.sort(key=lambda x: x[0], reverse=True)
        for score_sim, h, c in local_pairs[:max_pairs_per_cat]:
            pairs.append({
                'category': cat,
                'hallucination_index': h['index'],
                'correct_index': c['index'],
                'similarity_score': float(score_sim),
                'hallucination_score': float(h.get('score', 0.0) or 0.0),
                'correct_score': float(c.get('score', 0.0) or 0.0)
            })
    return pairs


def build_llm_refine_prompt(pair: Dict[str, Any],
                            h_item: Dict[str, Any], c_item: Dict[str, Any]) -> str:
    prompt = f"""
请评估以下一对问答是否构成高质量的对比样例（幻觉 vs 正确），并仅返回 JSON：
字段：{{
  "is_good_pair": true/false,
  "overall_score": 0-10,
  "reasons": ["...", "..."],
  "improvement": "..."
}}

类别: {pair.get('category')}

幻觉样本:
- index: {h_item.get('index')}
- question: {h_item.get('question')}
- answer: {h_item.get('model_answer')}
- confidence: {h_item.get('score')}

正确样本:
- index: {c_item.get('index')}
- question: {c_item.get('question')}
- answer: {c_item.get('model_answer')}
- confidence: {c_item.get('score')}

请仅输出 JSON，不要包含多余文字。
"""
    return prompt.strip()


def refine_pairs_with_llm(
    pairs: List[Dict[str, Any]],
    all_items: Dict[int, Dict[str, Any]],
    vllm_url: str,
    api_key: str,
    top_k_overall: int
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for p in pairs:
        h = all_items.get(p['hallucination_index'])
        c = all_items.get(p['correct_index'])
        if h is None or c is None:
            logger.warning(f"LLM复筛跳过：索引缺失 h={p['hallucination_index']} c={p['correct_index']}")
            continue
        prompt = build_llm_refine_prompt(p, h, c)
        try:
            resp = call_vllm_api(prompt, vllm_url=vllm_url, api_key=api_key, max_tokens=10000, temperature=0.1)
            data = parse_json_response(resp)
            is_good = bool(data.get('is_good_pair', False))
            overall = float(data.get('overall_score', 0.0) or 0.0)
            out = {
                'category': p.get('category'),
                'hallucination_index': p['hallucination_index'],
                'correct_index': p['correct_index'],
                'similarity_score': p.get('similarity_score', 0.0),
                'overall_score': overall,
                'is_good_pair': is_good,
                'reasons': data.get('reasons', []),
                'improvement': data.get('improvement', '')
            }
            results.append(out)
        except Exception as e:
            logger.warning(f"LLM复筛失败，跳过该对: h={p['hallucination_index']} c={p['correct_index']} 错误: {e}")
            continue
    # 选择总分最高的若干
    results.sort(key=lambda x: x.get('overall_score', 0.0), reverse=True)
    return results[:top_k_overall]


def save_outputs(
    output_path: str,
    categories: List[str],
    question_to_category: Dict[int, str],
    direct_pairs: List[Dict[str, Any]],
    llm_refined_pairs: List[Dict[str, Any]]
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'categories': categories,
        'question_to_category': {str(k): v for k, v in question_to_category.items()},
        'direct_pairs': direct_pairs,
        'llm_refined_pairs': llm_refined_pairs
    }
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Question Contrastive Selector - 精简对比选择')
    parser.add_argument('--correct_jsonl', type=str, required=True, help='正确样本 JSONL 路径')
    parser.add_argument('--incorrect_jsonl', type=str, required=True, help='幻觉样本 JSONL 路径')
    parser.add_argument('--output', type=str, required=True, help='输出 JSON 路径')
    parser.add_argument('--vllm_url', type=str, default=DEFAULT_VLLM_URL, help='vLLM API 地址')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY, help='API 密钥')
    parser.add_argument('--min_categories', type=int, default=5, help='最少类别数(KMeans最小簇数)')
    parser.add_argument('--max_categories', type=int, default=10, help='最多类别数(KMeans最大簇数)')
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2', help='句向量模型')
    parser.add_argument('--min_words', type=int, default=10, help='答案最少词数')
    parser.add_argument('--max_words', type=int, default=100, help='答案最多词数')
    parser.add_argument('--min_confidence', type=float, default=0.5, help='最小置信度阈值')
    parser.add_argument('--pairs_per_category', type=int, default=5, help='每类直接配对数量上限')
    parser.add_argument('--refined_pairs_topk', type=int, default=30, help='LLM复筛全局保留数量')
    parser.add_argument('--use_llm', action='store_true', default=False, help='是否启用 LLM 分类与复筛')
    parser.add_argument('--skip_existing', action='store_true', help='结果存在则跳过')
    args = parser.parse_args()

    # 跳过已存在
    if args.skip_existing and os.path.exists(args.output):
        logger.info(f"输出已存在且 skip_existing 启用，跳过: {args.output}")
        return

    # 读取数据
    correct = load_jsonl(args.correct_jsonl)
    incorrect = load_jsonl(args.incorrect_jsonl)

    # 汇总到统一索引表
    all_items: Dict[int, Dict[str, Any]] = {}
    indexed_questions: List[Tuple[int, str]] = []
    for item in correct:
        idx = int(item.get('index'))
        q = item.get('question', '')
        all_items[idx] = {
            'index': idx,
            'question': q,
            'model_answer': item.get('model_answer', ''),
            'score': item.get('score', 0.0),
            'is_correct': True
        }
        indexed_questions.append((idx, q))
    for item in incorrect:
        idx = int(item.get('index'))
        q = item.get('question', '')
        all_items[idx] = {
            'index': idx,
            'question': q,
            'model_answer': item.get('model_answer', ''),
            'score': item.get('score', 0.0),
            'is_correct': False
        }
        indexed_questions.append((idx, q))

    # 基于嵌入的聚类（替代 LLM 分类）
    categories: List[str] = []
    question_to_category: Dict[int, str] = {}
    try:
        questions = [q for _, q in indexed_questions]
        if len(questions) < max(2, args.min_categories):
            raise ValueError("问题数量不足以进行聚类")
        embeddings = embed_questions(questions, model_name=args.embed_model)
        # 尝试 K=min..max，选 silhouette 最优
        best_k = None
        best_score = -1.0
        best_labels = None
        for k in range(max(2, args.min_categories), max(2, args.max_categories) + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = km.fit_predict(embeddings)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
            except Exception as e:
                logger.warning(f"KMeans(k={k}) 失败或评分失败: {e}")
                continue
        if best_labels is None:
            raise ValueError("无法完成有效聚类")
        logger.info(f"最优聚类: K={best_k}, silhouette={best_score:.4f}")
        # 赋予类别名并建立映射
        categories = [f"cluster_{i}" for i in range(best_k)]
        # 建立 index -> category
        for (idx, _), label in zip(indexed_questions, best_labels):
            question_to_category[idx] = f"cluster_{int(label)}"
        # 准备 index->embedding
        index_to_embedding: Dict[int, np.ndarray] = {}
        for (idx, _), emb in zip(indexed_questions, embeddings):
            index_to_embedding[idx] = emb
    except Exception as e:
        logger.error(f"聚类失败: {e}")
        raise

    # 构造类别到索引映射
    category_to_indices: Dict[str, List[int]] = {}
    for idx, cat in question_to_category.items():
        category_to_indices.setdefault(cat, []).append(idx)

    # 直接配对
    direct_pairs = make_direct_pairs_by_category(
        category_to_indices=category_to_indices,
        all_items=all_items,
        min_words=args.min_words,
        max_words=args.max_words,
        min_confidence=args.min_confidence,
        max_pairs_per_cat=args.pairs_per_category,
        index_to_embedding=index_to_embedding,
    )
    logger.info(f"直接配对完成，共 {len(direct_pairs)} 对")

    # LLM 复筛
    llm_refined_pairs: List[Dict[str, Any]] = []
    if args.use_llm and direct_pairs:
        llm_refined_pairs = refine_pairs_with_llm(
            pairs=direct_pairs,
            all_items=all_items,
            vllm_url=args.vllm_url,
            api_key=args.api_key,
            top_k_overall=args.refined_pairs_topk,
        )
        logger.info(f"LLM 复筛完成，保留 {len(llm_refined_pairs)} 对")

    # 保存
    save_outputs(
        output_path=args.output,
        categories=categories,
        question_to_category=question_to_category,
        direct_pairs=direct_pairs,
        llm_refined_pairs=llm_refined_pairs
    )


if __name__ == "__main__":
    main()
