# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import torch
from tqdm import tqdm
import pandas as pd
import random
from sae_lens import SAE, HookedSAETransformer
import re
import numpy as np
import scipy.sparse
import json
import gc
import glob
import logging
import datetime
import sys
from pathlib import Path

try:
    from neurocogmap_release.paths import data_path, env_path_str, output_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from neurocogmap_release.paths import data_path, env_path_str, output_path

# 数据筛选说明：根据capability_dataset_stats.json统计，以下数据集平均长度超过1024，将被筛选掉
# - hotpotqa: 5981.737 (问题平均长度)
# - cnndaily_mail: 3722.597 (问题平均长度)  
# - narrativeqa: 3192.609 (问题平均长度)
# - drop: 1128.291 (问题平均长度)
# - mlqa: 1209.607 (问题平均长度)
# - adversarialqa: 1065.162 (问题平均长度)

# 设备选择
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sae_for_layers(layer_paths, sae_release, sae_local_base_dir, n_devices=1, force_device=None):
    """
    加载SAE模型到指定设备
    Args:
        layer_paths: SAE层路径列表
        sae_release: SAE发布版本
        sae_local_base_dir: 本地SAE基础目录
        n_devices: 设备数量
        force_device: 强制指定设备（用于错误重试）
    """
    sae_list = []
    hook_names = []
    for path in layer_paths:
        m = re.search(r"l(\d+)r", path)
        if m:
            id = m.group(1)
            real_path = f"Llama3_1-8B-Base-L{id}R-8x"
        else:
            real_path = path

        # 检查 Llama Scope SAE 的本地文件路径 (final.safetensors 在 checkpoints 子目录)
        llama_scope_local_path = os.path.join(sae_local_base_dir, real_path, "checkpoints", "final.safetensors")
        # 检查 Gemma SAE 的本地文件路径 (params.npz)
        gemma_local_path = os.path.join(sae_local_base_dir, path, "params.npz")
        
        if os.path.exists(llama_scope_local_path):
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=path,
                device=device,
                local_path=llama_scope_local_path
            )
        elif os.path.exists(gemma_local_path):
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=path,
                device=device,
                local_path=gemma_local_path
            )
        else:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=path,
                device=device,
            )
        if force_device is not None:
            # 强制使用指定设备（用于错误重试）
            sae.to(force_device)
        elif n_devices > 1:  
            sae.to("cuda:1")
        else:
            sae.to(device)
        sae.use_error_term = True
        sae_list.append(sae)
        hook_names.append(sae.cfg.metadata.hook_name if hasattr(sae.cfg, 'metadata') else sae.cfg.hook_name)
    return sae_list, hook_names

def format_qa_prompt(question, answer):
    return f"Question: {question}\nAnswer: {answer}"

def format_qa_prompt_instruct(question, answer, tokenizer):
    """为 instruct 模型格式化 QA prompt，使用 tokenizer 的 apply_chat_template"""
    try:
        # 构建对话格式的消息
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        
        # 使用 tokenizer 的 apply_chat_template 方法
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return prompt
    except Exception as e:
        # 如果出错，回退到简单格式
        print(f"Warning: Failed to apply chat template: {e}")
        return f"Question: {question}\nAnswer: {answer}"

def get_formatted_prompt(question, answer, is_instruct=False, tokenizer=None):
    """根据模型类型选择合适的 prompt 格式化方法"""
    if is_instruct and tokenizer is not None:
        return format_qa_prompt_instruct(question, answer, tokenizer)
    else:
        return format_qa_prompt(question, answer)



def split_sentences(text):
    """按中英文标点分句，返回[(start, end, sentence), ...]"""
    # 更智能的句子切分模式，避免在数字、数学公式等中错误分句
    
    # 预处理：标记需要保护的区域（数字、数学公式等）
    protected_spans = []
    
    # 保护数字（包括小数、科学计数法等）
    number_pattern = r'\d+\.\d+|\d+\.|\d+e[+-]?\d+|\d+'
    for match in re.finditer(number_pattern, text):
        protected_spans.append((match.start(), match.end()))
    
    # 保护数学公式（简单的数学表达式）
    math_pattern = r'[0-9+\-*/()=<>≤≥≠≈±∞∑∏∫∂√^]+'
    for match in re.finditer(math_pattern, text):
        # 检查是否是连续的数学表达式
        start, end = match.start(), match.end()
        # 扩展匹配范围，包含相邻的数学符号
        while end < len(text) and text[end] in '0123456789+-*/()=<>≤≥≠≈±∞∑∏∫∂√^.,;':
            end += 1
        while start > 0 and text[start-1] in '0123456789+-*/()=<>≤≥≠≈±∞∑∏∫∂√^.,;':
            start -= 1
        protected_spans.append((start, end))
    
    # 保护引号内的内容（包括嵌套引号）- 改进版本
    # 处理不同类型的引号
    quote_patterns = [
        r'["""].*?["""]',  # 中文引号
        r"[''].*?['']",    # 中文单引号
        r'"[^"]*"',        # 英文双引号
        r"'[^']*'",        # 英文单引号
    ]
    for pattern in quote_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            protected_spans.append((match.start(), match.end()))
    
    # 保护括号内的内容
    bracket_pattern = r'\([^)]*\)'
    for match in re.finditer(bracket_pattern, text):
        protected_spans.append((match.start(), match.end()))
    
    # 保护常见缩写（改进的模式）
    abbreviation_patterns = [
        r'\b[A-Z]\.\s*[A-Z]\.\s*[A-Z]\.',  # U.S.A., U.K., etc.
        r'\b[A-Z][a-z]*\.\s*[A-Z][a-z]+',  # Mr. Smith, Dr. Wang, etc.
        r'\b[A-Z]\.\s*[A-Z]\.',  # U.S., U.K., etc.
        r'\b[A-Z][a-z]*\.',  # Mr., Dr., etc.
    ]
    for pattern in abbreviation_patterns:
        for match in re.finditer(pattern, text):
            protected_spans.append((match.start(), match.end()))
    
    # 合并重叠的保护区域
    if protected_spans:
        protected_spans.sort()
        merged_spans = [protected_spans[0]]
        for start, end in protected_spans[1:]:
            last_start, last_end = merged_spans[-1]
            if start <= last_end:
                merged_spans[-1] = (last_start, max(end, last_end))
            else:
                merged_spans.append((start, end))
        protected_spans = merged_spans
    
    # 创建保护标记
    protected_chars = set()
    for start, end in protected_spans:
        for i in range(start, end):
            protected_chars.add(i)
    
    # 预处理：将连续的重复标点标记为保护区域
    i = 0
    while i < len(text):
        if text[i] in '!?' and i + 1 < len(text) and text[i + 1] == text[i]:
            # 找到连续重复的结束
            repeat_end = i
            while repeat_end < len(text) and text[repeat_end] == text[i]:
                repeat_end += 1
            # 将整个重复序列标记为保护区域
            for j in range(i, repeat_end):
                protected_chars.add(j)
            i = repeat_end
        else:
            i += 1
    
    # 智能分句：只在非保护区域的分句符号处分割
    sentences = []
    current_start = 0
    i = 0
    
    while i < len(text):
        char = text[i]
        
        # 检查是否是句子结束标记
        if char in '。！？!?.' and i not in protected_chars:
            # 检查是否是省略号
            if char == '.' and i + 2 < len(text) and text[i:i+3] == '...':
                i += 3
                continue
            
            # 检查前面的字符，避免在数字后的小数点分割
            if i > 0:
                prev_char = text[i - 1]
                if prev_char.isdigit() and char == '.':
                    i += 1
                    continue
            
            # 检查是否是缩写（更严格的检查）
            if char == '.':
                # 检查前面是否有大写字母
                j = i - 1
                while j >= 0 and text[j].isupper():
                    j -= 1
                if j >= 0 and text[j].isspace():
                    # 可能是缩写，继续检查
                    i += 1
                    continue
            
            # 真正的句子结束
            sentence = text[current_start:i+1].strip()
            if sentence:
                sentences.append((current_start, i+1, sentence))
            current_start = i + 1
        
        i += 1
    
    # 处理剩余的文本
    if current_start < len(text):
        remaining = text[current_start:].strip()
        if remaining:
            sentences.append((current_start, len(text), remaining))
    
    # 如果没有找到任何句子，将整个文本作为一个句子
    if not sentences:
        sentences = [(0, len(text), text)]
    
    return sentences

# 加载 capability_data 目录下所有 json 文件
def load_all_qa_datasets(data_dir, logger=None, is_test_mode=False):
    """
    兼容两种数据格式：
    - 训练/常规模式：*_qa.json（如 adversarial_qa.json）
    - test 模式：*_test.json（如 adversarial_test.json）
    """
    # 需要筛选掉的数据集（平均长度超过1024）——只在常规模式下使用
    filtered_datasets = [
        # "proofwriter",
        # "race",
        # "narrativeqa",
        # "timedial",
        # "math",
        # "cnn_dailymail",
        # "mlqa",
        # "tabfact",
        # "drop",
    ]

    datasets = {}
    if logger:
        logger.info(f"正在扫描目录: {data_dir}")

    pattern = '*_test.json' if is_test_mode else '*_qa.json'
    for path in glob.glob(os.path.join(data_dir, pattern)):
        base = os.path.splitext(os.path.basename(path))[0]
        if is_test_mode:
            name = base.replace('_test', '')
        else:
            name = base.replace('_qa', '')

        # 常规模式下按统计做筛选；test 模式默认不筛（避免误伤）
        if (not is_test_mode) and (name in filtered_datasets):
            if logger:
                logger.info(f"跳过数据集 {name}")
            continue

        if logger:
            logger.info(f"加载数据集文件: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if logger:
            logger.info(f"数据集 {name} 加载完成，样本数: {len(data)}")
        datasets[name] = data
    if logger:
        logger.info(f"共加载数据集数量: {len(datasets)}")
    return datasets




def process_dataset_with_layers(qa_data, dataset_name, sae_layer_paths, sae_release, sae_local_base_dir, layer_start=0, layer_end=25, layers_per_batch=2, output_dir="./output", baseline_direct_neural=False, is_instruct=False, is_rerun=False):
    # 如果output_dir存在则清除里面的文件，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    if is_rerun:
        sparse_filename = f"{dataset_name}_layer{layer_start}_sparse.npz"
        if os.path.exists(os.path.join(output_dir, sparse_filename)):
            os.remove(os.path.join(output_dir, sparse_filename))
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
            print(f"删除文件: {file}")
    meta_filename = f"{dataset_name}_meta.json"
    meta_path = os.path.join(output_dir, meta_filename)
    logger.info(f"数据集 {dataset_name} 开始分层处理，层范围: {layer_start}-{layer_end}，每 batch {layers_per_batch} 层")
    
    # 首先生成meta信息（只生成一次）
    logger.info(f"生成数据集 {dataset_name} 的meta信息...")
    all_meta = []
    sae_row_idx = 0
    
    for item in tqdm(qa_data, desc=f"Generating meta for {dataset_name}"):
        question = item["question"]
        answer = item["answer"]
        prompt = get_formatted_prompt(question, answer, is_instruct=is_instruct, tokenizer=model.tokenizer)
        
        # 获取tokenizer和token信息
        tokens = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True, padding=True, max_length=1024, truncation=True)
        input_ids = tokens["input_ids"].to(device)
        offsets = tokens["offset_mapping"][0]
        answer_start = prompt.rfind(answer)
        answer_end = answer_start + len(answer)
        
        sent_spans = split_sentences(answer)
        sent_meta = []
        
        for sent_idx, (s_start, s_end, sent) in enumerate(sent_spans):
            # 计算句子对应的token范围
            sent_abs_start = answer_start + s_start
            sent_abs_end = answer_start + s_end
            # 包含跨越句子边界的token：token的结束位置在句子范围内，或者token的开始位置在句子范围内
            sent_token_indices = [i for i, (ofs_s, ofs_e) in enumerate(offsets.tolist()) 
                                if (ofs_e > sent_abs_start and ofs_e <= sent_abs_end) or 
                                   (ofs_s < sent_abs_start and ofs_e > sent_abs_start)]
            
            # 过滤特殊token
            ids = input_ids[0].tolist()
            special_ids = set(model.tokenizer.all_special_ids) if hasattr(model.tokenizer, 'all_special_ids') else set()
            sent_valid_indices = [i for i in sent_token_indices if ids[i] not in special_ids]
            
            sent_meta.append({
                "sentence": sent,
                "start": s_start,  # 字符起始位置
                "end": s_end,      # 字符结束位置
                "token_start": sent_valid_indices[0] if sent_valid_indices else -1,  # token起始位置
                "token_end": sent_valid_indices[-1] if sent_valid_indices else -1,   # token结束位置
                "token_count": len(sent_valid_indices),  # 有效token数量
                "sae_row_idx": sae_row_idx
            })
            sae_row_idx += 1
            
        all_meta.append({
            "question": question,
            "answer": answer,
            "sentences": sent_meta
        })
    
    # 保存meta信息（只保存一次）
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    logger.info(f"Meta信息已保存，共生成 {sae_row_idx} 个句子级激活")
    
    # 解析传入的sae路径对应的实际层号
    def extract_layer_num(path_str):
        # 优先匹配新格式：Llama3_1-8B-Base-L0R-8x 中的 L0R
        m = re.search(r"layer_(\d+)/", path_str)
        if m:
            return int(m.group(1))
        # 匹配旧格式：layer_0/ 或 layer_1/
        m = re.search(r"l(\d+)r", path_str)
        return int(m.group(1)) if m else None
    layer_num_to_path = []
    for p in sae_layer_paths:
        ln = extract_layer_num(p)
        if ln is not None and layer_start <= ln <= layer_end:
            layer_num_to_path.append((ln, p))
    # 如果过滤后为空，则直接返回
    if not layer_num_to_path:
        logger.warning(f"在给定层范围 {layer_start}-{layer_end} 内没有匹配到任何SAE路径，退出当前数据集处理")
        return
    # 按层号排序
    layer_num_to_path.sort(key=lambda x: x[0])
    
    # 然后按层批次处理激活数据
    layer_numbers = [ln for ln, _ in layer_num_to_path]
    for batch_start_idx in range(0, len(layer_numbers), layers_per_batch):
        batch_slice = layer_num_to_path[batch_start_idx: batch_start_idx + layers_per_batch]
        current_layers = [ln for ln, _ in batch_slice]
        logger.info(f"处理数据集 {dataset_name}，层 {current_layers[0]}-{current_layers[-1]}")
        current_layer_paths = [p for _, p in batch_slice]
        
        # 设备错误重试逻辑
        max_retries = 3
        retry_count = 0
        sae_list = None
        hook_names = None
        all_acts_per_layer = None  # 在外部初始化，用于异常处理
        current_idx = 0  # 跟踪当前处理的数据项索引，用于CUDA OOM错误处理
        
        while retry_count <= max_retries:
            try:
                # 根据重试次数选择设备
                if retry_count == 0:
                    # 第一次尝试：正常加载
                    sae_list, hook_names = load_sae_for_layers(current_layer_paths, sae_release=sae_release, sae_local_base_dir=sae_local_base_dir, n_devices=args.n_devices)
                elif retry_count == 1:
                    # 第二次尝试：强制使用cuda:0
                    logger.warning(f"设备不匹配错误，重试：将SAE从cuda:1切换到cuda:0")
                    sae_list, hook_names = load_sae_for_layers(current_layer_paths, sae_release=sae_release, sae_local_base_dir=sae_local_base_dir, n_devices=args.n_devices, force_device="cuda:0")
                else:
                    # 第三次尝试：强制使用cuda:1
                    logger.warning(f"设备不匹配错误，重试：将SAE从cuda:0切换到cuda:1")
                    sae_list, hook_names = load_sae_for_layers(current_layer_paths, sae_release=sae_release, sae_local_base_dir=sae_local_base_dir, n_devices=args.n_devices, force_device="cuda:1")
                
                # 处理激活数据
                all_acts_per_layer = [[] for _ in range(len(sae_list))]
                current_idx = 0  # 重置当前索引
                
                for idx, item in enumerate(tqdm(qa_data, desc=f"Processing activations for {dataset_name}")):
                    current_idx = idx  # 更新当前索引
                    meta_item = all_meta[idx]
                    question = item["question"]
                    answer = item["answer"]
                    prompt = get_formatted_prompt(question, answer, is_instruct=is_instruct, tokenizer=model.tokenizer)
                    tokens = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True, padding=True, max_length=1024, truncation=True)
                    input_ids = tokens["input_ids"].to(device)
                    offsets = tokens["offset_mapping"][0]
                    answer_start = prompt.rfind(answer)
                    answer_end = answer_start + len(answer)
                    
                    # 计算整个答案的激活
                    answer_token_indices = [i for i, (ofs_s, ofs_e) in enumerate(offsets.tolist()) 
                                            if (ofs_e > answer_start and ofs_e <= answer_end) or 
                                           (ofs_s < answer_start and ofs_e > answer_start)]
                    ids = input_ids[0].tolist()
                    special_ids = set(model.tokenizer.all_special_ids) if hasattr(model.tokenizer, 'all_special_ids') else set()
                    valid_indices = [i for i in answer_token_indices if ids[i] not in special_ids]
                    
                    if not valid_indices:
                        # 如果没有有效token，为每个句子生成零激活
                        for sent_meta in meta_item["sentences"]:
                            for l in range(len(sae_list)):
                                avg_act = torch.zeros(sae_list[l].cfg.d_sae if not baseline_direct_neural else sae_list[l].cfg.d_in, device=device)
                                all_acts_per_layer[l].append(avg_act.detach().cpu().float().numpy())
                    else:
                        # 计算整个答案的激活
                        pos_slice = slice(valid_indices[0], valid_indices[-1]+1)
                        layers = [int(re.search(r"\.(\d+)\.", h).group(1)) for h in hook_names]
                        if baseline_direct_neural:
                            sae_keys = [f"{h}.hook_sae_input" for h in hook_names]
                        else:
                            sae_keys = [f"{h}.hook_sae_acts_post" for h in hook_names]
                        _, cache = model.run_with_cache_with_saes(
                            input_ids,
                            saes=sae_list,
                            names_filter=sae_keys,
                            stop_at_layer=max(layers)+1,
                            pos_slice=pos_slice
                        )
                        acts_per_layer = [cache[k] for k in sae_keys]  # 每层: [1, answer_len, d_sae]
                        relative_valid_indices = [i - valid_indices[0] for i in valid_indices]
                        
                        # 按句子切分激活
                        for sent_meta in meta_item["sentences"]:
                            # 直接使用预计算的token信息
                            token_start = sent_meta["token_start"]
                            token_end = sent_meta["token_end"]
                            token_count = sent_meta["token_count"]
                            
                            if token_count == 0 or token_start == -1:
                                # 句子没有有效token
                                for l in range(len(sae_list)):
                                    avg_act = torch.zeros(sae_list[l].cfg.d_sae if not baseline_direct_neural else sae_list[l].cfg.d_in, device=device)
                                    all_acts_per_layer[l].append(avg_act.detach().cpu().float().numpy())
                            else:
                                # 计算句子对应的激活
                                sent_relative_indices = [i - valid_indices[0] for i in range(token_start, token_end + 1)]
                                for l, acts in enumerate(acts_per_layer):
                                    if sent_relative_indices:
                                        valid_acts = acts[0, sent_relative_indices, :]
                                        avg_act = valid_acts.mean(dim=0)
                                    else:
                                        avg_act = torch.zeros(acts.shape[2], device=acts.device)
                                    all_acts_per_layer[l].append(avg_act.detach().cpu().float().numpy())
                
                # 如果成功执行到这里，跳出重试循环
                break
                
            except RuntimeError as e:
                error_msg = str(e)
                if "Expected all tensors to be on the same device" in error_msg and retry_count < max_retries:
                    retry_count += 1
                    logger.warning(f"设备不匹配错误 (尝试 {retry_count}/{max_retries+1}): {error_msg}")
                    
                    # 清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 如果还有重试机会，继续循环
                    if retry_count <= max_retries:
                        continue
                
                # 处理 CUDA out of memory 错误：为剩余数据项生成零激活
                elif "CUDA out of memory" in error_msg or "out of memory" in error_msg:
                    logger.warning(f"CUDA 显存不足错误，为剩余数据项生成零激活: {error_msg}")
                    
                    # 清理显存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 确保 all_acts_per_layer 已初始化
                    if sae_list is not None:
                        if all_acts_per_layer is None:
                            all_acts_per_layer = [[] for _ in range(len(sae_list))]
                        
                        # 为当前数据项及剩余数据项的所有句子生成零激活
                        for remaining_idx in range(current_idx, len(qa_data)):
                            remaining_meta_item = all_meta[remaining_idx]
                            for sent_meta in remaining_meta_item["sentences"]:
                                for l in range(len(sae_list)):
                                    avg_act = torch.zeros(sae_list[l].cfg.d_sae if not baseline_direct_neural else sae_list[l].cfg.d_in, device=device)
                                    all_acts_per_layer[l].append(avg_act.detach().cpu().float().numpy())
                        logger.info(f"已为从索引 {current_idx} 开始的 {len(qa_data) - current_idx} 个数据项生成零激活")
                        # 跳出重试循环，继续执行保存步骤
                        break
                else:
                    # 如果不是设备错误或者重试次数用完，重新抛出异常
                    raise e
        
        # 保存稀疏矩阵
        if all_acts_per_layer is None:
            logger.error(f"处理数据集 {dataset_name} 层 {current_layers[0]}-{current_layers[-1]} 时，all_acts_per_layer 未初始化，跳过保存")
            return
        
        for i, (layer, acts) in enumerate(zip(current_layers, all_acts_per_layer)):
            sparse_filename = f"{dataset_name}_layer{layer}_sparse.npz"
            sparse_path = os.path.join(output_dir, sparse_filename)
            if os.path.exists(sparse_path):
                old_sparse = scipy.sparse.load_npz(sparse_path)
                new_rows = scipy.sparse.csr_matrix(np.stack(acts, axis=0))
                combined = scipy.sparse.vstack([old_sparse, new_rows])
            else:
                combined = scipy.sparse.csr_matrix(np.stack(acts, axis=0))
            scipy.sparse.save_npz(sparse_path, combined)
        
        del sae_list, hook_names
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"层 {current_layers[0]}-{current_layers[-1]} 处理完成，已清理显存")

if __name__ == "__main__":

    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理SAE激活数据，支持并行处理与模型/SAE参数化')
    parser.add_argument('--data_dir', type=str,
                       default=None,
                       help='数据目录路径；默认使用 release 内 data/neurocogmap_construction/capability_qa 或 capability_test')
    parser.add_argument('--start', type=int, default=0, 
                       help='起始数据集索引（从0开始）')
    parser.add_argument('--end', type=int, default=-1, 
                       help='结束数据集索引（-1表示处理到最后一个数据集）')
    parser.add_argument('--layers_per_batch', type=int, default=8,
                       help='每批处理的层数')
    parser.add_argument('--layer_start', type=int, default=0,
                       help='起始层索引（将用于过滤 --sae_paths 中的层号）')
    parser.add_argument('--layer_end', type=int, default=25,
                       help='结束层索引（将用于过滤 --sae_paths 中的层号）')
    parser.add_argument('--output_base_dir', type=str,
                       default=str(output_path("neurocogmap_construction", "qa_sae_output")),
                       help='输出基础目录')
    parser.add_argument('--list_datasets', action='store_true',
                       help='列出所有可用的数据集并退出')

    # 新增参数：模型与SAE资源
    parser.add_argument('--model_name', type=str, default="google/gemma-2-2b",
                       help='Transformer 模型名称，如 google/gemma-2-2b 或 google/gemma-2-9b-it')
    parser.add_argument('--sae_release', type=str, default="gemma-scope-2b-pt-res",
                       help='SAE 发布名，如 gemma-scope-2b-pt-res 或 gemma-scope-9b-it-res')
    parser.add_argument('--sae_local_base_dir', type=str,
                       default=env_path_str("NEUROCOGMAP_GEMMA2_SAE_DIR"),
                       help='本地SAE权重的根目录（可用 NEUROCOGMAP_GEMMA2_SAE_DIR 设置；其下应包含 layer_x/.../params.npz）')
    parser.add_argument('--sae_paths', type=str, default="",
                       help='逗号分隔的SAE路径列表，例如 "layer_9/width_16k/average_l0_88,layer_20/width_16k/average_l0_91,layer_31/width_16k/average_l0_76"；若为空则使用内置默认列表')
    parser.add_argument('--n_devices', type=int, default=1,
                       help='模型使用的设备数量')
    parser.add_argument('--log_file', type=str, default=None,
                       help='日志文件路径，如果不指定则自动生成')
    parser.add_argument('--baseline_direct_neural', action='store_true',
                       help='是否使用直接神经激活作为基线，默认使用SAE激活')
    parser.add_argument('--is_instruct', action='store_true',
                       help='是否使用 instruct 模型')
    parser.add_argument('--is_rerun', action='store_true',
                       help='是否重跑')
    parser.add_argument('--test_mode', action='store_true',default=False,
                       help='是否使用 test 模式：读取 *_test.json（如 adversarial_test.json），而非 *_qa.json')
    args = parser.parse_args()

    if args.data_dir is None:
        default_leaf = "capability_test" if args.test_mode else "capability_qa"
        args.data_dir = str(data_path("neurocogmap_construction", default_leaf))
    
    # 设置日志记录
    def setup_logging(log_file=None):
        """设置日志记录，同时输出到控制台和文件"""
        if log_file is None:
            # 自动生成日志文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(output_path("logs", f"sae_activation_log_{timestamp}.log"))
        
        # 创建日志目录
        log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )
        
        return logging.getLogger(__name__)
    
    # 初始化日志记录器
    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("SAE激活数据处理开始")
    logger.info("=" * 60)
    
    # 输出日志文件路径
    if args.log_file:
        logger.info(f"日志文件路径: {os.path.abspath(args.log_file)}")
    else:
        logger.info(f"自动生成日志文件目录: {output_path('logs')}")
    
    # 如果只是列出数据集，则加载并显示后退出
    if args.list_datasets:
        datasets = load_all_qa_datasets(args.data_dir, logger, is_test_mode=args.test_mode)
        dataset_names = list(datasets.keys())
        logger.info(f"\n共找到 {len(dataset_names)} 个数据集:")
        for i, name in enumerate(dataset_names):
            logger.info(f"  {i}: {name}")
        logger.info(f"\n使用 --start 和 --end 参数指定要处理的数据集范围")
        if args.test_mode:
            logger.info(f"例如: python get_sae_act.py --test_mode --start 0 --end 5  # 处理前5个 test 数据集")
            logger.info(f"例如: python get_sae_act.py --test_mode --start 5 --end -1  # 处理第6个到最后的 test 数据集")
        else:
            logger.info(f"例如: python get_sae_act.py --start 0 --end 5  # 处理前5个数据集")
            logger.info(f"例如: python get_sae_act.py --start 5 --end -1  # 处理第6个到最后的数据集")
        exit(0)

    # 加载Gemma模型
    logger.info(f"加载模型: {args.model_name} 到设备: {device}")
    if "9b" in args.model_name:
        model = HookedSAETransformer.from_pretrained(args.model_name, device=device, dtype=torch.bfloat16, n_devices=args.n_devices)
    elif "8b" in args.model_name:
        model = HookedSAETransformer.from_pretrained(args.model_name, device=device, dtype=torch.bfloat16, n_devices=args.n_devices)
    else:
        model = HookedSAETransformer.from_pretrained(args.model_name, device=device, dtype=torch.bfloat16)

    # SAE加载信息
    if args.sae_paths.strip():
        sae_layer_paths = [p.strip() for p in args.sae_paths.split(',') if p.strip()]
        logger.info(f"使用传入的SAE路径，共 {len(sae_layer_paths)} 个: {sae_layer_paths}")
    else:
        # 兼容旧默认：2B-PT 的全层
        sae_layer_paths = [
            "layer_0/width_16k/average_l0_105",
            "layer_1/width_16k/average_l0_102",
            "layer_2/width_16k/average_l0_141",
            "layer_3/width_16k/average_l0_59",
            "layer_4/width_16k/average_l0_124",
            "layer_5/width_16k/average_l0_68",
            "layer_6/width_16k/average_l0_70",
            "layer_7/width_16k/average_l0_69",
            "layer_8/width_16k/average_l0_71",
            "layer_9/width_16k/average_l0_73",
            "layer_10/width_16k/average_l0_77",
            "layer_11/width_16k/average_l0_80",
            "layer_12/width_16k/average_l0_82",
            "layer_13/width_16k/average_l0_84",
            "layer_14/width_16k/average_l0_84",
            "layer_15/width_16k/average_l0_78",
            "layer_16/width_16k/average_l0_78",
            "layer_17/width_16k/average_l0_77",
            "layer_18/width_16k/average_l0_74",
            "layer_19/width_16k/average_l0_73",
            "layer_20/width_16k/average_l0_71",
            "layer_21/width_16k/average_l0_70",
            "layer_22/width_16k/average_l0_72",
            "layer_23/width_16k/average_l0_75",
            "layer_24/width_16k/average_l0_73",
            "layer_25/width_16k/average_l0_116",
        ]
        logger.info(f"未提供 --sae_paths，使用默认的2B-PT SAE路径，共 {len(sae_layer_paths)} 个")
    
    if args.test_mode:
        logger.info(f"开始批量处理 test 数据集目录下所有数据集（*_test.json） ...")
    else:
        logger.info(f"开始批量处理 capability_data 目录下所有数据集（*_qa.json） ...")
    logger.info(f"参数设置: start={args.start}, end={args.end}, layers_per_batch={args.layers_per_batch}")
    logger.info(f"SAE设置: release={args.sae_release}, local_base_dir={args.sae_local_base_dir}")
    
    datasets = load_all_qa_datasets(args.data_dir, logger, is_test_mode=args.test_mode)
    dataset_names = list(datasets.keys())
    
    # 确定处理的数据集范围
    if args.end == -1:
        end_idx = len(dataset_names)
    else:
        end_idx = min(args.end, len(dataset_names))
    
    start_idx = max(0, args.start)
    
    if start_idx >= len(dataset_names):
        logger.error(f"起始索引 {start_idx} 超出数据集范围 (0-{len(dataset_names)-1})")
        exit(1)
    
    logger.info(f"将处理数据集索引范围: {start_idx} 到 {end_idx-1}")
    logger.info(f"对应的数据集: {dataset_names[start_idx:end_idx]}")
    for i in range(start_idx, end_idx):
        dataset_name = dataset_names[i]
        qa_data = datasets[dataset_name]
        logger.info(f"\n开始处理数据集 {i+1}/{end_idx}: {dataset_name}")
        try:
            process_dataset_with_layers(
                qa_data=qa_data,
                dataset_name=dataset_name,
                sae_layer_paths=sae_layer_paths,
                sae_release=args.sae_release,
                sae_local_base_dir=args.sae_local_base_dir,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                layers_per_batch=args.layers_per_batch,
                output_dir=f"{args.output_base_dir}/{dataset_name}",
                baseline_direct_neural=args.baseline_direct_neural,
                is_instruct=args.is_instruct,
                is_rerun=args.is_rerun
                )
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时发生错误: {e}")
            continue
    logger.info(f"数据集 {start_idx} 到 {end_idx-1} 处理完毕！")
    logger.info("=" * 60)
    logger.info("SAE激活数据处理结束")
    logger.info("=" * 60)

"""
# 如何还原每层稀疏激活矩阵
import scipy.sparse
import json
import os

# 读取指定数据集指定层的元信息
dataset_name = "dataset1"
layer = 24
output_dir = "./qa_sae_output"
meta_filename = f"{dataset_name}_meta.json" # Changed to meta_filename
sparse_filename = f"{dataset_name}_layer{layer}_sparse.npz"

with open(os.path.join(output_dir, meta_filename), "r", encoding='utf-8') as f:
    meta = json.load(f)
sparse_mat = scipy.sparse.load_npz(os.path.join(output_dir, sparse_filename))

# 还原为稠密矩阵（如需）
dense_mat = sparse_mat.toarray()
# 例如第i条QA的激活向量：sparse_mat[i].toarray().squeeze()
"""
