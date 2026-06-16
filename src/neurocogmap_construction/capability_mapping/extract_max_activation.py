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
import pickle
from collections import defaultdict

# 设备选择
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_sae_for_layers(layer_paths, sae_release, sae_local_base_dir):
    """加载指定层的SAE模型（兼容 Gemma 与 Llama Scope LXR-8x）"""
    sae_list = []
    hook_names = []
    for path in layer_paths:
        # 对于 Llama Scope LXR-8x，sae_id 形如 l0r_8x，需要映射到本地真实路径
        m = re.search(r"l(\d+)r", path)
        if m:
            sae_id_num = m.group(1)
            real_path = f"Llama3_1-8B-Base-L{sae_id_num}R-8x"
        else:
            real_path = path

        # Llama Scope 本地文件: checkpoints/final.safetensors
        llama_scope_local_path = os.path.join(sae_local_base_dir, real_path, "checkpoints", "final.safetensors")
        # Gemma SAE 本地文件: params.npz
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
            # 如果本地文件不存在，则走默认的 HuggingFace 下载/缓存逻辑
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=path,
                device=device,
            )

        if args.n_devices > 1:
            sae.to("cuda:1")
        else:
            sae.to(device)
        sae.use_error_term = True
        sae_list.append(sae)
        hook_names.append(sae.cfg.metadata.hook_name if hasattr(sae.cfg, 'metadata') else sae.cfg.hook_name)
    return sae_list, hook_names

def format_qa_prompt(question, answer):
    """格式化QA提示"""
    return f"Question: {question}\nAnswer: {answer}"

def format_qa_prompt_instruct(question, answer, tokenizer):
    """为 instruct 模型格式化 QA prompt，使用 tokenizer 的 apply_chat_template"""
    try:
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return prompt
    except Exception as e:
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

def load_all_qa_datasets(data_dir, logger=None, max_activation_dir=None):
    """加载 capability_data 目录下所有 json 文件，并按已存在的 max_activation 结果跳过数据集。
    
    max_activation_dir:
        - 若为 None，则默认使用 9b 目录（向后兼容）
        - 否则使用传入目录（例如 8b 对应的 output_dir）
    """
    filtered_datasets = []
    # 根据不同模型/输出目录决定扫描哪个 max_activation 目录
    if max_activation_dir is None:
        max_activation_dir = "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation/dataset_9b"
    if os.path.exists(max_activation_dir):
        for path in glob.glob(os.path.join(max_activation_dir, '*_max_activation.pkl')):
            name = os.path.splitext(os.path.basename(path))[0].replace('_max_activation','')
            filtered_datasets.append(name)
    if logger:
        logger.info(f"已存在最大激活结果的数据集（将被跳过）: {filtered_datasets}")
    datasets = {}
    if logger:
        logger.info(f"正在扫描目录: {data_dir}")

    for path in glob.glob(os.path.join(data_dir, '*_qa.json')):
        name = os.path.splitext(os.path.basename(path))[0].replace('_qa','')
        
        # 检查是否是需要筛选的数据集
        if name in filtered_datasets:
            if logger:
                logger.info(f"跳过数据集 {name}")
            continue
            
        if logger:
            logger.info(f"加载数据集文件: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)[:100]
        if logger:
            logger.info(f"数据集 {name} 加载完成，样本数: {len(data)}")
        datasets[name] = data
    if logger:
        logger.info(f"共加载数据集数量: {len(datasets)}")
    return datasets

def extract_max_activation_per_feature(qa_data, dataset_name, sae_layer_paths, sae_release, sae_local_base_dir, 
                                     layer_start=0, layer_end=25, layers_per_batch=2, 
                                     output_dir="./output", baseline_direct_neural=False):
    """提取每个feature的最大激活状态，不进行平均"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化最大激活字典
    max_activation_dict = {}
    
    logger.info(f"数据集 {dataset_name} 开始分层处理，层范围: {layer_start}-{layer_end}，每 batch {layers_per_batch} 层")
    
    # 解析传入的sae路径对应的实际层号
    def extract_layer_num(path_str):
        # gemma 系列: 路径形如 layer_9/width_16k/...
        if "gemma" in sae_release.lower():
            m = re.search(r"layer_(\d+)/", path_str)
            return int(m.group(1)) if m else None
        else:
            # llama LXR-8x 系列: 路径形如 l0r_8x,l1r_8x
            m = re.search(r"l(\d+)r_8x", path_str)
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
    
    # 按层批次处理激活数据
    layer_numbers = [ln for ln, _ in layer_num_to_path]
    for batch_start_idx in range(0, len(layer_numbers), layers_per_batch):
        batch_slice = layer_num_to_path[batch_start_idx: batch_start_idx + layers_per_batch]
        current_layers = [ln for ln, _ in batch_slice]
        logger.info(f"处理数据集 {dataset_name}，层 {current_layers[0]}-{current_layers[-1]}")
        current_layer_paths = [p for _, p in batch_slice]
        sae_list, hook_names = load_sae_for_layers(current_layer_paths, sae_release=sae_release, sae_local_base_dir=sae_local_base_dir)
        
        # 处理激活数据
        for idx, item in enumerate(tqdm(qa_data[:20], desc=f"Processing activations for {dataset_name}")):
            question = item["question"]
            answer = item["answer"]
            prompt = get_formatted_prompt(question, answer, is_instruct=args.is_instruct, tokenizer=model.tokenizer)
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
                continue
            
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
            
            # 对每个层和每个feature，找到最大激活值
            for layer_idx, (layer_num, acts) in enumerate(zip(current_layers, acts_per_layer)):
                # acts shape: [1, answer_len, d_sae]
                # 在token维度上找到每个feature的最大值
                max_acts = torch.max(acts[0, relative_valid_indices, :], dim=0)[0]  # [d_sae]
                
                # 更新最大激活字典
                for feature_idx in range(max_acts.shape[0]):
                    key = (layer_num, feature_idx)
                    current_max = max_activation_dict.get(key, float('-inf'))
                    new_max = max_acts[feature_idx].item()
                    if new_max > current_max:
                        max_activation_dict[key] = new_max
        
        del sae_list, hook_names
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"层 {current_layers[0]}-{current_layers[-1]} 处理完成，已清理显存")
    
    # 保存最大激活字典
    output_file = os.path.join(output_dir, f"{dataset_name}_max_activation.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(max_activation_dict, f)
    
    logger.info(f"数据集 {dataset_name} 的最大激活已保存到: {output_file}")
    logger.info(f"共记录了 {len(max_activation_dict)} 个 (layer_id, latent_id) 的最大激活值")
    
    return max_activation_dict

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='提取SAE每个feature的最大激活状态')
    parser.add_argument('--data_dir', type=str, 
                       default="/path/to/project_root/neural_area/capability_data_v2",
                       help='数据目录路径')
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
    parser.add_argument('--output_dir', type=str, 
                       default="/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation",
                       help='输出目录')
    parser.add_argument('--list_datasets', action='store_true',
                       help='列出所有可用的数据集并退出')

    # 新增参数：模型与SAE资源
    parser.add_argument('--model_name', type=str, default="google/gemma-2-2b",
                       help='Transformer 模型名称，如 google/gemma-2-2b 或 google/gemma-2-9b-it')
    parser.add_argument('--sae_release', type=str, default="gemma-scope-2b-pt-res",
                       help='SAE 发布名，如 gemma-scope-2b-pt-res 或 gemma-scope-9b-it-res')
    parser.add_argument('--sae_local_base_dir', type=str, default="/path/to/local_models/gemma-scope-2b-pt-res",
                       help='本地SAE权重的根目录（其下应包含 layer_x/.../params.npz）')
    parser.add_argument('--sae_paths', type=str, default="",
                       help='逗号分隔的SAE路径列表，例如 "layer_9/width_16k/average_l0_88,layer_20/width_16k/average_l0_91,layer_31/width_16k/average_l0_76"；若为空则使用内置默认列表')
    parser.add_argument('--n_devices', type=int, default=1,
                       help='模型使用的设备数量')
    parser.add_argument('--log_file', type=str, default=None,
                       help='日志文件路径，如果不指定则自动生成')
    parser.add_argument('--baseline_direct_neural', action='store_true',
                       help='是否使用直接神经激活作为基线，默认使用SAE激活')
    parser.add_argument('--is_instruct', action='store_true',
                       help='是否使用 instruct 模型（使用 chat template 构造QA文本）')
    
    args = parser.parse_args()
    
    # 设置日志记录
    def setup_logging(log_file=None):
        """设置日志记录，同时输出到控制台和文件"""
        if log_file is None:
            # 自动生成日志文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"./logs/max_activation_log_{timestamp}.log"
        
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
    logger.info("SAE最大激活提取开始")
    logger.info("=" * 60)
    
    # 输出日志文件路径
    if args.log_file:
        logger.info(f"日志文件路径: {os.path.abspath(args.log_file)}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_log_file = f"max_activation_log_{timestamp}.log"
        logger.info(f"自动生成日志文件路径: {os.path.abspath(auto_log_file)}")
    
    # 加载Gemma模型
    logger.info(f"加载模型: {args.model_name} 到设备: {device}")
    if "9b" in args.model_name:
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

    # 如果只是列出数据集，则加载并显示后退出
    if args.list_datasets:
        datasets = load_all_qa_datasets(args.data_dir, logger, max_activation_dir=args.output_dir)
        dataset_names = list(datasets.keys())
        logger.info(f"\n共找到 {len(dataset_names)} 个数据集:")
        for i, name in enumerate(dataset_names):
            logger.info(f"  {i}: {name}")
        logger.info(f"\n使用 --start 和 --end 参数指定要处理的数据集范围")
        logger.info(f"例如: python extract_max_activation.py --start 0 --end 5  # 处理前5个数据集")
        logger.info(f"例如: python extract_max_activation.py --start 5 --end -1  # 处理第6个到最后的数据集")
        exit(0)
    
    logger.info(f"开始批量处理 capability_data 目录下所有数据集 ...")
    logger.info(f"参数设置: start={args.start}, end={args.end}, layers_per_batch={args.layers_per_batch}")
    logger.info(f"SAE设置: release={args.sae_release}, local_base_dir={args.sae_local_base_dir}")
    
    datasets = load_all_qa_datasets(args.data_dir, logger, max_activation_dir=args.output_dir)
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
    
    # 处理每个数据集
    for i in range(start_idx, end_idx):
        dataset_name = dataset_names[i]
        qa_data = datasets[dataset_name]
        logger.info(f"\n开始处理数据集 {i+1}/{end_idx}: {dataset_name}")
        try:
            max_activation_dict = extract_max_activation_per_feature(
                qa_data=qa_data,
                dataset_name=dataset_name,
                sae_layer_paths=sae_layer_paths,
                sae_release=args.sae_release,
                sae_local_base_dir=args.sae_local_base_dir,
                layer_start=args.layer_start,
                layer_end=args.layer_end,
                layers_per_batch=args.layers_per_batch,
                output_dir=args.output_dir,
                baseline_direct_neural=args.baseline_direct_neural
            )
            logger.info(f"数据集 {dataset_name} 处理完成，最大激活字典大小: {len(max_activation_dict)}")
        except Exception as e:
            logger.error(f"处理数据集 {dataset_name} 时发生错误: {e}")
            continue
    
    logger.info(f"数据集 {start_idx} 到 {end_idx-1} 处理完毕！")
    logger.info("=" * 60)
    logger.info("SAE最大激活提取结束")
    logger.info("=" * 60)

"""
# 如何读取保存的最大激活字典
import pickle
import os

# 读取指定数据集的最大激活字典
dataset_name = "tinystories_continuation"
output_dir = "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation"
output_file = os.path.join(output_dir, f"{dataset_name}_max_activation.pkl")

with open(output_file, 'rb') as f:
    max_activation_dict = pickle.load(f)

# 查看字典结构
print(f"字典大小: {len(max_activation_dict)}")
print(f"前5个键值对: {list(max_activation_dict.items())[:5]}")

# 获取特定层和feature的最大激活值
layer_id, latent_id = 0, 100
max_act = max_activation_dict.get((layer_id, latent_id), None)
print(f"Layer {layer_id}, Feature {latent_id} 的最大激活值: {max_act}")
""" 