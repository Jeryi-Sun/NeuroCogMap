#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 LLM (gemma-2-2b) 预测人类认知行为
处理 kool2016when_exp2_reformatted.jsonl 和 kool2017cost_exp2_reformatted.jsonl
计算 " <<" 之后预测的 NLL (负对数似然)
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import re
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from typing import List, Tuple
import numpy as np


class TokenLimitedExtractor:
    """Token 限制提取器，参考 base_extractor.py"""
    
    def __init__(self, tokenizer, max_tokens: int = 1024):
        """
        Args:
            tokenizer: tokenizer 对象
            max_tokens: 最大 token 数限制
        """
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
    
    def truncate_to_token_limit(
        self, 
        instruction: str, 
        experiments: List[str], 
        current_experiment_idx: int = 0
    ) -> str:
        """
        根据 token 限制截断 prompt，保留 instruction 和尽可能多的最近实验
        参考 base_extractor.py 的 truncate_to_token_limit 方法
        
        Args:
            instruction: instruction 部分
            experiments: 实验列表（累积的）
            current_experiment_idx: 当前要处理的实验索引（在 experiments 中的索引）
            
        Returns:
            截断后的完整 prompt 文本
        """
        # 首先尝试包含所有到当前实验的内容
        context_experiments = experiments[:current_experiment_idx + 1]
        full_text = instruction + '\n' + '\n'.join(context_experiments)
        
        # 计算 token 数量
        tokenized = self.tokenizer(full_text, return_tensors='pt', add_special_tokens=False)
        num_tokens = tokenized.input_ids.shape[1]
        
        if num_tokens <= self.max_tokens:
            return full_text
        
        # 如果超过限制，从 instruction 后的早期实验开始删除
        # 保留 instruction 和当前实验，尽可能多地保留最近的实验
        current_experiment = experiments[current_experiment_idx]
        
        # 尝试只保留 instruction + 当前实验
        minimal_text = instruction + '\n' + current_experiment
        tokenized_minimal = self.tokenizer(minimal_text, return_tensors='pt', add_special_tokens=False)
        num_tokens_minimal = tokenized_minimal.input_ids.shape[1]
        
        if num_tokens_minimal > self.max_tokens:
            # 如果连 instruction + 当前实验都超过限制，需要截断 instruction
            print(f"警告: instruction + 当前实验 ({num_tokens_minimal} tokens) 超过限制 ({self.max_tokens} tokens)")
            # 截断 instruction，保留最后部分
            instruction_tokens = self.tokenizer(instruction, return_tensors='pt', add_special_tokens=False)
            current_experiment_tokens = self.tokenizer(current_experiment, return_tensors='pt', add_special_tokens=False)
            current_experiment_token_count = current_experiment_tokens.input_ids.shape[1]
            
            available_tokens = self.max_tokens - current_experiment_token_count - 10  # 留一些余量
            if available_tokens > 0:
                # 截断 instruction
                instruction_ids = instruction_tokens.input_ids[0][-available_tokens:]
                truncated_instruction = self.tokenizer.decode(instruction_ids, skip_special_tokens=True)
                return truncated_instruction + '\n' + current_experiment
            else:
                # 如果当前实验本身就超过限制，只返回当前实验的一部分
                available_for_experiment = self.max_tokens - 10
                experiment_ids = current_experiment_tokens.input_ids[0][:available_for_experiment]
                return self.tokenizer.decode(experiment_ids, skip_special_tokens=True)
        
        # 从后往前添加实验，直到达到 token 限制
        result_experiments = [current_experiment]
        result_text = instruction + '\n' + current_experiment
        
        for i in range(current_experiment_idx - 1, -1, -1):
            test_experiments = experiments[i:current_experiment_idx + 1]
            test_text = instruction + '\n' + '\n'.join(test_experiments)
            tokenized_test = self.tokenizer(test_text, return_tensors='pt', add_special_tokens=False)
            num_tokens_test = tokenized_test.input_ids.shape[1]
            
            if num_tokens_test <= self.max_tokens:
                result_experiments.insert(0, experiments[i])
                result_text = test_text
            else:
                break
        
        return result_text


def collect_all_possible_options(experiments: List[str]) -> set:
    """
    收集所有 exp_list 中 <<>> 中间的所有可能选项
    
    Args:
        experiments: 实验列表
        
    Returns:
        所有可能的选项集合
    """
    all_options = set()
    pattern = r' <<([^>]+)>>'
    
    for experiment in experiments:
        for match in re.finditer(pattern, experiment):
            action = match.group(1).strip()
            if action:
                all_options.add(action)
    
    return all_options


def find_action_positions(text: str, all_possible_options: set = None) -> List[Tuple[int, str]]:
    """
    找到文本中所有 " <<" 之后的位置和对应的动作
    
    Args:
        text: 输入文本
        all_possible_options: 所有可能的选项集合（用于验证）
        
    Returns:
        List of (position, action) tuples
        position 是 " <<" 之后第一个字符的位置
        action 是选择的动作
    """
    positions = []
    pattern = r' <<([^>]+)>>'
    
    for match in re.finditer(pattern, text):
        # 找到 " <<" 之后的位置（即 " <<" 的结束位置）
        pos = match.end() - len(match.group(0)) + len(' <<')
        action = match.group(1).strip()
        
        # 如果提供了所有可能选项，验证动作是否在选项中
        if all_possible_options is not None:
            if action not in all_possible_options:
                continue
        
        positions.append((pos, action))
    
    return positions


def compute_nll_for_actions(
    model,
    tokenizer,
    text: str,
    action_positions: List[Tuple[int, str]],
    all_possible_options: set
) -> Tuple[float, float]:
    """
    计算 " <<" 之后预测的 NLL
    对所有可能的选项计算 softmax
    
    Args:
        model: LLM 模型
        tokenizer: tokenizer
        text: 完整文本
        action_positions: (position, action) 列表
        all_possible_options: 所有可能的选项集合
        
    Returns:
        (平均 NLL, 总 NLL)
    """
    if not action_positions or not all_possible_options:
        return 0.0, 0.0
    
    # Tokenize 整个文本
    tokenized = tokenizer(text, return_tensors='pt', add_special_tokens=False)
    input_ids = tokenized.input_ids.to(model.device)
    
    # 获取 " <<" 的 token IDs
    left_token = " <<"
    left_tokenized = tokenizer(left_token, add_special_tokens=False)
    if len(left_tokenized.input_ids) == 0:
        print(f"警告: 无法 tokenize '{left_token}'")
        return 0.0, 0.0
    
    # l_id = tokenizer(" <<").input_ids[1:] 之后的第一个 token
    if len(left_tokenized.input_ids) > 1:
        l_ids = left_tokenized.input_ids[1:]  # 跳过第一个 token
    else:
        l_ids = left_tokenized.input_ids
    
    # 前向传播获取 logits
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    # 为所有选项获取 token IDs（取第一个 token）
    option_to_token_id = {}
    option_list = sorted(list(all_possible_options))  # 排序以保证一致性
    
    for option in option_list:
        option_tokenized = tokenizer(option, add_special_tokens=False)
        if len(option_tokenized.input_ids) > 0:
            option_to_token_id[option] = option_tokenized.input_ids[0]
        else:
            print(f"警告: 无法 tokenize 选项 '{option}'")
    
    if not option_to_token_id:
        print(f"警告: 没有有效的选项 token IDs")
        return 0.0, 0.0
    
    nlls = []
    seq_len = input_ids.shape[1]
    
    # 找到所有 " <<" 在 tokenized 文本中的位置
    left_bracket_positions = []
    for i in range(seq_len - len(l_ids)):
        match = True
        for j, l_id in enumerate(l_ids):
            if i + j >= seq_len or input_ids[0, i + j] != l_id:
                match = False
                break
        if match:
            left_bracket_positions.append(i + len(l_ids))  # " <<" 之后的位置
    
    # 对于每个动作位置，找到对应的 " <<" 位置
    used_positions = set()  # 记录已使用的位置，确保每个 " <<" 只匹配一次
    
    for text_pos, action in action_positions:
        if action not in option_to_token_id:
            print(f"警告: 动作 '{action}' 不在选项集合中")
            continue
        
        action_token_id = option_to_token_id[action]
        
        # 找到对应的 " <<" 位置（找到第一个未使用的位置，其后的 token 匹配动作）
        found = False
        for target_pos in left_bracket_positions:
            if target_pos in used_positions:
                continue
            
            if target_pos >= seq_len:
                continue
            
            # 检查该位置的 token 是否匹配动作
            true_token_id = input_ids[0, target_pos]
            
            # 获取所有选项的 logits
            logits_at_pos = logits[0, target_pos - 1, :]  # 预测 target_pos 位置的 logits
            
            # 收集所有选项的 logits
            option_logits_list = []
            option_to_idx = {}  # 选项到索引的映射
            true_idx = -1
            
            for option in option_list:
                if option in option_to_token_id:
                    option_token_id = option_to_token_id[option]
                    option_logit = logits_at_pos[option_token_id].item()
                    current_idx = len(option_logits_list)
                    option_logits_list.append(option_logit)
                    option_to_idx[option] = current_idx
                    
                    # 找到真实选择的索引（检查 token ID 是否匹配）
                    if option_token_id == true_token_id:
                        true_idx = current_idx
            
            # 如果直接匹配失败，尝试在选项的所有 token 中查找
            if true_idx == -1:
                action_tokenized = tokenizer(action, add_special_tokens=False)
                action_all_tokens = action_tokenized.input_ids
                
                # 检查真实 token 是否在动作的所有 token 中
                if true_token_id in action_all_tokens:
                    # 如果动作在选项列表中，使用动作对应的索引
                    if action in option_to_idx:
                        true_idx = option_to_idx[action]
                    else:
                        # 尝试找到包含该 token 的选项
                        for option in option_list:
                            if option in option_to_token_id:
                                option_tokenized = tokenizer(option, add_special_tokens=False)
                                if true_token_id in option_tokenized.input_ids:
                                    true_idx = option_to_idx[option]
                                    break
            
            if true_idx == -1:
                print(f"警告: 无法找到动作 '{action}' (token_id={true_token_id}) 在选项列表中的索引")
                continue
            
            # 对所有选项计算 softmax 和 NLL
            option_logits_tensor = torch.tensor([option_logits_list])
            target_tensor = torch.tensor([true_idx])
            nll = F.cross_entropy(option_logits_tensor, target_tensor).item()
            nlls.append(nll)
            used_positions.add(target_pos)
            found = True
            break
        
        if not found:
            print(f"警告: 无法找到动作 '{action}' 对应的 ' <<' 位置或匹配 token")
    
    if not nlls:
        print(f"警告: 未能计算任何 NLL")
        return 0.0, 0.0
    
    avg_nll = np.mean(nlls)
    total_nll = np.sum(nlls)
    return avg_nll, total_nll


def process_participant(
    model,
    tokenizer,
    extractor,
    instruction: str,
    experiments: List[str],
    participant_id: str,
    all_possible_options: set
) -> Tuple[float, float, float]:
    """
    处理单个 participant 的数据，计算平均 NLL 和 AIC
    
    Args:
        model: LLM 模型
        tokenizer: tokenizer
        extractor: TokenLimitedExtractor
        instruction: instruction 部分
        experiments: 实验列表
        participant_id: participant ID
        all_possible_options: 所有可能的选项集合
        
    Returns:
        (平均 NLL, 平均 reward, AIC)
    """
    all_nlls = []
    all_rewards = []
    total_nll = 0.0  # 总的 NLL，用于计算 AIC
    num_actions = 0  # 动作总数
    
    # 累积实验历史
    accumulated_experiments = []
    
    for exp_idx, experiment in enumerate(experiments):
        accumulated_experiments.append(experiment)
        
        # 使用 truncate_to_token_limit 构建输入
        truncated_text = extractor.truncate_to_token_limit(
            instruction, 
            accumulated_experiments, 
            exp_idx
        )
        
        # 只计算当前实验（最后一条）中的动作的 NLL
        # 在截断文本中找到所有动作位置
        all_action_positions = find_action_positions(truncated_text, all_possible_options)
        
        # 找到当前实验在截断文本中的位置（使用 rfind 找到最后一个匹配，即当前实验）
        experiment_start_in_text = truncated_text.rfind(experiment)
        experiment_end_in_text = experiment_start_in_text + len(experiment) if experiment_start_in_text != -1 else -1
        
        # 只保留在当前实验范围内的动作
        current_exp_action_positions = []
        if experiment_start_in_text != -1:
            for pos, action in all_action_positions:
                # 检查动作位置是否在当前实验范围内
                if experiment_start_in_text <= pos < experiment_end_in_text:
                    current_exp_action_positions.append((pos, action))
        else:
            # 如果找不到当前实验，使用最后一个动作（假设是当前实验的）
            if all_action_positions:
                current_exp_action_positions = [all_action_positions[-1]]
        
        if current_exp_action_positions:
            # 只计算当前实验的 NLL（每个动作的平均 NLL 和总 NLL）
            avg_nll_exp, total_nll_exp = compute_nll_for_actions(
                model, tokenizer, truncated_text, 
                current_exp_action_positions, 
                all_possible_options
            )
            all_nlls.append(avg_nll_exp)
            
            # 累加总的 NLL
            total_nll += total_nll_exp
            num_actions += len(current_exp_action_positions)
        
        # 提取 reward（从实验文本中）
        reward_match = re.search(r'You find (\d+) pieces? of space treasure', experiment)
        if reward_match:
            reward = float(reward_match.group(1))
            all_rewards.append(reward)
    
    avg_nll = np.mean(all_nlls) if all_nlls else 0.0
    avg_reward = np.mean(all_rewards) if all_rewards else 0.0
    
    # 计算 AIC = 2k + 2 * NLL_total，其中 k=0
    # 所以 AIC = 2 * NLL_total
    aic = 2 * total_nll if num_actions > 0 else 0.0
    
    return avg_nll, avg_reward, aic


def main():
    parser = argparse.ArgumentParser(description='使用 LLM 预测人类认知行为')
    parser.add_argument('--model_name', type=str, 
                       default='google/gemma-2-2b',
                       help='模型名称')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='最大 token 数限制')
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入的 reformatted jsonl 文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出的 CSV 文件路径')
    parser.add_argument('--skip_existing', action='store_true',
                       help='如果输出文件已存在则跳过')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查输出文件是否已存在
    if args.skip_existing and os.path.exists(args.output_file):
        print(f"输出文件 {args.output_file} 已存在，跳过处理")
        return
    
    print(f"加载模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
        device_map='auto' if args.device == 'cuda' else None
    )
    if args.device == 'cpu':
        model = model.to(args.device)
    model.eval()
    
    print(f"模型加载完成，设备: {args.device}")
    
    # 创建 extractor
    extractor = TokenLimitedExtractor(tokenizer, max_tokens=args.max_tokens)
    
    # 读取输入文件
    print(f"读取文件: {args.input_file}")
    
    # 第一步：收集所有可能的选项
    print("第一步：收集所有可能的选项...")
    all_experiments = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                exp_list = item.get('exp_list', [])
                all_experiments.extend(exp_list)
            except:
                continue
    
    all_possible_options = collect_all_possible_options(all_experiments)
    print(f"收集到 {len(all_possible_options)} 个可能的选项: {sorted(all_possible_options)}")
    
    # 第二步：处理每个 participant
    print("\n第二步：处理每个 participant...")
    data = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                participant_id = item.get('participant', '')
                instruction = item.get('instruction', '')
                exp_list = item.get('exp_list', [])
                
                if not instruction or not exp_list:
                    print(f"警告: 第 {line_num} 行缺少 instruction 或 exp_list，跳过")
                    continue
                
                # 处理 participant
                print(f"处理 participant {participant_id} (第 {line_num} 个)")
                avg_nll, avg_reward, aic = process_participant(
                    model, tokenizer, extractor,
                    instruction, exp_list, participant_id,
                    all_possible_options
                )
                
                # 对于 twostep 实验，param 设为 0（因为我们只计算 NLL）
                data.append([participant_id, 0.0, avg_reward, avg_nll, aic])  # param=0, aic 已计算
                
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 保存结果
    df = pd.DataFrame(data, columns=['participant', 'param', 'reward', 'nll', 'aic'])
    print(f"\n结果统计:")
    print(df)
    df.to_csv(args.output_file, index=False)
    print(f"\n结果已保存到: {args.output_file}")


if __name__ == '__main__':
    main()

