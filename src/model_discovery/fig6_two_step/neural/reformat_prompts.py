#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新格式化 prompts.jsonl 文件
在原始数据基础上新增 instruction 和 exp_list 字段
"""

import json
import re
import argparse
import os

def split_instruction_and_experiments(text):
    """
    将文本分割为 instruction 和实验列表
    Args:
        text: 完整的 prompt 文本
    Returns:
        instruction: instruction 部分（第一次 "You are presented with" 出现前的所有内容）
        experiments: 实验列表，每个实验是一个字符串
    """
    # 找到第一次 "You are presented with" 出现的位置
    first_presented = text.find("You are presented with")
    
    if first_presented == -1:
        # 如果没有找到，整个文本都是 instruction
        return text, []
    
    instruction = text[:first_presented].strip()
    
    # 从第一次 "You are presented with" 开始，分割所有实验
    experiments_text = text[first_presented:]
    
    # 使用正则表达式分割实验（每次 "You are presented with" 开始一个新实验）
    # 但要注意保留换行符
    experiment_pattern = r'(?=You are presented with)'
    experiments = re.split(experiment_pattern, experiments_text)
    
    # 过滤掉空字符串
    experiments = [exp.strip() for exp in experiments if exp.strip()]
    
    return instruction, experiments

def main():
    parser = argparse.ArgumentParser(description='重新格式化 prompts.jsonl 文件')
    parser.add_argument('--input', type=str, 
                       default='prompts.jsonl',
                       help='输入的 prompts.jsonl 文件路径')
    parser.add_argument('--output', type=str,
                       default='prompts_reformatted.jsonl',
                       help='输出的重新格式化后的文件路径')
    parser.add_argument('--skip_existing', action='store_true',
                       help='如果输出文件已存在则跳过')
    
    args = parser.parse_args()
    
    # 检查输出文件是否已存在
    if args.skip_existing and os.path.exists(args.output):
        print(f"输出文件 {args.output} 已存在，跳过处理")
        return
    
    input_path = os.path.join(os.path.dirname(__file__), args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(os.path.dirname(__file__), args.output) if not os.path.isabs(args.output) else args.output
    
    print(f"读取文件: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                
                if not text:
                    print(f"警告: 第 {line_num} 行没有 text 字段，跳过")
                    continue
                
                instruction, experiments = split_instruction_and_experiments(text)
                
                # 在原始数据基础上新增 instruction 和 exp_list 字段
                output_data = data.copy()  # 保留原始所有字段
                output_data['instruction'] = instruction
                output_data['exp_list'] = experiments
                print(len(experiments))
                f_out.write(json.dumps(output_data, ensure_ascii=False) + '\n')
                
                if line_num % 10 == 0:
                    print(f"已处理 {line_num} 行")
                    
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"处理完成，输出文件: {output_path}")

if __name__ == '__main__':
    main()

