#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版的提取脚本，使用模块化的提取器
支持三种提取方式：
1. Language Model (HookedTransformer)
2. SAE Model (重构后的 embedding)
3. SAE Activation (激活值)
"""
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch
import json
import argparse
from typing import List, Dict

# 导入提取器
from extraction.language_model_extractor import LanguageModelTokenLimitedExtractor
from extraction.sae_model_extractor import SAEModelTokenLimitedExtractor
from extraction.saeact_model_extractor import SAEActModelTokenLimitedExtractor
from extraction.embeddings_extractor import EmbeddingTokenLimitedExtractor

LAYER_BASED_EXTRACTORS = {"language_model", "bert_model", "language_model_attention"}


def main():
    parser = argparse.ArgumentParser(description='提取激活表示（带 token 限制）')
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input", type=str, 
                      default="feher2023rethinking/prompts_reformatted.jsonl",
                      help="输入的重新格式化后的 prompts 文件")
    parser.add_argument("--max_tokens", type=int, default=1024, 
                       help="最大 token 数限制")
    parser.add_argument("--extractor_type", type=str, 
                      choices=["language_model", "bert_model", "language_model_attention", "sae_model", "saeact_model", "sae_model_pre_nozero_patch", "embeddings"],
                      default="language_model",
                      help="提取器类型")
    parser.add_argument("--layers", type=str, default="0,10,20,30,40",
                       help="要提取的层，用逗号分隔（仅用于 language_model）")
    parser.add_argument("--layer_idx", type=int, default=None,
                       help="要提取的层索引（仅用于 language_model，如果指定则覆盖 layers，用于单层提取）")
    parser.add_argument("--parcel_id", type=int, default=None,
                       help="Parcel ID（用于 sae_model 和 saeact_model），如果指定 --extract_all_parcels 则忽略此参数")
    parser.add_argument("--extract_all_parcels", action='store_true',
                       help="一次性提取所有 Parcel 的特征（仅用于 sae_model 和 saeact_model），大幅提升性能")
    parser.add_argument("--parcel_mapping_path", type=str, default=None,
                       help="Parcel 映射文件路径（用于 sae_model 和 saeact_model）")
    parser.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res",
                       help="SAE release ID（用于 sae_model 和 saeact_model）")
    parser.add_argument("--sae_local_base_dir", type=str, default=None,
                       help="SAE 本地基础目录（用于 sae_model 和 saeact_model）")
    parser.add_argument("--sae_paths", type=str, default=None,
                       help="SAE 路径列表，用逗号分隔（用于 sae_model 和 saeact_model）")
    parser.add_argument("--skip_existing", action='store_true',
                       help="如果输出文件已存在则跳过")
    parser.add_argument("--reverse_order", action='store_true',
                       help="如果为 True，则倒序处理实验")
    parser.add_argument("--vector_path", type=str, default=None,
                       help="静态词向量文件路径（extractor_type=embeddings 时必填）")
    parser.add_argument("--embedding_lowercase", type=int, default=1,
                       help="embeddings 模式是否小写化输入文本（1/0）")
    parser.add_argument("--embedding_oov_handling", type=str, default="copy_prev",
                       choices=["copy_prev", "zero", "skip", "error"],
                       help="embeddings 模式 OOV 处理策略")
    parser.add_argument("--embedding_pooling", type=str, default="mean",
                       choices=["mean", "max"],
                       help="embeddings 模式池化策略")
    
    args = parser.parse_args()
    
    # 根据提取器类型创建提取器（所有提取器都在内部加载模型）
    if args.extractor_type in LAYER_BASED_EXTRACTORS:
        # 解析层列表
        if args.layer_idx is not None:
            layers = [args.layer_idx]
        else:
            layers = [int(x.strip()) for x in args.layers.split(',')]

        hook_type = "hook_attn_out" if args.extractor_type == "language_model_attention" else "hook_resid_pre"
        
        # 创建提取器（使用第一个层，如果需要多层需要循环）
        extractor = LanguageModelTokenLimitedExtractor(
            model_name=args.model,
            tokenizer=None,  # 将在提取器中从模型获取
            layer_idx=layers[0],
            hook_type=hook_type,
            max_tokens=args.max_tokens
        )
    elif "sae_model" in args.extractor_type:
        if not args.extract_all_parcels and args.parcel_id is None:
            raise ValueError("sae_model 提取器需要 --parcel_id 参数，或使用 --extract_all_parcels 一次性提取所有 Parcel")
        if args.parcel_mapping_path is None:
            raise ValueError("sae_model 提取器需要 --parcel_mapping_path 参数")
        if args.sae_local_base_dir is None:
            raise ValueError("sae_model 提取器需要 --sae_local_base_dir 参数")
        
        sae_paths = args.sae_paths.split(',') if args.sae_paths else None
        if sae_paths is None:
            # 使用默认路径
            sae_paths = SAEModelTokenLimitedExtractor.DEFAULT_SAE_PATHS_2B_PT
        
        # 如果提取所有 Parcel，使用第一个 Parcel ID 初始化（仅用于初始化，实际会提取所有 Parcel）
        init_parcel_id = args.parcel_id if args.parcel_id is not None else 0
        
        extractor = SAEModelTokenLimitedExtractor(
            model_name=args.model,
            tokenizer=None,  # 将在提取器中从模型获取
            parcel_id=init_parcel_id,
            parcel_mapping_path=args.parcel_mapping_path,
            sae_release=args.sae_release,
            sae_local_base_dir=args.sae_local_base_dir,
            sae_paths=sae_paths,
            max_tokens=args.max_tokens
        )
    elif "saeact_model" in args.extractor_type:
        if not args.extract_all_parcels and args.parcel_id is None:
            raise ValueError("saeact_model 提取器需要 --parcel_id 参数，或使用 --extract_all_parcels 一次性提取所有 Parcel")
        if args.parcel_mapping_path is None:
            raise ValueError("saeact_model 提取器需要 --parcel_mapping_path 参数")
        if args.sae_local_base_dir is None:
            raise ValueError("saeact_model 提取器需要 --sae_local_base_dir 参数")
        
        sae_paths = args.sae_paths.split(',') if args.sae_paths else None
        if sae_paths is None:
            # 使用默认路径
            sae_paths = SAEActModelTokenLimitedExtractor.DEFAULT_SAE_PATHS_2B_PT
        
        # 如果提取所有 Parcel，使用第一个 Parcel ID 初始化（仅用于初始化，实际会提取所有 Parcel）
        init_parcel_id = args.parcel_id if args.parcel_id is not None else 0
        
        extractor = SAEActModelTokenLimitedExtractor(
            model_name=args.model,
            tokenizer=None,  # 将在提取器中从模型获取
            parcel_id=init_parcel_id,
            parcel_mapping_path=args.parcel_mapping_path,
            sae_release=args.sae_release,
            sae_local_base_dir=args.sae_local_base_dir,
            sae_paths=sae_paths,
            max_tokens=args.max_tokens
        )
    elif args.extractor_type == "embeddings":
        if args.vector_path is None:
            raise ValueError("embeddings 提取器需要 --vector_path 参数")
        extractor = EmbeddingTokenLimitedExtractor(
            vector_path=args.vector_path,
            lowercase=bool(args.embedding_lowercase),
            oov_handling=args.embedding_oov_handling,
            pooling=args.embedding_pooling,
        )
    else:
        raise ValueError(f"不支持的 extractor_type: {args.extractor_type}")
    
    # 加载数据
    input_path = args.input if os.path.isabs(args.input) else \
                 os.path.join(os.path.dirname(__file__), args.input)
    
    print(f"读取数据: {input_path}")
    
    # 处理每个参与者（每行一个参与者）
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if args.reverse_order:
            lines = lines[::-1]
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
                participant_id = str(data.get('participant', '0'))
                instruction = data.get('instruction', '')
                exp_list = data.get('exp_list', [])
                
                if not instruction:
                    print(f"警告: 第 {line_num} 行没有 instruction 字段，跳过")
                    continue
                
                if not exp_list:
                    print(f"警告: 第 {line_num} 行没有 exp_list 字段或为空，跳过")
                    continue
                
                # 构建输出路径，按提取器类型分别保存在不同的 cache 目录下
                cache_dir = f'results/cache_{args.extractor_type}'
                os.makedirs(cache_dir, exist_ok=True)

                def build_output_path(param_value: str) -> str:
                    """根据 layer 或 parcel 的取值，构建分层存储路径。"""
                    if args.extractor_type in LAYER_BASED_EXTRACTORS:
                        sub_dir = f'layer_{param_value}'
                    elif "sae" in args.extractor_type:
                        sub_dir = f'parcel_{param_value}'
                    else:
                        sub_dir = "general"
                    full_dir = os.path.join(cache_dir, sub_dir)
                    os.makedirs(full_dir, exist_ok=True)
                    return os.path.join(
                        full_dir,
                        f'model={args.model.replace("/", "-")}_extractor={args.extractor_type}_{sub_dir}_participant={participant_id}.pth'
                    )
                
                # 处理所有 Parcel 的情况
                if args.extract_all_parcels and "sae" in args.extractor_type:
                    # 一次性提取所有 Parcel 的特征
                    print(f"处理参与者 {participant_id}: {len(exp_list)} 个实验，一次性提取所有 Parcel")
                    
                    # 收集所有 Parcel 的所有激活点位的表示
                    all_parcel_representations: Dict[int, List[torch.Tensor]] = {}
                    
                    # 累积处理每个实验
                    for exp_idx, experiment in enumerate(exp_list): # 增加倒序并行处理功能
                        try:
                            # 一次性提取所有 Parcel 的特征（只运行一次模型前向传播）
                            parcel_features = extractor.extract_all_parcels_from_experiment(
                                instruction, exp_list, exp_idx
                            )
                            
                            # 将每个 Parcel 的特征添加到对应的列表
                            for parcel_id, features in parcel_features.items():
                                if parcel_id not in all_parcel_representations:
                                    all_parcel_representations[parcel_id] = []
                                if features.numel() > 0:
                                    all_parcel_representations[parcel_id].append(features)
                            
                            if (exp_idx + 1) % 50 == 0:
                                print(f"  已处理 {exp_idx + 1}/{len(exp_list)} 个实验")
                                
                        except Exception as e:
                            print(f"错误: 处理实验 {exp_idx} 失败: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # 保存每个 Parcel 的结果
                    for parcel_id, representations in all_parcel_representations.items():
                        output_file = build_output_path(str(parcel_id))
                        
                        if args.skip_existing and os.path.exists(output_file):
                            print(f"跳过已存在的文件: {output_file}")
                            continue
                        
                        if representations:
                            # 将所有激活点位的表示 concatenate 在一起
                            concatenated = torch.cat(representations, dim=0)
                            torch.save(concatenated, output_file)
                            print(f"保存参与者 {participant_id} Parcel {parcel_id} 的表示: {concatenated.shape[0]} 个激活点位，特征维度: {concatenated.shape[1]}")
                        else:
                            print(f"警告: 参与者 {participant_id} Parcel {parcel_id} 没有提取到任何表示")
                
                else:
                    # 处理多层 language_model 的情况
                    if args.extractor_type in LAYER_BASED_EXTRACTORS and len(layers) > 1:
                        # 一次性提取所有层的特征
                        print(f"处理参与者 {participant_id}: {len(exp_list)} 个实验，一次性提取所有层")
                        
                        # 收集所有层的所有激活点位的表示
                        all_layer_representations: Dict[int, List[torch.Tensor]] = {}
                        
                        # 累积处理每个实验
                        for exp_idx, experiment in enumerate(exp_list):
                            try:
                                # 一次性提取所有层的特征（只运行一次模型前向传播）
                                layer_features = extractor.extract_all_layers_from_experiment(
                                    instruction, exp_list, exp_idx, layers
                                )
                                
                                # 将每个层的特征添加到对应的列表
                                for layer_idx in layers:
                                    if layer_idx not in all_layer_representations:
                                        all_layer_representations[layer_idx] = []
                                    if layer_idx in layer_features and layer_features[layer_idx].numel() > 0:
                                        all_layer_representations[layer_idx].append(layer_features[layer_idx])
                                
                                if (exp_idx + 1) % 50 == 0:
                                    print(f"  已处理 {exp_idx + 1}/{len(exp_list)} 个实验")
                                    
                            except Exception as e:
                                print(f"错误: 处理实验 {exp_idx} 失败: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        # 保存每个层的结果
                        for layer_idx in layers:
                            output_file = build_output_path(str(layer_idx))
                            
                            if args.skip_existing and os.path.exists(output_file):
                                print(f"跳过已存在的文件: {output_file}")
                                continue
                            
                            if layer_idx in all_layer_representations and all_layer_representations[layer_idx]:
                                # 将所有激活点位的表示 concatenate 在一起
                                concatenated = torch.cat(all_layer_representations[layer_idx], dim=0)
                                torch.save(concatenated, output_file)
                                print(f"保存参与者 {participant_id} 层 {layer_idx} 的表示: {concatenated.shape[0]} 个激活点位，特征维度: {concatenated.shape[1]}")
                            else:
                                print(f"警告: 参与者 {participant_id} 层 {layer_idx} 没有提取到任何表示")
                    
                    else:
                        # 单个 Parcel 或单层提取
                        # 构建输出文件名，包含 layer_idx 或 parcel_id
                        if args.extractor_type in LAYER_BASED_EXTRACTORS:
                            layer_idx = args.layer_idx if args.layer_idx is not None else layers[0]
                            output_file = build_output_path(str(layer_idx))
                        elif "sae" in args.extractor_type:
                            output_file = build_output_path(str(args.parcel_id))
                        else:
                            output_file = build_output_path("general")
                        
                        if args.skip_existing and os.path.exists(output_file):
                            print(f"跳过已存在的文件: {output_file}")
                            continue
                        
                        print(f"处理参与者 {participant_id}: {len(exp_list)} 个实验")
                        
                        # 收集所有激活点位的表示
                        all_representations = []
                        
                        # 累积处理每个实验
                        for exp_idx, experiment in enumerate(exp_list):
                            try:
                                # 单层提取
                                reprs = extractor.extract_from_experiment(
                                    instruction, exp_list, exp_idx
                                )
                                if reprs.numel() > 0:
                                    all_representations.append(reprs)
                                
                                if (exp_idx + 1) % 50 == 0:
                                    print(f"  已处理 {exp_idx + 1}/{len(exp_list)} 个实验")
                                    
                            except Exception as e:
                                print(f"错误: 处理实验 {exp_idx} 失败: {e}")
                                import traceback
                                traceback.print_exc()
                                continue
                        
                        # 保存结果
                        if all_representations:
                            # 将所有激活点位的表示 concatenate 在一起
                            concatenated = torch.cat(all_representations, dim=0)
                            torch.save(concatenated, output_file)
                            print(f"保存参与者 {participant_id} 的表示: {concatenated.shape[0]} 个激活点位，特征维度: {concatenated.shape[1]}")
                        else:
                            print(f"警告: 参与者 {participant_id} 没有提取到任何表示")
                
            except json.JSONDecodeError as e:
                print(f"错误: 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                print(f"错误: 第 {line_num} 行处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("处理完成")


if __name__ == '__main__':
    main()
