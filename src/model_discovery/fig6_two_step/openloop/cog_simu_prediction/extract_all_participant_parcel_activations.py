#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取所有参与者的 parcel 激活特征
从JSONL文件中提取所有参与者数据，然后提取每个participant的parcel激活
将所有trial的激活合并为一个列表，保存格式为 {participant_id: [activation]}
"""
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import sys
import json
import argparse
import torch
from typing import List, Dict, Set
from tqdm import tqdm
from pathlib import Path

# 添加neural目录到路径
neural_path = Path(__file__).parent.parent.parent / "neural"
sys.path.insert(0, str(neural_path))

from extraction.saeact_model_extractor import SAEActModelTokenLimitedExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取所有参与者的parcel激活特征"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入JSONL文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径（JSON格式）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="模型名称（如：google/gemma-2-2b）"
    )
    parser.add_argument(
        "--parcel-mapping-path",
        type=str,
        required=True,
        help="Parcel映射文件路径"
    )
    parser.add_argument(
        "--sae-release",
        type=str,
        default="gemma-scope-2b-pt-res",
        help="SAE release ID"
    )
    parser.add_argument(
        "--sae-local-base-dir",
        type=str,
        required=True,
        help="SAE本地基础目录"
    )
    parser.add_argument(
        "--sae-paths",
        type=str,
        default=None,
        help="SAE路径列表，用逗号分隔（如果为空则使用默认值）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="最大token数限制"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出文件已存在则跳过"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划操作，不实际执行"
    )
    return parser.parse_args()


def load_all_participants(input_path: str) -> Dict[str, Dict]:
    """
    从JSONL文件中加载所有参与者数据
    
    Args:
        input_path: 输入JSONL文件路径（每行一个JSON对象）
        
    Returns:
        参与者数据字典，键为participant_id，值为参与者数据
    """
    print(f"[INFO] 加载数据文件: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    all_participants: Dict[str, Dict] = {}
    line_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                line_count += 1
                
                # 从JSONL中提取participant ID
                participant_id = data.get("participant", "")
                if not participant_id:
                    print(f"[WARN] 第 {line_num} 行没有 participant 字段，跳过")
                    continue
                
                # 使用字符串形式的participant_id作为键
                participant_id_str = str(participant_id)
                
                # 检查必需字段
                if "instruction" not in data:
                    print(f"[WARN] 参与者 {participant_id_str} 没有 instruction 字段，跳过")
                    continue
                
                if "exp_list" not in data:
                    print(f"[WARN] 参与者 {participant_id_str} 没有 exp_list 字段，跳过")
                    continue
                
                # 如果已经存在相同的participant_id，则跳过或警告
                if participant_id_str in all_participants:
                    print(f"[WARN] 参与者ID '{participant_id_str}' 已存在，跳过重复项")
                    continue
                
                all_participants[participant_id_str] = data
                
            except json.JSONDecodeError as e:
                print(f"[ERROR] 第 {line_num} 行 JSON 解析失败: {e}")
                continue
            except Exception as e:
                import traceback
                print(f"[ERROR] 第 {line_num} 行处理失败: {e}")
                print(f"[ERROR] 异常类型: {type(e).__name__}")
                print(f"[ERROR] 完整traceback:")
                traceback.print_exc()
                continue
    
    print(f"[INFO] 总共读取 {line_count} 行，找到 {len(all_participants)} 个唯一参与者")
    return all_participants


def extract_parcel_activations_for_participant(
    participant_id: str,
    participant_data: Dict,
    extractor: SAEActModelTokenLimitedExtractor,
    num_parcels: int
) -> List[float]:
    """
    为单个参与者提取所有trial的parcel激活，并合并为一个列表
    
    Args:
        participant_id: 参与者ID
        participant_data: 参与者数据
        extractor: SAE激活提取器
        num_parcels: Parcel总数
        
    Returns:
        合并后的激活列表，格式为 [activation_value1, activation_value2, ...]
        将所有trial的所有token的所有parcel激活展平为一个一维列表
    """
    instruction = participant_data.get("instruction", "")
    exp_list = participant_data.get("exp_list", [])[:100]
    for i in range(len(exp_list)):
        if "<<" not in exp_list[i]:
            instruction += exp_list[i]
        else:
            break
    exp_list = exp_list[i:]
    
    if not instruction:
        print(f"[WARN] 参与者 {participant_id} 没有instruction字段")
        return []
    
    if not exp_list:
        print(f"[WARN] 参与者 {participant_id} 没有exp_list字段或为空")
        return []
    
    print(f"[INFO] 处理参与者 {participant_id}: {len(exp_list)} 个trial")
    
    all_activations = []
    
    # 处理每个trial
    for trial_idx, experiment in enumerate(tqdm(exp_list, desc=f"参与者 {participant_id}")):
        try:
            # 一次性提取所有parcel的激活
            parcel_features = extractor.extract_all_parcels_from_experiment(
                instruction=instruction,
                experiments=exp_list,
                current_experiment_idx=trial_idx,
                only_left_token=True
            )
            # 按parcel_id排序，确保顺序一致
            sorted_parcel_ids = sorted(parcel_features.keys())
            
            # 检查是否有激活
            if not parcel_features:
                print(f"[WARN] 参与者 {participant_id} trial {trial_idx} 没有提取到任何parcel激活")
                continue
            
            # 获取第一个parcel的激活数量（所有parcel应该有相同的num_relevant_tokens）
            first_parcel_features = parcel_features[sorted_parcel_ids[0]]
            num_relevant_tokens = first_parcel_features.shape[0]
            
            if num_relevant_tokens == 0:
                print(f"[WARN] 参与者 {participant_id} trial {trial_idx} 没有relevant_tokens")
                continue
            
            # 收集所有parcel的激活
            parcel_activation_list = []
            for parcel_id in sorted_parcel_ids:
                features = parcel_features[parcel_id]  # [num_relevant_tokens, num_latents]
                
                # 对每个parcel的latents求和（因为每个parcel可能有多个latents）
                # 得到每个relevant_token在该parcel上的总激活
                parcel_activation = features.sum(dim=1)  # [num_relevant_tokens]
                parcel_activation_list.append(parcel_activation)
            
            if parcel_activation_list:
                # 将所有parcel的激活拼接成 [num_relevant_tokens, num_parcels]
                trial_activation = torch.stack(parcel_activation_list, dim=1)  # [num_relevant_tokens, num_parcels]
                # 展平为一维列表并添加到总激活列表
                # 格式：[token0_parcel0, token0_parcel1, ..., token1_parcel0, token1_parcel1, ...]
                trial_activation_flat = trial_activation.flatten().cpu().tolist()
                all_activations.append(trial_activation_flat)
            else:
                print(f"[WARN] 参与者 {participant_id} trial {trial_idx} 没有提取到激活")
                
        except Exception as ex:
            import traceback
            print(f"[ERROR] 处理参与者 {participant_id} trial {trial_idx} 失败: {ex}")
            print(f"[ERROR] 异常类型: {type(ex).__name__}")
            print(f"[ERROR] 完整traceback:")
            traceback.print_exc()
            # 继续处理下一个trial，不添加空激活
    
    return all_activations


def get_num_parcels(parcel_mapping_path: str) -> int:
    """从parcel映射文件中获取parcel总数"""
    try:
        with open(parcel_mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        parcel_to_latents = data.get("parcel_to_latents", {})
        return len(parcel_to_latents)
    except Exception as ex:
        import traceback
        print(f"[ERROR] 无法读取parcel映射文件: {ex}")
        print(f"[ERROR] 异常类型: {type(ex).__name__}")
        print(f"[ERROR] 完整traceback:")
        traceback.print_exc()
        raise


def main():
    args = parse_args()
    
    # 检查输出文件是否已存在
    if args.skip_existing and os.path.exists(args.output):
        print(f"[SKIP] 输出文件已存在: {args.output}")
        return
    
    if args.dry_run:
        print(f"[DRY-RUN] 将处理输入文件: {args.input}")
        print(f"[DRY-RUN] 将保存到输出文件: {args.output}")
        return
    
    # 加载所有参与者（所有键下的所有参与者）
    all_participants = load_all_participants(args.input)
    
    if not all_participants:
        print("[ERROR] 没有找到任何参与者数据")
        return
    
    # 获取parcel总数
    num_parcels = get_num_parcels(args.parcel_mapping_path)
    print(f"[INFO] Parcel总数: {num_parcels}")
    
    # 解析SAE路径
    if args.sae_paths:
        sae_paths = [p.strip() for p in args.sae_paths.split(',') if p.strip()]
    else:
        sae_paths = SAEActModelTokenLimitedExtractor.DEFAULT_SAE_PATHS_2B_PT
    
    # 创建提取器（使用第一个parcel_id初始化，实际会提取所有parcel）
    print(f"[INFO] 初始化提取器...")
    print(f"[INFO] 模型: {args.model_name}")
    print(f"[INFO] SAE release: {args.sae_release}")
    print(f"[INFO] SAE本地目录: {args.sae_local_base_dir}")
    
    extractor = SAEActModelTokenLimitedExtractor(
        model_name=args.model_name,
        tokenizer=None,
        parcel_id=0,  # 仅用于初始化，实际会提取所有parcel
        parcel_mapping_path=args.parcel_mapping_path,
        sae_release=args.sae_release,
        sae_local_base_dir=args.sae_local_base_dir,
        sae_paths=sae_paths,
        max_tokens=args.max_tokens
    )
    
    # 处理每个参与者
    all_results: Dict[str, List[float]] = {}
    
    for participant_id, participant_data in tqdm(all_participants.items(), desc="处理参与者"):
        try:
            activations = extract_parcel_activations_for_participant(
                participant_id=participant_id,
                participant_data=participant_data,
                extractor=extractor,
                num_parcels=num_parcels
            )
            print(f"[INFO] 参与者 {participant_id} 的激活长度: {len(activations)}")
            # 沿着 dim 0 求平均
            all_results[participant_id] = torch.mean(torch.tensor(activations), dim=0).tolist()
            print(f"[INFO] 参与者 {participant_id} 完成: {len(activations)} 个激活值")
        except Exception as ex:
            import traceback
            print(f"[ERROR] 处理参与者 {participant_id} 失败: {ex}")
            print(f"[ERROR] 异常类型: {type(ex).__name__}")
            print(f"[ERROR] 完整traceback:")
            traceback.print_exc()
            continue
    
    # 保存结果
    print(f"[INFO] 保存结果到: {args.output}")
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] 处理完成，共处理 {len(all_results)} 个参与者")
    print(f"[INFO] 结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

