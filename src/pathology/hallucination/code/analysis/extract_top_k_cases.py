#!/usr/bin/env python3
"""
分析代码：提取指定 Parcel ID 上激活最大的 K 个案例

该脚本读取 token_parcels.jsonl 文件（包含每个案例的 Parcel 激活数据）和原始 correct.jsonl 文件，
计算每个案例在指定 Parcel ID 上的总激活值，并返回激活最大的 K 个案例的完整信息。
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import traceback


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="提取指定 Parcel ID 上激活最大的 K 个案例"
    )
    parser.add_argument(
        "--parcel-id",
        type=int,
        required=True,
        help="要分析的 Parcel ID (0-based)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="返回的 Top-K 案例数量 (默认: 10)"
    )
    parser.add_argument(
        "--token-parcels-file",
        type=str,
        default="/path/to/project_root/safety_explanation/hallucination/results/truthfulqa_gemma-2-2b/parcels_token_acts/correct/token_parcels.jsonl",
        help="token_parcels.jsonl 文件路径"
    )
    parser.add_argument(
        "--original-file",
        type=str,
        default="/path/to/project_root/safety_explanation/hallucination/results/truthfulqa_gemma-2-2b/correct.jsonl",
        help="原始 correct.jsonl 文件路径"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出文件路径 (默认: parcel_{parcel_id}_top_{k}_cases.jsonl)"
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default="sum",
        choices=["sum", "max", "mean"],
        help="激活值聚合方法: sum(求和), max(最大值), mean(平均值) (默认: sum)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果输出文件已存在则跳过处理"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出信息"
    )
    
    return parser.parse_args()


def load_token_parcels_data(file_path: str) -> Dict[int, Dict]:
    """
    加载 token_parcels.jsonl 文件数据
    
    Args:
        file_path: token_parcels.jsonl 文件路径
        
    Returns:
        Dict[int, Dict]: 以 index 为键的字典，包含每个案例的 Parcel 激活数据
    """
    token_parcels_data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    index = data.get('index')
                    if index is None:
                        print(f"[WARN] 第 {line_idx} 行缺少 index 字段，跳过", file=sys.stderr)
                        continue
                    
                    token_parcels_data[index] = data
                    
                except json.JSONDecodeError as e:
                    print(f"[ERROR] 第 {line_idx} 行 JSON 解析失败: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"[ERROR] 第 {line_idx} 行处理失败: {e}", file=sys.stderr)
                    print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    continue
                    
    except FileNotFoundError:
        print(f"[ERROR] 文件不存在: {file_path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}", file=sys.stderr)
        print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    
    print(f"[INFO] 成功加载 {len(token_parcels_data)} 个案例的 Parcel 激活数据")
    return token_parcels_data


def load_original_data(file_path: str) -> Dict[int, Dict]:
    """
    加载原始 correct.jsonl 文件数据
    
    Args:
        file_path: correct.jsonl 文件路径
        
    Returns:
        Dict[int, Dict]: 以 index 为键的字典，包含每个案例的完整信息
    """
    original_data = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    index = data.get('index')
                    if index is None:
                        print(f"[WARN] 第 {line_idx} 行缺少 index 字段，跳过", file=sys.stderr)
                        continue
                    
                    original_data[index] = data
                    
                except json.JSONDecodeError as e:
                    print(f"[ERROR] 第 {line_idx} 行 JSON 解析失败: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    print(f"[ERROR] 第 {line_idx} 行处理失败: {e}", file=sys.stderr)
                    print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    continue
                    
    except FileNotFoundError:
        print(f"[ERROR] 文件不存在: {file_path}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}", file=sys.stderr)
        print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    
    print(f"[INFO] 成功加载 {len(original_data)} 个案例的原始数据")
    return original_data


def calculate_parcel_activation(
    token_parcels_data: Dict[int, Dict], 
    parcel_id: int, 
    aggregation_method: str = "sum"
) -> List[Tuple[int, float]]:
    """
    计算每个案例在指定 Parcel ID 上的激活值
    
    Args:
        token_parcels_data: token_parcels 数据
        parcel_id: 目标 Parcel ID
        aggregation_method: 聚合方法 ("sum", "max", "mean")
        
    Returns:
        List[Tuple[int, float]]: (index, activation_value) 的列表，按激活值降序排列
    """
    case_activations = []
    
    for index, data in token_parcels_data.items():
        try:
            token_parcel_acts = data.get('token_parcel_acts')
            if not token_parcel_acts:
                print(f"[WARN] 案例 {index} 缺少 token_parcel_acts 数据，跳过", file=sys.stderr)
                continue
            
            parcel_dim = data.get('parcel_dim', 0)
            if parcel_id >= parcel_dim:
                print(f"[WARN] Parcel ID {parcel_id} 超出范围 [0, {parcel_dim-1}]，跳过案例 {index}", file=sys.stderr)
                continue
            
            # 提取指定 Parcel ID 的激活值
            parcel_activations = []
            for token_acts in token_parcel_acts:
                if token_acts and len(token_acts) > parcel_id:
                    parcel_activations.append(token_acts[parcel_id])
            
            if not parcel_activations:
                print(f"[WARN] 案例 {index} 在 Parcel {parcel_id} 上没有激活数据，跳过", file=sys.stderr)
                continue
            
            # 根据聚合方法计算总激活值
            if aggregation_method == "sum":
                total_activation = sum(parcel_activations)
            elif aggregation_method == "max":
                total_activation = max(parcel_activations)
            elif aggregation_method == "mean":
                total_activation = sum(parcel_activations) / len(parcel_activations)
            else:
                raise ValueError(f"不支持的聚合方法: {aggregation_method}")
            
            case_activations.append((index, total_activation))
            
        except Exception as e:
            print(f"[ERROR] 计算案例 {index} 的 Parcel {parcel_id} 激活值失败: {e}", file=sys.stderr)
            print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
    
    # 按激活值降序排列
    case_activations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"[INFO] 成功计算 {len(case_activations)} 个案例的 Parcel {parcel_id} 激活值")
    return case_activations


def extract_top_k_cases(
    case_activations: List[Tuple[int, float]], 
    original_data: Dict[int, Dict], 
    k: int
) -> List[Dict]:
    """
    提取 Top-K 案例的完整信息
    
    Args:
        case_activations: 案例激活值列表
        original_data: 原始案例数据
        k: 返回的案例数量
        
    Returns:
        List[Dict]: Top-K 案例的完整信息列表
    """
    top_k_cases = []
    
    for i, (index, activation_value) in enumerate(case_activations[:k]):
        if index not in original_data:
            print(f"[WARN] 案例 {index} 在原始数据中不存在，跳过", file=sys.stderr)
            continue
        
        case_info = original_data[index].copy()
        case_info['parcel_activation'] = activation_value
        case_info['rank'] = i + 1
        
        top_k_cases.append(case_info)
    
    print(f"[INFO] 成功提取 {len(top_k_cases)} 个 Top-{k} 案例")
    return top_k_cases


def save_results(top_k_cases: List[Dict], output_file: str, parcel_id: int, k: int) -> None:
    """
    保存结果到文件
    
    Args:
        top_k_cases: Top-K 案例列表
        output_file: 输出文件路径
        parcel_id: Parcel ID
        k: 案例数量
    """
    try:
        # 创建输出目录
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for case in top_k_cases:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        print(f"[INFO] 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"[ERROR] 保存结果失败: {e}", file=sys.stderr)
        print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


def print_summary(top_k_cases: List[Dict], parcel_id: int, k: int, aggregation_method: str) -> None:
    """
    打印结果摘要
    
    Args:
        top_k_cases: Top-K 案例列表
        parcel_id: Parcel ID
        k: 案例数量
        aggregation_method: 聚合方法
    """
    print(f"\n{'='*80}")
    print(f"Parcel {parcel_id} Top-{k} 案例摘要 (聚合方法: {aggregation_method})")
    print(f"{'='*80}")
    
    for i, case in enumerate(top_k_cases, 1):
        print(f"\n排名 {i}:")
        print(f"  索引: {case.get('index', 'N/A')}")
        print(f"  激活值: {case.get('parcel_activation', 'N/A'):.4f}")
        print(f"  问题: {case.get('question', 'N/A')}")
        print(f"  模型答案: {case.get('model_answer', 'N/A')}")
        print(f"  正确答案: {case.get('answer_true', 'N/A')}")
        if case.get('context'):
            print(f"  上下文: {case.get('context', 'N/A')}")


def main() -> None:
    """主函数"""
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.token_parcels_file):
        print(f"[ERROR] token_parcels 文件不存在: {args.token_parcels_file}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.original_file):
        print(f"[ERROR] 原始文件不存在: {args.original_file}", file=sys.stderr)
        sys.exit(1)
    
    # 设置输出文件路径
    if args.output_file is None:
        args.output_file = f"parcel_{args.parcel_id}_top_{args.k}_cases.jsonl"
    
    # 检查是否跳过已存在的文件
    if args.skip_existing and os.path.exists(args.output_file):
        print(f"[SKIP] 输出文件已存在，跳过处理: {args.output_file}")
        return
    
    try:
        # 加载数据
        print(f"[INFO] 加载 token_parcels 数据: {args.token_parcels_file}")
        token_parcels_data = load_token_parcels_data(args.token_parcels_file)
        
        print(f"[INFO] 加载原始数据: {args.original_file}")
        original_data = load_original_data(args.original_file)
        
        # 计算 Parcel 激活值
        print(f"[INFO] 计算 Parcel {args.parcel_id} 的激活值 (聚合方法: {args.aggregation_method})")
        case_activations = calculate_parcel_activation(
            token_parcels_data, 
            args.parcel_id, 
            args.aggregation_method
        )
        
        if not case_activations:
            print(f"[ERROR] 没有找到任何案例在 Parcel {args.parcel_id} 上的激活数据", file=sys.stderr)
            sys.exit(1)
        
        # 提取 Top-K 案例
        print(f"[INFO] 提取 Top-{args.k} 案例")
        top_k_cases = extract_top_k_cases(case_activations, original_data, args.k)
        
        if not top_k_cases:
            print(f"[ERROR] 没有成功提取任何案例", file=sys.stderr)
            sys.exit(1)
        
        # 保存结果
        print(f"[INFO] 保存结果到: {args.output_file}")
        save_results(top_k_cases, args.output_file, args.parcel_id, args.k)
        
        # 打印摘要
        if args.verbose:
            print_summary(top_k_cases, args.parcel_id, args.k, args.aggregation_method)
        
        print(f"\n[SUCCESS] 成功完成 Parcel {args.parcel_id} Top-{args.k} 案例提取")
        print(f"结果文件: {args.output_file}")
        print(f"提取案例数: {len(top_k_cases)}")
        
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}", file=sys.stderr)
        print(f"[ERROR] 异常类型: {type(e).__name__}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
