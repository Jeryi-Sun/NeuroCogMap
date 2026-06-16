#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幻觉干预运行脚本

支持多种运行模式和配置选项的幻觉干预实验。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from hallucination_intervention import HallucinationIntervention
from data_loader import HallucinationDataLoader, HallucinationEvaluator


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger("hallucination_intervention")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def run_single_experiment(config: Dict[str, Any], 
                         dataset_path: str,
                         parcel_ids: List[int],
                         intervention_strength: float = 1.0,
                         max_samples: int = 0,
                         enable_evaluation: bool = True,
                         output_file: Optional[str] = None) -> Dict[str, Any]:
    """运行单个干预实验"""
    logger = logging.getLogger("hallucination_intervention")
    
    # 从数据集路径提取数据集名称
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    model_name = config.get("model_name", "google/gemma-2-2b")
    
    # 动态生成parcel_json_path
    parcel_json_path_template = config.get("parcel_json_path_template")
    if parcel_json_path_template:
        # 替换模板中的占位符
        parcel_json_path = parcel_json_path_template.format(
            dataset_name=dataset_name
        )
    else:
        # 如果没有模板，使用旧的固定路径
        parcel_json_path = config.get("parcel_json_path")
    
    logger.info(f"使用parcel_json_path: {parcel_json_path}")
    
    # 创建干预系统
    intervention = HallucinationIntervention(
        model_name=model_name,
        sae_release=config.get("sae_release", "gemma-scope-2b-pt-res"),
        sae_local_base_dir=config.get("sae_local_base_dir", "/path/to/local_models/gemma-scope-2b-pt-res"),
        parcel_json_path=parcel_json_path,
        latent_parcel_assignments_path=config.get("latent_parcel_assignments_path"),
        max_activation_dir=config.get("max_activation_dir", "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation"),
        results_dir=config.get("results_dir", "/path/to/project_root/safety_explanation/hallucination/results/intervention"),
        is_instruct=config.get("is_instruct", False),
        lambda_scale=config.get("lambda_scale", 0.3),
        smooth=config.get("smooth", 80.0),
        min_scale=config.get("min_scale", -1.0),
        max_scale=config.get("max_scale", 1.0),
        strength=intervention_strength
    )
    
    # 加载模型和SAE
    logger.info("加载模型和SAE...")
    intervention.load_model_and_tokenizer()
    intervention.load_saes(
        config.get("sae_paths", []), 
        max_saes=config.get("max_saes", 10)
    )
    
    # 加载测试数据
    logger.info(f"加载测试数据: {dataset_path}")
    data_loader = HallucinationDataLoader(logger)
    test_data = data_loader.load_dataset(dataset_path, max_samples)
    # 运行干预实验
    logger.info(f"开始干预实验，parcel_ids={parcel_ids}, strength={intervention_strength}")
    results = intervention.run_intervention_experiment(
        test_data, parcel_ids, save_results=False, use_incontext=config.get("use_incontext", False)  # 不自动保存，我们手动保存
    )
    
    # 评估干预效果
    if enable_evaluation and config.get("vllm_url"):
        logger.info("开始评估干预效果...")
        try:
            evaluator = HallucinationEvaluator(
                vllm_url=config.get("vllm_url", "http://127.0.0.1:8001/v1"),
                api_key=config.get("api_key", "abcabc"),
                logger=logger
            )
            
            # 获取干预结果进行评估
            intervention_results = results["intervention_results"].get(str(intervention_strength), {})
            if "intervention_results" in intervention_results:
                eval_results = evaluator.evaluate_batch(
                    test_data,
                    intervention_results["baseline_results"],
                    intervention_results["intervention_results"]
                )
                results["evaluation"] = eval_results
                
                logger.info(f"评估完成 - 基线准确率: {eval_results['baseline_stats']['accuracy']:.3f}, "
                           f"干预准确率: {eval_results['intervention_stats']['accuracy']:.3f}, "
                           f"改善: {eval_results['improvement']['relative_improvement']:.1f}%")
            else:
                logger.warning("未找到干预结果，跳过评估")
        except Exception as e:
            logger.error(f"评估失败: {e}")
            print(f"评估失败: {e}")
            results["evaluation"] = {
                "error": str(e),
                "note": "评估服务不可用或失败，仅返回干预结果"
            }
    
    # 添加数据集和模型信息到结果中
    results["dataset_info"] = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "model_name": model_name,
        "parcel_ids": parcel_ids,
        "intervention_strength": intervention_strength,
        "max_samples": max_samples,
        "enable_evaluation": enable_evaluation
    }
    
    # 如果指定了输出文件，直接保存
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"实验结果已保存至: {output_path}")
    else:
        # 保持向后兼容：如果没有指定输出文件，仍然输出到 stdout
        print("\n" + "="*50)
        print("EXPERIMENT_RESULTS_JSON_START")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        print("EXPERIMENT_RESULTS_JSON_END")
        print("="*50)
    
    return results


def run_batch_experiments(config: Dict[str, Any], 
                         dataset_paths: List[str],
                         parcel_id_groups: List[List[int]],
                         intervention_strengths: List[float] = None) -> Dict[str, Any]:
    """运行批量干预实验"""
    logger = logging.getLogger("hallucination_intervention")
    
    if intervention_strengths is None:
        intervention_strengths = [0.5, 1.0, 2.0, 4.0]
    
    all_results = {
        "batch_info": {
            "num_datasets": len(dataset_paths),
            "num_parcel_groups": len(parcel_id_groups),
            "intervention_strengths": intervention_strengths
        },
        "experiments": {}
    }
    
    for i, dataset_path in enumerate(dataset_paths):
        logger.info(f"处理数据集 {i+1}/{len(dataset_paths)}: {dataset_path}")
        
        dataset_name = Path(dataset_path).stem
        dataset_results = {}
        
        for j, parcel_ids in enumerate(parcel_id_groups):
            logger.info(f"  测试parcel组 {j+1}/{len(parcel_id_groups)}: {parcel_ids}")
            
            group_results = {}
            for strength in intervention_strengths:
                try:
                    result = run_single_experiment(
                        config, dataset_path, parcel_ids, strength,
                        max_samples=config.get("max_samples", 0),
                        enable_evaluation=config.get("enable_evaluation", True)
                    )
                    group_results[str(strength)] = result
                    
                except Exception as e:
                    logger.error(f"    强度 {strength} 实验失败: {e}")
                    group_results[str(strength)] = {"error": str(e)}
            
            dataset_results[f"parcel_group_{j}"] = group_results
        
        all_results["experiments"][dataset_name] = dataset_results
    
    # 保存批量结果
    results_file = os.path.join(
        config.get("results_dir", "/path/to/project_root/safety_explanation/hallucination/results/intervention"),
        "batch_intervention_results.json"
    )
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"批量实验结果已保存到: {results_file}")
    
    return all_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="幻觉干预实验运行脚本")
    
    # 基本配置
    parser.add_argument("--config", type=str, 
                       default="/path/to/project_root/safety_explanation/hallucination/code/intervention/config.json",
                       help="配置文件路径")
    parser.add_argument("--mode", type=str, choices=["single", "batch", "eval_only"], default="single",
                       help="运行模式：single=单个实验，batch=批量实验，eval_only=仅评估已有结果")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    
    # 数据配置
    parser.add_argument("--dataset", type=str, help="数据集路径（单个文件或目录）")
    parser.add_argument("--datasets", type=str, nargs="+", help="多个数据集路径（批量模式）")
    parser.add_argument("--max_samples", type=int, default=0, help="最大样本数，0表示全部")
    
    # 干预配置
    parser.add_argument("--parcel_ids", type=int, nargs="+", help="要干预的parcel ID列表")
    parser.add_argument("--parcel_groups", type=str, help="parcel组配置文件路径（批量模式）")
    parser.add_argument("--intervention_strength", type=float, default=1.0, help="干预强度")
    parser.add_argument("--intervention_strengths", type=float, nargs="+", help="多个干预强度（批量模式）")
    
    # 评估配置
    parser.add_argument("--enable_evaluation", action="store_true", help="启用评估")
    parser.add_argument("--output_file", type=str, help="结果文件保存路径（JSON 格式）")
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8001/v1", help="vLLM服务地址")
    parser.add_argument("--api_key", type=str, default="abcabc", help="API密钥")
    parser.add_argument("--eval_only_from_file", type=str, help="仅评估模式：从已有结果文件读取并评估")
    parser.add_argument("--use_incontext", action="store_true", help="使用in-context learning")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 加载配置
    try:
        config = load_config(args.config)
        logger.info(f"配置文件加载成功: {args.config}")
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        return
    
    # 覆盖命令行参数
    if args.dataset:
        config["dataset"] = args.dataset
    if args.datasets:
        config["datasets"] = args.datasets
    if args.max_samples > 0:
        config["max_samples"] = args.max_samples
    if args.parcel_ids:
        config["parcel_ids"] = args.parcel_ids
    config["intervention_strength"] = args.intervention_strength
    if args.intervention_strengths:
        config["intervention_strengths"] = args.intervention_strengths
    if args.enable_evaluation:
        config["enable_evaluation"] = True
    if args.vllm_url:
        config["vllm_url"] = args.vllm_url
    if args.api_key:
        config["api_key"] = args.api_key
    if args.use_incontext:
        config["use_incontext"] = args.use_incontext
    try:
        if args.mode == "single":
            # 单个实验模式
            if not config.get("dataset"):
                logger.error("单个实验模式需要指定 --dataset 参数")
                return
            
            if not config.get("parcel_ids"):
                logger.error("需要指定 --parcel_ids 参数")
                return
            results = run_single_experiment(
                config,
                config["dataset"],
                config["parcel_ids"],
                config.get("intervention_strength", 1.0),
                config.get("max_samples", 0),
                config.get("enable_evaluation", False),
                output_file=args.output_file
            )
            
            logger.info("单个实验完成")
            
        elif args.mode == "batch":
            # 批量实验模式
            if not config.get("datasets"):
                logger.error("批量实验模式需要指定 --datasets 参数")
                return
            
            # 加载parcel组配置
            parcel_groups = []
            if args.parcel_groups and os.path.exists(args.parcel_groups):
                with open(args.parcel_groups, 'r', encoding='utf-8') as f:
                    parcel_groups = json.load(f)
            elif config.get("parcel_ids"):
                parcel_groups = [config["parcel_ids"]]
            else:
                logger.error("需要指定parcel组配置")
                return
            
            results = run_batch_experiments(
                config,
                config["datasets"],
                parcel_groups,
                config.get("intervention_strengths", [0.5, 1.0, 2.0, 4.0])
            )
            
            logger.info("批量实验完成")
        
        elif args.mode == "eval_only" or args.eval_only_from_file:
            # 仅评估模式：从已有结果文件读取并评估
            eval_file = args.eval_only_from_file or args.dataset
            if not eval_file:
                logger.error("仅评估模式需要指定 --eval_only_from_file 或 --dataset 参数")
                return
            
            if not os.path.exists(eval_file):
                logger.error(f"结果文件不存在: {eval_file}")
                return
            
            logger.info(f"从文件读取结果: {eval_file}")
            
            # 读取已有结果
            with open(eval_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # 获取需要评估的信息
            intervention_results = existing_results.get("intervention_results", {})
            dataset_path = None
            
            # 尝试从结果中获取数据集信息
            dataset_info = existing_results.get("dataset_info", {})
            if dataset_info and dataset_info.get("dataset_path"):
                dataset_path = dataset_info["dataset_path"]
                logger.info(f"从结果中读取数据集路径: {dataset_path}")
            
            # 如果结果中没有数据集路径，尝试从文件名推断
            if not dataset_path:
                # 尝试从文件名提取数据集名称，然后查找原数据集
                # 文件名格式通常是: model_name_dataset_name_intervention.json
                import re
                match = re.search(r'_([^_]+)_intervention\.json$', eval_file)
                if match:
                    dataset_name = match.group(1)
                    # 尝试在数据集目录中查找
                    default_dataset_dir = "/path/to/project_root/safety_explanation/hallucination/dataset"
                    potential_path = os.path.join(default_dataset_dir, f"{dataset_name}.csv")
                    if os.path.exists(potential_path):
                        dataset_path = potential_path
                        logger.info(f"推断数据集路径: {dataset_path}")
                    else:
                        logger.warning(f"无法找到数据集文件: {potential_path}")
            
            if not dataset_path or not os.path.exists(dataset_path):
                logger.error(f"无法确定数据集路径，评估将使用已有结果中的测试数据")
                dataset_path = None
            
            # 评估干预效果
            if config.get("vllm_url"):
                logger.info("开始评估干预效果...")
                evaluator = HallucinationEvaluator(
                    vllm_url=config.get("vllm_url", "http://127.0.0.1:8001/v1"),
                    api_key=config.get("api_key", "abcabc"),
                    logger=logger
                )
                
                # 对每个干预强度进行评估
                for strength, result_data in intervention_results.items():
                    if "intervention_results" not in result_data:
                        logger.warning(f"强度 {strength} 的结果格式不正确，跳过")
                        continue
                    
                    baseline_results = result_data.get("baseline_results", [])
                    intervention_results_list = result_data.get("intervention_results", [])
                    
                    if not baseline_results or not intervention_results_list:
                        logger.warning(f"强度 {strength} 的结果为空，跳过")
                        continue
                    
                    logger.info(f"评估强度 {strength} 的结果...")
                    
                    # 如果找不到数据集，直接从结果中提取问题进行评估
                    if dataset_path:
                        data_loader = HallucinationDataLoader(logger)
                        test_data = data_loader.load_dataset(dataset_path, max_samples=config.get("max_samples", 0))
                        # 确保test_data的长度匹配
                        min_len = min(len(test_data), len(baseline_results), len(intervention_results_list))
                        test_data = test_data[:min_len]
                        baseline_results = baseline_results[:min_len]
                        intervention_results_list = intervention_results_list[:min_len]
                    else:
                        # 从结果中提取问题信息（仅作为后备）
                        test_data = []
                        for i, baseline in enumerate(baseline_results):
                            if i < len(intervention_results_list):
                                test_data.append({
                                    "question": baseline.get("question", baseline.get("prompt", "")),
                                    "context": baseline.get("context", []),
                                    "answer_true": baseline.get("answer_true", [])
                                })
                    
                    try:
                        # 现在 evaluate_batch 可以直接从 baseline_results 和 intervention_results_list 中提取所需信息
                        eval_results = evaluator.evaluate_batch(
                            test_data,  # 作为后备，主要用于日志
                            baseline_results,
                            intervention_results_list
                        )
                        
                        # 更新结果中的评估信息
                        if "evaluation" not in existing_results:
                            existing_results["evaluation"] = {}
                        existing_results["evaluation"][strength] = eval_results
                        
                        logger.info(f"评估完成 - 基线准确率: {eval_results['baseline_stats']['accuracy']:.3f}, "
                                   f"干预准确率: {eval_results['intervention_stats']['accuracy']:.3f}, "
                                   f"改善: {eval_results['improvement']['relative_improvement']:.1f}%")
                    except Exception as e:
                        logger.error(f"评估强度 {strength} 失败: {e}")
                        print(f"评估强度 {strength} 失败: {e}")
                
                # 保存更新后的结果
                output_file = eval_file  # 覆盖原文件
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"评估结果已保存到: {output_file}")
            else:
                logger.error("未配置vLLM地址，无法进行评估")
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
