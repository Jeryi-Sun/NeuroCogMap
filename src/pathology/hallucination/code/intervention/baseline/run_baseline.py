#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 干预运行脚本

使用 10 条 correct 和 incorrect 数据对比得到 steer vector，然后进行干预。
评测方案与上级目录的 intervention 保持一致。
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import argparse
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# 使用本地的 data_loader（独立实现，不依赖上级目录）
from data_loader import HallucinationDataLoader, HallucinationEvaluator

from baseline_intervention import BaselineIntervention


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger("baseline_intervention")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def run_baseline_experiment(config: Dict[str, Any], 
                           steer_vector_path: str,
                           test_dataset_path: str,
                           intervention_strength: float = 1.0,
                           max_samples: int = 0,
                           enable_evaluation: bool = True,
                           output_file: Optional[str] = None,
                           skip_if_exists: bool = True) -> Dict[str, Any]:
    """运行 baseline 干预实验"""
    logger = logging.getLogger("baseline_intervention")
    
    # 检查输出文件是否已存在
    if skip_if_exists and output_file and os.path.exists(output_file):
        logger.info(f"输出文件已存在，跳过: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    model_name = config.get("model_name", "google/gemma-2-2b")
    
    # 创建干预系统
    intervention = BaselineIntervention(
        model_name=model_name,
        sae_release=config.get("sae_release", "gemma-scope-2b-pt-res"),
        sae_local_base_dir=config.get("sae_local_base_dir", "/path/to/local_models/gemma-scope-2b-pt-res"),
        results_dir=config.get("results_dir", "/path/to/project_root/safety_explanation/hallucination/results/intervention/baseline"),
        is_instruct=config.get("is_instruct", False),
        layers=config.get("layers", None)  # None 表示使用所有层
    )
    
    # 加载 steer vectors
    intervention.load_steer_vectors(steer_vector_path)
    
    # 加载模型和SAE
    logger.info("加载模型和SAE...")
    intervention.load_model_and_tokenizer()
    intervention.load_saes(
        config.get("sae_paths", []), 
        max_saes=config.get("max_saes", 100)
    )
    
    # 加载测试数据
    logger.info(f"加载测试数据: {test_dataset_path}")
    data_loader = HallucinationDataLoader(logger)
    test_data = data_loader.load_dataset(test_dataset_path, max_samples)
    
    # 运行干预实验
    logger.info(f"开始干预实验，strength={intervention_strength}")
    results = intervention.run_intervention_experiment(
        test_data, 
        save_results=False, 
        use_incontext=config.get("use_incontext", False),
        intervention_strength=intervention_strength
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
        "steer_vector_path": steer_vector_path,
        "test_dataset_path": test_dataset_path,
        "model_name": model_name,
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Baseline 干预实验运行脚本")
    
    # 基本配置
    parser.add_argument("--config", type=str, 
                       default="/path/to/project_root/safety_explanation/hallucination/code/intervention/config.json",
                       help="配置文件路径")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    
    # 数据配置
    parser.add_argument("--steer_vector_file", type=str, required=True,
                       help="预计算的 steer vector 文件路径")
    parser.add_argument("--test_dataset", type=str, required=True,
                       help="测试数据集路径（CSV文件）")
    parser.add_argument("--max_samples", type=int, default=0, help="最大样本数，0表示全部")
    
    # 干预配置
    parser.add_argument("--intervention_strength", type=float, default=1.0, help="干预强度")
    parser.add_argument("--layers", type=int, nargs="+", help="要干预的层列表，不指定则使用所有层")
    
    # 评估配置
    parser.add_argument("--enable_evaluation", action="store_true", help="启用评估")
    parser.add_argument("--output_file", type=str, help="结果文件保存路径（JSON 格式）")
    parser.add_argument("--skip_if_exists", action="store_true", default=True,
                       help="如果输出文件已存在则跳过（默认开启）")
    parser.add_argument("--vllm_url", type=str, default="http://127.0.0.1:8001/v1", help="vLLM服务地址")
    parser.add_argument("--api_key", type=str, default="abcabc", help="API密钥")
    parser.add_argument("--use_incontext", action="store_true", help="使用in-context learning")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 加载配置
    try:
        if os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"配置文件加载成功: {args.config}")
        else:
            logger.warning(f"配置文件不存在: {args.config}，使用默认配置")
            config = {}
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        config = {}
    
    # 覆盖命令行参数
    if args.layers:
        config["layers"] = args.layers
    if args.enable_evaluation:
        config["enable_evaluation"] = True
    if args.vllm_url:
        config["vllm_url"] = args.vllm_url
    if args.api_key:
        config["api_key"] = args.api_key
    if args.use_incontext:
        config["use_incontext"] = True
    
    try:
        # 检查文件是否存在
        if not os.path.exists(args.steer_vector_file):
            logger.error(f"steer vector 文件不存在: {args.steer_vector_file}")
            return
        
        if not os.path.exists(args.test_dataset):
            logger.error(f"测试数据集文件不存在: {args.test_dataset}")
            return
        
        results = run_baseline_experiment(
            config,
            args.steer_vector_file,
            args.test_dataset,
            args.intervention_strength,
            args.max_samples,
            args.enable_evaluation,
            args.output_file,
            args.skip_if_exists
        )
        
        logger.info("Baseline 干预实验完成")
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()

