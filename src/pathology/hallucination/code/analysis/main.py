#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幻觉机制分析主执行脚本

整合所有分析模块，提供统一的命令行接口来运行完整的幻觉机制分析流程。

作者: Jeryi
日期: 2025
"""

import argparse
import sys
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time
import json

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入分析模块
from analysis_parcel_level import ParcelLevelAnalyzer
from analysis_capability_level import CapabilityLevelAnalyzer
from analysis_llm_summary import LLMReportGenerator
from config import get_default_config, load_env_config, validate_config, save_config, create_dynamic_config
from utils import setup_logging, FileManager

# 设置日志
logger = logging.getLogger(__name__)

class HallucinationAnalysisPipeline:
    """幻觉机制分析流水线"""
    
    def __init__(self, config: Optional[Dict] = None, config_file: Optional[str] = None):
        """
        初始化分析流水线
        
        Args:
            config: 配置字典
            config_file: 配置文件路径
        """
        if config_file and os.path.exists(config_file):
            from config import load_config
            self.config = load_config(config_file)
        elif config:
            self.config = config
        else:
            self.config = load_env_config()
        
        # 验证配置
        errors = validate_config(self.config)
        if errors:
            logger.error("配置验证失败:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("配置验证失败")
        
        # 设置日志
        log_file = Path(self.config['experiment_config'].experiment_name + '.log')
        setup_logging(log_level='INFO', log_file=str(log_file))
        
        # 创建输出目录
        self._create_output_directories()
        
        # 初始化分析器
        self.parcel_analyzer = None
        self.capability_analyzer = None
        self.llm_generator = None
        
    def _create_output_directories(self) -> None:
        """创建输出目录"""
        data_paths = self.config['data_paths']
        
        # 创建所有输出目录
        directories = [
            data_paths.parcel_level_output,
            data_paths.capability_level_output,
            data_paths.llm_analysis_output
        ]
        
        for directory in directories:
            FileManager.ensure_dir(directory)
            logger.info(f"创建输出目录: {directory}")
    
    def run_parcel_analysis(self) -> bool:
        """运行Parcel级别分析"""
        try:
            logger.info("=" * 60)
            logger.info("开始Parcel级别分析")
            logger.info("=" * 60)
            
            data_paths = self.config['data_paths']
            model_config = self.config['model_config']
            # 检查输入文件
            if not os.path.exists(data_paths.correct_activations):
                logger.error(f"正确样本激活文件不存在: {data_paths.correct_activations}")
                return False
            
            if not os.path.exists(data_paths.incorrect_activations):
                logger.error(f"幻觉样本激活文件不存在: {data_paths.incorrect_activations}")
                return False
            
            # 创建分析器
            self.parcel_analyzer = ParcelLevelAnalyzer(
                correct_jsonl_path=data_paths.correct_activations,
                incorrect_jsonl_path=data_paths.incorrect_activations,
                output_dir=data_paths.parcel_level_output,
                parcel_info_path=data_paths.parcel_descriptions,
                epsilon=model_config.epsilon,
                significance_threshold=model_config.significance_threshold,
                skip_existing=getattr(model_config, 'skip_existing', False),
                max_tokens=model_config.max_tokens,
                use_pca_connectivity=model_config.use_pca_connectivity,
            )
            
            # 运行分析
            start_time = time.time()
            self.parcel_analyzer.run_analysis()
            end_time = time.time()
            
            logger.info(f"Parcel级别分析完成，耗时: {end_time - start_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"Parcel级别分析失败: {e}")
            return False
    
    def run_capability_analysis(self) -> bool:
        """运行Capability级别分析"""
        try:
            logger.info("=" * 60)
            logger.info("开始Capability级别分析")
            logger.info("=" * 60)
            
            data_paths = self.config['data_paths']
            model_config = self.config['model_config']
            
            # 检查输入文件
            if not os.path.exists(data_paths.capability_parcel_mapping):
                logger.error(f"Capability-Parcel映射文件不存在: {data_paths.capability_parcel_mapping}")
                return False
            
            if not os.path.exists(data_paths.correct_activations):
                logger.error(f"正确样本激活文件不存在: {data_paths.correct_activations}")
                return False
            
            if not os.path.exists(data_paths.incorrect_activations):
                logger.error(f"幻觉样本激活文件不存在: {data_paths.incorrect_activations}")
                return False
            
            # 创建分析器
            self.capability_analyzer = CapabilityLevelAnalyzer(
                mapping_json_path=data_paths.capability_parcel_mapping,
                correct_jsonl_path=data_paths.correct_activations,
                incorrect_jsonl_path=data_paths.incorrect_activations,
                output_dir=data_paths.capability_level_output,
                capability_desc_path=data_paths.capability_descriptions,
                epsilon=model_config.epsilon,
                significance_threshold=model_config.significance_threshold,
                skip_existing=getattr(model_config, 'skip_existing', False),
                max_tokens=model_config.max_tokens,
                use_pca_connectivity=model_config.use_pca_connectivity,
            )
            
            # 运行分析
            start_time = time.time()
            self.capability_analyzer.run_analysis()
            end_time = time.time()
            
            logger.info(f"Capability级别分析完成，耗时: {end_time - start_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"Capability级别分析失败: {e}")
            return False
    
    def run_llm_report_generation(self) -> bool:
        """运行LLM报告生成"""
        try:
            logger.info("=" * 60)
            logger.info("开始LLM报告生成")
            logger.info("=" * 60)
            
            data_paths = self.config['data_paths']
            llm_config = self.config['llm_config']
            
            # 检查输入文件
            parcel_diff_file = os.path.join(data_paths.parcel_level_output, "top_anomalous_parcels.json")
            cap_diff_file = os.path.join(data_paths.capability_level_output, "top_anomalous_capabilities.json")
            
            if not os.path.exists(parcel_diff_file):
                logger.error(f"Parcel分析结果文件不存在: {parcel_diff_file}")
                return False
            
            if not os.path.exists(cap_diff_file):
                logger.error(f"Capability分析结果文件不存在: {cap_diff_file}")
                return False
            
            # 创建报告生成器
            self.llm_generator = LLMReportGenerator(
                parcel_desc_path=data_paths.parcel_descriptions,
                cap_desc_path=data_paths.capability_descriptions,
                parcel_diff_path=parcel_diff_file,
                cap_diff_path=cap_diff_file,
                output_path=os.path.join(data_paths.llm_analysis_output, "nature_style_report.md"),
                vllm_url=llm_config.vllm_url,
                api_key=llm_config.api_key
            )
            
            # 运行报告生成
            start_time = time.time()
            self.llm_generator.run_analysis()
            end_time = time.time()
            
            logger.info(f"LLM报告生成完成，耗时: {end_time - start_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"LLM报告生成失败: {e}")
            return False
    
    def run_full_analysis(self) -> bool:
        """运行完整分析流程"""
        try:
            logger.info("=" * 80)
            logger.info("开始完整幻觉机制分析流程")
            logger.info("=" * 80)
            
            experiment_config = self.config['experiment_config']
            start_time = time.time()
            
            success_count = 0
            total_steps = 0
            
            # 1. Parcel级别分析
            if experiment_config.analyze_parcel_level:
                total_steps += 1
                if self.run_parcel_analysis():
                    success_count += 1
                else:
                    logger.error("Parcel级别分析失败，停止后续分析")
                    return False
            
            # 2. Capability级别分析
            if experiment_config.analyze_capability_level:
                total_steps += 1
                if self.run_capability_analysis():
                    success_count += 1
                else:
                    logger.error("Capability级别分析失败，停止后续分析")
                    return False
            
            # 3. LLM报告生成
            # if experiment_config.generate_llm_report:
            #     total_steps += 1
            #     if self.run_llm_report_generation():
            #         success_count += 1
            #     else:
            #         logger.warning("LLM报告生成失败，但分析结果已保存")
            
            end_time = time.time()
            
            # 输出总结
            logger.info("=" * 80)
            logger.info("分析流程完成")
            logger.info(f"成功步骤: {success_count}/{total_steps}")
            logger.info(f"总耗时: {end_time - start_time:.2f}秒")
            logger.info("=" * 80)
            
            # 保存分析总结
            #self._save_analysis_summary(success_count, total_steps, end_time - start_time)
            
            return success_count == total_steps
            
        except Exception as e:
            logger.error(f"完整分析流程失败: {e}")
            return False
    
    def _save_analysis_summary(self, success_count: int, total_steps: int, total_time: float) -> None:
        """保存分析总结"""
        try:
            summary = {
                'experiment_name': self.config['experiment_config'].experiment_name,
                'experiment_version': self.config['experiment_config'].experiment_version,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success_count': success_count,
                'total_steps': total_steps,
                'total_time_seconds': total_time,
                'success_rate': success_count / total_steps if total_steps > 0 else 0,
                'config': {
                    'model_name': self.config['model_config'].model_name,
                    'dataset_name': self.config['experiment_config'].dataset_name,
                    'parcel_dim': self.config['model_config'].parcel_dim,
                    'significance_threshold': self.config['model_config'].significance_threshold,
                    'skip_existing': getattr(self.config['model_config'], 'skip_existing', False),
                    'has_parcel_descriptions': os.path.exists(self.config['data_paths'].parcel_descriptions),
                    'has_capability_descriptions': os.path.exists(self.config['data_paths'].capability_descriptions)
                }
            }
            
            summary_file = os.path.join(
                self.config['data_paths'].llm_analysis_output, 
                'analysis_summary.json'
            )
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析总结已保存到: {summary_file}")
            
        except Exception as e:
            logger.warning(f"保存分析总结失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='幻觉机制分析主程序')
    
    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--experiment_name', type=str, help='实验名称')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    
    # 分析步骤控制
    parser.add_argument('--parcel_only', action='store_true', help='只运行Parcel级别分析')
    parser.add_argument('--capability_only', action='store_true', help='只运行Capability级别分析')
    parser.add_argument('--llm_only', action='store_true', help='只运行LLM报告生成')
    parser.add_argument('--skip_parcel', action='store_true', help='跳过Parcel级别分析')
    parser.add_argument('--skip_capability', action='store_true', help='跳过Capability级别分析')
    parser.add_argument('--skip_llm', action='store_true', help='跳过LLM报告生成')
    
    # 数据路径参数
    parser.add_argument('--correct_jsonl', type=str, help='正确样本激活数据路径')
    parser.add_argument('--incorrect_jsonl', type=str, help='幻觉样本激活数据路径')
    parser.add_argument('--mapping_json', type=str, help='Capability-Parcel映射文件路径')
    parser.add_argument('--parcel_desc', type=str, help='Parcel功能描述文件路径')
    parser.add_argument('--cap_desc', type=str, help='Capability描述文件路径')
    
    # 模型参数
    parser.add_argument('--significance_threshold', type=float, help='统计显著性阈值')
    parser.add_argument('--epsilon', type=float, help='L2归一化小常数')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的结果文件')
    
    # LLM参数
    parser.add_argument('--vllm_url', type=str, help='vLLM API地址')
    parser.add_argument('--api_key', type=str, help='API密钥')
    parser.add_argument('--max_tokens', type=int, help='最大token数')
    
    # 其他参数
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--dry_run', action='store_true', help='干运行（不执行实际分析）')
    parser.add_argument('--use_pca_connectivity', action='store_true', help='使用PCA连接性计算')
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        setup_logging(log_level='DEBUG')
    else:
        setup_logging(log_level='INFO')
    
    try:
        # 加载配置
        if args.config:
            from config import load_config
            config = load_config(args.config)
        else:
            # 使用动态配置，支持命令行参数
            config = create_dynamic_config(
                correct_activations=args.correct_jsonl,
                incorrect_activations=args.incorrect_jsonl,
                parcel_descriptions=args.parcel_desc,
                capability_descriptions=args.cap_desc,
                capability_parcel_mapping=args.mapping_json,
                output_dir=args.output_dir
            )
        
        # 覆盖其他配置参数
        if args.experiment_name:
            config['experiment_config'].experiment_name = args.experiment_name
        
        if args.significance_threshold:
            config['model_config'].significance_threshold = args.significance_threshold
        if args.epsilon:
            config['model_config'].epsilon = args.epsilon
        if args.skip_existing:
            config['model_config'].skip_existing = args.skip_existing
        
        if args.vllm_url:
            config['llm_config'].vllm_url = args.vllm_url
        if args.api_key:
            config['llm_config'].api_key = args.api_key
        if args.max_tokens:
            config['model_config'].max_tokens = args.max_tokens
        if args.use_pca_connectivity:
            config['model_config'].use_pca_connectivity = args.use_pca_connectivity
        # 干运行模式
        if args.dry_run:
            logger.info("干运行模式 - 显示配置但不执行分析")
            logger.info(f"实验名称: {config['experiment_config'].experiment_name}")
            logger.info(f"输出目录: {config['data_paths'].parcel_level_output}")
            logger.info(f"正确样本: {config['data_paths'].correct_activations}")
            logger.info(f"幻觉样本: {config['data_paths'].incorrect_activations}")
            logger.info(f"Parcel描述: {config['data_paths'].parcel_descriptions}")
            logger.info(f"Capability描述: {config['data_paths'].capability_descriptions}")
            logger.info(f"跳过已存在文件: {getattr(config['model_config'], 'skip_existing', False)}")
            return 0
        
        # 创建分析流水线
        pipeline = HallucinationAnalysisPipeline(config=config)
        
        # 根据参数决定运行哪些步骤
        if args.parcel_only:
            success = pipeline.run_parcel_analysis()
        elif args.capability_only:
            success = pipeline.run_capability_analysis()
        elif args.llm_only:
            success = pipeline.run_llm_report_generation()
        else:
            # 修改配置以跳过某些步骤
            if args.skip_parcel:
                config['experiment_config'].analyze_parcel_level = False
            if args.skip_capability:
                config['experiment_config'].analyze_capability_level = False
            if args.skip_llm:
                config['experiment_config'].generate_llm_report = False
            
            success = pipeline.run_full_analysis()
        
        if success:
            logger.info("分析完成！")
            return 0
        else:
            logger.error("分析失败！")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
