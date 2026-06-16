#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件

定义幻觉机制分析实验的配置参数，包括路径、模型参数、分析参数等。

作者: Jeryi
日期: 2025
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# 基础路径配置
BASE_DIR = Path("/path/to/project_root")
HALLUCINATION_DIR = BASE_DIR / "safety_explanation" / "hallucination"
CODE_DIR = HALLUCINATION_DIR / "code"
ANALYSIS_DIR = CODE_DIR / "analysis"
RESULTS_DIR = HALLUCINATION_DIR / "results"

# 数据路径配置
@dataclass
class DataPaths:
    """数据路径配置"""
    
    # 激活数据路径
    correct_activations: str = str(RESULTS_DIR / "truthfulqa_gemma-2-2b" / "parcels_token_acts" / "correct" / "token_parcels.jsonl")
    incorrect_activations: str = str(RESULTS_DIR / "truthfulqa_gemma-2-2b" / "parcels_token_acts" / "incorrect" / "token_parcels.jsonl")
    
    # 描述文件路径
    parcel_descriptions: str = str(BASE_DIR / "neural_area" / "divide_area_by_sae_act" / "cluster_output_2b_pt" / "clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270" / "latent_parcel_topsamples_functionality_summary.json")
    capability_descriptions: str = str(BASE_DIR / "capability_analysis" / "data" / "capability_descriptions" / "capability_descriptions_run2.json")
    
    # 映射文件路径
    capability_parcel_mapping: str = str(BASE_DIR / "neural_area" / "connect_cap_parcel" / "results" / "aggrate_final" / "final_capability_parcel_all.json")
    
    # 输出路径（这些路径将在运行时被动态设置）
    parcel_level_output: str = ""
    capability_level_output: str = ""
    llm_analysis_output: str = ""
    
    def __post_init__(self):
        """验证路径是否存在（只验证非空的路径）"""
        for path_name, path_value in self.__dict__.items():
            if path_value and not os.path.exists(path_value):
                print(f"警告: {path_name} 路径不存在: {path_value}")
    
    def update_paths(self, **kwargs):
        """更新路径配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def set_output_paths(self, base_output_dir: str, model_data: str = ""):
        """设置输出路径"""
        if model_data:
            # 如果有模型数据，使用模型特定的输出目录
            self.parcel_level_output = str(Path(base_output_dir) / model_data / "parcel_level")
            self.capability_level_output = str(Path(base_output_dir) / model_data / "capability_level")
            self.llm_analysis_output = str(Path(base_output_dir) / model_data)
        else:
            # 否则使用默认输出目录
            self.parcel_level_output = str(Path(base_output_dir) / "parcel_level")
            self.capability_level_output = str(Path(base_output_dir) / "capability_level")
            self.llm_analysis_output = str(Path(base_output_dir) / "llm_analysis")

# 模型配置
@dataclass
class ModelConfig:
    """模型配置"""
    
    # 模型基本信息
    model_name: str = "gemma-2-2b"
    model_path: str = "/path/to/local_models/gpt-oss-20b"
    
    # 激活维度
    parcel_dim: int = 270
    max_sequence_length: int = 2048
    
    # 数据处理参数
    epsilon: float = 1e-8
    normalization_method: str = "l2"  # l2, zscore, minmax
    
    # 统计参数
    significance_threshold: float = 0.05
    multiple_comparison_correction: str = "fdr_bh"  # bonferroni, fdr_bh, none
    
    # 连接性分析参数
    connectivity_method: str = "cosine"  # cosine, pearson, spearman
    connectivity_threshold: float = 0.3
    
    # 异常检测参数
    anomaly_detection_method: str = "ttest"  # ttest, mannwhitney, zscore
    effect_size_threshold: float = 0.2  # Cohen's d阈值

    max_tokens: int = None
    use_pca_connectivity: bool = False


# LLM配置
@dataclass
class LLMConfig:
    """LLM配置"""
    
    # API配置
    vllm_url: str = "http://0.0.0.0:8001/v1"
    api_key: str = "abcabc"
    
    # 生成参数
    max_tokens: int = 2048
    temperature: float = 0.1
    timeout: int = 120
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 报告配置
    report_language: str = "zh"  # zh, en
    report_style: str = "nature"  # nature, science, arxiv
    max_report_length: int = 3000

# 分析配置
@dataclass
class AnalysisConfig:
    """分析配置"""
    
    # 并行处理
    n_jobs: int = -1  # -1表示使用所有CPU核心
    batch_size: int = 100
    
    # 内存管理
    memory_limit_gb: float = 16.0
    use_memory_mapping: bool = False
    
    # 可视化配置
    figure_dpi: int = 300
    figure_format: str = "png"  # png, pdf, svg
    colormap: str = "viridis"
    
    # 输出配置
    save_intermediate_results: bool = True
    compress_output: bool = False
    output_precision: int = 4

# 实验配置
@dataclass
class ExperimentConfig:
    """实验配置"""
    
    # 实验基本信息
    experiment_name: str = "hallucination_mechanism_analysis"
    experiment_version: str = "v1.0"
    experiment_description: str = "基于Parcel和Capability激活的幻觉机制分析"
    
    # 数据配置
    dataset_name: str = "TruthfulQA"
    model_version: str = "gemma-2-2b"
    
    # 分析范围
    analyze_parcel_level: bool = True
    analyze_capability_level: bool = True
    generate_llm_report: bool = True
    
    # 质量控制
    min_samples_per_group: int = 10
    max_missing_ratio: float = 0.1
    outlier_detection: bool = True
    outlier_threshold: float = 3.0  # 3倍标准差

# 路径验证函数
def validate_paths(data_paths: DataPaths) -> Dict[str, bool]:
    """
    验证所有路径是否存在
    
    Args:
        data_paths: 数据路径配置
        
    Returns:
        路径验证结果字典
    """
    validation_results = {}
    
    for path_name, path_value in data_paths.__dict__.items():
        exists = os.path.exists(path_value)
        validation_results[path_name] = exists
        
        if not exists:
            print(f"警告: {path_name} 路径不存在: {path_value}")
    
    return validation_results

# 创建默认配置实例
def get_default_config() -> Dict:
    """获取默认配置"""
    return {
        'data_paths': DataPaths(),
        'model_config': ModelConfig(),
        'llm_config': LLMConfig(),
        'analysis_config': AnalysisConfig(),
        'experiment_config': ExperimentConfig()
    }

# 创建动态配置实例（支持外部路径参数）
def create_dynamic_config(
    correct_activations: str = None,
    incorrect_activations: str = None,
    parcel_descriptions: str = None,
    capability_descriptions: str = None,
    capability_parcel_mapping: str = None,
    output_dir: str = None,
    model_data: str = None
) -> Dict:
    """
    创建动态配置，支持外部传入的路径参数
    
    Args:
        correct_activations: 正确样本激活数据路径
        incorrect_activations: 错误样本激活数据路径
        parcel_descriptions: Parcel描述文件路径
        capability_descriptions: Capability描述文件路径
        capability_parcel_mapping: Capability-Parcel映射文件路径
        output_dir: 输出目录
        model_data: 模型数据名称（用于创建模型特定的输出路径）
    
    Returns:
        配置字典
    """
    # 创建数据路径配置
    data_paths = DataPaths()
    
    # 更新传入的路径
    if correct_activations:
        data_paths.correct_activations = correct_activations
    if incorrect_activations:
        data_paths.incorrect_activations = incorrect_activations
    if parcel_descriptions:
        data_paths.parcel_descriptions = parcel_descriptions
    if capability_descriptions:
        data_paths.capability_descriptions = capability_descriptions
    if capability_parcel_mapping:
        data_paths.capability_parcel_mapping = capability_parcel_mapping
    
    # 设置输出路径
    if output_dir:
        data_paths.set_output_paths(output_dir, model_data)
    
    return {
        'data_paths': data_paths,
        'model_config': ModelConfig(),
        'llm_config': LLMConfig(),
        'analysis_config': AnalysisConfig(),
        'experiment_config': ExperimentConfig()
    }

# 环境变量配置
def load_env_config() -> Dict:
    """从环境变量加载配置"""
    config = get_default_config()
    
    # 从环境变量覆盖配置
    if 'VLLM_URL' in os.environ:
        config['llm_config'].vllm_url = os.environ['VLLM_URL']
    
    if 'API_KEY' in os.environ:
        config['llm_config'].api_key = os.environ['API_KEY']
    
    if 'MODEL_PATH' in os.environ:
        config['model_config'].model_path = os.environ['MODEL_PATH']
    
    if 'N_JOBS' in os.environ:
        config['analysis_config'].n_jobs = int(os.environ['N_JOBS'])
    
    return config

# 配置验证函数
def validate_config(config: Dict) -> List[str]:
    """
    验证配置的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        验证错误列表
    """
    errors = []
    
    # 验证模型配置
    model_config = config.get('model_config', ModelConfig())
    if model_config.parcel_dim <= 0:
        errors.append("parcel_dim必须大于0")
    
    if model_config.significance_threshold <= 0 or model_config.significance_threshold >= 1:
        errors.append("significance_threshold必须在(0,1)范围内")
    
    if model_config.epsilon <= 0:
        errors.append("epsilon必须大于0")
    
    # 验证LLM配置
    llm_config = config.get('llm_config', LLMConfig())
    if llm_config.max_tokens <= 0:
        errors.append("max_tokens必须大于0")
    
    if llm_config.temperature < 0 or llm_config.temperature > 2:
        errors.append("temperature必须在[0,2]范围内")
    
    # 验证分析配置
    analysis_config = config.get('analysis_config', AnalysisConfig())
    if analysis_config.batch_size <= 0:
        errors.append("batch_size必须大于0")
    
    if analysis_config.memory_limit_gb <= 0:
        errors.append("memory_limit_gb必须大于0")
    
    return errors

# 配置保存和加载函数
def save_config(config: Dict, file_path: str) -> None:
    """保存配置到文件"""
    import json
    from pathlib import Path
    
    # 确保目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 转换dataclass为字典
    config_dict = {}
    for key, value in config.items():
        if hasattr(value, '__dict__'):
            config_dict[key] = value.__dict__
        else:
            config_dict[key] = value
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def load_config(file_path: str) -> Dict:
    """从文件加载配置"""
    import json
    
    with open(file_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # 重建dataclass对象
    config = {}
    for key, value in config_dict.items():
        if key == 'data_paths':
            config[key] = DataPaths(**value)
        elif key == 'model_config':
            config[key] = ModelConfig(**value)
        elif key == 'llm_config':
            config[key] = LLMConfig(**value)
        elif key == 'analysis_config':
            config[key] = AnalysisConfig(**value)
        elif key == 'experiment_config':
            config[key] = ExperimentConfig(**value)
        else:
            config[key] = value
    
    return config

# 默认配置实例
DEFAULT_DATA_PATHS = DataPaths()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LLM_CONFIG = LLMConfig()
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# 快速配置函数
def get_quick_config(experiment_name: str = "hallucination_analysis") -> Dict:
    """获取快速配置"""
    config = get_default_config()
    config['experiment_config'].experiment_name = experiment_name
    return config

def get_production_config() -> Dict:
    """获取生产环境配置"""
    config = get_default_config()
    
    # 生产环境优化
    config['analysis_config'].n_jobs = 4  # 限制并行度
    config['analysis_config'].memory_limit_gb = 32.0  # 增加内存限制
    config['llm_config'].max_retries = 5  # 增加重试次数
    config['llm_config'].timeout = 300  # 增加超时时间
    
    return config

def get_development_config() -> Dict:
    """获取开发环境配置"""
    config = get_default_config()
    
    # 开发环境优化
    config['analysis_config'].n_jobs = 1  # 单线程便于调试
    config['analysis_config'].save_intermediate_results = True
    config['llm_config'].max_tokens = 1024  # 减少token数量
    config['llm_config'].temperature = 0.0  # 确定性输出
    
    return config
