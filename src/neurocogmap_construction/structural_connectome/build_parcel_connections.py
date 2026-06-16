#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel-Parcel结构连接构建主脚本
"""

import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path
from typing import Optional
import yaml

try:
    from neurocogmap_release.paths import artifact_path, env_path_str, output_path, release_root
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from neurocogmap_release.paths import artifact_path, env_path_str, output_path, release_root

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from connection_calculator import ParcelConnectionCalculator
from visualization import ParcelConnectionVisualizer

# 设置日志
_log_path = output_path("neurocogmap_construction", "structural_connectome", "build_connections.log")
_log_path.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_log_path))
    ]
)
logger = logging.getLogger(__name__)

MODEL_ATLAS_BY_KEY = {
    "gemma2_2b": ("gemma2_2b", "NEUROCOGMAP_GEMMA2_SAE_DIR"),
    "llama3_1_8b": ("llama3_1_8b", "NEUROCOGMAP_LLAMA3_8B_SAE_DIR"),
    "gemma2_9b_it": ("gemma2_9b_it", "NEUROCOGMAP_GEMMA2_9B_SAE_DIR"),
}


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="构建Parcel-Parcel结构连接",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 完整计算所有连接
  python build_parcel_connections.py --config configs/paths.yaml
  
  # 测试模式（只计算少量连接）
  python build_parcel_connections.py --config configs/paths.yaml --test --max_connections 100
  
  # 跳过已存在的文件
  python build_parcel_connections.py --config configs/paths.yaml --skip_existing
  
  # 不跳过已存在的文件（重新计算）
  python build_parcel_connections.py --config configs/paths.yaml --no_skip_existing
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=os.path.join(current_dir, 'configs', 'paths.yaml'),
        help='配置文件路径 (默认: configs/paths.yaml)'
    )
    parser.add_argument(
        '--model_key',
        choices=sorted(MODEL_ATLAS_BY_KEY.keys()),
        default=None,
        help='release atlas 模型键；用于在配置未指定 parcel_assignments_path 时选择默认 atlas'
    )
    parser.add_argument(
        '--sae_weights_base_dir',
        type=str,
        default=None,
        help='本地 SAE 权重目录；也可通过 NEUROCOGMAP_GEMMA2_SAE_DIR 等环境变量设置'
    )
    parser.add_argument(
        '--parcel_assignments_path',
        type=str,
        default=None,
        help='latent_parcel_assignments.json 路径；默认使用 release atlas 中对应模型的文件'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='测试模式，只计算少量连接'
    )
    
    parser.add_argument(
        '--max_connections', 
        type=int, 
        default=None,
        help='最大连接数限制（用于测试）'
    )
    
    parser.add_argument(
        '--skip_existing', 
        action='store_true',
        help='跳过已存在的结果文件'
    )
    
    parser.add_argument(
        '--no_skip_existing', 
        action='store_true',
        help='不跳过已存在的结果文件（重新计算）'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='生成可视化图表'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='输出目录（覆盖配置文件中的设置）'
    )
    
    return parser.parse_args()


def _resolve_config_path(raw_value: str, config_dir: Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(str(raw_value)))
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    if expanded.startswith(("artifacts/", "data/")):
        return str((release_root() / expanded).resolve())
    return str((config_dir / expanded).resolve())


def _infer_model_key(config: dict, config_path: Path, arg_model_key: Optional[str]) -> str:
    if arg_model_key:
        return arg_model_key
    configured = config.get("model_key")
    if configured:
        return str(configured)
    if "8b" in config_path.stem:
        return "llama3_1_8b"
    if "9b" in config_path.stem:
        return "gemma2_9b_it"
    return "gemma2_2b"


def build_effective_config(args: argparse.Namespace) -> str:
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    model_key = _infer_model_key(config, config_path, args.model_key)
    atlas_dir, sae_env_name = MODEL_ATLAS_BY_KEY[model_key]
    config["model_key"] = model_key

    if args.sae_weights_base_dir:
        config["sae_weights_base_dir"] = str(Path(args.sae_weights_base_dir).expanduser().resolve())
    else:
        configured_sae = str(config.get("sae_weights_base_dir") or "")
        env_sae = env_path_str(sae_env_name)
        legacy_env_sae = env_path_str("NEUROCOGMAP_GEMMA2_SAE_DIR") if model_key == "gemma2_2b" else ""
        if env_sae:
            config["sae_weights_base_dir"] = env_sae
        elif legacy_env_sae:
            config["sae_weights_base_dir"] = legacy_env_sae
        elif configured_sae and "/new_disk" not in configured_sae:
            config["sae_weights_base_dir"] = _resolve_config_path(configured_sae, config_path.parent)
        else:
            config["sae_weights_base_dir"] = ""

    if args.parcel_assignments_path:
        config["parcel_assignments_path"] = str(Path(args.parcel_assignments_path).expanduser().resolve())
    else:
        configured_assignments = str(config.get("parcel_assignments_path") or "")
        if configured_assignments and "/new_disk" not in configured_assignments:
            config["parcel_assignments_path"] = _resolve_config_path(configured_assignments, config_path.parent)
        else:
            config["parcel_assignments_path"] = str(
                artifact_path("neurocogmap_atlas", atlas_dir, "latent_parcel_assignments.json")
            )

    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
    else:
        configured_output = str(config.get("output_dir") or "")
        if configured_output and "/new_disk" not in configured_output:
            config["output_dir"] = _resolve_config_path(configured_output, config_path.parent)
        else:
            config["output_dir"] = str(output_path("neurocogmap_construction", "structural_connectome", model_key))

    effective_dir = output_path("neurocogmap_construction", "structural_connectome", "effective_configs")
    effective_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{config_path.stem}_", suffix=".yaml", dir=str(effective_dir))
    os.close(fd)
    effective_path = Path(tmp_name)
    with effective_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    logger.info(f"使用 effective config: {effective_path}")
    return str(effective_path)


def validate_config(config_path: str) -> bool:
    """验证配置文件"""
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查必要的配置项
        required_keys = [
            'sae_weights_base_dir', 
            'parcel_assignments_path', 
            'output_dir',
            'sae_layers',
            'latent_per_layer',
            'n_parcels'
        ]
        
        for key in required_keys:
            if key not in config:
                logger.error(f"❌ 配置文件缺少必要项: {key}")
                return False
        
        # 检查文件路径是否存在
        if not os.path.exists(config['sae_weights_base_dir']):
            logger.error(f"❌ SAE权重目录不存在: {config['sae_weights_base_dir']}")
            return False
        
        if not os.path.exists(config['parcel_assignments_path']):
            logger.error(f"❌ Parcel分配文件不存在: {config['parcel_assignments_path']}")
            return False
        
        logger.info("✅ 配置文件验证通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 配置文件验证失败: {e}")
        return False


def main():
    """主函数"""
    args = parse_arguments()
    effective_config = build_effective_config(args)
    
    # 验证配置文件
    if not validate_config(effective_config):
        sys.exit(1)
    
    # 确定是否跳过已存在的文件
    skip_existing = args.skip_existing
    if args.no_skip_existing:
        skip_existing = False
    
    # 确定最大连接数
    max_connections = args.max_connections
    if args.test and max_connections is None:
        max_connections = 100
        logger.info(f"🧪 测试模式：限制最大连接数为 {max_connections}")
    
    try:
        logger.info("🚀 开始构建Parcel-Parcel结构连接...")
        
        # 创建连接计算器
        calculator = ParcelConnectionCalculator(
            config_path=effective_config,
            skip_existing=skip_existing
        )
        
        # 运行完整计算
        connections, matrix = calculator.run_full_calculation(max_connections=max_connections)
        
        logger.info(f"✅ 连接计算完成！")
        logger.info(f"   计算了 {len(connections)} 个连接")
        logger.info(f"   矩阵形状: {matrix.shape}")
        
        # 生成可视化图表
        if args.visualize or args.test:
            logger.info("🎨 开始生成可视化图表...")
            
            visualizer = ParcelConnectionVisualizer(
                config_path=effective_config,
                output_dir=args.output_dir
            )
            
            generated_files = visualizer.create_all_visualizations(connections, matrix)
            
            logger.info(f"✅ 可视化完成！生成了 {len(generated_files)} 个文件:")
            for file_path in generated_files:
                logger.info(f"   - {file_path}")
        
        # 显示结果摘要
        if len(connections) > 0:
            strengths = [conn['connection_strength'] for conn in connections]
            logger.info("\n📊 结果摘要:")
            logger.info(f"   连接强度 - 均值: {sum(strengths)/len(strengths):.6f}")
            logger.info(f"   连接强度 - 范围: [{min(strengths):.6f}, {max(strengths):.6f}]")
            logger.info(f"   非零连接数: {sum(1 for s in strengths if s != 0)}")
        
        logger.info("🎉 所有任务完成！")
        
    except KeyboardInterrupt:
        logger.info("⚠️ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
