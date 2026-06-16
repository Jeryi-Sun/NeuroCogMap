#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收集每个 story 对应的 Y 值（stimuli * 100 个 Parcel）

从 assembly 文件中加载数据，将每个 story 的 brain_data（顶点级别）映射到 Parcel 级别（102 个 Parcel），
然后去除第 0 个和第 51 个分区（对应没有实际内容的分区），保留 100 个 human parcels，
并保存为 numpy 数组文件。

每个 story 的数据形状为 (n_stimuli, 100)，其中：
- n_stimuli: 该 story 的 stimuli 数量
- 100: 去除第 0 个和第 51 个分区后的 100 个 human parcels
"""

import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path
import logging

# 添加项目根目录到路径
# 从 data_preparation/collect_story_parcel_data.py 到 litcoder_core 需要向上 3 级
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from encoding.assembly.assembly_loader import load_assembly
from encoding.brain_projection.vertix2parcel import VertexToParcelMapper

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_story_parcel_data(
    assembly_path: str,
    output_dir: str,
    lh_annot_path: str,
    rh_annot_path: str,
    skip_existing: bool = True,
):
    """
    收集每个 story 的 Parcel 级别数据
    
    Args:
        assembly_path: assembly pickle 文件路径
        output_dir: 输出目录，每个 story 的数据将保存为 {output_dir}/{story_name}/{story_name}.npy
        lh_annot_path: 左半球 annot 文件路径
        rh_annot_path: 右半球 annot 文件路径
        skip_existing: 如果输出文件已存在，是否跳过
    """
    logger.info(f"加载 assembly 文件: {assembly_path}")
    assembly = load_assembly(assembly_path)
    
    logger.info(f"初始化 VertexToParcelMapper...")
    mapper = VertexToParcelMapper(
        lh_annot_path=lh_annot_path,
        rh_annot_path=rh_annot_path,
        drop_label_names=("???", "unknown"),
        use_nanmean=True,
    )
    
    logger.info(f"Parcel 数量: {len(mapper.parcel_names)}")
    logger.info(f"Story 数量: {len(assembly.stories)}")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 处理每个 story
    all_stories_data = {}
    for story_name in assembly.stories:
        logger.info(f"处理 story: {story_name}")
        
        # 为每个 story 创建单独的文件夹
        story_output_dir = output_path / story_name
        story_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = story_output_dir / f"{story_name}.npy"
        
        # 检查文件是否已存在
        if skip_existing and output_file.exists():
            logger.info(f"文件已存在，跳过: {output_file}")
            try:
                story_parcel_data = np.load(output_file)
                # 如果旧文件是 102 个 Parcel，需要过滤为 100 个
                if story_parcel_data.shape[1] == 102:
                    logger.info(f"  检测到旧格式（102 个 Parcel），正在过滤为 100 个 Parcel")
                    keep_indices = [i for i in range(102) if i != 0 and i != 51]
                    story_parcel_data = story_parcel_data[:, keep_indices]
                    # 保存过滤后的数据
                    np.save(output_file, story_parcel_data)
                    logger.info(f"  已更新为 100 个 Parcel，形状: {story_parcel_data.shape}")
                elif story_parcel_data.shape[1] != 100:
                    logger.warning(f"  文件形状异常: {story_parcel_data.shape}，期望第二维为 100，将重新生成")
                    raise ValueError(f"Unexpected parcel count: {story_parcel_data.shape[1]}")
                all_stories_data[story_name] = story_parcel_data
                logger.info(f"  已加载现有数据，形状: {story_parcel_data.shape}")
                continue
            except Exception as e:
                logger.warning(f"加载现有文件失败: {e}，将重新生成")
        
        # 获取该 story 的 brain_data
        story_data = assembly.story_data[story_name]
        brain_data = story_data.brain_data  # 形状: (n_stimuli, n_vertices)
        
        logger.info(f"  brain_data 形状: {brain_data.shape}")
        
        # 映射到 Parcel 级别
        try:
            parcel_data = mapper.project(brain_data)  # 形状: (n_stimuli, 102)
            logger.info(f"  parcel_data 原始形状: {parcel_data.shape}")
            
            # 去除第 0 个和第 51 个，保留 100 个对应 human parcels
            # 索引映射：原始索引 -> human parcel_id
            # 1-50: 对应 human parcel_id 1-50 (左脑)
            # 52-101: 对应 human parcel_id 51-100 (右脑)
            # 去除第 0 列和第 51 列（索引 0 和 51）
            keep_indices = [i for i in range(102) if i != 0 and i != 51]
            parcel_data = parcel_data[:, keep_indices]  # 形状: (n_stimuli, 100)
            logger.info(f"  parcel_data 过滤后形状: {parcel_data.shape}")
            
            # 保存数据
            np.save(output_file, parcel_data)
            logger.info(f"  已保存到: {output_file}")
            
            all_stories_data[story_name] = parcel_data
            
        except Exception as e:
            logger.error(f"处理 story {story_name} 时出错: {e}", exc_info=True)
            raise
    
    # 保存汇总信息
    # 过滤 parcel_names，去除第 0 个和第 51 个
    keep_indices = [i for i in range(len(mapper.parcel_names)) if i != 0 and i != 51]
    filtered_parcel_names = [mapper.parcel_names[i] for i in keep_indices]
    
    summary = {
        'stories': list(all_stories_data.keys()),
        'shapes': {story: data.shape for story, data in all_stories_data.items()},
        'n_parcels': 100,  # 去除第 0 个和第 51 个后保留 100 个 human parcels
        'parcel_names': filtered_parcel_names,
    }
    
    summary_file = output_path / "summary.json"
    import json
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总信息已保存到: {summary_file}")
    
    logger.info("=" * 60)
    logger.info("所有 story 数据处理完成")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"共处理 {len(all_stories_data)} 个 story")
    for story_name, data in all_stories_data.items():
        logger.info(f"  {story_name}: {data.shape}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="收集每个 story 对应的 Y 值（stimuli * 100 个 Parcel）"
    )
    parser.add_argument(
        "--assembly_path",
        type=str,
        required=True,
        help="assembly pickle 文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出目录，每个 story 的数据将保存为 {output_dir}/{story_name}/{story_name}.npy"
    )
    parser.add_argument(
        "--lh_annot_path",
        type=str,
        default="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/annotation/lh.Schaefer2018_100Parcels_7Networks_order.annot",
        help="左半球 annot 文件路径"
    )
    parser.add_argument(
        "--rh_annot_path",
        type=str,
        default="/path/to/project_root/Human_LLM_align/litcoder_core/dataset/annotation/rh.Schaefer2018_100Parcels_7Networks_order.annot",
        help="右半球 annot 文件路径"
    )
    # 默认行为是跳过已存在的文件（skip_existing=True）
    # 只提供 --no_skip_existing 选项来禁用此行为
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        default=True,
        help="禁用跳过已存在文件的功能，即使文件存在也重新生成（默认会跳过已存在的文件）"
    )
    
    args = parser.parse_args()
    
    collect_story_parcel_data(
        assembly_path=args.assembly_path,
        output_dir=args.output_dir,
        lh_annot_path=args.lh_annot_path,
        rh_annot_path=args.rh_annot_path,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()

