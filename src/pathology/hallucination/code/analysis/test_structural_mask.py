#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结构性连接mask功能

作者: Assistant
日期: 2025
"""

import sys
import os
import numpy as np
import logging

# 添加当前目录到Python路径
sys.path.append('.')

from structural_mask_utils import StructuralMaskProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_structural_mask():
    """测试结构性连接mask功能"""
    logger.info("开始测试结构性连接mask功能...")
    
    # 测试参数
    structural_matrix_path = "/path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv"
    parcel_dim = 270
    
    try:
        # 1. 测试加载结构性连接矩阵
        logger.info("1. 测试加载结构性连接矩阵...")
        processor = StructuralMaskProcessor(structural_matrix_path, parcel_dim)
        structural_matrix = processor.load_structural_matrix()
        
        logger.info(f"结构性连接矩阵形状: {structural_matrix.shape}")
        logger.info(f"矩阵统计信息:")
        logger.info(f"  最小值: {np.min(structural_matrix):.6f}")
        logger.info(f"  最大值: {np.max(structural_matrix):.6f}")
        logger.info(f"  平均值: {np.mean(structural_matrix):.6f}")
        logger.info(f"  非零元素: {np.sum(structural_matrix != 0)}")
        
        # 2. 测试创建mask矩阵
        logger.info("2. 测试创建mask矩阵...")
        mask_matrix = processor.create_mask_matrix(threshold=0.0)
        
        logger.info(f"Mask矩阵形状: {mask_matrix.shape}")
        logger.info(f"Mask统计信息:")
        logger.info(f"  非零元素: {np.sum(mask_matrix)}")
        logger.info(f"  密度: {np.sum(mask_matrix) / mask_matrix.size:.4f}")
        logger.info(f"  对角线元素: {np.sum(np.diag(mask_matrix))}")
        
        # 3. 测试mask应用
        logger.info("3. 测试mask应用...")
        
        # 创建测试功能连接矩阵
        test_connectivity = np.random.randn(parcel_dim, parcel_dim)
        test_connectivity = (test_connectivity + test_connectivity.T) / 2  # 对称化
        
        logger.info(f"原始功能连接矩阵统计:")
        logger.info(f"  非零元素: {np.sum(test_connectivity != 0)}")
        logger.info(f"  平均值: {np.mean(test_connectivity):.6f}")
        
        # 应用二进制mask
        masked_connectivity_binary = processor.apply_mask(
            test_connectivity, 
            threshold=0.0, 
            mask_type='binary'
        )
        
        logger.info(f"二进制mask后功能连接矩阵统计:")
        logger.info(f"  非零元素: {np.sum(masked_connectivity_binary != 0)}")
        logger.info(f"  平均值: {np.mean(masked_connectivity_binary):.6f}")
        
        # 应用加权mask
        masked_connectivity_weighted = processor.apply_mask(
            test_connectivity, 
            threshold=0.0, 
            mask_type='weighted'
        )
        
        logger.info(f"加权mask后功能连接矩阵统计:")
        logger.info(f"  非零元素: {np.sum(masked_connectivity_weighted != 0)}")
        logger.info(f"  平均值: {np.mean(masked_connectivity_weighted):.6f}")
        
        # 4. 测试不同阈值
        logger.info("4. 测试不同阈值...")
        thresholds = [0.0, 0.01, 0.05, 0.1]
        
        for threshold in thresholds:
            mask = processor.create_mask_matrix(threshold=threshold)
            density = np.sum(mask) / mask.size
            logger.info(f"  阈值 {threshold}: 密度 {density:.4f}")
        
        # 5. 获取统计信息
        logger.info("5. 获取mask统计信息...")
        stats = processor.get_mask_statistics()
        
        logger.info("Mask统计信息:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ 结构性连接mask功能测试完成！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parcel_analysis_with_mask():
    """测试Parcel级别分析使用mask功能"""
    logger.info("开始测试Parcel级别分析使用mask功能...")
    
    # 这里可以添加更复杂的测试，比如实际运行一小部分分析
    # 由于需要真实数据，这里只做基本检查
    try:
        from analysis_parcel_level import ParcelLevelAnalyzer
        
        # 检查是否能正确初始化
        analyzer = ParcelLevelAnalyzer(
            correct_jsonl_path="dummy.jsonl",
            incorrect_jsonl_path="dummy.jsonl", 
            output_dir="dummy_output",
            use_structural_mask=True,
            structural_matrix_path="/path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv"
        )
        
        logger.info("✅ Parcel级别分析器初始化成功（使用mask）")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parcel级别分析器初始化失败: {e}")
        return False

def test_capability_analysis_with_mask():
    """测试Capability级别分析使用mask功能"""
    logger.info("开始测试Capability级别分析使用mask功能...")
    
    try:
        from analysis_capability_level import CapabilityLevelAnalyzer
        
        # 检查是否能正确初始化
        analyzer = CapabilityLevelAnalyzer(
            mapping_json_path="dummy.json",
            correct_jsonl_path="dummy.jsonl",
            incorrect_jsonl_path="dummy.jsonl", 
            output_dir="dummy_output",
            use_structural_mask=True,
            structural_matrix_path="/path/to/project_root/neural_area/global_weight/outputs/parcel_connection_matrix.csv"
        )
        
        logger.info("✅ Capability级别分析器初始化成功（使用mask）")
        return True
        
    except Exception as e:
        logger.error(f"❌ Capability级别分析器初始化失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("结构性连接mask功能测试")
    logger.info("=" * 60)
    
    # 运行测试
    test1_passed = test_structural_mask()
    test2_passed = test_parcel_analysis_with_mask()
    test3_passed = test_capability_analysis_with_mask()
    
    logger.info("=" * 60)
    logger.info("测试结果汇总:")
    logger.info(f"  结构性连接mask功能: {'✅ 通过' if test1_passed else '❌ 失败'}")
    logger.info(f"  Parcel级别分析器: {'✅ 通过' if test2_passed else '❌ 失败'}")
    logger.info(f"  Capability级别分析器: {'✅ 通过' if test3_passed else '❌ 失败'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        logger.info("🎉 所有测试通过！")
        sys.exit(0)
    else:
        logger.error("❌ 部分测试失败！")
        sys.exit(1)
