#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PCA连接性计算功能

这个脚本用于测试新添加的PCA连接性计算功能是否正常工作
"""

import numpy as np
import json
import tempfile
import os
from pathlib import Path
from analysis_parcel_level import ParcelLevelAnalyzer
from analysis_capability_level import CapabilityLevelAnalyzer

def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    
    # 创建测试的激活数据
    n_samples = 10
    n_tokens = 20
    n_parcels = 50
    n_capabilities = 20
    
    # 生成正确样本数据
    correct_data = []
    for i in range(n_samples):
        # 生成随机激活数据
        activations = np.random.randn(n_tokens, n_parcels).astype(np.float32)
        correct_data.append({
            'token_parcel_acts': activations.tolist()
        })
    
    # 生成幻觉样本数据（添加一些偏差）
    incorrect_data = []
    for i in range(n_samples):
        # 生成随机激活数据，添加一些偏差
        activations = np.random.randn(n_tokens, n_parcels).astype(np.float32)
        # 在某些Parcel上添加偏差
        activations[:, :10] += 0.5  # 前10个Parcel增加激活
        activations[:, 10:20] -= 0.3  # 中间10个Parcel减少激活
        incorrect_data.append({
            'token_parcel_acts': activations.tolist()
        })
    
    # 保存测试数据
    correct_file = temp_dir / "correct_test.jsonl"
    incorrect_file = temp_dir / "incorrect_test.jsonl"
    
    with open(correct_file, 'w') as f:
        for data in correct_data:
            f.write(json.dumps(data) + '\n')
    
    with open(incorrect_file, 'w') as f:
        for data in incorrect_data:
            f.write(json.dumps(data) + '\n')
    
    # 创建测试的映射数据
    mapping_data = {}
    for i in range(n_capabilities):
        capability_name = f"capability_{i}"
        # 为每个capability分配一些parcels
        start_parcel = i * 2
        end_parcel = min(start_parcel + 5, n_parcels)
        ranking = []
        for j in range(start_parcel, end_parcel):
            ranking.append([f"parcel_{j}", np.random.rand()])
        mapping_data[capability_name] = {"ranking": ranking}
    
    mapping_file = temp_dir / "mapping_test.json"
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f)
    
    return temp_dir, correct_file, incorrect_file, mapping_file

def test_parcel_pca_connectivity():
    """测试Parcel级别的PCA连接性计算"""
    print("\n=== 测试Parcel级别PCA连接性计算 ===")
    
    temp_dir, correct_file, incorrect_file, mapping_file = create_test_data()
    
    try:
        # 创建分析器
        analyzer = ParcelLevelAnalyzer(
            correct_jsonl_path=str(correct_file),
            incorrect_jsonl_path=str(incorrect_file),
            output_dir=str(temp_dir / "parcel_output"),
            use_pca_connectivity=True,
            pca_explained_variance=0.8
        )
        
        # 加载数据
        analyzer.load_activation_data()
        
        # 测试PCA连接性计算
        print("测试PCA连接性计算...")
        pca_connectivity = analyzer.compute_pca_connectivity(analyzer.correct_activations)
        
        print(f"PCA连接矩阵形状: {pca_connectivity.shape}")
        print(f"PCA连接矩阵范围: [{pca_connectivity.min():.4f}, {pca_connectivity.max():.4f}]")
        print(f"PCA连接矩阵均值: {pca_connectivity.mean():.4f}")
        
        # 测试传统方法
        print("\n测试传统连接性计算...")
        traditional_connectivity = analyzer.compute_baseline_connectivity()
        
        print(f"传统连接矩阵形状: {traditional_connectivity.shape}")
        print(f"传统连接矩阵范围: [{traditional_connectivity.min():.4f}, {traditional_connectivity.max():.4f}]")
        print(f"传统连接矩阵均值: {traditional_connectivity.mean():.4f}")
        
        print("✓ Parcel级别PCA连接性计算测试通过")
        
    except Exception as e:
        print(f"✗ Parcel级别PCA连接性计算测试失败: {e}")
        raise
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

def test_capability_pca_connectivity():
    """测试Capability级别的PCA连接性计算"""
    print("\n=== 测试Capability级别PCA连接性计算 ===")
    
    temp_dir, correct_file, incorrect_file, mapping_file = create_test_data()
    
    try:
        # 创建分析器
        analyzer = CapabilityLevelAnalyzer(
            mapping_json_path=str(mapping_file),
            correct_jsonl_path=str(correct_file),
            incorrect_jsonl_path=str(incorrect_file),
            output_dir=str(temp_dir / "capability_output"),
            use_pca_connectivity=True,
            pca_explained_variance=0.8
        )
        
        # 加载数据
        analyzer.load_capability_parcel_mapping()
        analyzer.load_activation_data()
        
        # 构建映射矩阵
        analyzer.mapping_matrix = analyzer.build_mapping_matrix(analyzer.parcel_dim)
        
        # 聚合到Capability级别
        capability_activations_list = []
        for parcel_activations in analyzer.correct_activations:
            capability_activations = analyzer.aggregate_to_capabilities(parcel_activations)
            capability_activations_list.append(capability_activations)
        
        # 测试PCA连接性计算
        print("测试Capability PCA连接性计算...")
        pca_connectivity = analyzer.compute_pca_capability_connectivity(capability_activations_list)
        
        print(f"Capability PCA连接矩阵形状: {pca_connectivity.shape}")
        print(f"Capability PCA连接矩阵范围: [{pca_connectivity.min():.4f}, {pca_connectivity.max():.4f}]")
        print(f"Capability PCA连接矩阵均值: {pca_connectivity.mean():.4f}")
        
        # 测试传统方法
        print("\n测试Capability传统连接性计算...")
        traditional_connectivity = analyzer.compute_baseline_capability_connectivity()
        
        print(f"Capability传统连接矩阵形状: {traditional_connectivity.shape}")
        print(f"Capability传统连接矩阵范围: [{traditional_connectivity.min():.4f}, {traditional_connectivity.max():.4f}]")
        print(f"Capability传统连接矩阵均值: {traditional_connectivity.mean():.4f}")
        
        print("✓ Capability级别PCA连接性计算测试通过")
        
    except Exception as e:
        print(f"✗ Capability级别PCA连接性计算测试失败: {e}")
        raise
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)

def main():
    """主测试函数"""
    print("开始测试PCA连接性计算功能...")
    
    try:
        test_parcel_pca_connectivity()
        test_capability_pca_connectivity()
        print("\n🎉 所有测试通过！PCA连接性计算功能正常工作。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
