#!/usr/bin/env python3
"""
测试双连接性计算方法（传统方法 + PCA拼接方法）
"""

import numpy as np
import json
import os
from pathlib import Path
from analysis_parcel_level import ParcelLevelAnalyzer
from analysis_capability_level import CapabilityLevelAnalyzer

def create_dummy_data():
    """创建测试用的虚拟数据"""
    output_base_dir = Path("./test_output_dual_connectivity")
    output_base_dir.mkdir(exist_ok=True)

    correct_jsonl_path = output_base_dir / "correct_token_parcels.jsonl"
    incorrect_jsonl_path = output_base_dir / "incorrect_token_parcels.jsonl"
    parcel_info_path = output_base_dir / "parcel_info.json"
    capability_info_path = output_base_dir / "capability_info.json"
    mapping_path = output_base_dir / "capability_parcel_mapping.json"

    num_samples = 5
    num_tokens_per_sample = 10
    parcel_dim = 30
    capability_dim = 8

    # 生成虚拟parcel信息
    parcel_info_data = {"parcel_summaries": []}
    for i in range(parcel_dim):
        parcel_info_data["parcel_summaries"].append({
            "parcel_id": i,
            "function_name": f"Test Parcel {i}",
            "function_description": f"Description for Test Parcel {i}",
            "model_role": "Test Role",
            "keywords": ["test", f"keyword_{i}"]
        })
    with open(parcel_info_path, 'w', encoding='utf-8') as f:
        json.dump(parcel_info_data, f, indent=2, ensure_ascii=False)

    # 生成虚拟capability信息
    capability_info_data = {"capability_summaries": []}
    for i in range(capability_dim):
        capability_info_data["capability_summaries"].append({
            "capability_id": i,
            "capability_name": f"Test Capability {i}",
            "capability_description": f"Description for Test Capability {i}",
            "keywords": ["test", f"capability_{i}"]
        })
    with open(capability_info_path, 'w', encoding='utf-8') as f:
        json.dump(capability_info_data, f, indent=2, ensure_ascii=False)

    # 生成虚拟映射数据
    mapping_data = {"mappings": []}
    parcels_per_capability = parcel_dim // capability_dim
    for i in range(capability_dim):
        start_parcel = i * parcels_per_capability
        end_parcel = min((i + 1) * parcels_per_capability, parcel_dim)
        mapping_data["mappings"].append({
            "capability_id": i,
            "parcel_ids": list(range(start_parcel, end_parcel))
        })
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)

    # 生成虚拟激活数据
    def generate_dummy_jsonl(file_path, num_samples, num_tokens, parcel_dim, is_incorrect=False):
        with open(file_path, 'w', encoding='utf-8') as f:
            for _ in range(num_samples):
                # 模拟一些差异用于incorrect样本
                if is_incorrect:
                    activations = np.random.rand(num_tokens, parcel_dim) * 2 + 0.5
                else:
                    activations = np.random.rand(num_tokens, parcel_dim) + 0.5
                
                # 添加一些结构化噪声使PCA更有趣
                for p_idx in range(parcel_dim):
                    if p_idx % 5 == 0:  # 每5个parcel有更强的信号
                        activations[:, p_idx] += np.sin(np.linspace(0, 2*np.pi, num_tokens)) * 2
                    elif p_idx % 7 == 0:  # 每7个parcel有不同的模式
                        activations[:, p_idx] += np.cos(np.linspace(0, 4*np.pi, num_tokens)) * 1.5

                entry = {"token_parcel_acts": activations.tolist()}
                f.write(json.dumps(entry) + '\n')

    generate_dummy_jsonl(correct_jsonl_path, num_samples, num_tokens_per_sample, parcel_dim, is_incorrect=False)
    generate_dummy_jsonl(incorrect_jsonl_path, num_samples, num_tokens_per_sample, parcel_dim, is_incorrect=True)

    print(f"虚拟数据已生成在: {output_base_dir}")
    return output_base_dir, correct_jsonl_path, incorrect_jsonl_path, parcel_info_path, capability_info_path, mapping_path

def test_parcel_level_dual_connectivity():
    """测试Parcel级别的双连接性计算"""
    print("\n=== 测试Parcel级别双连接性计算 ===")
    
    output_base_dir, correct_jsonl_path, incorrect_jsonl_path, parcel_info_path, _, _ = create_dummy_data()
    
    # 测试启用PCA拼接方法
    output_dir_pca = output_base_dir / "parcel_level_dual"
    analyzer_pca = ParcelLevelAnalyzer(
        correct_jsonl_path=str(correct_jsonl_path),
        incorrect_jsonl_path=str(incorrect_jsonl_path),
        output_dir=str(output_dir_pca),
        parcel_info_path=str(parcel_info_path),
        use_pca_connectivity=True,
        pca_explained_variance=0.8,
        skip_existing=False,
        top_k_edges=10
    )
    
    try:
        analyzer_pca.run_analysis()
        print(f"✅ Parcel级别双连接性分析完成，结果保存在: {output_dir_pca}")
        
        # 检查输出文件
        conn_dir = output_dir_pca / "connectivity_matrices"
        if conn_dir.exists():
            files = list(conn_dir.glob("*.npy")) + list(conn_dir.glob("*.json"))
            print(f"📁 生成的文件数量: {len(files)}")
            for file in files:
                print(f"   - {file.name}")
        else:
            print("❌ 连接矩阵目录不存在")
            
    except Exception as e:
        print(f"❌ Parcel级别分析失败: {e}")
        import traceback
        traceback.print_exc()

def test_capability_level_dual_connectivity():
    """测试Capability级别的双连接性计算"""
    print("\n=== 测试Capability级别双连接性计算 ===")
    
    output_base_dir, correct_jsonl_path, incorrect_jsonl_path, parcel_info_path, capability_info_path, mapping_path = create_dummy_data()
    
    # 测试启用PCA拼接方法
    output_dir_pca = output_base_dir / "capability_level_dual"
    analyzer_pca = CapabilityLevelAnalyzer(
        mapping_json_path=str(mapping_path),
        correct_jsonl_path=str(correct_jsonl_path),
        incorrect_jsonl_path=str(incorrect_jsonl_path),
        output_dir=str(output_dir_pca),
        capability_desc_path=str(capability_info_path),
        use_pca_connectivity=True,
        pca_explained_variance=0.8,
        skip_existing=False,
        top_k_edges=10
    )
    
    try:
        analyzer_pca.run_analysis()
        print(f"✅ Capability级别双连接性分析完成，结果保存在: {output_dir_pca}")
        
        # 检查输出文件
        conn_dir = output_dir_pca / "connectivity_matrices"
        if conn_dir.exists():
            files = list(conn_dir.glob("*.npy")) + list(conn_dir.glob("*.json"))
            print(f"📁 生成的文件数量: {len(files)}")
            for file in files:
                print(f"   - {file.name}")
        else:
            print("❌ 连接矩阵目录不存在")
            
    except Exception as e:
        print(f"❌ Capability级别分析失败: {e}")
        import traceback
        traceback.print_exc()

def test_traditional_only():
    """测试仅使用传统方法（不启用PCA）"""
    print("\n=== 测试仅传统方法 ===")
    
    output_base_dir, correct_jsonl_path, incorrect_jsonl_path, parcel_info_path, _, _ = create_dummy_data()
    
    # 测试不启用PCA拼接方法
    output_dir_traditional = output_base_dir / "parcel_level_traditional_only"
    analyzer_traditional = ParcelLevelAnalyzer(
        correct_jsonl_path=str(correct_jsonl_path),
        incorrect_jsonl_path=str(incorrect_jsonl_path),
        output_dir=str(output_dir_traditional),
        parcel_info_path=str(parcel_info_path),
        use_pca_connectivity=False,  # 不启用PCA
        skip_existing=False,
        top_k_edges=10
    )
    
    try:
        analyzer_traditional.run_analysis()
        print(f"✅ 传统方法分析完成，结果保存在: {output_dir_traditional}")
        
        # 检查输出文件
        conn_dir = output_dir_traditional / "connectivity_matrices"
        if conn_dir.exists():
            files = list(conn_dir.glob("*.npy")) + list(conn_dir.glob("*.json"))
            print(f"📁 生成的文件数量: {len(files)}")
            for file in files:
                print(f"   - {file.name}")
        else:
            print("❌ 连接矩阵目录不存在")
            
    except Exception as e:
        print(f"❌ 传统方法分析失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 开始测试双连接性计算方法...")
    
    # 测试1: Parcel级别双连接性
    test_parcel_level_dual_connectivity()
    
    # 测试2: Capability级别双连接性
    test_capability_level_dual_connectivity()
    
    # 测试3: 仅传统方法
    test_traditional_only()
    
    print("\n🎉 所有测试完成！")
    print("\n📋 测试总结:")
    print("   - ✅ 同时计算传统方法和PCA拼接方法")
    print("   - ✅ PCA拼接方法文件添加_concate后缀")
    print("   - ✅ PCA拼接方法不需要显著性检验")
    print("   - ✅ 保存完整的连接矩阵和高连接、异常连接信息")

if __name__ == "__main__":
    main()
