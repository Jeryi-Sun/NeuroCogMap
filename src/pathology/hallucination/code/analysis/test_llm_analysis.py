#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM分析脚本测试脚本
"""

import json
import sys
from pathlib import Path

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载功能...")
    
    # 测试文件路径
    base_dir = Path("/path/to/project_root")
    results_dir = base_dir / "safety_explanation/hallucination/results/analysis_output/truthfulqa_gemma-2-2b"
    
    # 测试文件
    test_files = [
        results_dir / "parcel_level/top_activated_parcels.json",
        results_dir / "parcel_level/top_parcel_connections.json",
        results_dir / "parcel_level/anomalous_connections.json",
        results_dir / "capability_level/top_activated_capabilities.json",
        results_dir / "capability_level/top_capability_connections.json",
        results_dir / "capability_level/anomalous_capability_connections.json",
        base_dir / "capability_analysis/data/capability_cog_mapping.json"
    ]
    
    for file_path in test_files:
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✓ {file_path.name}: 加载成功")
                
                # 检查数据结构
                if 'top_correct_parcels' in data:
                    print(f"  - 正确样本Parcels: {len(data['top_correct_parcels'])}")
                if 'top_incorrect_parcels' in data:
                    print(f"  - 幻觉样本Parcels: {len(data['top_incorrect_parcels'])}")
                if 'top_correct_capabilities' in data:
                    print(f"  - 正确样本Capabilities: {len(data['top_correct_capabilities'])}")
                if 'top_incorrect_capabilities' in data:
                    print(f"  - 幻觉样本Capabilities: {len(data['top_incorrect_capabilities'])}")
                if 'anomalous_connections' in data:
                    print(f"  - 异常连接: {len(data['anomalous_connections'])}")
                    
            except Exception as e:
                print(f"✗ {file_path.name}: 加载失败 - {e}")
        else:
            print(f"✗ {file_path.name}: 文件不存在")
    
    print("\n数据加载测试完成！")

def test_cognitive_mapping():
    """测试认知层级映射功能"""
    print("\n测试认知层级映射功能...")
    
    cog_mapping_path = Path("/path/to/project_root/capability_analysis/data/capability_cog_mapping.json")
    
    if not cog_mapping_path.exists():
        print("✗ 认知映射文件不存在")
        return
    
    try:
        with open(cog_mapping_path, 'r', encoding='utf-8') as f:
            cog_mapping = json.load(f)
        
        print("✓ 认知映射文件加载成功")
        
        # 测试认知层级分类
        if 'capability_classification_system' in cog_mapping:
            levels = set()
            categories = set()
            
            for cap_name, cap_info in cog_mapping['capability_classification_system'].items():
                levels.add(cap_info.get('level', 'Unknown'))
                categories.add(cap_info.get('name', 'Unknown'))
            
            print(f"  - 认知层级数量: {len(levels)}")
            print(f"  - 认知类别数量: {len(categories)}")
            print(f"  - 认知层级: {list(levels)}")
            
            # 测试几个具体的Capability
            test_capabilities = [
                "Trustworthiness capability",
                "Verification capability", 
                "Counterfactual Reasoning Capability"
            ]
            
            for cap in test_capabilities:
                # 模拟get_cognitive_level函数
                capability_lower = cap.lower()
                found = False
                
                # 在capability_mappings中查找
                for category, capabilities in cog_mapping.get('capability_mappings', {}).items():
                    for cap_name, cap_info in capabilities.items():
                        if cap_name.lower() == capability_lower or cap_name.lower() in capability_lower:
                            print(f"  - {cap}: {cap_info.get('level', 'Unknown')} - {cap_info.get('category_name', 'Unknown')}")
                            found = True
                            break
                    if found:
                        break
                
                if not found:
                    print(f"  - {cap}: 未找到匹配")
        
    except Exception as e:
        print(f"✗ 认知映射测试失败: {e}")

def main():
    """主函数"""
    print("LLM分析脚本功能测试")
    print("=" * 50)
    
    test_data_loading()
    test_cognitive_mapping()
    
    print("\n测试完成！")
    print("\n如果所有测试都通过，可以运行以下命令生成LLM分析报告：")
    print("./run_llm_analysis_example.sh")

if __name__ == "__main__":
    main()
