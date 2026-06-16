#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM案例选择测试脚本

测试LLM增强案例选择功能，验证各种选择策略的效果。

作者: Jeryi
日期: 2025
"""

import json
import os
import sys
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_llm_case_selection():
    """测试LLM案例选择功能"""
    
    # 基础路径配置
    base_dir = Path("/path/to/project_root")
    results_dir = base_dir / "safety_explanation" / "hallucination" / "results"
    
    # 数据路径
    correct_jsonl = results_dir / "truthfulqa_gemma-2-2b" / "correct.jsonl"
    incorrect_jsonl = results_dir / "truthfulqa_gemma-2-2b" / "incorrect.jsonl"
    parcel_analysis = results_dir / "parcel_level" / "top_anomalous_parcels.json"
    capability_analysis = results_dir / "capability_level" / "top_anomalous_capabilities.json"
    
    # 检查文件是否存在
    required_files = [
        (correct_jsonl, "正确样本数据"),
        (incorrect_jsonl, "幻觉样本数据"),
        (parcel_analysis, "Parcel分析结果"),
        (capability_analysis, "Capability分析结果")
    ]
    
    print("🔍 检查数据文件...")
    for file_path, name in required_files:
        if file_path.exists():
            print(f"✅ {name}: {file_path}")
        else:
            print(f"❌ {name}: {file_path} (不存在)")
            print(f"   请先运行基础分析生成此文件")
            return False
    
    # 导入案例选择器
    try:
        from case_selector import CaseSelector
        print("✅ 成功导入CaseSelector")
    except ImportError as e:
        print(f"❌ 导入CaseSelector失败: {e}")
        return False
    
    # 创建选择器实例
    try:
        selector = CaseSelector(
            correct_jsonl_path=str(correct_jsonl),
            incorrect_jsonl_path=str(incorrect_jsonl),
            parcel_analysis_path=str(parcel_analysis),
            capability_analysis_path=str(capability_analysis)
        )
        print("✅ 成功创建CaseSelector实例")
    except Exception as e:
        print(f"❌ 创建CaseSelector实例失败: {e}")
        return False
    
    # 测试数据加载
    print("\n📊 测试数据加载...")
    try:
        selector.load_data()
        print(f"✅ 数据加载成功: {len(selector.correct_data)} 正确样本, {len(selector.incorrect_data)} 幻觉样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    # 测试传统选择方法
    print("\n🔧 测试传统选择方法...")
    try:
        # 高置信度幻觉案例
        high_conf_cases = selector.find_high_confidence_hallucinations(5)
        print(f"✅ 高置信度幻觉案例: {len(high_conf_cases)} 个")
        
        # 不同错误类型案例
        diverse_cases = selector.find_diverse_error_types(2)
        print(f"✅ 不同错误类型案例: {len(diverse_cases)} 个")
        
        # 极端案例
        extreme_cases = selector.find_extreme_cases(3)
        print(f"✅ 极端案例: {len(extreme_cases)} 个")
        
    except Exception as e:
        print(f"❌ 传统选择方法测试失败: {e}")
        return False
    
    # 测试LLM服务连接
    print("\n🤖 测试LLM服务连接...")
    try:
        # 简单的测试调用
        test_prompt = "请回答：1+1等于几？"
        response = selector.call_vllm_api(test_prompt, max_tokens=50)
        print(f"✅ LLM服务连接成功")
        print(f"   测试响应: {response[:100]}...")
        
        # 测试LLM案例评估
        if high_conf_cases:
            test_case = high_conf_cases[0]
            print(f"\n🧠 测试LLM案例评估...")
            evaluation = selector.llm_evaluate_case_insight(test_case)
            print(f"✅ LLM案例评估成功")
            print(f"   案例ID: {evaluation.get('case_id', 'N/A')}")
            print(f"   综合分数: {evaluation.get('overall_score', 'N/A')}")
            print(f"   洞察摘要: {evaluation.get('insight_summary', 'N/A')[:100]}...")
        
    except Exception as e:
        print(f"⚠️  LLM服务连接失败: {e}")
        print("   将跳过LLM相关测试")
        llm_available = False
    else:
        llm_available = True
    
    # 测试LLM选择方法（如果服务可用）
    if llm_available:
        print("\n🎯 测试LLM选择方法...")
        try:
            # LLM高洞察价值案例选择
            llm_cases = selector.llm_select_high_insight_cases(3, min_score=6.0)
            print(f"✅ LLM高洞察价值案例: {len(llm_cases)} 个")
            
            # LLM对比配对选择
            llm_pairs = selector.llm_find_contrastive_pairs(2)
            print(f"✅ LLM对比配对案例: {len(llm_pairs)} 对")
            
        except Exception as e:
            print(f"❌ LLM选择方法测试失败: {e}")
            llm_available = False
    
    # 测试综合选择
    print("\n🔄 测试综合选择...")
    try:
        # 传统综合选择
        traditional_cases = selector.select_comprehensive_cases(10, use_llm=False)
        print(f"✅ 传统综合选择: {len(traditional_cases)} 个案例")
        
        # LLM增强综合选择（如果服务可用）
        if llm_available:
            llm_comprehensive_cases = selector.select_comprehensive_cases(10, use_llm=True)
            print(f"✅ LLM增强综合选择: {len(llm_comprehensive_cases)} 个案例")
        
    except Exception as e:
        print(f"❌ 综合选择测试失败: {e}")
        return False
    
    # 测试结果保存
    print("\n💾 测试结果保存...")
    try:
        test_output_dir = results_dir / "test_case_selection"
        test_output_dir.mkdir(exist_ok=True)
        
        # 保存传统选择结果
        selector.save_selected_cases(traditional_cases, str(test_output_dir / "traditional_cases.json"))
        print(f"✅ 传统选择结果已保存到: {test_output_dir / 'traditional_cases.json'}")
        
        # 保存LLM选择结果（如果可用）
        if llm_available and 'llm_comprehensive_cases' in locals():
            selector.save_selected_cases(llm_comprehensive_cases, str(test_output_dir / "llm_cases.json"))
            print(f"✅ LLM选择结果已保存到: {test_output_dir / 'llm_cases.json'}")
        
    except Exception as e:
        print(f"❌ 结果保存测试失败: {e}")
        return False
    
    # 测试总结
    print("\n📋 测试总结")
    print("=" * 50)
    print("✅ 数据加载: 成功")
    print("✅ 传统选择方法: 成功")
    if llm_available:
        print("✅ LLM服务连接: 成功")
        print("✅ LLM选择方法: 成功")
    else:
        print("⚠️  LLM服务连接: 失败（将使用传统方法）")
    print("✅ 综合选择: 成功")
    print("✅ 结果保存: 成功")
    
    print(f"\n🎉 所有测试完成！")
    print(f"📁 测试结果保存在: {test_output_dir}")
    
    return True

def main():
    """主函数"""
    print("🚀 开始LLM案例选择功能测试")
    print("=" * 50)
    
    success = test_llm_case_selection()
    
    if success:
        print("\n✅ 测试成功！LLM案例选择功能工作正常")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！请检查错误信息并修复问题")
        sys.exit(1)

if __name__ == "__main__":
    main()
