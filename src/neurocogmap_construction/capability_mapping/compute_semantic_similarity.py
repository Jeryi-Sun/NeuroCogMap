#!/usr/bin/env python3
"""
计算capability descriptions和parcel functionality descriptions之间的语义相似性
生成capability-parcel相似性矩阵CSV文件
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CapabilityParcelSimilarityComputer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        初始化相似性计算器
        
        Args:
            model_name: 用于计算语义相似性的sentence transformer模型名称
        """
        self.model_name = model_name
        self.model = None
        self.capabilities = {}
        self.parcels = {}
        
    def load_model(self):
        """加载sentence transformer模型"""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Model loaded successfully")
        
    def load_capabilities(self, capability_file):
        """
        加载capability descriptions
        
        Args:
            capability_file: capability descriptions JSON文件路径
        """
        logger.info(f"Loading capabilities from: {capability_file}")
        with open(capability_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.capabilities = {}
        for cap_key, cap_data in data.items():
            # 提取capability的关键信息
            capability_name = cap_data.get('capability_name', cap_key)
            definition = cap_data.get('definition_refined', '')
            # 组合所有相关文本用于相似性计算
            combined_text = f"{capability_name}. {definition}"
            
            # 添加cognitive alignment信息
            if 'cognitive_alignment' in cap_data:
                cognitive_info = cap_data['cognitive_alignment']
                if isinstance(cognitive_info, dict):
                    for key, value in cognitive_info.items():
                        if isinstance(value, str):
                            combined_text += f" {value}"
                elif isinstance(cognitive_info, str):
                    combined_text += f" {cognitive_info}"
            
            # 添加manifestation信息
            if 'manifestation_in_llms' in cap_data:
                combined_text += f" {cap_data['manifestation_in_llms']}"
                
            self.capabilities[cap_key] = {
                'name': capability_name,
                'text': combined_text.strip()
            }
            
        logger.info(f"Loaded {len(self.capabilities)} capabilities")
        
    def load_parcels(self, parcel_file):
        """
        加载parcel functionality descriptions
        
        Args:
            parcel_file: parcel functionality analysis JSON文件路径
        """
        logger.info(f"Loading parcels from: {parcel_file}")
        with open(parcel_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.parcels = {}
        for parcel_key, parcel_data in data.items():
            parcel_id = parcel_data.get('parcel_id', parcel_key)
            functionality_desc = parcel_data.get('functionality_description', '')
            
            # 提取function name和description
            if '**Function Name:**' in functionality_desc:
                parts = functionality_desc.split('**Function Name:**')
                if len(parts) > 1:
                    function_name_part = parts[1].split('**Function Description:**')[0].strip()
                    function_desc_part = parts[1].split('**Function Description:**')[1] if '**Function Description:**' in parts[1] else ''
                    combined_text = f"{function_name_part}. {function_desc_part}"
                else:
                    combined_text = functionality_desc
            else:
                combined_text = functionality_desc
                
            self.parcels[parcel_key] = {
                'id': parcel_id,
                'text': combined_text.strip()
            }
            
        logger.info(f"Loaded {len(self.parcels)} parcels")
        
    def compute_embeddings(self):
        """计算capabilities和parcels的文本嵌入"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        logger.info("Computing embeddings for capabilities...")
        capability_texts = [cap['text'] for cap in self.capabilities.values()]
        self.capability_embeddings = self.model.encode(capability_texts)
        
        logger.info("Computing embeddings for parcels...")
        parcel_texts = [parcel['text'] for parcel in self.parcels.values()]
        self.parcel_embeddings = self.model.encode(parcel_texts)
        
        logger.info("Embeddings computed successfully")
        
    def compute_similarity_matrix(self):
        """计算capability-parcel相似性矩阵"""
        if not hasattr(self, 'capability_embeddings') or not hasattr(self, 'parcel_embeddings'):
            raise ValueError("Embeddings not computed. Call compute_embeddings() first.")
            
        logger.info("Computing similarity matrix...")
        
        # 计算余弦相似性
        similarity_matrix = cosine_similarity(self.capability_embeddings, self.parcel_embeddings)
        
        # 创建DataFrame
        capability_names = [cap['name'] for cap in self.capabilities.values()]
        parcel_ids = [parcel['id'] for parcel in self.parcels.values()]
        
        self.similarity_df = pd.DataFrame(
            similarity_matrix,
            index=capability_names,
            columns=[f"Parcel_{pid}" for pid in parcel_ids]
        )
        
        logger.info(f"Similarity matrix computed: {similarity_matrix.shape}")
        return self.similarity_df
        
    def save_results(self, output_file):
        """保存相似性矩阵到CSV文件"""
        if not hasattr(self, 'similarity_df'):
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
            
        logger.info(f"Saving similarity matrix to: {output_file}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存CSV文件
        self.similarity_df.to_csv(output_file)
        
        # 保存详细结果（包含原始文本）
        detailed_file = output_file.replace('.csv', '_detailed.csv')
        detailed_data = []
        
        for cap_key, cap_data in self.capabilities.items():
            for parcel_key, parcel_data in self.parcels.items():
                similarity_score = self.similarity_df.loc[cap_data['name'], f"Parcel_{parcel_data['id']}"]
                detailed_data.append({
                    'capability_key': cap_key,
                    'capability_name': cap_data['name'],
                    'parcel_key': parcel_key,
                    'parcel_id': parcel_data['id'],
                    'similarity_score': similarity_score,
                    'capability_text': cap_data['text'][:500] + '...' if len(cap_data['text']) > 500 else cap_data['text'],
                    'parcel_text': parcel_data['text'][:500] + '...' if len(parcel_data['text']) > 500 else parcel_data['text']
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df = detailed_df.sort_values('similarity_score', ascending=False)
        detailed_df.to_csv(detailed_file, index=False)
        
        logger.info(f"Results saved to {output_file} and {detailed_file}")
        
    def get_top_matches(self, top_k=10):
        """获取相似性最高的capability-parcel匹配"""
        if not hasattr(self, 'similarity_df'):
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
            
        # 获取所有匹配的相似性分数
        matches = []
        for cap_name in self.similarity_df.index:
            for parcel_col in self.similarity_df.columns:
                similarity_score = self.similarity_df.loc[cap_name, parcel_col]
                parcel_id = parcel_col.replace('Parcel_', '')
                matches.append({
                    'capability_name': cap_name,
                    'parcel_id': parcel_id,
                    'similarity_score': similarity_score
                })
        
        # 按相似性分数排序
        matches_df = pd.DataFrame(matches)
        matches_df = matches_df.sort_values('similarity_score', ascending=False)
        
        return matches_df.head(top_k)
        
    def run_analysis(self, capability_file, parcel_file, output_file):
        """运行完整的相似性分析"""
        logger.info("Starting capability-parcel similarity analysis...")
        
        # 加载模型和数据
        self.load_model()
        self.load_capabilities(capability_file)
        self.load_parcels(parcel_file)
        
        # 计算嵌入和相似性
        self.compute_embeddings()
        similarity_df = self.compute_similarity_matrix()
        
        # 保存结果
        self.save_results(output_file)
        
        # 显示top匹配
        top_matches = self.get_top_matches(10)
        logger.info("Top 10 capability-parcel matches:")
        print(top_matches.to_string(index=False))
        
        logger.info("Analysis completed successfully!")
        return similarity_df

def main():
    parser = argparse.ArgumentParser(description='Compute semantic similarity between capabilities and parcels')
    parser.add_argument('--capability_file', 
                       default='/path/to/project_root/capability_analysis/data/capability_descriptions/capability_descriptions_run2.json',
                       help='Path to capability descriptions JSON file')
    parser.add_argument('--parcel_file',
                       default='/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels195/latent_parcel_topsamples_functionality_analysis.json',
                       help='Path to parcel functionality analysis JSON file')
    parser.add_argument('--output_file',
                       default='/path/to/project_root/neural_area/connect_cap_parcel/code/capability_parcel_similarity_matrix.csv',
                       help='Path to output CSV file')
    parser.add_argument('--model_name',
                       default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    
    args = parser.parse_args()
    
    # 创建相似性计算器
    computer = CapabilityParcelSimilarityComputer(model_name=args.model_name)
    
    # 运行分析
    similarity_df = computer.run_analysis(
        capability_file=args.capability_file,
        parcel_file=args.parcel_file,
        output_file=args.output_file
    )
    
    print(f"\nSimilarity matrix shape: {similarity_df.shape}")
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
