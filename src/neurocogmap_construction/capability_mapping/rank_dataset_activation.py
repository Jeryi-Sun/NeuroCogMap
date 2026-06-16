#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capability-Parcel Connection System
连接能力(capability)和神经分区(parcel)的系统

实现两种方案：
1. 知识驱动：通过认知神经科学知识匹配能力和分区
2. 数据驱动：通过激活状态分析能力和分区的关联关系
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse
import glob
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import argparse


class CapabilityParcelConnector:
    """能力-分区连接器主类"""
    
    def __init__(self, 
                 capability_data_path: str,
                 parcel_assignments_path: str,
                 sae_output_path: str,
                 results_dir: str):
        """
        初始化连接器
        
        Args:
            capability_data_path: 能力数据集统计文件路径
            parcel_assignments_path: parcel分配结果文件路径
            sae_output_path: SAE激活数据目录路径
            results_dir: 结果存储目录
        """
        self.capability_data_path = capability_data_path
        self.parcel_assignments_path = parcel_assignments_path
        self.sae_output_path = sae_output_path
        self.results_dir = results_dir
        
        # 创建结果目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 数据存储
        self.capability_stats = None
        self.parcel_assignments = None
        self.sae_activations = {}
        self.parcel_activations = {}
        
        print("🔗 Capability-Parcel Connector 初始化完成")
    
    def load_capability_data(self) -> Dict[str, Any]:
        """
        加载能力数据集统计信息
        
        Returns:
            能力数据集统计信息字典
        """
        print("📊 加载能力数据集统计信息...")
        
        try:
            with open(self.capability_data_path, 'r', encoding='utf-8') as f:
                self.capability_stats = json.load(f)
            
            print(f"✅ 成功加载 {len(self.capability_stats)} 个能力数据集")
            return self.capability_stats
            
        except Exception as e:
            print(f"❌ 加载能力数据失败: {e}")
            raise
    
    def load_parcel_assignments(self) -> Dict[str, Any]:
        """
        加载parcel分配结果
        
        Returns:
            parcel分配结果字典
        """
        print("🧩 加载parcel分配结果...")
        
        try:
            with open(self.parcel_assignments_path, 'r', encoding='utf-8') as f:
                self.parcel_assignments = json.load(f)
            
            n_parcels = len(self.parcel_assignments.get('parcel_to_latents', {}))
            n_latents = len(self.parcel_assignments.get('latent_to_parcel', {}))
            
            print(f"✅ 成功加载 {n_parcels} 个parcels, {n_latents} 个latents")
            return self.parcel_assignments
            
        except Exception as e:
            print(f"❌ 加载parcel分配结果失败: {e}")
            raise
    
    def load_sae_activations(self, data_level: str = "sentence") -> Dict[str, np.ndarray]:
        """
        加载SAE激活数据
        
        Args:
            data_level: 数据级别 ("sentence", "example", "dataset")
            
        Returns:
            数据集名称到激活矩阵的映射
        """
        print(f"🧠 加载SAE激活数据 (级别: {data_level})...")
        
        self.sae_activations = {}
        dataset_dirs = sorted(os.listdir(self.sae_output_path))
        import pdb; pdb.set_trace()
        for dataset_dir in tqdm(dataset_dirs, desc="加载数据集"):
            dataset_path = os.path.join(self.sae_output_path, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
            
            # 检查是否有meta文件
            meta_path = os.path.join(dataset_path, f"{dataset_dir}_meta.json")
            if not os.path.exists(meta_path):
                print(f"⚠️  跳过 {dataset_dir}，未找到meta文件")
                continue
            
            # 加载meta信息
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            
            # 获取所有层文件并按层号排序
            layer_files = glob.glob(os.path.join(dataset_path, f"{dataset_dir}_layer*_sparse.npz"))
            if not layer_files:
                print(f"⚠️  跳过 {dataset_dir}，未找到层文件")
                continue
            
            # 按层号排序文件
            def extract_layer_number(filename):
                import re
                match = re.search(r'_layer(\d+)_', filename)
                return int(match.group(1)) if match else 0
            
            layer_files.sort(key=extract_layer_number)
            
            # 加载所有层的激活数据
            layer_acts = []
            for lf in layer_files:
                mat = scipy.sparse.load_npz(lf).toarray()
                layer_acts.append(mat)
            
            if not layer_acts:
                continue
            
            # 将所有层的激活连接起来：
            # 如果某些层的样本数（维度 0）更长，这里统一截断到最小长度，避免 concat 报错
            n_rows_list = [m.shape[0] for m in layer_acts]
            min_rows = min(n_rows_list)
            if len(set(n_rows_list)) > 1:
                print(f"⚠️  数据集 {dataset_dir} 各层样本数不一致，将所有层截断到最小长度 {min_rows}。原始行数: {n_rows_list}")
            aligned_layer_acts = []
            for idx, m in enumerate(layer_acts):
                if m.shape[0] != min_rows:
                    print(f"  ↳ 截断层索引 {idx}：{m.shape[0]} -> {min_rows}")
                    aligned_layer_acts.append(m[:min_rows])
                else:
                    aligned_layer_acts.append(m)
            
            acts = np.concatenate(aligned_layer_acts, axis=1)  # [n_sentences, n_layers * SAE_DIM]
            
            # 根据data_level进行数据聚合
            if data_level == "sentence":
                final_acts = acts
            elif data_level == "example":
                final_acts = self._aggregate_sentences_to_example(acts, meta)
            elif data_level == "dataset":
                example_acts = self._aggregate_sentences_to_example(acts, meta)
                final_acts = example_acts.mean(axis=0, keepdims=True)
            else:
                raise ValueError(f"未知的data_level: {data_level}")
            
            self.sae_activations[dataset_dir] = final_acts
            print(f"  📁 {dataset_dir}: {final_acts.shape}")
        
        print(f"✅ 成功加载 {len(self.sae_activations)} 个数据集的SAE激活")
        return self.sae_activations
    
    def _aggregate_sentences_to_example(self, acts: np.ndarray, meta: List[Dict]) -> np.ndarray:
        """
        将句子级别的激活聚合为example级别
        
        Args:
            acts: 句子级别的激活矩阵 [n_sentences, latent_dim]
            meta: 元数据信息
            
        Returns:
            example级别的激活矩阵 [n_examples, latent_dim]
        """
        example_acts = []
        
        for qa_item in meta:
            sentences = qa_item["sentences"]
            example_vector = np.zeros(acts.shape[1])
            total_tokens = 0
            
            for sentence in sentences:
                sentence_row = sentence["sae_row_idx"]
                token_count = sentence["token_end"] - sentence["token_start"] + 1
                if token_count <= 0:
                    token_count = 1
                
                sentence_vector = acts[sentence_row]
                example_vector += sentence_vector * token_count
                total_tokens += token_count
            
            if total_tokens > 0:
                example_vector /= total_tokens
            else:
                example_vector = np.zeros(acts.shape[1])
            
            example_acts.append(example_vector)
        
        return np.array(example_acts)
    
    def calculate_parcel_activations(self, normalize_method: str = "l2") -> Dict[str, Dict[str, float]]:
        """
        计算每个数据集的parcel激活强度
        
        Args:
            normalize_method: 归一化方法 ("l2", "zscore", "positive", "none")
            
        Returns:
            数据集名称到parcel激活强度字典的映射
        """
        print("⚡ 计算parcel激活强度...")
        if not self.parcel_assignments or not self.sae_activations:
            raise ValueError("请先加载parcel分配结果和SAE激活数据")
        
        parcel_to_latents = self.parcel_assignments['parcel_to_latents']
        self.parcel_activations = {}
        
        for dataset_name, activation_matrix in tqdm(self.sae_activations.items(), desc="计算parcel激活"):
            # 对激活矩阵进行归一化
            if normalize_method == "l2":
                # L2范数归一化
                norms = np.linalg.norm(activation_matrix, axis=0, keepdims=True)
                norms = np.clip(norms, 1e-8, None)
                normalized_acts = activation_matrix / norms
            elif normalize_method == "zscore":
                # Z-score标准化
                scaler = StandardScaler()
                normalized_acts = scaler.fit_transform(activation_matrix)
            elif normalize_method == "positive":
                # 只保留正值，负值设为0
                normalized_acts = np.maximum(activation_matrix, 0)
            else:
                normalized_acts = activation_matrix
            
            # 计算每个parcel的激活强度
            parcel_activation = {}
            for parcel_name, latent_ids in parcel_to_latents.items():
                # 计算该parcel包含的latent的平均激活强度
                parcel_latent_acts = normalized_acts[:, latent_ids]
                # 对所有样本的激活强度取平均
                if normalize_method == "positive":
                    # 对于positive方法，已经只保留正值，直接取平均
                    avg_activation = np.mean(parcel_latent_acts)
                else:
                    # 其他方法使用绝对值
                    avg_activation = np.mean(np.abs(parcel_latent_acts))
                parcel_activation[parcel_name] = float(avg_activation)
            
            self.parcel_activations[dataset_name] = parcel_activation
        
        print(f"✅ 完成 {len(self.parcel_activations)} 个数据集的parcel激活计算")
        return self.parcel_activations
    
    def generate_activation_rankings(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        生成每个数据集的parcel激活排序
        
        Returns:
            数据集名称到parcel激活排序的映射
        """
        print("📈 生成parcel激活排序...")
        
        if not self.parcel_activations:
            raise ValueError("请先计算parcel激活强度")
        
        rankings = {}
        for dataset_name, parcel_acts in self.parcel_activations.items():
            # 按激活强度降序排序
            sorted_parcels = sorted(parcel_acts.items(), key=lambda x: x[1], reverse=True)
            rankings[dataset_name] = sorted_parcels
        
        return rankings
    
    def save_results(self, rankings: Dict[str, List[Tuple[str, float]]], 
                    output_prefix: str = "parcel_activation"):
        """
        保存结果到文件
        
        Args:
            rankings: parcel激活排序结果
            output_prefix: 输出文件前缀
        """
        print("💾 保存结果...")
        
        # 保存parcel激活强度
        activation_file = os.path.join(self.results_dir, f"{output_prefix}_strengths.json")
        with open(activation_file, 'w', encoding='utf-8') as f:
            json.dump(self.parcel_activations, f, ensure_ascii=False, indent=2)
        
        # 保存parcel激活排序
        ranking_file = os.path.join(self.results_dir, f"{output_prefix}_rankings.json")
        with open(ranking_file, 'w', encoding='utf-8') as f:
            json.dump(rankings, f, ensure_ascii=False, indent=2)
        
        # 生成CSV格式的排序表
        csv_file = os.path.join(self.results_dir, f"{output_prefix}_rankings.csv")
        self._save_rankings_csv(rankings, csv_file)
        
        # 生成可视化图表
        plot_file = os.path.join(self.results_dir, f"{output_prefix}_heatmap.png")
        self._plot_activation_heatmap(rankings, plot_file)
        
        print(f"✅ 结果已保存到: {self.results_dir}")
        print(f"  📄 激活强度: {activation_file}")
        print(f"  📄 激活排序: {ranking_file}")
        print(f"  📊 CSV表格: {csv_file}")
        print(f"  📈 热力图: {plot_file}")
    
    def _save_rankings_csv(self, rankings: Dict[str, List[Tuple[str, float]]], csv_file: str):
        """保存排序结果为CSV格式"""
        # 获取所有parcel名称
        all_parcels = set()
        for dataset_rankings in rankings.values():
            all_parcels.update([parcel for parcel, _ in dataset_rankings])
        
        # 创建DataFrame
        data = {}
        for dataset_name, dataset_rankings in rankings.items():
            parcel_dict = dict(dataset_rankings)
            for parcel in all_parcels:
                if parcel not in data:
                    data[parcel] = {}
                data[parcel][dataset_name] = parcel_dict.get(parcel, 0.0)
        
        df = pd.DataFrame(data).T
        df.to_csv(csv_file, encoding='utf-8')
    
    def _plot_activation_heatmap(self, rankings: Dict[str, List[Tuple[str, float]]], plot_file: str):
        """绘制激活强度热力图"""
        # 准备数据
        datasets = list(rankings.keys())
        all_parcels = set()
        for dataset_rankings in rankings.values():
            all_parcels.update([parcel for parcel, _ in dataset_rankings])
        
        # 创建数据矩阵
        data_matrix = []
        parcel_names = sorted(all_parcels)
        
        for parcel in parcel_names:
            row = []
            for dataset in datasets:
                dataset_rankings = dict(rankings[dataset])
                row.append(dataset_rankings.get(parcel, 0.0))
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # 绘制热力图
        plt.figure(figsize=(max(12, len(datasets) * 0.8), max(8, len(parcel_names) * 0.3)))
        sns.heatmap(data_matrix, 
                   xticklabels=datasets,
                   yticklabels=parcel_names,
                   cmap='viridis',
                   annot=True,
                   fmt='.3f',
                   cbar_kws={'label': 'Activation Strength'})
        
        plt.title('Parcel Activation Strength by Dataset', fontsize=14, pad=20)
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Parcel', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def knowledge_driven_analysis(self) -> Dict[str, Any]:
        """
        知识驱动的能力-分区连接分析
        
        Returns:
            知识驱动分析结果
        """
        print("🧠 执行知识驱动分析...")
        
        # 这里可以实现基于认知神经科学知识的匹配逻辑
        # 例如：通过能力描述和分区功能描述的语义相似度进行匹配
        
        # 示例实现（需要根据实际需求完善）
        knowledge_results = {
            "method": "knowledge_driven",
            "description": "基于认知神经科学知识的语义匹配",
            "matches": {},
            "confidence_scores": {}
        }
        
        # 保存知识驱动结果
        knowledge_file = os.path.join(self.results_dir, "knowledge_driven_results.json")
        with open(knowledge_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 知识驱动分析结果已保存到: {knowledge_file}")
        return knowledge_results
    
    def data_driven_analysis(self, top_k: int = 10) -> Dict[str, Any]:
        """
        数据驱动的能力-分区连接分析
        
        Args:
            top_k: 取前k个最相关的parcel
            
        Returns:
            数据驱动分析结果
        """
        print("📊 执行数据驱动分析...")
        
        if not self.parcel_activations:
            raise ValueError("请先计算parcel激活强度")
        
        data_results = {
            "method": "data_driven",
            "description": "基于激活强度的数据驱动分析",
            "top_parcels_by_dataset": {},
            "parcel_importance_scores": {},
            "intervention_analysis": {}
        }
        
        # 为每个数据集找出最重要的parcel
        for dataset_name, parcel_acts in self.parcel_activations.items():
            sorted_parcels = sorted(parcel_acts.items(), key=lambda x: x[1], reverse=True)
            data_results["top_parcels_by_dataset"][dataset_name] = sorted_parcels[:top_k]
        
        # 计算每个parcel的总体重要性（跨数据集）
        parcel_importance = {}
        for dataset_acts in self.parcel_activations.values():
            for parcel, activation in dataset_acts.items():
                if parcel not in parcel_importance:
                    parcel_importance[parcel] = []
                parcel_importance[parcel].append(activation)
        
        # 计算平均重要性
        for parcel, activations in parcel_importance.items():
            data_results["parcel_importance_scores"][parcel] = {
                "mean_activation": float(np.mean(activations)),
                "std_activation": float(np.std(activations)),
                "max_activation": float(np.max(activations)),
                "activation_count": len(activations)
            }
        
        # 保存数据驱动结果
        data_file = os.path.join(self.results_dir, "data_driven_results.json")
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 数据驱动分析结果已保存到: {data_file}")
        return data_results
    
    def run_complete_analysis(self, data_level: str = "sentence", 
                            normalize_method: str = "l2",
                            top_k: int = 10) -> Dict[str, Any]:
        """
        运行完整的分析流程
        
        Args:
            data_level: 数据级别
            normalize_method: 归一化方法
            top_k: 数据驱动分析中的top_k参数
            
        Returns:
            完整分析结果
        """
        print("🚀 开始完整分析流程...")
        print("=" * 60)
        
        # 1. 加载数据
        self.load_capability_data()
        self.load_parcel_assignments()
        self.load_sae_activations(data_level)
        
        # 2. 计算parcel激活
        self.calculate_parcel_activations(normalize_method)
        
        # 3. 生成排序
        rankings = self.generate_activation_rankings()
        
        # 4. 保存基础结果
        self.save_results(rankings)
        
        # 5. 知识驱动分析
        knowledge_results = self.knowledge_driven_analysis()
        
        # 6. 数据驱动分析
        data_results = self.data_driven_analysis(top_k)
        
        # 7. 生成综合分析报告
        summary = {
            "analysis_summary": {
                "total_datasets": len(self.sae_activations),
                "total_parcels": len(self.parcel_assignments.get('parcel_to_latents', {})),
                "data_level": data_level,
                "normalize_method": normalize_method,
                "top_k": top_k
            },
            "knowledge_driven": knowledge_results,
            "data_driven": data_results,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # 保存综合分析报告
        summary_file = os.path.join(self.results_dir, "analysis_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("=" * 60)
        print("✅ 完整分析流程完成！")
        print(f"📋 综合分析报告: {summary_file}")
        
        return summary


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Capability-Parcel Connection Analysis')
    parser.add_argument('--capability_data', type=str, 
                       default='/path/to/project_root/neural_area/capability_data_v2/data_stastic/final_merged_capability_dataset_stats.json',
                       help='能力数据集统计文件路径')
    parser.add_argument('--parcel_assignments', type=str,
                       default='/path/to/project_root/neural_area/divide_area_by_sae_act/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels195/latent_parcel_assignments.json',
                       help='parcel分配结果文件路径')
    parser.add_argument('--sae_output', type=str,
                       default='/path/to/project_root/neural_area/divide_area_by_sae_act/qa_sae_output_2b_pt',
                       help='SAE激活数据目录路径')
    parser.add_argument('--results_dir', type=str,
                       default='/path/to/project_root/neural_area/connect_cap_parcel/results',
                       help='结果存储目录')
    parser.add_argument('--data_level', type=str, default='sentence',
                       choices=['sentence', 'example', 'dataset'],
                       help='数据级别')
    parser.add_argument('--normalize_method', type=str, default='l2',
                       choices=['l2', 'zscore', 'none', 'positive'],
                       help='归一化方法')
    parser.add_argument('--top_k', type=int, default=10,
                       help='数据驱动分析中的top_k参数')
    
    args = parser.parse_args()
    
    # 创建连接器并运行分析
    connector = CapabilityParcelConnector(
        capability_data_path=args.capability_data,
        parcel_assignments_path=args.parcel_assignments,
        sae_output_path=args.sae_output,
        results_dir=args.results_dir
    )
    
    # 运行完整分析
    results = connector.run_complete_analysis(
        data_level=args.data_level,
        normalize_method=args.normalize_method,
        top_k=args.top_k
    )
    
    print("🎉 分析完成！")


if __name__ == "__main__":
    main() 