#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算 Human Parcel 和 LLM Parcel 之间的语义相似度矩阵

使用 Qwen3-8b-embedding 模型计算嵌入，构建 H×L 相似度矩阵
H: Human Parcels (按 parcel_id 排序)
L: LLM Parcels (按 parcel_id 排序)
行名和列名都使用 function_name
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from pathlib import Path
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# 设置日志（默认只输出到控制台，可通过 setup_logging 函数添加文件输出）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_logging(log_file=None):
    """设置日志配置，添加文件输出"""
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class ParcelSimilarityComputer:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-8B', device=None):
        """
        初始化相似性计算器
        
        Args:
            model_name: Qwen3-8b-embedding 模型名称
            device: 设备 ('cuda', 'cpu', 或 None 自动选择)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.human_parcels = []
        self.llm_parcels = []
        
        # 设置设备
        if device is None:
            # 自动选择设备：如果有 GPU 则使用 GPU，否则使用 CPU
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"GPU available, using: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device('cpu')
                logger.warning("GPU not available, using CPU")
        else:
            self.device = torch.device(device)
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
        
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """加载 Qwen3-8b-embedding 模型并移动到指定设备"""
        logger.info(f"Loading Qwen3-8b-embedding model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            # 将模型移动到指定设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_human_parcels(self, human_parcel_file):
        """
        加载 Human Parcel descriptions
        
        Args:
            human_parcel_file: Human Parcel JSON 文件路径
        """
        logger.info(f"Loading human parcels from: {human_parcel_file}")
        with open(human_parcel_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Human parcel file should be a list, got {type(data)}")
        
        # 提取信息并按 parcel_id 排序
        self.human_parcels = []
        for item in data:
            parcel_id = item.get('parcel_id')
            if parcel_id is None:
                logger.warning(f"Skipping human parcel without parcel_id: {item.get('parcel_name', 'unknown')}")
                continue
            
            function_name = item.get('function_name', '')
            function_description = item.get('function_description', '')
            
            # 组合文本用于嵌入
            combined_text = f"{function_name}. {function_description}"
            
            self.human_parcels.append({
                'parcel_id': parcel_id,
                'parcel_name': item.get('parcel_name', ''),
                'function_name': function_name,
                'text': combined_text.strip()
            })
        
        # 按 parcel_id 排序
        self.human_parcels.sort(key=lambda x: x['parcel_id'])
        logger.info(f"Loaded {len(self.human_parcels)} human parcels")
    
    def load_llm_parcels(self, llm_parcel_file):
        """
        加载 LLM Parcel descriptions
        
        Args:
            llm_parcel_file: LLM Parcel JSON 文件路径
        """
        logger.info(f"Loading LLM parcels from: {llm_parcel_file}")
        with open(llm_parcel_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取 parcel_summaries
        if isinstance(data, dict) and 'parcel_summaries' in data:
            parcel_summaries = data['parcel_summaries']
        elif isinstance(data, list):
            parcel_summaries = data
        else:
            raise ValueError(f"Unexpected LLM parcel file format: {type(data)}")
        
        # 提取信息并按 parcel_id 排序
        self.llm_parcels = []
        for item in parcel_summaries:
            parcel_id = item.get('parcel_id')
            if parcel_id is None:
                logger.warning(f"Skipping LLM parcel without parcel_id")
                continue
            
            function_name = item.get('function_name', '')
            function_description = item.get('function_description', '')
            
            # 清理 function_name（去除可能的 ** 标记和多余空格）
            function_name = function_name.replace('**', '').strip()
            # 去除前后多余的空格和换行
            function_name = ' '.join(function_name.split())
            
            # 组合文本用于嵌入
            combined_text = f"{function_name}. {function_description}"
            
            self.llm_parcels.append({
                'parcel_id': parcel_id,
                'function_name': function_name,
                'text': combined_text.strip()
            })
        
        # 按 parcel_id 排序
        self.llm_parcels.sort(key=lambda x: x['parcel_id'])
        logger.info(f"Loaded {len(self.llm_parcels)} LLM parcels")
    
    def get_embedding(self, text, batch_size=32):
        """
        使用 Qwen3-8b-embedding 模型计算文本嵌入
        
        Args:
            text: 输入文本
            batch_size: 批处理大小
            
        Returns:
            嵌入向量
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 对文本进行编码（单条文本也按 batch 处理，保持接口一致）
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        )

        # 移动到模型设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 计算嵌入：last token pooling + L2 归一化（对齐官方示例）
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state  # [B, T, D]
            attention_mask = inputs.get("attention_mask", None)

            if attention_mask is None:
                logger.warning("attention_mask 未找到，get_embedding 退化为使用最后一个 token 做 pooling")
                token_embeddings = last_hidden_state[:, -1, :]
            else:
                token_counts = attention_mask.sum(dim=1)  # [B]
                if torch.any(token_counts == 0):
                    logger.warning("get_embedding 中存在 attention_mask 全 0 的样本，使用第一个 token 兜底")
                last_indices = torch.clamp(token_counts - 1, min=0)  # [B]
                batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                token_embeddings = last_hidden_state[batch_indices, last_indices, :]  # [B, D]

            token_embeddings = F.normalize(token_embeddings, p=2, dim=1)

        # 对单条文本返回一维向量
        return token_embeddings.squeeze(0).cpu().numpy()
    
    def compute_embeddings(self, batch_size=32):
        """批量计算所有 parcels 的文本嵌入"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Computing embeddings on device: {self.device}")
        
        # 计算 Human Parcel 嵌入
        logger.info("Computing embeddings for human parcels...")
        human_texts = [parcel['text'] for parcel in self.human_parcels]
        human_embeddings = []

        for i in range(0, len(human_texts), batch_size):
            batch_texts = human_texts[i:i + batch_size]
            batch_inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                last_hidden_state = outputs.last_hidden_state  # [B, T, D]
                attention_mask = batch_inputs.get("attention_mask", None)

                if attention_mask is None:
                    logger.warning("attention_mask 未找到，human parcels 退化为使用最后一个 token 做 pooling")
                    token_embeddings = last_hidden_state[:, -1, :]
                else:
                    token_counts = attention_mask.sum(dim=1)  # [B]
                    if torch.any(token_counts == 0):
                        logger.warning("human parcels 中存在 attention_mask 全 0 的样本，使用第一个 token 兜底")
                    last_indices = torch.clamp(token_counts - 1, min=0)  # [B]
                    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                    token_embeddings = last_hidden_state[batch_indices, last_indices, :]  # [B, D]

                token_embeddings = F.normalize(token_embeddings, p=2, dim=1)
                batch_embeddings = token_embeddings.cpu().numpy()

            human_embeddings.append(batch_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(human_texts)} human parcels")

        self.human_embeddings = np.vstack(human_embeddings)
        logger.info(f"Human parcel embeddings shape: {self.human_embeddings.shape}")
        
        # 计算 LLM Parcel 嵌入
        logger.info("Computing embeddings for LLM parcels...")
        llm_texts = [parcel['text'] for parcel in self.llm_parcels]
        llm_embeddings = []

        for i in range(0, len(llm_texts), batch_size):
            batch_texts = llm_texts[i:i + batch_size]
            batch_inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                last_hidden_state = outputs.last_hidden_state  # [B, T, D]
                attention_mask = batch_inputs.get("attention_mask", None)

                if attention_mask is None:
                    logger.warning("attention_mask 未找到，LLM parcels 退化为使用最后一个 token 做 pooling")
                    token_embeddings = last_hidden_state[:, -1, :]
                else:
                    token_counts = attention_mask.sum(dim=1)  # [B]
                    if torch.any(token_counts == 0):
                        logger.warning("LLM parcels 中存在 attention_mask 全 0 的样本，使用第一个 token 兜底")
                    last_indices = torch.clamp(token_counts - 1, min=0)  # [B]
                    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
                    token_embeddings = last_hidden_state[batch_indices, last_indices, :]  # [B, D]

                token_embeddings = F.normalize(token_embeddings, p=2, dim=1)
                batch_embeddings = token_embeddings.cpu().numpy()

            llm_embeddings.append(batch_embeddings)
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch_texts)}/{len(llm_texts)} LLM parcels")

        self.llm_embeddings = np.vstack(llm_embeddings)
        print("--------------------------------")
        print(self.llm_embeddings.shape)
        print(self.human_embeddings.shape)
        print(llm_texts[0])
        print(human_texts[0])
        logger.info(f"LLM parcel embeddings shape: {self.llm_embeddings.shape}")
    
    def compute_similarity_matrix(self):
        """计算 Human Parcel × LLM Parcel 相似性矩阵"""
        if not hasattr(self, 'human_embeddings') or not hasattr(self, 'llm_embeddings'):
            raise ValueError("Embeddings not computed. Call compute_embeddings() first.")
        
        logger.info("Computing similarity matrix...")
        
        # 计算余弦相似性 (H × L)
        similarity_matrix = cosine_similarity(self.human_embeddings, self.llm_embeddings)
        
        # 使用 parcel_id 作为行名和列名
        human_parcel_ids = [f"Human_Parcel_{parcel['parcel_id']}" for parcel in self.human_parcels]
        llm_parcel_ids = [f"LLM_Parcel_{parcel['parcel_id']}" for parcel in self.llm_parcels]
        
        # 保存 parcel_id 到 function_name 的映射
        self.human_parcel_id_to_name = {parcel['parcel_id']: parcel['function_name'] for parcel in self.human_parcels}
        self.llm_parcel_id_to_name = {parcel['parcel_id']: parcel['function_name'] for parcel in self.llm_parcels}
        
        self.similarity_df = pd.DataFrame(
            similarity_matrix,
            index=human_parcel_ids,
            columns=llm_parcel_ids
        )
        
        logger.info(f"Similarity matrix shape: {self.similarity_df.shape} (H={len(human_parcel_ids)}, L={len(llm_parcel_ids)})")
        return self.similarity_df
    
    def save_results(self, output_file):
        """保存相似性矩阵到 CSV 文件，并保存 parcel_id 到 function_name 的映射"""
        if not hasattr(self, 'similarity_df'):
            raise ValueError("Similarity matrix not computed. Call compute_similarity_matrix() first.")
        
        logger.info(f"Saving similarity matrix to: {output_file}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存 CSV 文件
        self.similarity_df.to_csv(output_file)
        logger.info(f"Results saved to {output_file}")
        
        # 保存映射文件（parcel_id -> function_name）
        mapping_file = output_file.replace('.csv', '_parcel_id_to_function_name.json')
        mapping_data = {
            'human_parcels': self.human_parcel_id_to_name,
            'llm_parcels': self.llm_parcel_id_to_name
        }
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Parcel ID to function name mapping saved to: {mapping_file}")
    
    def run_analysis(self, human_parcel_file, llm_parcel_file, output_file, batch_size=32):
        """运行完整的相似性分析"""
        logger.info("Starting Human-LLM Parcel similarity analysis...")
        logger.info(f"Device: {self.device}")
        
        # 加载模型和数据
        self.load_model()
        self.load_human_parcels(human_parcel_file)
        self.load_llm_parcels(llm_parcel_file)
        
        # 计算嵌入和相似性
        self.compute_embeddings(batch_size=batch_size)
        similarity_df = self.compute_similarity_matrix()
        
        # 保存结果
        self.save_results(output_file)
        
        logger.info("Analysis completed successfully!")
        return similarity_df


def main():
    parser = argparse.ArgumentParser(description='Compute semantic similarity between Human and LLM Parcels')
    parser.add_argument('--human_parcel_file',
                       default='/path/to/project_root/Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json',
                       help='Path to Human Parcel descriptions JSON file')
    parser.add_argument('--llm_parcel_file',
                       default='/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_topsamples_functionality_summary.json',
                       help='Path to LLM Parcel functionality summary JSON file')
    parser.add_argument('--output_file',
                       default='/path/to/project_root/Human_LLM_align/litcoder_core/data_analysis/draw_graphs/data_preparation/semantic_matrix_gemma2_2b.csv',
                       help='Path to output CSV file')
    parser.add_argument('--model_name',
                       default='Qwen/Qwen3-Embedding-8B',
                       help='Qwen3-8b-embedding model name')
    parser.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help='Batch size for embedding computation')
    parser.add_argument('--log_file',
                       default=None,
                       help='Path to log file (optional, if not specified, logs only to console)')
    parser.add_argument('--device',
                       default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (cuda or cpu). Default: None (auto-detect, prefer GPU)')
    
    args = parser.parse_args()
    
    # 设置日志
    if args.log_file:
        logger = setup_logging(args.log_file)
    else:
        setup_logging()
        logger = logging.getLogger(__name__)
    
    # 创建相似性计算器
    computer = ParcelSimilarityComputer(model_name=args.model_name, device=args.device)
    
    # 运行分析
    similarity_df = computer.run_analysis(
        human_parcel_file=args.human_parcel_file,
        llm_parcel_file=args.llm_parcel_file,
        output_file=args.output_file,
        batch_size=args.batch_size
    )
    
    print(f"\nSimilarity matrix shape: {similarity_df.shape}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()

