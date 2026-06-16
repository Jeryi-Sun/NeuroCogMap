import os
import numpy as np
import scipy.sparse
import json
import glob
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from tqdm import tqdm
import torch
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import TruncatedSVD
import random
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from neurocogmap_release.paths import output_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from neurocogmap_release.paths import output_path

def get_memory_usage():
    """
    获取当前内存使用情况
    Returns:
        memory_mb: 内存使用量（MB）
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    except ImportError:
        return 0.0

def _log_mem(stage: str):
    """
    打印当前CPU内存与GPU显存使用情况
    """
    print_memory_usage(f"[build_cap_matrix][{stage}] ")
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"[build_cap_matrix][{stage}] GPU 显存: allocated={allocated:.1f} MB, reserved={reserved:.1f} MB")
    except Exception as e:
        print(f"[build_cap_matrix][{stage}] 获取GPU显存信息失败: {e}")

def print_memory_usage(prefix=""):
    """
    打印当前内存使用情况
    Args:
        prefix: 前缀字符串
    """
    memory_mb = get_memory_usage()
    print(f"{prefix}内存使用: {memory_mb:.1f} MB")

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
import torch

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 1. 加载激活数据，适配新的文件结构

def extract_content_from_meta(meta, data_level):
    """
    从meta文件中提取具体内容
    Args:
        meta: meta文件内容
        data_level: 数据级别 ("sentence", "example", "dataset")
    Returns:
        content_list: 内容列表 [(question, content), ...]
    """
    content_list = []
    
    if data_level == "sentence":
        # 提取所有句子内容
        for qa_item in meta:
            question = qa_item.get("question", "")
            answer = qa_item.get("answer", "")
            sentences = qa_item.get("sentences", [])
            for sentence in sentences:
                sentence_text = sentence.get("sentence", "")
                content_list.append((question, answer, sentence_text))
                
    elif data_level == "example":
        # 提取所有QA对的内容
        for qa_item in meta:
            question = qa_item.get("question", "")
            answer = qa_item.get("answer", "")
            content_list.append((question, answer))
            
    elif data_level == "dataset":
        # dataset级别不需要具体内容，使用数据集名称
        content_list = [("", "")]
        
    return content_list

def load_all_activations(output_root, data_level="sentence", test_mode=False):
    """
    加载所有数据集的SAE激活数据
    Args:
        output_root: qa_sae_output目录路径
        data_level: 数据级别 ("sentence", "example", "dataset")
        test_mode: 测试模式，只加载少量数据
    Returns:
        capabilities: 数据集名称列表
        cap2vecs: 数据集名称到激活矩阵的映射
        latent_dim: SAE激活维度 (SAE_DIM)
        level_mappings: 各级别的映射记录
        sample_mappings: 样本到具体内容的映射
    """
    capabilities = []
    cap2vecs = {}
    latent_dim = 0
    current_row_count = 0  # 跟踪当前的总行数
    sample_mappings = []  # 存储样本到具体内容的映射
    level_mappings = {
        "sentence_level": {},
        "example_level": {},
        "dataset_level": {}
    }
    
    # 遍历所有数据集目录
    dataset_dirs = sorted(os.listdir(output_root))
    if test_mode:
        dataset_dirs = dataset_dirs[:3]  # 只加载前3个数据集进行测试
        print(f"测试模式：只加载前3个数据集")
    
    for dataset_dir in dataset_dirs:
        filtered_datasets = [
            "advglue",
            "agnews",
            "commongen",
            #"commonsenseqa",
            "crows_pairs",
            "gap_coref",
            "scan",
            "spider",
            #"stereoset",
            "tinystories_continuation",
            "wsc",
            #"drop"
        ]
        is_filtered = False
        for filtered_dataset in filtered_datasets:
            if filtered_dataset in dataset_dir:
                is_filtered = True
                break
        if is_filtered:
            continue
        dataset_path = os.path.join(output_root, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
            
        # 检查是否有meta文件
        meta_path = os.path.join(dataset_path, f"{dataset_dir}_meta.json")
        if not os.path.exists(meta_path):
            print(f"跳过 {dataset_dir}，未找到meta文件")
            continue
            
        # 加载meta信息
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        # 获取所有层文件并按层号排序
        layer_files = glob.glob(os.path.join(dataset_path, f"{dataset_dir}_layer*_sparse.npz"))
        if not layer_files:
            print(f"跳过 {dataset_dir}，未找到层文件")
            continue
        
        # 按层号排序文件（而不是按文件名排序）
        def extract_layer_number(filename):
            import re
            match = re.search(r'_layer(\d+)_', filename)
            return int(match.group(1)) if match else 0
        
        layer_files.sort(key=extract_layer_number)
        print(f"数据集 {dataset_dir} 的层文件按层号排序: {[os.path.basename(f) for f in layer_files]}")
        
        # 加载所有层的激活数据
        layer_acts = []
        layer_numbers = []
        for lf in layer_files:
            layer_num = extract_layer_number(lf)
            layer_numbers.append(layer_num)
            mat = scipy.sparse.load_npz(lf).toarray()  # [n_sentences, SAE_DIM]
            layer_acts.append(mat)
            print(f"  加载层 {layer_num}: {mat.shape}")
        # 取 layer_acts 中 mat 的第一维，最小的维度进行截取
        min_dim = min(mat.shape[0] for mat in layer_acts)
        layer_acts = [mat[:min_dim] for mat in layer_acts]
        print(f"截取后 layer_acts 的维度: {layer_acts[0].shape}")
        
        if not layer_acts:
            print(f"跳过 {dataset_dir}，激活数据为空")
            continue
        
        # 验证层号连续性
        if "9b" in output_root:
            expected_layers = [9, 20, 31]
        elif "2b" in output_root:
            expected_layers = list(range(26))
        elif "8b" in output_root:
            expected_layers = list(range(32))
        else:
            raise ValueError(f"未知的模型: {dataset_dir}")

        if layer_numbers != expected_layers:
            print(f"⚠️  警告：数据集 {dataset_dir} 的层号不连续！")
            print(f"   实际层号: {layer_numbers}")
            print(f"   期望层号: {expected_layers}")
            print(f"   缺失的层: {set(expected_layers) - set(layer_numbers)}")
            print(f"   多余的层: {set(layer_numbers) - set(expected_layers)}")
            print(f"跳过 {dataset_dir}，层号不连续")
            continue
        
        # 将所有层的激活连接起来
        _log_mem("开始连接所有层的激活")
        acts = np.concatenate(layer_acts, axis=1)  # [n_sentences, n_layers * SAE_DIM]
        print(f"  连接后总维度: {acts.shape} (层数: {len(layer_numbers)})")
        _log_mem("连接所有层的激活完成")
        if latent_dim == 0:
            latent_dim = acts.shape[1]
        else:
            assert acts.shape[1] == latent_dim, f"latent dim mismatch: {acts.shape[1]} vs {latent_dim}"
        
        # 提取具体内容
        content_list = extract_content_from_meta(meta, data_level)
        
        # 根据data_level进行数据聚合
        if data_level == "sentence":
            # sentence_level: 直接使用原始数据
            final_acts = acts
            
            # 构建句子级别的样本映射
            for sent_idx in range(acts.shape[0]):
                if sent_idx < len(content_list):
                    question, answer, sentence_text = content_list[sent_idx]
                    content = f"Question: {question}\nAnswer: {answer}\nActivated Sentence: {sentence_text}"
                else:
                    content = f"{dataset_dir}_sentence_{sent_idx}"
                
                sample_mappings.append({
                    "dataset": dataset_dir,
                    "type": "sentence",
                    "index": sent_idx,
                    "content": content
                })
            
            level_mappings["sentence_level"][dataset_dir] = {
                "start_row": current_row_count,
                "end_row": current_row_count + acts.shape[0],
                "total_sentences": acts.shape[0]
            }
            current_row_count += acts.shape[0]
            print(f"加载数据集 {dataset_dir} (sentence_level): {acts.shape}")
            
        elif data_level == "example":
            # example_level: 将每个QA对的所有句子聚合为一个向量
            final_acts, example_mapping = aggregate_sentences_to_example(acts, meta, dataset_dir)
            
            # 构建example级别的样本映射
            for example_idx in range(final_acts.shape[0]):
                if example_idx < len(content_list):
                    question, answer = content_list[example_idx]
                    content = f"Question: {question}\nAnswer: {answer}"
                else:
                    content = f"{dataset_dir}_example_{example_idx}"
                
                sample_mappings.append({
                    "dataset": dataset_dir,
                    "type": "example",
                    "index": example_idx,
                    "content": content
                })
            
            level_mappings["example_level"][dataset_dir] = {
                "start_row": current_row_count,
                "end_row": current_row_count + final_acts.shape[0],
                "total_examples": final_acts.shape[0],
                "example_mapping": example_mapping
            }
            current_row_count += final_acts.shape[0]
            print(f"加载数据集 {dataset_dir} (example_level): {final_acts.shape}")
            
        elif data_level == "dataset":
            # dataset_level: 对整个数据集取平均，每个数据集一行
            example_acts, example_mapping = aggregate_sentences_to_example(acts, meta, dataset_dir)
            final_acts = example_acts.mean(axis=0, keepdims=True)  # [1, latent_dim]
            
            # 构建dataset级别的样本映射
            sample_mappings.append({
                "dataset": dataset_dir,
                "type": "dataset",
                "index": 0,
                "content": dataset_dir
            })
            
            level_mappings["dataset_level"][dataset_dir] = {
                "start_row": current_row_count,
                "end_row": current_row_count + 1,
                "total_examples": example_acts.shape[0],
                "example_mapping": example_mapping
            }
            current_row_count += 1
            print(f"加载数据集 {dataset_dir} (dataset_level): {final_acts.shape}")
            
        else:
            raise ValueError(f"未知的data_level: {data_level}")
            
        capabilities.append(dataset_dir)
        cap2vecs[dataset_dir] = final_acts
        
    return capabilities, cap2vecs, latent_dim, level_mappings, sample_mappings

def aggregate_sentences_to_example(acts, meta, dataset_name):
    """
    将句子级别的激活聚合为example级别
    Args:
        acts: 句子级别的激活矩阵 [n_sentences, latent_dim]
        meta: 元数据信息
        dataset_name: 数据集名称
    Returns:
        example_acts: example级别的激活矩阵 [n_examples, latent_dim]
        example_mapping: example映射信息
    """
    example_acts = []
    example_mapping = {}
    
    for example_idx, qa_item in enumerate(meta):
        sentences = qa_item["sentences"]
        example_vector = np.zeros(acts.shape[1])
        total_tokens = 0
        sentence_rows = []
        token_counts = []
        
        for sentence in sentences:
            sentence_row = sentence["sae_row_idx"]
            # 计算token数量（闭区间）
            token_count = sentence["token_end"] - sentence["token_start"] + 1
            if token_count <= 0:  # 处理边界情况
                token_count = 1
                
            sentence_vector = acts[sentence_row]
            example_vector += sentence_vector * token_count
            total_tokens += token_count
            sentence_rows.append(sentence_row)
            token_counts.append(token_count)
        
        if total_tokens > 0:
            example_vector /= total_tokens
        else:
            # 如果没有有效token，使用零向量
            example_vector = np.zeros(acts.shape[1])
            
        example_acts.append(example_vector)
        example_mapping[f"example_{example_idx}"] = {
            "sentence_rows": sentence_rows,
            "token_counts": token_counts,
            "total_tokens": total_tokens
        }
    
    return np.array(example_acts), example_mapping

def save_level_mappings(level_mappings, output_dir):
    """
    保存各级别的映射记录
    Args:
        level_mappings: 映射信息字典
        output_dir: 输出目录
    """
    for level_name, mappings in level_mappings.items():
        if mappings:  # 只保存非空的映射
            mapping_file = os.path.join(output_dir, f"{level_name}_mapping.json")
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)
            print(f"已保存 {level_name} 映射到: {mapping_file}")

# 2. 构造能力×latent矩阵

def _prefilter_features_before_concat(
    capabilities,
    cap2vecs,
    latent_dim: int,
    *,
    nz_eps: float = 1e-5,
    min_activation_rate: float = 0.02,
    min_var: float = 1e-6,
):
    """
    在真正构造整块大矩阵 A 之前，基于“每个数据集单独的矩阵”做一次全局 feature 统计，
    然后按照 (1536-1542) 中的逻辑对列做一次筛选，得到 kept_feature_idx，并立刻
    用这个索引裁剪每个数据集的矩阵列数，从而在后续拼接 A 时显著降低内存占用。

    这里的统计等价于：
        col_nonzero_ratio = (A > nz_eps).mean(axis=0)
        col_var = A.var(axis=0)
    只是改为通过逐数据集累加、而不是一次性构造 A。
    """
    if len(capabilities) == 0 or latent_dim == 0:
        return np.arange(latent_dim, dtype=int), None, None

    total_examples = 0
    total_nz = np.zeros(latent_dim, dtype=np.int64)
    sum_vals = np.zeros(latent_dim, dtype=np.float64)
    sum_sq_vals = np.zeros(latent_dim, dtype=np.float64)

    for cap in capabilities:
        Z = cap2vecs[cap]
        if Z.shape[1] != latent_dim:
            raise ValueError(f"[prefilter] 数据集 {cap} 的latent维度 {Z.shape[1]} 与预期 {latent_dim} 不一致")

        n = Z.shape[0]
        if n == 0:
            continue
        total_examples += n

        nz_mask = Z > nz_eps
        total_nz += nz_mask.sum(axis=0)

        sum_vals += Z.sum(axis=0, dtype=np.float64)
        sum_sq_vals += np.square(Z, dtype=np.float64).sum(axis=0)

    if total_examples == 0:
        print("[prefilter] 警告：total_examples 为 0，跳过预筛选")
        return np.arange(latent_dim, dtype=int), None, None

    total_examples_f = float(total_examples)
    col_nonzero_ratio = total_nz.astype(np.float64) / total_examples_f

    mean = sum_vals / total_examples_f
    col_var = sum_sq_vals / total_examples_f - mean ** 2
    col_var = np.maximum(col_var, 0.0)

    keep_cols_mask = (col_nonzero_ratio > min_activation_rate) & (col_var > min_var)
    kept_feature_idx = np.where(keep_cols_mask)[0].astype(int)

    print(f"[prefilter] 全局列筛选: 原始 latent_dim={latent_dim}, 保留 {len(kept_feature_idx)}/{latent_dim} 个feature")
    print(f"[prefilter] 非零比例范围: [{col_nonzero_ratio.min():.4e}, {col_nonzero_ratio.max():.4e}]")
    print(f"[prefilter] 方差范围: [{col_var.min():.4e}, {col_var.max():.4e}]")

    if len(kept_feature_idx) == 0:
        print("[prefilter] 警告：列筛选后没有任何 feature 被保留，将退化为保留全部 feature")
        kept_feature_idx = np.arange(latent_dim, dtype=int)

    # 就地裁剪每个数据集的矩阵列数，后面构造 A 时只在这些列上拼接
    for cap in capabilities:
        Z = cap2vecs[cap]
        if Z.shape[1] != latent_dim:
            raise ValueError(f"[prefilter] 截断前数据集 {cap} 的latent维度 {Z.shape[1]} 与预期 {latent_dim} 不一致")
        cap2vecs[cap] = Z[:, kept_feature_idx]

    return kept_feature_idx, col_nonzero_ratio, col_var


def build_capability_matrix(capabilities, cap2vecs, latent_dim, use_preprocessing=True, preprocessing_config=None, 
                          use_svd_reduction=False, svd_config=None, model_name="llama_8b_pt"):
    """
    构建样本×latent矩阵（根据data_level不同，样本可能是句子、example或数据集）
    Args:
        capabilities: 数据集名称列表
        cap2vecs: 数据集名称到激活矩阵的映射
        latent_dim: 总latent维度
        use_preprocessing: 是否使用预处理筛选
        preprocessing_config: 预处理配置参数
        use_svd_reduction: 是否使用SVD降维
        svd_config: SVD降维配置参数
    Returns:
        A: 原始样本矩阵 [n_samples, latent_dim]
        A_z: 预处理后的矩阵 [n_samples, filtered_latent_dim] 或降维后的矩阵 [reduced_dim, latent_dim]
        preprocessing_info: 预处理信息（包含索引映射）
    """
    # 内部小工具：监控构建矩阵阶段的内存 / 显存

    _log_mem("开始前")

    # 1) 如开启 use_preprocessing，则在拼接大矩阵前做一次 feature 预筛选（按 1536-1542 的规则）
    prefilter_kept_feature_idx = None
    if use_preprocessing and preprocessing_config is not None:
        min_activation_rate = preprocessing_config.get("min_activation_rate", 0.02)
        min_var = preprocessing_config.get("min_var", 1e-6)
        nz_eps = preprocessing_config.get("nz_eps", 1e-5)
        prefilter_kept_feature_idx, _, _ = _prefilter_features_before_concat(
            capabilities,
            cap2vecs,
            latent_dim,
            nz_eps=nz_eps,
            min_activation_rate=min_activation_rate,
            min_var=min_var,
        )
        effective_latent_dim = len(prefilter_kept_feature_idx)
    else:
        effective_latent_dim = latent_dim

    # 2) 在预筛选后的列空间上拼接大矩阵
    # 收集所有样本的激活向量（低内存版本：预分配 + 分块拷贝，避免 np.vstack 额外拷贝）
    if len(capabilities) == 0:
        A_reduced = np.zeros((0, effective_latent_dim), dtype=np.float32)
    else:
        # 先统计总样本数和统一dtype
        total_samples = 0
        first_cap = capabilities[0]
        first_Z = cap2vecs[first_cap]
        base_dtype = first_Z.dtype
        for cap in capabilities:
            Z = cap2vecs[cap]
            if Z.shape[1] != effective_latent_dim:
                raise ValueError(f"数据集 {cap} 的latent维度 {Z.shape[1]} 与预期 {effective_latent_dim} 不一致")
            total_samples += Z.shape[0]
        # 预分配一次性大矩阵，然后逐块拷贝
        A_reduced = np.empty((total_samples, effective_latent_dim), dtype=base_dtype)
        row_cursor = 0
        # 逐数据集拷贝 + 立刻释放已拷贝的块，避免长期双份占用
        for cap in capabilities:
            Z = cap2vecs[cap]
            n_rows = Z.shape[0]
            A_reduced[row_cursor:row_cursor + n_rows, :] = Z
            row_cursor += n_rows
            # 立刻释放已拷贝的数据块，降低峰值内存
            try:
                del cap2vecs[cap]
                import gc
                gc.collect()
            except Exception as e:
                print(f"[build_cap_matrix] 清理cap2vecs[{cap}]失败: {e}")
    _log_mem(f"拼接 A_reduced 完成, A_reduced.shape={A_reduced.shape}")

    # 逻辑上的“原始 A”：后续所有列索引统一回到原始 latent 空间
    A = A_reduced
    
    if use_preprocessing and preprocessing_config is not None:
        # 使用预处理筛选（此时 A_reduced 已经做过一次列筛选，这里不再重复 1536-1542 的筛选）
        print("开始预处理筛选（基于预筛后的 A_reduced）...")
        _log_mem("预处理筛选前")
        # 只做行筛选 / Gini / 标准化，不再做第 1 步激活率+方差列筛
        prep_result = preprocess_sae_for_feature_clustering(
            A_reduced,
            skip_first_col_filter=True,
            **preprocessing_config,
        )
        
        # 获取筛选后的矩阵和索引映射（索引目前是相对于 A_reduced 的）
        A_filtered = prep_result["X_feat"].T  # 转置回 [n_samples, filtered_latent_dim]
        kept_example_idx = prep_result["kept_example_idx"]
        kept_feature_idx_stage2 = prep_result["kept_feature_idx"]
        
        # 合并两次索引，得到“原始 latent 空间”的 kept_feature_idx
        if prefilter_kept_feature_idx is None:
            final_kept_feature_idx = kept_feature_idx_stage2
        else:
            final_kept_feature_idx = prefilter_kept_feature_idx[kept_feature_idx_stage2]
        
        # 构建完整的索引映射（包括未参与聚类的latent），下游全部用原始 latent 维度
        labels_by_original_feature = np.full(shape=latent_dim, fill_value=-1, dtype=int)
        labels_by_original_feature[final_kept_feature_idx] = np.arange(len(final_kept_feature_idx))
        
        _log_mem(f"预处理筛选后, A_filtered.shape={A_filtered.shape}")

        # 如果同时使用SVD降维，在筛选后进行降维
        if use_svd_reduction and svd_config is not None:
            print("在筛选后进行SVD降维...")
            _log_mem("SVD 降维前（筛选后）")
            svd_result = reduce_matrix_dimension_with_svd(A_filtered, **svd_config, model_name=model_name)
            
            # 获取降维后的矩阵
            A_z = svd_result["A_z_reduced"]  # [reduced_dim, filtered_latent_dim]
            reduced_dim = A_z.shape[0]
            
            preprocessing_info = {
                "kept_example_idx": kept_example_idx,
                "kept_feature_idx": final_kept_feature_idx,
                "labels_by_original_feature": labels_by_original_feature,
                "prep_result": prep_result,
                "svd_result": svd_result,
                "filtered_latent_dim": reduced_dim,  # 降维后的维度
                "use_svd_reduction": True,
                "use_preprocessing": True
            }
            
            print(f"筛选+降维完成: 原始 {A.shape} -> 筛选后 {A_filtered.shape} -> 降维后 {A_z.shape}")
            _log_mem("SVD 降维后（筛选+降维）")
            return A, A_z, preprocessing_info
        else:
            # 只使用预处理筛选
            A_z = A_filtered
            
            preprocessing_info = {
                "kept_example_idx": kept_example_idx,
                "kept_feature_idx": final_kept_feature_idx,
                "labels_by_original_feature": labels_by_original_feature,
                "prep_result": prep_result,
                "filtered_latent_dim": A_z.shape[1],
                "use_svd_reduction": False,
                "use_preprocessing": True
            }
            
            print(f"预处理完成: 原始 {A.shape} -> 筛选后 {A_z.shape}")
            _log_mem("仅预处理完成")
            return A, A_z, preprocessing_info
            
    elif use_svd_reduction and svd_config is not None:
        # 只使用SVD降维（不进行预处理筛选）
        print("开始SVD降维...")
        _log_mem("SVD 降维前（无预处理）")
        svd_result = reduce_matrix_dimension_with_svd(A, **svd_config, model_name=model_name)
        
        # 获取降维后的矩阵
        A_z = svd_result["A_z_reduced"]  # [reduced_dim, latent_dim] - gwMRF期望的形状
        reduced_dim = A_z.shape[0]
        
        preprocessing_info = {
            "kept_example_idx": np.arange(A.shape[0]),  # 所有样本都保留
            "kept_feature_idx": np.arange(latent_dim),  # 所有latent都保留
            "labels_by_original_feature": np.arange(latent_dim),
            "svd_result": svd_result,
            "filtered_latent_dim": reduced_dim,  # 降维后的维度
            "use_svd_reduction": True,
            "use_preprocessing": False
        }
        
        print(f"SVD降维完成: 原始 {A.shape} -> 降维后 {A_z.shape}")
        _log_mem("SVD 降维后（仅降维）")
        return A, A_z, preprocessing_info
    else:
        # 不使用预处理，保持原有逻辑
        # 对每个latent的激活进行归一化（L2范数归一化，按列）
        # 这样每个latent的激活分布被标准化，便于latent聚类
        latent_norms = np.linalg.norm(A, axis=0, keepdims=True)
        latent_norms = np.clip(latent_norms, 1e-8, None)  # 避免除零
        A_z = A / latent_norms
        
        preprocessing_info = {
            "kept_example_idx": np.arange(A.shape[0]),
            "kept_feature_idx": np.arange(A.shape[1]),
            "labels_by_original_feature": np.arange(A.shape[1]),
            "prep_result": None,
            "filtered_latent_dim": A_z.shape[1],
            "use_svd_reduction": False,
            "use_preprocessing": False
        }
        
        return A, A_z, preprocessing_info

# 3. 构建 latent 层信息和编号

def build_latent_layer_info(n_layers=26, sae_dim=16384, kept_feature_idx=None):
    """
    构建latent层信息
    Args:
        n_layers: 层数 (默认26层)
        sae_dim: 每层SAE维度 (默认16384)
        kept_feature_idx: 筛选后保留的latent索引
    Returns:
        core2layer_latent: latent索引到(层号, 层内索引)的映射
        latent2layer: 每个latent对应的层号
        idx_in_layer: 每个latent在层内的索引
    """
    core2layer_latent = {}
    latent2layer = []
    idx_in_layer = []
    acc = 0
    
    for layer_id in range(n_layers):
        for i in range(sae_dim):
            core2layer_latent[acc] = (layer_id, i)
            latent2layer.append(layer_id)
            idx_in_layer.append(i)
            acc += 1
    
    latent2layer = np.array(latent2layer)
    idx_in_layer = np.array(idx_in_layer)
    
    if kept_feature_idx is not None:
        # 只保留筛选后的latent信息
        latent2layer = latent2layer[kept_feature_idx]
        idx_in_layer = idx_in_layer[kept_feature_idx]
        print(f"筛选后的层信息: {len(latent2layer)} 个latent")
            
    return core2layer_latent, latent2layer, idx_in_layer

# 4. 构建同层邻域

def build_neighbors(latent2layer, idx_in_layer, sae_dim=16384):
    """
    构建同层邻域关系
    Args:
        latent2layer: 每个latent对应的层号
        idx_in_layer: 每个latent在层内的索引
        sae_dim: 每层SAE维度
    Returns:
        neighbors: 每个latent的邻居列表
    """
    neighbors = {}
    for idx, (layer, lid) in enumerate(zip(latent2layer, idx_in_layer)):
        layer_mask = (latent2layer == layer)
        # 在SAE维度上构建邻域（这里简化为相邻索引）
        lid_mask = (idx_in_layer == lid - 1) | (idx_in_layer == lid + 1)
        nb_idx = np.where(layer_mask & lid_mask)[0]
        neighbors[idx] = nb_idx.tolist()
    return neighbors

# 5. GPU加速的gwMRF 聚类主循环

def gwMRF_latent_clustering(A_z, neighbors, latent2layer, idx_in_layer, n_parcels=20, n_iter=10, spatial_weight=0.1, pairwise_weight=1.0, random_cluster_baseline=False):
    """
    优化后的gwMRF聚类主函数
    Args:
        A_z: L2归一化后的样本矩阵
        neighbors: 邻居关系
        latent2layer: 层信息
        idx_in_layer: 层内索引
        n_parcels: 聚类数量
        n_iter: 迭代次数
        spatial_weight: 空间权重
        pairwise_weight: 成对权重
    Returns:
        parcel_assign: 聚类分配结果
        cost_history: 成本变化历史记录
    """
    M = A_z.shape[1]  # latent维度
    C = A_z.shape[0]  # 样本数量

    # 内部小工具：带阶段标记的内存打印，便于定位峰值
    def _log_mem(stage: str):
        """
        在聚类过程中打印带标签的内存使用情况
        """
        print_memory_usage(f"[gwMRF][{stage}] ")
    
    print(f"A_z shape: {A_z.shape}")
    print(f"A_z range: [{A_z.min():.4f}, {A_z.max():.4f}]")
    print(f"A_z has NaN: {np.isnan(A_z).any()}")
    print(f"A_z has Inf: {np.isinf(A_z).any()}")
    _log_mem("初始化后")
    
    # 检查数据规模，如果样本数量太大，使用CPU计算
    if C > 10000:  # 如果样本数量超过10000，使用CPU
        print(f"样本数量 {C} 较大，使用CPU计算以避免GPU内存不足")
        use_gpu = False
    else:
        use_gpu = True
        print(f"样本数量 {C} 较小，使用GPU计算")
    
    if use_gpu:
        # GPU版本
        A_z_tensor = torch.tensor(A_z, dtype=torch.float32, device=device)
        latent2layer_tensor = torch.tensor(latent2layer, dtype=torch.long, device=device)
        parcel_assign = torch.arange(M, device=device) % n_parcels
    else:
        # CPU版本
        A_z_tensor = torch.tensor(A_z, dtype=torch.float32, device='cpu')
        latent2layer_tensor = torch.tensor(latent2layer, dtype=torch.long, device='cpu')
        parcel_assign = torch.arange(M, device='cpu') % n_parcels

    _log_mem("A_z_tensor / parcel_assign 创建后")

    # 确保A_z_tensor的列向量是L2归一化的（真正的cosine相似度）
    A_z_tensor = torch.nn.functional.normalize(A_z_tensor, p=2, dim=0, eps=1e-8)
    _log_mem("A_z_tensor 归一化后")
    
    # 初始化成本历史记录
    cost_history = {
        'epochs': [],
        'cost1_mean': [],
        'cost1_std': [],
        'cost3_mean': [],
        'cost3_std': [],
        'total_cost_mean': [],
        'total_cost_std': [],
        'parcel_distribution': []
    }
    
    for epoch in tqdm(range(n_iter), desc="gwMRF聚类"):
        _log_mem(f"Epoch {epoch} 开始前")
        # 计算每个parcel的平均向量
        parcel_means = []
        for p in range(n_parcels):
            idxs = torch.where(parcel_assign == p)[0]
            if len(idxs) == 0:
                # 修复：空簇处理 - 选择一个未被该簇覆盖的列向量作为均值
                if use_gpu:
                    # 随机选择一个latent作为新中心
                    fallback_idx = torch.randint(0, M, (1,), device=A_z_tensor.device)
                    mean_vec = A_z_tensor[:, fallback_idx].squeeze(1)
                else:
                    fallback_idx = torch.randint(0, M, (1,), device='cpu')
                    mean_vec = A_z_tensor[:, fallback_idx].squeeze(1)
            else:
                mean_vec = A_z_tensor[:, idxs].mean(dim=1)
            parcel_means.append(mean_vec)
        
        _log_mem(f"Epoch {epoch} parcel_means 计算后")

        # 批量计算所有parcel的平均向量并L2归一化
        parcel_means_tensor = torch.stack(parcel_means)  # [n_parcels, C]
        parcel_means_tensor = torch.nn.functional.normalize(parcel_means_tensor, p=2, dim=1, eps=1e-8)
        _log_mem(f"Epoch {epoch} parcel_means_tensor 归一化后")
        
        # 优化：预计算每个parcel的层分布，避免重复计算（这一部分对所有chunk共享）
        max_layer = int(latent2layer_tensor.max().item()) + 1
        layer_bins = torch.arange(max_layer, device=latent2layer_tensor.device)
        
        # 每个parcel只计算一次层分布
        parcel_layer_ratios = []
        for p in range(n_parcels):
            idxs = torch.where(parcel_assign == p)[0]
            if len(idxs) == 0:
                parcel_layer_ratios.append(None)
                continue
            counts = torch.bincount(latent2layer_tensor[idxs], minlength=max_layer).float()
            if counts.sum() > 0:
                parcel_layer_ratios.append(counts / counts.sum())
            else:
                parcel_layer_ratios.append(None)
        
        # 预计算所有latent的层与所有层的距离矩阵（这一矩阵按chunk切片使用）
        this_layers = latent2layer_tensor.float()  # shape [M]
        layer_bins_float = layer_bins.float()
        abs_diff = torch.abs(layer_bins_float[None, :] - this_layers[:, None])  # [M, L]
        
        # 分块计算 cost1 / cost3，并在线更新分配与统计，避免构造完整的 [n_parcels, M] 大矩阵
        chunk_size = 10000  # 每次处理10000个latent
        new_assign_full = torch.empty(M, dtype=torch.long, device=parcel_means_tensor.device)
        # 统计量：用于计算均值和方差
        sum_cost1 = 0.0
        sum_cost1_sq = 0.0
        sum_cost3 = 0.0
        sum_cost3_sq = 0.0
        sum_total = 0.0
        sum_total_sq = 0.0
        # 记录被选中成本的min/max（仅用于epoch 0调试打印）
        chosen_cost1_min = None
        chosen_cost1_max = None
        chosen_cost3_min = None
        chosen_cost3_max = None
        chosen_total_min = None
        chosen_total_max = None
        # 聚类分布统计
        counts_all = torch.zeros(n_parcels, dtype=torch.long, device=parcel_means_tensor.device)
        
        for chunk_start in range(0, M, chunk_size):
            chunk_end = min(chunk_start + chunk_size, M)
            idx_range = torch.arange(chunk_start, chunk_end, device=parcel_means_tensor.device)
            # 相似度成本（真正的cosine）
            similarity_chunk = torch.mm(parcel_means_tensor, A_z_tensor[:, chunk_start:chunk_end])
            if random_cluster_baseline:
                cost1_chunk = torch.rand_like(similarity_chunk)
            else:
                cost1_chunk = 1.0 - similarity_chunk.clamp(min=-1.0, max=1.0)
            
            # 空间成本：利用 abs_diff 的对应切片
            abs_diff_chunk = abs_diff[chunk_start:chunk_end, :]  # [chunk_len, L]
            if use_gpu:
                cost3_chunk = torch.zeros_like(cost1_chunk, device=device)
            else:
                cost3_chunk = torch.zeros_like(cost1_chunk, device='cpu')
            for p in range(n_parcels):
                if parcel_layer_ratios[p] is None:
                    continue
                weighted_chunk = abs_diff_chunk @ parcel_layer_ratios[p]  # [chunk_len]
                cost3_chunk[p] = spatial_weight * (weighted_chunk + 1.0)
            
            # 总成本 & argmin
            if random_cluster_baseline:
                total_cost_chunk = cost1_chunk
            else:
                total_cost_chunk = cost1_chunk + cost3_chunk
            new_assign_chunk = torch.argmin(total_cost_chunk, dim=0)
            new_assign_full[chunk_start:chunk_end] = new_assign_chunk
            
            # 只统计被选中的成本
            idx_local = torch.arange(chunk_end - chunk_start, device=parcel_means_tensor.device)
            chosen_cost1_chunk = cost1_chunk[new_assign_chunk, idx_local]
            chosen_cost3_chunk = cost3_chunk[new_assign_chunk, idx_local]
            chosen_total_chunk = chosen_cost1_chunk + chosen_cost3_chunk
            
            # 更新统计量
            sum_cost1 += chosen_cost1_chunk.sum().item()
            sum_cost1_sq += (chosen_cost1_chunk ** 2).sum().item()
            sum_cost3 += chosen_cost3_chunk.sum().item()
            sum_cost3_sq += (chosen_cost3_chunk ** 2).sum().item()
            sum_total += chosen_total_chunk.sum().item()
            sum_total_sq += (chosen_total_chunk ** 2).sum().item()
            
            # 更新min/max（仅epoch 0需要）
            if epoch == 0:
                c1_min = chosen_cost1_chunk.min().item()
                c1_max = chosen_cost1_chunk.max().item()
                c3_min = chosen_cost3_chunk.min().item()
                c3_max = chosen_cost3_chunk.max().item()
                ct_min = chosen_total_chunk.min().item()
                ct_max = chosen_total_chunk.max().item()
                chosen_cost1_min = c1_min if chosen_cost1_min is None else min(chosen_cost1_min, c1_min)
                chosen_cost1_max = c1_max if chosen_cost1_max is None else max(chosen_cost1_max, c1_max)
                chosen_cost3_min = c3_min if chosen_cost3_min is None else min(chosen_cost3_min, c3_min)
                chosen_cost3_max = c3_max if chosen_cost3_max is None else max(chosen_cost3_max, c3_max)
                chosen_total_min = ct_min if chosen_total_min is None else min(chosen_total_min, ct_min)
                chosen_total_max = ct_max if chosen_total_max is None else max(chosen_total_max, ct_max)
            
            # 更新聚类计数
            counts_chunk = torch.bincount(new_assign_chunk, minlength=n_parcels)
            counts_all += counts_chunk
        
        _log_mem(f"Epoch {epoch} cost1/cost3 分块计算并更新分配后")
        
        # 当前epoch的最终分配
        new_assign = new_assign_full
        
        # 记录成本统计（只看被选中的成本）
        N = float(M)
        cost1_mean = sum_cost1 / N
        cost3_mean = sum_cost3 / N
        total_cost_mean = sum_total / N
        # 方差 = E[x^2] - (E[x])^2
        cost1_var = max(sum_cost1_sq / N - cost1_mean ** 2, 0.0)
        cost3_var = max(sum_cost3_sq / N - cost3_mean ** 2, 0.0)
        total_cost_var = max(sum_total_sq / N - total_cost_mean ** 2, 0.0)
        cost1_std = cost1_var ** 0.5
        cost3_std = cost3_var ** 0.5
        total_cost_std = total_cost_var ** 0.5
        counts = counts_all.detach().cpu().numpy()
        
        # 添加到历史记录
        cost_history['epochs'].append(epoch)
        cost_history['cost1_mean'].append(cost1_mean)
        cost_history['cost1_std'].append(cost1_std)
        cost_history['cost3_mean'].append(cost3_mean)
        cost_history['cost3_std'].append(cost3_std)
        cost_history['total_cost_mean'].append(total_cost_mean)
        cost_history['total_cost_std'].append(total_cost_std)
        cost_history['parcel_distribution'].append([int(x) for x in counts.tolist()])
        
        # 添加调试信息
        if epoch == 0 and chosen_cost1_min is not None:
            print(f"Chosen Cost1 range: [{chosen_cost1_min:.4f}, {chosen_cost1_max:.4f}]")
            print(f"Chosen Cost3 range: [{chosen_cost3_min:.4f}, {chosen_cost3_max:.4f}]")
            print(f"Chosen Total cost range: [{chosen_total_min:.4f}, {chosen_total_max:.4f}]")
            
            # 详细调试信息
            print(f"parcel_means_tensor shape: {parcel_means_tensor.shape}")
            print(f"parcel_means_tensor range: [{parcel_means_tensor.min().item():.4f}, {parcel_means_tensor.max().item():.4f}]")
            print(f"A_z_tensor range: [{A_z_tensor.min().item():.4f}, {A_z_tensor.max().item():.4f}]")
            
            # 检查是否有NaN（这里只检查静态张量，避免额外构造大矩阵）
            print(f"parcel_means_tensor has NaN: {torch.isnan(parcel_means_tensor).any()}")
            print(f"A_z_tensor has NaN: {torch.isnan(A_z_tensor).any()}")
        
        parcel_assign = new_assign
        _log_mem(f"Epoch {epoch} 结束前/清理前")
    
        # 清理内存
        if use_gpu:
            torch.cuda.empty_cache()
        
        # 强制垃圾回收（每5个epoch或最后）
        if epoch % 5 == 0 or epoch == n_iter - 1:
            import gc
            gc.collect()
        
        # 打印当前聚类分布
        if epoch % 5 == 0 or epoch == n_iter - 1:
            counts = torch.bincount(parcel_assign, minlength=n_parcels)
            if use_gpu:
                print(f"Epoch {epoch}: {counts.cpu().numpy()}")
            else:
                print(f"Epoch {epoch}: {counts.numpy()}")
    
    # 最终内存清理
    if use_gpu:
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    if use_gpu:
        return parcel_assign.cpu().numpy(), cost_history
    else:
        return parcel_assign.numpy(), cost_history

# 6. 结果分析与可视化

def analyze_parcels(parcel_assign, A_z, sample_mappings, data_level, n_top=-1):
    """
    分析聚类结果，根据激活强度排序
    Args:
        parcel_assign: 聚类分配结果
        A_z: L2归一化后的样本矩阵
        sample_mappings: 样本到具体内容的映射
        data_level: 数据级别 ("sentence", "example", "dataset")
        n_top: 每个parcel显示的前N个样本
    Returns:
        parcel2topsamples: 每个parcel对应的top样本
    """
    M = len(parcel_assign)
    n_parcels = np.max(parcel_assign) + 1
    parcel2topsamples = {}
    
    # 在gwMRF中，我们总是对latent进行聚类，但analyze_parcels应该分析样本信息
    # 所以我们需要分析每个样本在每个parcel中的激活情况
    print(f"分析样本在parcel中的激活情况: A_z.shape = {A_z.shape}, parcel_assign长度 = {len(parcel_assign)}")
    
    for p in range(n_parcels):
        # 找到属于该parcel的latent
        parcel_latent_idxs = np.where(parcel_assign == p)[0]
        if len(parcel_latent_idxs) == 0:
            continue
            
        # 计算每个样本在该parcel中的平均激活强度
        sample_scores = []
        for sample_idx in range(A_z.shape[0]):  # 遍历所有样本
            if sample_idx < len(sample_mappings):
                sample_info = sample_mappings[sample_idx]
                # 计算该样本在该parcel的latent上的平均激活强度
                parcel_activations = A_z[sample_idx, parcel_latent_idxs]
                avg_activation = np.mean(np.abs(parcel_activations))
                sample_scores.append({
                    "content": sample_info["content"],
                    "dataset": sample_info["dataset"],
                    "type": sample_info["type"],
                    "index": sample_info["index"],
                    "avg_activation": float(avg_activation)
                })
        
        # 按激活强度从大到小排序，取top N
        sorted_samples = sorted(sample_scores, key=lambda x: x["avg_activation"], reverse=True)
        if n_top == -1:
            parcel2topsamples[p] = sorted_samples
        else:
            parcel2topsamples[p] = sorted_samples[:n_top]
        
    return parcel2topsamples

def plot_layer_parcels(parcel_assign, latent2layer, idx_in_layer, out_dir, sae_dim=16384):
    """
    可视化每层的聚类结果
    Args:
        parcel_assign: 聚类分配结果
        latent2layer: 层信息
        idx_in_layer: 层内索引
        out_dir: 输出目录
        sae_dim: 每层SAE维度
    """
    n_layers = np.max(latent2layer) + 1
    
    for l in range(n_layers):
        mask = (latent2layer == l)
        ids = idx_in_layer[mask]
        parcels = parcel_assign[mask]
        
        plt.figure(figsize=(15, 4))
        plt.scatter(ids, parcels, c=parcels, cmap='tab20', s=10, alpha=0.7)
        plt.title(f"Layer {l} SAE latent parcel assignment")
        plt.xlabel("SAE latent index")
        plt.ylabel("parcel")
        plt.colorbar(label='Parcel ID')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"layer{l}_parcel.png"), dpi=150, bbox_inches='tight')
        plt.close()

def plot_cost_history(cost_history, out_dir, save_name="cost_history.png"):
    """
    绘制成本变化历史图（优化版本）
    Args:
        cost_history: 成本历史记录字典
        out_dir: 输出目录
        save_name: 保存文件名
    """
    epochs = cost_history['epochs']
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 总成本变化（被选中的成本）
    axes[0, 0].plot(epochs, cost_history['total_cost_mean'], 'b-', linewidth=2, label='Mean')
    axes[0, 0].fill_between(epochs, 
                            np.array(cost_history['total_cost_mean']) - np.array(cost_history['total_cost_std']),
                            np.array(cost_history['total_cost_mean']) + np.array(cost_history['total_cost_std']),
                            alpha=0.3, color='blue', label='±1 Std')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Cost (Chosen)')
    axes[0, 0].set_title('Total Cost Evolution (Selected Assignments)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. 相似度成本变化（被选中的成本）
    axes[0, 1].plot(epochs, cost_history['cost1_mean'], 'r-', linewidth=2, label='Mean')
    axes[0, 1].fill_between(epochs, 
                            np.array(cost_history['cost1_mean']) - np.array(cost_history['cost1_std']),
                            np.array(cost_history['cost1_mean']) + np.array(cost_history['cost1_std']),
                            alpha=0.3, color='red', label='±1 Std')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Similarity Cost (Cost1)')
    axes[0, 1].set_title('Similarity Cost Evolution (1 - cosine)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3. 空间成本变化（被选中的成本）
    axes[1, 0].plot(epochs, cost_history['cost3_mean'], 'g-', linewidth=2, label='Mean')
    axes[1, 0].fill_between(epochs, 
                            np.array(cost_history['cost3_mean']) - np.array(cost_history['cost3_std']),
                            np.array(cost_history['cost3_mean']) + np.array(cost_history['cost3_std']),
                            alpha=0.3, color='green', label='±1 Std')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Spatial Cost (Cost3)')
    axes[1, 0].set_title('Spatial Cost Evolution (Layer Distribution)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Parcel分布变化
    parcel_distributions = np.array(cost_history['parcel_distribution'])
    for i in range(min(parcel_distributions.shape[1], 10)):  # 最多显示10个parcel
        axes[1, 1].plot(epochs, parcel_distributions[:, i], 
                        label=f'Parcel {i}', alpha=0.7, linewidth=1)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Number of Latents')
    axes[1, 1].set_title('Parcel Distribution Evolution')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"成本变化图已保存到: {os.path.join(out_dir, save_name)}")
    
    # 打印成本变化趋势分析
    if len(epochs) > 1:
        total_cost_trend = cost_history['total_cost_mean'][-1] - cost_history['total_cost_mean'][0]
        cost1_trend = cost_history['cost1_mean'][-1] - cost_history['cost1_mean'][0]
        cost3_trend = cost_history['cost3_mean'][-1] - cost_history['cost3_mean'][0]
        
        print(f"\n成本变化趋势分析:")
        print(f"总成本变化: {total_cost_trend:.4f} ({'下降' if total_cost_trend < 0 else '上升'})")
        print(f"相似度成本变化: {cost1_trend:.4f} ({'下降' if cost1_trend < 0 else '上升'})")
        print(f"空间成本变化: {cost3_trend:.4f} ({'下降' if cost3_trend < 0 else '上升'})")
        
        if total_cost_trend > 0:
            print("⚠️  警告：总成本在上升，可能需要调整参数（如降低spatial_weight）")
        else:
            print("✅ 总成本在下降，聚类优化正常")

def load_parcel_assignments(result_dir):
    """
    加载保存的聚类分配结果
    Args:
        result_dir: 结果目录路径
    Returns:
        parcel_assign: 聚类分配数组
        assignments_info: 分配信息字典
    """
    # 加载numpy数组
    parcel_assign = np.load(os.path.join(result_dir, "parcel_assignments.npy"))
    
    # 加载JSON信息
    with open(os.path.join(result_dir, "latent_parcel_assignments.json"), 'r', encoding='utf-8') as f:
        assignments_info = json.load(f)
    
    return parcel_assign, assignments_info

def fast_silhouette_score(X, labels, sample_size=10000, n_trials=5, random_state=42):
    """
    快速计算silhouette_score，通过多次采样取平均提高准确性
    Args:
        X: 特征矩阵 [n_samples, n_features]
        labels: 聚类标签
        sample_size: 采样数量
        n_trials: 采样试验次数
        random_state: 随机种子
    Returns:
        silhouette_score: 轮廓系数（多次采样的平均值）
    """
    n_samples = X.shape[0]
    
    # 如果样本数太大，进行多次采样
    if n_samples > sample_size:
        print(f"样本数 {n_samples} 过大，进行 {n_trials} 次采样，每次 {sample_size} 个样本")
        scores = []
        
        for trial in range(n_trials):
            # 每次使用不同的随机种子
            random.seed(random_state + trial)
            sample_indices = random.sample(range(n_samples), sample_size)
            X_sampled = X[sample_indices]
            labels_sampled = labels[sample_indices]
            
            try:
                score = silhouette_score(X_sampled, labels_sampled)
                scores.append(score)
                print(f"  试验 {trial + 1}/{n_trials}: score={score:.4f}")
            except Exception as e:
                print(f"  试验 {trial + 1}/{n_trials} 失败: {e}")
                continue
        
        if scores:
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
            print(f"  平均score: {mean_score:.4f} ± {std_score:.4f}")
            return mean_score
        else:
            print("所有试验都失败了")
            return None
    else:
        # 样本数不大，直接计算
        try:
            score = silhouette_score(X, labels)
            return float(score)  # 确保返回Python float类型
        except Exception as e:
            print(f"silhouette_score计算失败: {e}")
            return None

def alternative_clustering_metrics(X, labels, sample_size=10000, n_trials=3, random_state=42):
    """
    计算替代的聚类评估指标，通过采样加速计算
    Args:
        X: 特征矩阵
        labels: 聚类标签
        sample_size: 采样数量
        n_trials: 采样试验次数
        random_state: 随机种子
    Returns:
        metrics: 包含多个评估指标的字典
    """
    n_samples = X.shape[0]
    
    # 如果样本数太大，进行采样
    if n_samples > sample_size:
        print(f"样本数 {n_samples} 过大，采样 {sample_size} 个样本计算替代指标")
        
        all_metrics = []
        for trial in range(n_trials):
            random.seed(random_state + trial)
            sample_indices = random.sample(range(n_samples), sample_size)
            X_sampled = X[sample_indices]
            labels_sampled = labels[sample_indices]
            
            metrics = _compute_clustering_metrics(X_sampled, labels_sampled)
            all_metrics.append(metrics)
            print(f"  试验 {trial + 1}/{n_trials}: WSS={metrics['wss']:.2e}, 质量比={metrics['cluster_quality_ratio']:.3f}")
        
        # 取平均值
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[key] = float(np.mean(values))  # 确保转换为Python float类型
        
        print(f"  平均指标: WSS={final_metrics['wss']:.2e}, 质量比={final_metrics['cluster_quality_ratio']:.3f}")
        return final_metrics
    else:
        # 样本数不大，直接计算
        return _compute_clustering_metrics(X, labels)

def debug_clustering_parameters(A_z, neighbors, latent2layer, idx_in_layer, n_parcels=20, n_iter=5, random_cluster_baseline=False):
    """
    调试聚类参数，快速测试不同配置的效果
    Args:
        A_z: 样本矩阵
        neighbors: 邻居关系
        latent2layer: 层信息
        idx_in_layer: 层内索引
        n_parcels: 聚类数量
        n_iter: 迭代次数
    Returns:
        results: 不同参数配置的结果
    """
    print("🔍 开始调试聚类参数...")
    
    # 测试不同的空间权重
    spatial_weights = [0.001, 0.01, 0.1, 0.5, 1.0]
    results = {}
    
    for spatial_weight in spatial_weights:
        print(f"\n测试 spatial_weight = {spatial_weight}")
        
        # 运行短时间的聚类
        parcel_assign, cost_history = gwMRF_latent_clustering(
            A_z, neighbors, latent2layer, idx_in_layer,
            n_parcels=n_parcels, n_iter=n_iter,
            spatial_weight=spatial_weight, pairwise_weight=1.0,
            random_cluster_baseline=random_cluster_baseline
        )
        
        # 计算最终成本
        final_total_cost = cost_history['total_cost_mean'][-1]
        final_cost1 = cost_history['cost1_mean'][-1]
        final_cost3 = cost_history['cost3_mean'][-1]
        
        # 计算成本变化趋势
        total_trend = cost_history['total_cost_mean'][-1] - cost_history['total_cost_mean'][0]
        cost1_trend = cost_history['cost1_mean'][-1] - cost_history['cost1_mean'][0]
        cost3_trend = cost_history['cost3_mean'][-1] - cost_history['cost3_mean'][0]
        
        results[spatial_weight] = {
            'final_total_cost': float(final_total_cost),
            'final_cost1': float(final_cost1),
            'final_cost3': float(final_cost3),
            'total_trend': float(total_trend),
            'cost1_trend': float(cost1_trend),
            'cost3_trend': float(cost3_trend),
            # 不保存完整的cost_history以避免JSON序列化问题
            'cost_history_summary': {
                'epochs': cost_history['epochs'],
                'final_total_cost': cost_history['total_cost_mean'][-1],
                'final_cost1': cost_history['cost1_mean'][-1],
                'final_cost3': cost_history['cost3_mean'][-1]
            }
        }
        
        print(f"  最终总成本: {final_total_cost:.4f}")
        print(f"  最终相似度成本: {final_cost1:.4f}")
        print(f"  最终空间成本: {final_cost3:.4f}")
        print(f"  总成本趋势: {total_trend:.4f} ({'下降' if total_trend < 0 else '上升'})")
        print(f"  相似度成本趋势: {cost1_trend:.4f} ({'下降' if cost1_trend < 0 else '上升'})")
        print(f"  空间成本趋势: {cost3_trend:.4f} ({'下降' if cost3_trend < 0 else '上升'})")
    
    # 分析结果
    print(f"\n📊 参数调试结果分析:")
    best_spatial_weight = min(spatial_weights, key=lambda w: results[w]['final_total_cost'])
    print(f"最佳空间权重: {best_spatial_weight} (总成本: {results[best_spatial_weight]['final_total_cost']:.4f})")
    
    # 检查是否有成本下降的趋势
    improving_weights = [w for w in spatial_weights if results[w]['total_trend'] < 0]
    if improving_weights:
        print(f"成本下降的权重: {improving_weights}")
    else:
        print("⚠️  警告：所有权重都导致成本上升，可能需要调整其他参数")
    
    return results

def _compute_clustering_metrics(X, labels, random_state=42):
    """
    计算聚类指标的核心函数
    Args:
        X: 特征矩阵
        labels: 聚类标签
        random_state: 随机种子
    Returns:
        metrics: 包含多个评估指标的字典
    """
    metrics = {}
    
    # 1. 计算聚类内平方和 (WSS) - 肘部法则的基础
    unique_labels = np.unique(labels)
    wss = 0
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            wss += np.sum(np.linalg.norm(cluster_points - centroid, axis=1) ** 2)
    metrics['wss'] = float(wss)
    
    # 2. 计算聚类间距离
    centroids = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 0:
            centroids.append(np.mean(cluster_points, axis=0))
    
    if len(centroids) > 1:
        # 计算聚类间距离矩阵
        centroid_distances = euclidean_distances(centroids)
        # 平均聚类间距离
        inter_cluster_distance = np.mean(centroid_distances[np.triu_indices_from(centroid_distances, k=1)])
        metrics['inter_cluster_distance'] = float(inter_cluster_distance)
    else:
        metrics['inter_cluster_distance'] = 0.0
    
    # 3. 计算聚类大小分布
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    metrics['cluster_size_std'] = float(np.std(cluster_sizes))
    metrics['cluster_size_cv'] = float(np.std(cluster_sizes) / np.mean(cluster_sizes))  # 变异系数
    
    # 4. 计算聚类内平均距离（采样计算以提高速度）
    intra_cluster_distances = []
    for label in unique_labels:
        cluster_points = X[labels == label]
        if len(cluster_points) > 1:
            # 如果聚类太大，采样计算距离
            if len(cluster_points) > 1000:
                sample_size = min(1000, len(cluster_points))
                random.seed(random_state)
                sample_indices = random.sample(range(len(cluster_points)), sample_size)
                cluster_points_sampled = cluster_points[sample_indices]
                distances = euclidean_distances(cluster_points_sampled)
            else:
                distances = euclidean_distances(cluster_points)
            
            # 取上三角矩阵的平均值
            upper_tri = distances[np.triu_indices_from(distances, k=1)]
            intra_cluster_distances.append(float(np.mean(upper_tri)))
    
    if intra_cluster_distances:
        metrics['intra_cluster_distance'] = float(np.mean(intra_cluster_distances))
    else:
        metrics['intra_cluster_distance'] = 0.0
    
    # 5. 计算聚类质量指标 (inter/intra ratio)
    if metrics['intra_cluster_distance'] > 0:
        metrics['cluster_quality_ratio'] = float(metrics['inter_cluster_distance'] / metrics['intra_cluster_distance'])
    else:
        metrics['cluster_quality_ratio'] = 0.0
    
    return metrics

def ensure_json_serializable(obj):
    """
    确保对象是JSON可序列化的，将numpy类型转换为Python原生类型
    Args:
        obj: 要检查的对象
    Returns:
        json_serializable_obj: JSON可序列化的对象
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None:
        return None
    else:
        return obj

def gini_col(col: np.ndarray) -> float:
    """
    计算列的Gini系数（选择性指标）
    Args:
        col: 一维数组
    Returns:
        gini: Gini系数，数值越大越"挑"
    """
    C = len(col)
    if C == 0 or np.all(col == 0):
        return 0.0
    # 集中度（选择性）指标：数值越大越"挑"
    num = (np.sum(np.abs(col)) / C) ** 2
    den = (np.sum(col ** 2) / C)
    return 1.0 - (num / den)

def reduce_matrix_dimension_with_svd(
    A: np.ndarray,
    *,
    target_variance: float = 0.8,
    random_state: int = 42,
    use_sparse: bool = True,
    load_cached: bool = False,
    cache_dir: Optional[str] = None,
    max_fit_components: int = 5000,
    model_name: str = "llama_8b_pt",
) -> Dict[str, Any]:
    """
    使用TruncatedSVD按累计解释方差阈值自动选择维度，并支持结果缓存。
    Args:
        A: 原始矩阵 (N, M) - N个样本，M个latent
        target_variance: 目标累计解释方差阈值，例如0.8表示保留80%的信息
        random_state: 随机种子
        use_sparse: 是否使用稀疏矩阵（如果A很稀疏）
        load_cached: 是否尝试从缓存加载已降维结果
        cache_dir: 缓存结果存放目录
        max_fit_components: 为计算累计方差而拟合的最大主成分数上限
    Returns:
        svd_info: 包含降维结果与元信息的字典
    """
    print(f"开始SVD降维: 原始矩阵形状 {A.shape} -> 按累计方差阈值 {target_variance}")
    print_memory_usage("SVD降维前")

    if cache_dir is None:
        cache_dir = str(output_path("neurocogmap_construction", "svd_cache"))
    os.makedirs(cache_dir, exist_ok=True)

    N, M = A.shape
    print(f"原始矩阵: {N} 个样本, {M} 个latent")

    # 生成稳定的轻量指纹用于缓存键（仅取左上角子块避免高开销）
    sub_N = min(16, N)
    sub_M = min(16, M)
    try:
        sub_block = A[:sub_N, :sub_M]
        if scipy.sparse.issparse(sub_block):
            sub_vals = sub_block.toarray().astype(np.float64, copy=False)
        else:
            sub_vals = np.asarray(sub_block, dtype=np.float64)
        mean = float(np.round(sub_vals.mean(dtype=np.float64), 8))
        std = float(np.round(sub_vals.std(dtype=np.float64), 8))
        fingerprint = f"{mean}_{std}"
    except Exception:
        fingerprint = f"N{N}_M{M}"

    # 规范化阈值字符串用于路径
    tv_str = f"{target_variance:.3f}".replace(".", "p")
    fp_hex = hashlib.blake2s(fingerprint.encode("utf-8"), digest_size=4).hexdigest()
    cache_name = f"svd_N{N}_M{M}_var{tv_str}_rs{random_state}_sp{int(bool(use_sparse))}_fp{fp_hex}_model{model_name}"
    cache_path = os.path.join(cache_dir, cache_name)

    # 尝试加载缓存
    if load_cached and os.path.isdir(cache_path):
        info_file = os.path.join(cache_path, "svd_info.json")
        az_file = os.path.join(cache_path, "A_z_reduced.npy")
        comps_file = os.path.join(cache_path, "svd_components.npy")
        sv_file = os.path.join(cache_path, "svd_singular_values.npy")
        try:
            print(f"尝试从缓存加载SVD结果: {cache_path}")
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
            A_z_reduced = np.load(az_file)
            components = np.load(comps_file)
            singular_values = np.load(sv_file)
            print(f"已从缓存加载：降维后 {A_z_reduced.shape}, 解释度 {info['total_variance_explained']:.4f}")
            return {
                "A_z_reduced": A_z_reduced,
                "feature_embeddings": components.T * singular_values,  # (M, K)
                "svd_model": None,
                "components": components,
                "singular_values": singular_values,
                "explained_variance_ratio": np.array(info["explained_variance_ratio"]),
                "total_variance_explained": float(info["total_variance_explained"]),
                "original_shape": tuple(info["original_shape"]),
                "reduced_shape": tuple(info["reduced_shape"]),
                "config": {
                    "target_variance": float(info["config"]["target_variance"]),
                    "random_state": int(info["config"]["random_state"]),
                    "use_sparse": bool(info["config"]["use_sparse"]),
                    "chosen_n_components": int(info["config"]["chosen_n_components"]),
                    "cache_path": cache_path,
                },
            }
        except Exception as e:
            print(f"缓存加载失败，将重新计算: {e}")

    # 检查是否需要转换为稀疏矩阵
    if use_sparse and scipy.sparse.issparse(A):
        print("使用稀疏矩阵进行SVD")
        A_sparse = A
    elif use_sparse and (A == 0).sum() / A.size > 0.5:  # 如果超过50%是零
        print("检测到稀疏数据，转换为稀疏矩阵")
        A_sparse = scipy.sparse.csr_matrix(A)
    else:
        print("使用密集矩阵进行SVD")
        A_sparse = A

    # 设置拟合的上限主成分数
    fit_n_components = max(2, min(M - 1, max_fit_components))
    print(f"拟合TruncatedSVD以获取累计方差，拟合上限维度: {fit_n_components}")
    svd = TruncatedSVD(n_components=fit_n_components, random_state=random_state)
    svd.fit(A_sparse)

    # 依据累计解释方差选择最小K
    evr = svd.explained_variance_ratio_
    cumsum = np.cumsum(evr)
    chosen_k = int(np.searchsorted(cumsum, target_variance, side='left') + 1)
    if chosen_k > fit_n_components:
        print(f"警告：目标方差 {target_variance} 超出当前拟合上限，使用上限维度 {fit_n_components}。")
        chosen_k = fit_n_components
    total_variance_explained = float(cumsum[chosen_k - 1])

    # 计算列（latent）的K维表示：V_K * Σ_K
    feature_embeddings_full = svd.components_.T * svd.singular_values_  # (M, fit_n_components)
    feature_embeddings = feature_embeddings_full[:, :chosen_k]  # (M, K)
    print(f"列嵌入形状: {feature_embeddings.shape} (M={M}, K={chosen_k})")

    # 转置为gwMRF期望的形状：(K, M) - 每列是一个latent的K维表示
    A_z_reduced = feature_embeddings.T  # (K, M)
    print(f"降维后矩阵形状: {A_z_reduced.shape} (K={chosen_k}, M={M})")

    # L2归一化每个列向量（每个latent的表示）
    A_z_reduced = A_z_reduced / np.maximum(
        np.linalg.norm(A_z_reduced, axis=0, keepdims=True), 1e-8
    )
    print(f"L2归一化完成")

    # 打印前10个主成分的方差解释比例（截到K）
    print(f"信息保留度: {total_variance_explained:.4f} ({total_variance_explained*100:.2f}%)")
    print(f"前10个主成分的方差解释比例:")
    for i in range(min(10, chosen_k)):
        print(f"  主成分 {i+1}: {evr[i]:.4f} ({evr[i]*100:.2f}%)")

    print_memory_usage("SVD降维后")

    # 保存到缓存
    try:
        os.makedirs(cache_path, exist_ok=True)
        info = {
            "target_variance": float(target_variance),
            "total_variance_explained": total_variance_explained,
            "explained_variance_ratio": evr[:chosen_k].tolist(),
            "original_shape": [int(N), int(M)],
            "reduced_shape": [int(chosen_k), int(M)],
            "config": {
                "target_variance": float(target_variance),
                "random_state": int(random_state),
                "use_sparse": bool(use_sparse),
                "chosen_n_components": int(chosen_k),
            }
        }
        with open(os.path.join(cache_path, "svd_info.json"), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(cache_path, "A_z_reduced.npy"), A_z_reduced)
        np.save(os.path.join(cache_path, "svd_components.npy"), svd.components_[:chosen_k, :])
        np.save(os.path.join(cache_path, "svd_singular_values.npy"), svd.singular_values_[:chosen_k])
        print(f"SVD降维结果已缓存到: {cache_path}")
    except Exception as e:
        print(f"缓存保存失败: {e}")

    return {
        "A_z_reduced": A_z_reduced,  # (K, M)
        "feature_embeddings": feature_embeddings,  # (M, K)
        "svd_model": svd,
        "components": svd.components_[:chosen_k, :],  # (K, M)
        "singular_values": svd.singular_values_[:chosen_k],  # (K,)
        "explained_variance_ratio": evr[:chosen_k],
        "total_variance_explained": total_variance_explained,
        "original_shape": A.shape,
        "reduced_shape": A_z_reduced.shape,
        "config": {
            "target_variance": target_variance,
            "random_state": random_state,
            "use_sparse": use_sparse,
            "chosen_n_components": chosen_k,
            "cache_path": cache_path,
        }
    }

def preprocess_sae_for_feature_clustering(
    A: np.ndarray,
    *,
    nz_eps: float = 1e-5,
    min_activation_rate: float = 0.02,   # 2%
    min_var: float = 1e-6,
    drop_row_low_sum_quantile: float = 5.0,  # 去掉激活和处于底部 5% 的行
    use_gini: bool = True,
    gini_keep_quantile: float = 0.8,     # 保留 Gini 位于后 80%（中高选择性）
    standardize: str = "l2",         # 'zscore' | 'l2' | 'none'
    skip_first_col_filter: bool = False,  # 若为 True，则假设外部已按 (1536-1542) 做过列筛选，这里不再重复
) -> Dict[str, Any]:
    """
    返回用于列聚类的矩阵与索引映射。
    输出中的矩阵形状均为"样本×特征向量维"（即：特征作为样本，所以是 F'×N'）
    这样可直接喂给大多数聚类器（按"行样本"聚类）。
    """
    
    assert A.ndim == 2, "A must be 2D (N_examples x F_features)"
    N, F = A.shape
    print(f"原始矩阵形状: {A.shape}")

    # —— 1) 初步列筛选：激活率 + 方差
    col_nonzero_ratio = (A > nz_eps).mean(axis=0)     # (F,)
    col_var = A.var(axis=0)                           # (F,)

    if skip_first_col_filter:
        # 已在外部按相同规则筛过一次，这里只做统计但不再丢列
        kept_feature_idx = np.arange(F)
        A1 = A
        print(f"跳过第一步列筛选（外部已完成），当前列数: {F}")
    else:
        keep_cols_mask = (col_nonzero_ratio > min_activation_rate) & (col_var > min_var)
        kept_feature_idx = np.where(keep_cols_mask)[0]
        A1 = A[:, kept_feature_idx]                       # (N, F1)
        print(f"列筛选后: {A1.shape}, 保留 {len(kept_feature_idx)}/{F} 个latent")

    # —— 2) 行筛选：去掉总激活极低的行
    row_sum = A1.sum(axis=1)                          # (N,)
    thresh = np.percentile(row_sum, drop_row_low_sum_quantile)
    keep_rows_mask = row_sum > thresh
    kept_example_idx = np.where(keep_rows_mask)[0]
    A2 = A1[kept_example_idx, :]                      # (N', F1)
    print(f"行筛选后: {A2.shape}, 保留 {len(kept_example_idx)}/{N} 个样本")

    # —— 3) 选择性筛选（Gini）
    if use_gini and A2.shape[1] > 0:
        gini_scores = np.apply_along_axis(gini_col, 0, A2)  # (F1,)
        gq = np.percentile(gini_scores, 100 * gini_keep_quantile)
        keep_cols2_mask = gini_scores >= gq
        kept_feature_idx = kept_feature_idx[keep_cols2_mask]
        A3 = A2[:, keep_cols2_mask]                  # (N', F')
        print(f"Gini筛选后: {A3.shape}, 保留 {len(kept_feature_idx)} 个latent")
    else:
        A3 = A2                                      # (N', F')

    # —— 4) 标准化（面向"列聚类"，故把每一列视为一个样本向量：先转置再标）
    # 我们把特征作为样本：X_feat = (F' x N')，每行是一个 latent 的"激活轮廓"
    X_feat = A3.T  # (F', N')

    scaler = None
    if standardize == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_feat = scaler.fit_transform(X_feat)
    elif standardize == "l2":
        X_feat = normalize(X_feat, norm="l2", axis=1)
    elif standardize == "none":
        pass
    else:
        raise ValueError("standardize must be one of {'zscore','l2','none'}")

    print(f"最终预处理矩阵形状: {X_feat.shape} (latent x samples)")

    return {
        # 用于聚类的矩阵（样本=特征）：F' x N'
        "X_feat": X_feat,
        # 保留的原始索引映射
        "kept_example_idx": kept_example_idx,  # 映射到原始 example 索引（行）
        "kept_feature_idx": kept_feature_idx,  # 映射到原始 latent 索引（列）
        # 过程中的统计（可选）
        "col_nonzero_ratio": col_nonzero_ratio,
        "col_var": col_var,
        "gini_scores": gini_scores if use_gini and A2.shape[1] > 0 else None,
        "standardizer": scaler,
        "config": {
            "nz_eps": nz_eps,
            "min_activation_rate": min_activation_rate,
            "min_var": min_var,
            "drop_row_low_sum_quantile": drop_row_low_sum_quantile,
            "use_gini": use_gini,
            "gini_keep_quantile": gini_keep_quantile,
            "standardize": standardize,
        },
    }

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='gwMRF latent clustering with different data levels')
    parser.add_argument('--output_root', type=str,
                       default=str(output_path("neurocogmap_construction", "qa_sae_output")),
                       help='输入数据目录路径')
    parser.add_argument('--out_dir', type=str,
                       default=str(output_path("neurocogmap_construction", "cluster_output")),
                       help='输出目录路径')
    parser.add_argument('--data_level', type=str, default="sentence",
                       choices=["sentence", "example", "dataset"],
                       help='数据级别: sentence, example, dataset')
    parser.add_argument('--n_parcels', type=int, default=20,
                       help='聚类数量')
    parser.add_argument('--n_iter', type=int, default=10,
                       help='迭代次数')
    parser.add_argument('--spatial_weight', type=float, default=0.001,
                       help='空间权重（建议使用较小的值，如0.01-0.1）')
    parser.add_argument('--pairwise_weight', type=float, default=1.0,
                       help='成对权重')
    parser.add_argument('--test_mode', action='store_true',
                       help='测试模式，只加载前3个数据集进行调试')
    parser.add_argument('--auto_n_parcels', action='store_true',
                       help='自动遍历多组n_parcels并计算silhouette_score')
    parser.add_argument('--n_parcels_range', type=str, default='10,20,30,40,50',
                       help='自动模式下遍历的n_parcels列表，用逗号分隔')
    parser.add_argument('--use_preprocessing', action='store_true',
                       help='是否使用预处理筛选')
    parser.add_argument('--min_activation_rate', type=float, default=0.02,
                       help='最小激活率阈值')
    parser.add_argument('--min_var', type=float, default=1e-6,
                       help='最小方差阈值')
    parser.add_argument('--gini_keep_quantile', type=float, default=0.3,
                       help='Gini筛选保留分位数')
    parser.add_argument('--drop_row_low_sum_quantile', type=float, default=5.0,
                       help='行筛选分位数')
    parser.add_argument('--debug_params', action='store_true',
                       help='启用参数调试模式，快速测试不同spatial_weight的效果')
    parser.add_argument('--use_svd_reduction', action='store_true',
                       help='是否使用SVD降维')
    # 替换原来的手动维度设置为按累计方差阈值与缓存选项
    parser.add_argument('--svd_target_variance', type=float, default=0.8,
                       help='SVD累计解释方差阈值，例如0.8表示保留80%%的信息')
    parser.add_argument('--svd_random_state', type=int, default=42,
                       help='SVD降维的随机种子')
    parser.add_argument('--svd_use_sparse', action='store_true', default=True,
                       help='SVD降维是否使用稀疏矩阵')
    parser.add_argument('--svd_load_cached', action='store_true',
                       help='是否优先从缓存加载已降维结果')
    parser.add_argument('--svd_cache_dir', type=str,
                       default=str(output_path("neurocogmap_construction", "svd_cache")),
                       help='SVD降维结果缓存目录')
    parser.add_argument('--random_cluster_baseline', action='store_true',
                       help='是否使用随机簇基线')
    parser.add_argument('--sae_dim', type=int, default=16384,
                       help='SAE维度')
    parser.add_argument('--n_layer', type=int, default=26,
                       help='层数')
    parser.add_argument('--model_name', type=str, default="llama_8b_pt",
                       help='模型名称')
    args = parser.parse_args()

    # 打印使用说明
    print("🚀 gwMRF Latent Clustering 优化版本")
    print("=" * 50)
    print("主要优化:")
    print("1. 修复了成本监控指标（只统计被选中的成本）")
    print("2. 改进了空簇处理（使用真实向量而非随机噪声）")
    print("3. 优化了空间成本计算（避免重复计算）")
    print("4. 使用真正的cosine相似度（L2归一化）")
    print("5. 改进了成本函数（1 - cosine 而非 -cosine）")
    print("6. 新增SVD降维功能，支持按累计方差阈值自动选维并缓存结果")
    print("=" * 50)

    # 参数验证和建议
    if args.spatial_weight > 0.5:
        print(f"⚠️  警告：spatial_weight={args.spatial_weight} 可能过大，建议使用 0.01-0.1 范围")
        print("   如果成本持续上升，请尝试降低spatial_weight或使用 --debug_params")

    if args.n_iter < 5:
        print(f"⚠️  警告：n_iter={args.n_iter} 可能过少，建议至少10次迭代")

    print("💡 使用建议:")
    print("   - 首次运行建议使用 --debug_params 找到最佳spatial_weight")
    print("   - 如果成本上升，降低spatial_weight或增加n_iter")
    print("   - 使用 --test_mode 在小数据集上快速测试")
    print("   - 对于高维数据，建议使用 --use_svd_reduction 进行降维")
    print("   - 可使用 --svd_target_variance 指定信息保留比例，并用 --svd_load_cached 复用缓存")
    print("=" * 50)

    output_root = args.output_root
    data_level = args.data_level
    SAE_DIM = args.sae_dim
    N_LAYERS = args.n_layer
    # 加载所有数据集的激活数据
    capabilities, cap2vecs, latent_dim, level_mappings, sample_mappings = load_all_activations(output_root, data_level, args.test_mode)
    # 构建输出路径，包含重要参数
    path_components = []

    # 基础标识
    if args.test_mode:
        path_components.append("test")

    # 数据级别
    path_components.append(data_level)

    # 预处理参数
    if args.use_preprocessing:
        path_components.append(f"prep{args.min_activation_rate}_{args.gini_keep_quantile}")

    # SVD降维参数（使用方差阈值标识）
    if args.use_svd_reduction:
        tv_str = f"{args.svd_target_variance:.2f}".replace(".", "p")
        path_components.append(f"svdvar{tv_str}")

    # 聚类参数
    path_components.append(f"parcels{args.n_parcels}")
    path_components.append(f"iter{args.n_iter}")
    path_components.append(f"spatial{args.spatial_weight}")

    # 组合路径
    out_dir_base = os.path.join(args.out_dir, f"clustering_results_{'_'.join(path_components)}")
    os.makedirs(out_dir_base, exist_ok=True)
    save_level_mappings(level_mappings, out_dir_base)

    # 构建预处理配置
    preprocessing_config = None
    if args.use_preprocessing:
        preprocessing_config = {
            "min_activation_rate": args.min_activation_rate,
            "min_var": args.min_var,
            "gini_keep_quantile": args.gini_keep_quantile,
            "drop_row_low_sum_quantile": args.drop_row_low_sum_quantile,
            "use_gini": True,
            "standardize": "l2"
        }
        print(f"使用预处理筛选，配置: {preprocessing_config}")

    # 构建SVD降维配置（按方差阈值 + 缓存）
    svd_config = None
    if args.use_svd_reduction:
        svd_config = {
            "target_variance": args.svd_target_variance,
            "random_state": args.svd_random_state,
            "use_sparse": args.svd_use_sparse,
            "load_cached": args.svd_load_cached,
            "cache_dir": args.svd_cache_dir,
        }
        print(f"使用SVD降维，配置: {svd_config}")

    # 构建样本矩阵
    A, A_z, preprocessing_info = build_capability_matrix(
        capabilities, cap2vecs, latent_dim,
        use_preprocessing=args.use_preprocessing,
        preprocessing_config=preprocessing_config,
        use_svd_reduction=args.use_svd_reduction,
        svd_config=svd_config,
        model_name=args.model_name
    )

    # 构建latent层信息 (26层，每层SAE_DIM维)
    # 根据预处理和SVD降维的组合情况决定latent筛选
    if args.use_preprocessing:
        # 如果使用了预处理筛选，使用筛选后的latent索引
        kept_feature_idx = preprocessing_info["kept_feature_idx"]
    elif args.use_svd_reduction:
        # 如果只使用SVD降维，保留所有latent
        kept_feature_idx = None
    else:
        # 都不使用，保留所有latent
        kept_feature_idx = None

    core2layer_latent, latent2layer, idx_in_layer = build_latent_layer_info(
        n_layers=N_LAYERS, sae_dim=SAE_DIM, kept_feature_idx=kept_feature_idx
    )
    # 构建邻居关系
    neighbors = build_neighbors(latent2layer, idx_in_layer, sae_dim=SAE_DIM)
    
    if args.auto_n_parcels:
        n_parcels_list = [int(x) for x in args.n_parcels_range.split(',')]
        silhouette_dict = {}
        
        print(f"开始自动模式，将测试 {len(n_parcels_list)} 个n_parcels: {n_parcels_list}")
        print_memory_usage("初始")
        
        for n_parcels in n_parcels_list:
            print(f"\n=== 聚类 n_parcels={n_parcels} ===")
            print_memory_usage("开始前")
            out_dir = out_dir_base + f"_nparcels{n_parcels}"
            os.makedirs(out_dir, exist_ok=True)
            
            # 执行聚类
            parcel_assign, cost_history = gwMRF_latent_clustering(
                A_z, neighbors, latent2layer, idx_in_layer, 
                n_parcels=n_parcels, n_iter=args.n_iter,
                spatial_weight=args.spatial_weight, pairwise_weight=args.pairwise_weight, 
                random_cluster_baseline=args.random_cluster_baseline
            )
            
            # 保存聚类分配结果
            parcel_assignments = {
                "latent_to_parcel": {},
                "parcel_to_latents": {},
                "total_latents": len(parcel_assign),
                "n_parcels": n_parcels,
                "data_level": data_level
            }
            
            # 根据预处理和SVD降维的组合情况保存结果
            if args.use_preprocessing and args.use_svd_reduction:
                # 筛选+降维：需要映射回原始latent空间
                kept_feature_idx = preprocessing_info["kept_feature_idx"]
                labels_by_original_feature = preprocessing_info["labels_by_original_feature"]
                
                # 将聚类结果映射回原始latent空间
                original_parcel_assign = np.full(shape=latent_dim, fill_value=-1, dtype=int)
                for i, parcel_id in enumerate(parcel_assign):
                    original_latent_id = kept_feature_idx[i]
                    original_parcel_assign[original_latent_id] = parcel_id
                
                # 保存原始latent空间的分配结果
                for latent_id, parcel_id in enumerate(original_parcel_assign):
                    if parcel_id != -1:  # 只保存参与聚类的latent
                        parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
                
                for parcel_id in range(n_parcels):
                    latents_in_parcel = np.where(original_parcel_assign == parcel_id)[0]
                    parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
                
                # 保存预处理信息
                preprocessing_file = os.path.join(out_dir, "preprocessing_info.json")
                with open(preprocessing_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "kept_feature_idx": kept_feature_idx.tolist(),
                        "kept_example_idx": preprocessing_info["kept_example_idx"].tolist(),
                        "labels_by_original_feature": labels_by_original_feature.tolist(),
                        "filtered_latent_dim": preprocessing_info["filtered_latent_dim"]
                    }, f, ensure_ascii=False, indent=2)
                
                # 保存SVD降维信息
                svd_file = os.path.join(out_dir, "svd_info.json")
                with open(svd_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "target_variance": preprocessing_info["svd_result"]["config"]["target_variance"],
                        "chosen_n_components": preprocessing_info["svd_result"]["config"]["chosen_n_components"],
                        "random_state": preprocessing_info["svd_result"]["config"]["random_state"],
                        "use_sparse": preprocessing_info["svd_result"]["config"]["use_sparse"],
                        "cache_path": preprocessing_info["svd_result"]["config"].get("cache_path", None),
                        "total_variance_explained": preprocessing_info["svd_result"]["total_variance_explained"],
                        "explained_variance_ratio": preprocessing_info["svd_result"]["explained_variance_ratio"].tolist(),
                        "original_shape": preprocessing_info["svd_result"]["original_shape"],
                        "reduced_shape": preprocessing_info["svd_result"]["reduced_shape"]
                    }, f, ensure_ascii=False, indent=2)
                
                np.save(os.path.join(out_dir, "parcel_assignments.npy"), original_parcel_assign)
                np.save(os.path.join(out_dir, "filtered_parcel_assignments.npy"), parcel_assign)
                np.save(os.path.join(out_dir, "svd_components.npy"), preprocessing_info["svd_result"]["components"])
                np.save(os.path.join(out_dir, "svd_singular_values.npy"), preprocessing_info["svd_result"]["singular_values"])
                
            elif args.use_preprocessing:
                # 只使用预处理筛选
                kept_feature_idx = preprocessing_info["kept_feature_idx"]
                labels_by_original_feature = preprocessing_info["labels_by_original_feature"]
                
                # 将聚类结果映射回原始latent空间
                original_parcel_assign = np.full(shape=latent_dim, fill_value=-1, dtype=int)
                for i, parcel_id in enumerate(parcel_assign):
                    original_latent_id = kept_feature_idx[i]
                    original_parcel_assign[original_latent_id] = parcel_id
                
                # 保存原始latent空间的分配结果
                for latent_id, parcel_id in enumerate(original_parcel_assign):
                    if parcel_id != -1:  # 只保存参与聚类的latent
                        parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
                
                for parcel_id in range(n_parcels):
                    latents_in_parcel = np.where(original_parcel_assign == parcel_id)[0]
                    parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
                
                # 保存预处理信息
                preprocessing_file = os.path.join(out_dir, "preprocessing_info.json")
                with open(preprocessing_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "kept_feature_idx": kept_feature_idx.tolist(),
                        "kept_example_idx": preprocessing_info["kept_example_idx"].tolist(),
                        "labels_by_original_feature": labels_by_original_feature.tolist(),
                        "filtered_latent_dim": preprocessing_info["filtered_latent_dim"]
                    }, f, ensure_ascii=False, indent=2)
                
                np.save(os.path.join(out_dir, "parcel_assignments.npy"), original_parcel_assign)
                np.save(os.path.join(out_dir, "filtered_parcel_assignments.npy"), parcel_assign)
                
            elif args.use_svd_reduction:
                # 只使用SVD降维：所有latent都参与聚类，直接保存
                for latent_id, parcel_id in enumerate(parcel_assign):
                    parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
                for parcel_id in range(n_parcels):
                    latents_in_parcel = np.where(parcel_assign == parcel_id)[0]
                    parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
                
                # 保存SVD降维信息
                svd_file = os.path.join(out_dir, "svd_info.json")
                with open(svd_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "target_variance": preprocessing_info["svd_result"]["config"]["target_variance"],
                        "chosen_n_components": preprocessing_info["svd_result"]["config"]["chosen_n_components"],
                        "random_state": preprocessing_info["svd_result"]["config"]["random_state"],
                        "use_sparse": preprocessing_info["svd_result"]["config"]["use_sparse"],
                        "cache_path": preprocessing_info["svd_result"]["config"].get("cache_path", None),
                        "total_variance_explained": preprocessing_info["svd_result"]["total_variance_explained"],
                        "explained_variance_ratio": preprocessing_info["svd_result"]["explained_variance_ratio"].tolist(),
                        "original_shape": preprocessing_info["svd_result"]["original_shape"],
                        "reduced_shape": preprocessing_info["svd_result"]["reduced_shape"]
                    }, f, ensure_ascii=False, indent=2)
                
                np.save(os.path.join(out_dir, "parcel_assignments.npy"), parcel_assign)
                np.save(os.path.join(out_dir, "svd_components.npy"), preprocessing_info["svd_result"]["components"])
                np.save(os.path.join(out_dir, "svd_singular_values.npy"), preprocessing_info["svd_result"]["singular_values"])
            else:
                # 没有预处理，直接保存
                for latent_id, parcel_id in enumerate(parcel_assign):
                    parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
                for parcel_id in range(n_parcels):
                    latents_in_parcel = np.where(parcel_assign == parcel_id)[0]
                    parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
                np.save(os.path.join(out_dir, "parcel_assignments.npy"), parcel_assign)
            
            assignment_file = os.path.join(out_dir, "latent_parcel_assignments.json")
            with open(assignment_file, 'w', encoding='utf-8') as f:
                json.dump(parcel_assignments, f, ensure_ascii=False, indent=2)
            
            # 保存成本历史
            cost_history_file = os.path.join(out_dir, "cost_history.json")
            # 确保成本历史是JSON可序列化的
            json_serializable_cost_history = ensure_json_serializable(cost_history)
            with open(cost_history_file, 'w', encoding='utf-8') as f:
                json.dump(json_serializable_cost_history, f, ensure_ascii=False, indent=2)
            
            # 绘制成本变化图
            plot_cost_history(cost_history, out_dir, f"cost_history_nparcels{n_parcels}.png")
            
            # silhouette_score 计算（以latent为样本，A_z.T）
            try:
                # 使用快速silhouette_score计算（多次采样取平均）
                score = fast_silhouette_score(A_z.T, parcel_assign, sample_size=5000, n_trials=3)
                print(f"快速silhouette_score计算完成: {score}")
            except Exception as e:
                print(f"silhouette_score计算失败: {e}")
                score = None
            
            # 计算替代指标（使用采样加速）
            alternative_metrics = alternative_clustering_metrics(A_z.T, parcel_assign, 
                                                             sample_size=5000, n_trials=3)
            print(f"替代指标: WSS={alternative_metrics['wss']:.2e}, "
                  f"聚类质量比={alternative_metrics['cluster_quality_ratio']:.3f}")
            
            # 确保所有值都是JSON可序列化的
            if score is not None:
                score = float(score)
            
            silhouette_dict[n_parcels] = {
                'silhouette_score': score,
                'alternative_metrics': alternative_metrics
            }
            print(f"n_parcels={n_parcels}, silhouette_score={score}")
            
            # 分析并保存每个 n_parcels 的top样本
            try:
                if args.use_preprocessing and args.use_svd_reduction:
                    # 使用预处理后的样本矩阵（未降维的样本×特征矩阵）与筛选后的样本映射
                    kept_example_idx = preprocessing_info["kept_example_idx"]
                    sample_mappings_for_tops = [sample_mappings[i] for i in kept_example_idx]
                    A_for_tops = preprocessing_info["prep_result"]["X_feat"].T  # (N', F')
                elif args.use_preprocessing:
                    kept_example_idx = preprocessing_info["kept_example_idx"]
                    sample_mappings_for_tops = [sample_mappings[i] for i in kept_example_idx]
                    A_for_tops = A_z  # (N', F')
                elif args.use_svd_reduction:
                    # 使用原始样本矩阵按列L2归一化
                    latent_norms = np.linalg.norm(A, axis=0, keepdims=True)
                    latent_norms = np.clip(latent_norms, 1e-8, None)
                    A_for_tops = A / latent_norms  # (N, M)
                    sample_mappings_for_tops = sample_mappings
                else:
                    A_for_tops = A_z  # (N, M)
                    sample_mappings_for_tops = sample_mappings
                
                parcel2topsamples = analyze_parcels(parcel_assign, A_for_tops, sample_mappings_for_tops, data_level, n_top=1000)
                result_file = os.path.join(out_dir, "latent_parcel_topsamples.json")
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(parcel2topsamples, f, ensure_ascii=False, indent=2)
                print(f"top样本已保存到: {result_file}")
            except Exception as e:
                print(f"保存top样本失败: {e}")
            
            # 内存清理
            print(f"清理内存...")
            import gc
            
            # 删除大数组
            del parcel_assign
            del parcel_assignments
            del alternative_metrics
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"GPU缓存已清理")
            
            # 强制垃圾回收
            gc.collect()
            print(f"垃圾回收完成")
            
            # 显示当前内存使用情况
            print_memory_usage("清理后")
            
        # 保存所有n_parcels的silhouette_score
        # 确保所有数据都是JSON可序列化的
        json_serializable_silhouette_dict = ensure_json_serializable(silhouette_dict)
        with open(os.path.join(out_dir_base, "silhouette_scores.json"), 'w', encoding='utf-8') as f:
            json.dump(json_serializable_silhouette_dict, f, ensure_ascii=False, indent=2)
        print(f"所有n_parcels的silhouette_score已保存到: {os.path.join(out_dir_base, 'silhouette_scores.json')}")
    else:
        # 原有单次聚类流程
        # 使用与自动模式相同的路径构建逻辑
        path_components = []
        
        # 基础标识
        if args.test_mode:
            path_components.append("test")
        
        # 数据级别
        path_components.append(data_level)
        
        # 预处理参数
        if args.use_preprocessing:
            path_components.append(f"prep{args.min_activation_rate}_{args.gini_keep_quantile}")
        
        # SVD降维参数（使用方差阈值标识）
        if args.use_svd_reduction:
            tv_str = f"{args.svd_target_variance:.2f}".replace(".", "p")
            path_components.append(f"svdvar{tv_str}")
        
        # 聚类参数
        path_components.append(f"parcels{args.n_parcels}")
        path_components.append(f"iter{args.n_iter}")
        path_components.append(f"spatial{args.spatial_weight}")
        
        # 组合路径
        out_dir = os.path.join(args.out_dir, f"clustering_results_{'_'.join(path_components)}")
        os.makedirs(out_dir, exist_ok=True)
        # 注意：A, A_z, preprocessing_info 已经在主函数中构建过了
        # 这里直接使用已经构建好的数据，避免重复降维
        
        # 如果启用调试模式，先进行参数调试
        if args.debug_params:
            print("🔧 启用参数调试模式...")
            debug_results = debug_clustering_parameters(
                A_z, neighbors, latent2layer, idx_in_layer,
                n_parcels=args.n_parcels, n_iter=5  # 调试时使用较少的迭代次数
            )
            
            # 使用最佳参数进行正式聚类
            best_spatial_weight = min(debug_results.keys(), 
                                   key=lambda w: debug_results[w]['final_total_cost'])
            print(f"\n🎯 使用最佳空间权重 {best_spatial_weight} 进行正式聚类...")
            args.spatial_weight = best_spatial_weight
        
        # gwMRF主聚类
        parcel_assign, cost_history = gwMRF_latent_clustering(
            A_z, neighbors, latent2layer, idx_in_layer, 
            n_parcels=args.n_parcels, n_iter=args.n_iter,
            spatial_weight=args.spatial_weight, pairwise_weight=args.pairwise_weight,
            random_cluster_baseline=args.random_cluster_baseline
        )
        
        # 保存聚类分配结果
        print(f"保存聚类分配结果...")
        parcel_assignments = {
            "latent_to_parcel": {},
            "parcel_to_latents": {},
            "total_latents": len(parcel_assign),
            "n_parcels": args.n_parcels,
            "data_level": data_level
        }
        
        # 根据预处理和SVD降维的组合情况保存结果
        if args.use_preprocessing and args.use_svd_reduction:
            # 筛选+降维：需要映射回原始latent空间
            kept_feature_idx = preprocessing_info["kept_feature_idx"]
            labels_by_original_feature = preprocessing_info["labels_by_original_feature"]
            
            # 将聚类结果映射回原始latent空间
            original_parcel_assign = np.full(shape=latent_dim, fill_value=-1, dtype=int)
            for i, parcel_id in enumerate(parcel_assign):
                original_latent_id = kept_feature_idx[i]
                original_parcel_assign[original_latent_id] = parcel_id
            
            # 保存原始latent空间的分配结果
            for latent_id, parcel_id in enumerate(original_parcel_assign):
                if parcel_id != -1:  # 只保存参与聚类的latent
                    parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
            
            for parcel_id in range(args.n_parcels):
                latents_in_parcel = np.where(original_parcel_assign == parcel_id)[0]
                parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
            
            # 保存预处理信息
            preprocessing_file = os.path.join(out_dir, "preprocessing_info.json")
            with open(preprocessing_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "kept_feature_idx": kept_feature_idx.tolist(),
                    "kept_example_idx": preprocessing_info["kept_example_idx"].tolist(),
                    "labels_by_original_feature": labels_by_original_feature.tolist(),
                    "filtered_latent_dim": preprocessing_info["filtered_latent_dim"]
                }, f, ensure_ascii=False, indent=2)
            
            # 保存SVD降维信息
            svd_file = os.path.join(out_dir, "svd_info.json")
            with open(svd_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "target_variance": preprocessing_info["svd_result"]["config"]["target_variance"],
                    "chosen_n_components": preprocessing_info["svd_result"]["config"]["chosen_n_components"],
                    "random_state": preprocessing_info["svd_result"]["config"]["random_state"],
                    "use_sparse": preprocessing_info["svd_result"]["config"]["use_sparse"],
                    "cache_path": preprocessing_info["svd_result"]["config"].get("cache_path", None),
                    "total_variance_explained": float(preprocessing_info["svd_result"]["total_variance_explained"]),
                    "explained_variance_ratio": preprocessing_info["svd_result"]["explained_variance_ratio"].tolist(),
                    "original_shape": preprocessing_info["svd_result"]["original_shape"],
                    "reduced_shape": preprocessing_info["svd_result"]["reduced_shape"]
                }, f, ensure_ascii=False, indent=2)
            
            np.save(os.path.join(out_dir, "parcel_assignments.npy"), original_parcel_assign)
            np.save(os.path.join(out_dir, "filtered_parcel_assignments.npy"), parcel_assign)
            np.save(os.path.join(out_dir, "svd_components.npy"), preprocessing_info["svd_result"]["components"])
            np.save(os.path.join(out_dir, "svd_singular_values.npy"), preprocessing_info["svd_result"]["singular_values"])
            
        elif args.use_preprocessing:
            # 只使用预处理筛选
            kept_feature_idx = preprocessing_info["kept_feature_idx"]
            labels_by_original_feature = preprocessing_info["labels_by_original_feature"]
            
            # 将聚类结果映射回原始latent空间
            original_parcel_assign = np.full(shape=latent_dim, fill_value=-1, dtype=int)
            for i, parcel_id in enumerate(parcel_assign):
                original_latent_id = kept_feature_idx[i]
                original_parcel_assign[original_latent_id] = parcel_id
            
            # 保存原始latent空间的分配结果
            for latent_id, parcel_id in enumerate(original_parcel_assign):
                if parcel_id != -1:  # 只保存参与聚类的latent
                    parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
            
            for parcel_id in range(args.n_parcels):
                latents_in_parcel = np.where(original_parcel_assign == parcel_id)[0]
                parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
            
            # 保存预处理信息
            preprocessing_file = os.path.join(out_dir, "preprocessing_info.json")
            with open(preprocessing_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "kept_feature_idx": kept_feature_idx.tolist(),
                    "kept_example_idx": preprocessing_info["kept_example_idx"].tolist(),
                    "labels_by_original_feature": labels_by_original_feature.tolist(),
                    "filtered_latent_dim": preprocessing_info["filtered_latent_dim"]
                }, f, ensure_ascii=False, indent=2)
            
            np.save(os.path.join(out_dir, "parcel_assignments.npy"), original_parcel_assign)
            np.save(os.path.join(out_dir, "filtered_parcel_assignments.npy"), parcel_assign)
            
        elif args.use_svd_reduction:
            # 只使用SVD降维：所有latent都参与聚类，直接保存
            for latent_id, parcel_id in enumerate(parcel_assign):
                parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
            
            for parcel_id in range(args.n_parcels):
                latents_in_parcel = np.where(parcel_assign == parcel_id)[0]
                parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
            
            # 保存SVD降维信息
            svd_file = os.path.join(out_dir, "svd_info.json")
            with open(svd_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "target_variance": preprocessing_info["svd_result"]["config"]["target_variance"],
                    "chosen_n_components": preprocessing_info["svd_result"]["config"]["chosen_n_components"],
                    "random_state": preprocessing_info["svd_result"]["config"]["random_state"],
                    "use_sparse": preprocessing_info["svd_result"]["config"]["use_sparse"],
                    "cache_path": preprocessing_info["svd_result"]["config"].get("cache_path", None),
                    "total_variance_explained": float(preprocessing_info["svd_result"]["total_variance_explained"]),
                    "explained_variance_ratio": preprocessing_info["svd_result"]["explained_variance_ratio"].tolist(),
                    "original_shape": preprocessing_info["svd_result"]["original_shape"],
                    "reduced_shape": preprocessing_info["svd_result"]["reduced_shape"]
                }, f, ensure_ascii=False, indent=2)
            
            np.save(os.path.join(out_dir, "parcel_assignments.npy"), parcel_assign)
            np.save(os.path.join(out_dir, "svd_components.npy"), preprocessing_info["svd_result"]["components"])
            np.save(os.path.join(out_dir, "svd_singular_values.npy"), preprocessing_info["svd_result"]["singular_values"])
        else:
            # 没有预处理，直接保存
            for latent_id, parcel_id in enumerate(parcel_assign):
                parcel_assignments["latent_to_parcel"][f"latent_{latent_id}"] = int(parcel_id)
            
            for parcel_id in range(args.n_parcels):
                latents_in_parcel = np.where(parcel_assign == parcel_id)[0]
                parcel_assignments["parcel_to_latents"][f"parcel_{parcel_id}"] = [int(latent_id) for latent_id in latents_in_parcel]
            
            np.save(os.path.join(out_dir, "parcel_assignments.npy"), parcel_assign)
        
        # 保存为JSON文件
        assignment_file = os.path.join(out_dir, "latent_parcel_assignments.json")
        with open(assignment_file, 'w', encoding='utf-8') as f:
            json.dump(parcel_assignments, f, ensure_ascii=False, indent=2)
        
        # 保存成本历史
        cost_history_file = os.path.join(out_dir, "cost_history.json")
        # 确保成本历史是JSON可序列化的
        json_serializable_cost_history = ensure_json_serializable(cost_history)
        with open(cost_history_file, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_cost_history, f, ensure_ascii=False, indent=2)
        
        # 绘制成本变化图
        plot_cost_history(cost_history, out_dir, "cost_history.png")
        
        print(f"聚类分配结果已保存到: {assignment_file}")
        print(f"numpy数组已保存到: {os.path.join(out_dir, 'parcel_assignments.npy')}")
        print(f"成本历史已保存到: {cost_history_file}")
        
        # 打印每个parcel的latent数量统计
        print("每个parcel的latent数量:")
        for parcel_id in range(args.n_parcels):
            count = np.sum(parcel_assign == parcel_id)
            print(f"Parcel {parcel_id}: {count} latents")
        
        # 结果分析
        if args.use_preprocessing and args.use_svd_reduction:
            # 筛选+降维：使用筛选后的样本映射
            kept_example_idx = preprocessing_info["kept_example_idx"]
            filtered_sample_mappings = [sample_mappings[i] for i in kept_example_idx]
            parcel2topsamples = analyze_parcels(parcel_assign, A_z, filtered_sample_mappings, data_level, n_top=-1)
        elif args.use_preprocessing:
            # 只使用预处理筛选：使用筛选后的样本映射
            kept_example_idx = preprocessing_info["kept_example_idx"]
            filtered_sample_mappings = [sample_mappings[i] for i in kept_example_idx]
            parcel2topsamples = analyze_parcels(parcel_assign, A_z, filtered_sample_mappings, data_level, n_top=-1)
        elif args.use_svd_reduction:
            # 只使用SVD降维：使用所有样本映射
            parcel2topsamples = analyze_parcels(parcel_assign, A_z, sample_mappings, data_level, n_top=-1)
        else:
            # 都不使用：使用所有样本映射
            parcel2topsamples = analyze_parcels(parcel_assign, A_z, sample_mappings, data_level, n_top=-1)
        
        result_file = os.path.join(out_dir, f"latent_parcel_topsamples.json")
        # 确保parcel2topsamples是JSON可序列化的
        json_serializable_parcel2topsamples = ensure_json_serializable(parcel2topsamples)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_parcel2topsamples, f, ensure_ascii=False, indent=2)
            
        print("每个parcel的高激活样本:")
        for p, tops in parcel2topsamples.items():
            print(f"Parcel {p}:")
            for i, sample in enumerate(tops[:3]):  # 只显示前3个
                content_preview = sample['content'][:100] + "..." if len(sample['content']) > 100 else sample['content']
                print(f"  {i+1}. {content_preview} (激活强度: {sample['avg_activation']:.4f})")
        
        # 可视化每层分区
        if args.use_preprocessing and args.use_svd_reduction:
            # 筛选+降维：使用原始latent空间的分配结果进行可视化
            original_parcel_assign = np.load(os.path.join(out_dir, "parcel_assignments.npy"))
            plot_layer_parcels(original_parcel_assign, np.arange(latent_dim) // SAE_DIM, np.arange(latent_dim) % SAE_DIM, out_dir, sae_dim=SAE_DIM)
        elif args.use_preprocessing:
            # 只使用预处理筛选：使用原始latent空间的分配结果进行可视化
            original_parcel_assign = np.load(os.path.join(out_dir, "parcel_assignments.npy"))
            plot_layer_parcels(original_parcel_assign, np.arange(latent_dim) // SAE_DIM, np.arange(latent_dim) % SAE_DIM, out_dir, sae_dim=SAE_DIM)
        elif args.use_svd_reduction:
            # 只使用SVD降维：直接使用聚类结果进行可视化
            plot_layer_parcels(parcel_assign, latent2layer, idx_in_layer, out_dir, sae_dim=SAE_DIM)
        else:
            # 都不使用：直接使用聚类结果进行可视化
            plot_layer_parcels(parcel_assign, latent2layer, idx_in_layer, out_dir, sae_dim=SAE_DIM)
        print("已保存每层parcel分布图")
        
        # 保存聚类参数
        params = {
            "data_level": data_level,
            "n_parcels": args.n_parcels,
            "n_iter": args.n_iter,
            "spatial_weight": args.spatial_weight,
            "pairwise_weight": args.pairwise_weight,
            "test_mode": args.test_mode,
            "capabilities": capabilities,
            "latent_dim": latent_dim,
            "use_preprocessing": args.use_preprocessing,
            "use_svd_reduction": args.use_svd_reduction
        }
        
        if args.use_preprocessing and args.use_svd_reduction:
            # 筛选+降维：保存两种方法的参数
            params.update({
                "min_activation_rate": args.min_activation_rate,
                "min_var": args.min_var,
                "gini_keep_quantile": args.gini_keep_quantile,
                "drop_row_low_sum_quantile": args.drop_row_low_sum_quantile,
                "svd_target_variance": args.svd_target_variance,
                "svd_random_state": args.svd_random_state,
                "svd_use_sparse": args.svd_use_sparse,
                "filtered_latent_dim": preprocessing_info["filtered_latent_dim"]
            })
        elif args.use_preprocessing:
            # 只使用预处理筛选
            params.update({
                "min_activation_rate": args.min_activation_rate,
                "min_var": args.min_var,
                "gini_keep_quantile": args.gini_keep_quantile,
                "drop_row_low_sum_quantile": args.drop_row_low_sum_quantile,
                "filtered_latent_dim": preprocessing_info["filtered_latent_dim"]
            })
        elif args.use_svd_reduction:
            # 只使用SVD降维
            params.update({
                "svd_target_variance": args.svd_target_variance,
                "svd_random_state": args.svd_random_state,
                "svd_use_sparse": args.svd_use_sparse,
                "reduced_latent_dim": preprocessing_info["filtered_latent_dim"]
            })
        params_file = os.path.join(out_dir, "clustering_params.json")
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        
        print(f"聚类完成！结果保存在: {out_dir}") 
