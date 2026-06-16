#!/usr/bin/env python3
"""
合并所有激活字典文件，保留每个latent的最大激活值
"""

import pickle
import os
import glob
from collections import defaultdict
import numpy as np

def merge_max_activations():
    """合并所有激活文件，保留每个latent的最大激活值"""
    
    # 激活文件目录
    activation_dir = "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation/dataset_8b/"
    
    # 获取所有pickle文件
    pickle_files = glob.glob(os.path.join(activation_dir, "*_max_activation.pkl"))
    
    print(f"找到 {len(pickle_files)} 个激活文件")
    
    # 用于存储每个latent的最大激活值
    max_activations = defaultdict(float)
    
    # 遍历所有文件
    for i, file_path in enumerate(pickle_files):
        filename = os.path.basename(file_path)
        print(f"处理文件 {i+1}/{len(pickle_files)}: {filename}")
        
        try:
            # 读取pickle文件
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"  - 文件包含 {len(data)} 个激活值")
            
            # 遍历该文件中的所有激活值
            for latent_key, activation_value in data.items():
                # 更新最大激活值
                if activation_value > max_activations[latent_key]:
                    max_activations[latent_key] = activation_value
                    
        except Exception as e:
            print(f"  - 错误：无法处理文件 {filename}: {e}")
            continue
    
    print(f"\n合并完成！")
    print(f"总共找到 {len(max_activations)} 个唯一的latent键")
    print(f"最大激活值: {max(max_activations.values()):.6f}")
    print(f"最小激活值: {min(max_activations.values()):.6f}")
    print(f"平均激活值: {np.mean(list(max_activations.values())):.6f}")
    
    # 保存合并后的结果
    output_file = "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation/merged_max_activations_8b.pkl"
    
    with open(output_file, 'wb') as f:
        pickle.dump(dict(max_activations), f)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 显示一些统计信息
    print(f"\n统计信息:")
    activations_array = np.array(list(max_activations.values()))
    print(f"激活值分布:")
    print(f"  - 25%分位数: {np.percentile(activations_array, 25):.6f}")
    print(f"  - 50%分位数 (中位数): {np.percentile(activations_array, 50):.6f}")
    print(f"  - 75%分位数: {np.percentile(activations_array, 75):.6f}")
    print(f"  - 95%分位数: {np.percentile(activations_array, 95):.6f}")
    print(f"  - 99%分位数: {np.percentile(activations_array, 99):.6f}")
    
    # 显示激活值最高的前10个latent
    sorted_activations = sorted(max_activations.items(), key=lambda x: x[1], reverse=True)
    print(f"\n激活值最高的前10个latent:")
    for i, (latent_key, activation) in enumerate(sorted_activations[:10]):
        print(f"  {i+1}. {latent_key}: {activation:.6f}")

if __name__ == "__main__":
    merge_max_activations() 