#!/usr/bin/env python3
"""
将 metrics_language_parcels.json（实际是 pickle 格式）转换为真正的 JSON 格式

功能：
1. 扫描 data 目录下所有子目录
2. 检查 metrics_language_parcels.json 文件是否为 pickle 格式
3. 如果是 pickle 格式，转换为 JSON 格式
"""

import json
import pickle
from pathlib import Path
import numpy as np


def convert_to_json_serializable(obj):
    """将包含 numpy 数组的对象转换为 JSON 可序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def is_pickle_file(file_path: Path) -> bool:
    """检查文件是否为 pickle 格式"""
    try:
        with open(file_path, "rb") as f:
            # 读取前几个字节检查 pickle 魔数
            magic = f.read(2)
            # pickle 文件通常以特定字节开头
            return magic in [b'\x80\x03', b'\x80\x04', b'\x80\x05']
    except Exception:
        return False


def convert_file(pkl_path: Path, json_path: Path) -> bool:
    """转换单个文件"""
    try:
        # 读取 pickle 文件
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # 转换为 JSON 可序列化格式
        json_data = convert_to_json_serializable(data)
        
        # 保存 JSON 文件
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] 已转换: {pkl_path.name} -> {json_path.name}")
        return True
    except Exception as e:
        print(f"[ERROR] 转换失败 {pkl_path.name}: {e}")
        return False


def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return
    
    print(f"[INFO] 扫描目录: {data_dir}")
    
    # 查找所有 metrics_language_parcels.json 文件
    json_files = list(data_dir.glob("*/metrics_language_parcels.json"))
    
    if not json_files:
        print("[WARN] 未找到任何 metrics_language_parcels.json 文件")
        return
    
    print(f"[INFO] 找到 {len(json_files)} 个文件")
    
    converted = 0
    skipped = 0
    failed = 0
    
    for json_file in json_files:
        # 检查是否为 pickle 格式
        if is_pickle_file(json_file):
            print(f"[INFO] 检测到 pickle 格式文件: {json_file}")
            # 备份原文件
            backup_path = json_file.with_suffix(".json.pkl_backup")
            if not backup_path.exists():
                import shutil
                shutil.copy2(json_file, backup_path)
                print(f"[INFO] 已备份到: {backup_path.name}")
            
            # 转换文件
            if convert_file(json_file, json_file):
                converted += 1
            else:
                failed += 1
        else:
            # 尝试读取 JSON 验证
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json.load(f)
                print(f"[SKIP] 已经是 JSON 格式: {json_file.name}")
                skipped += 1
            except Exception:
                print(f"[WARN] 无法识别格式: {json_file.name}")
                skipped += 1
    
    print("\n" + "="*60)
    print(f"转换完成: 成功 {converted}，跳过 {skipped}，失败 {failed}")
    print("="*60)


if __name__ == "__main__":
    main()
