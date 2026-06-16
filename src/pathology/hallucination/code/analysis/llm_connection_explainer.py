#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 连接组结果解释器

读取 connection_analysis_plus 的输出结果，构建提示并调用 gpt-oss（vLLM API）
生成一份对比幻觉行为与 Truthful 行为的科学报告（Markdown）。

作者: AI Assistant
日期: 2025
"""

import json
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import requests
import numpy as np

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置（参考 analysis_llm_summary.py）
DEFAULT_VLLM_URL = "http://0.0.0.0:8001/v1"
DEFAULT_API_KEY = "abcabc"
DEFAULT_MODEL_PATH = "/path/to/local_models/gpt-oss-20b"


def load_connection_results(results_dir: Path) -> Dict:
    """加载 connection_analysis_plus 的结果文件"""
    results_path = results_dir / "connection_analysis_results.json"
    delta_mat_path = results_dir / "delta_connectivity_matrix.npy"
    report_path = results_dir / "analysis_report.txt"

    data = {
        "results": {},
        "delta_matrix": None,
        "text_report": ""
    }
    if results_path.exists():
        with open(results_path, 'r', encoding='utf-8') as f:
            data["results"] = json.load(f)
    else:
        raise FileNotFoundError(f"缺少结果文件: {results_path}")

    if delta_mat_path.exists():
        try:
            data["delta_matrix"] = np.load(delta_mat_path)
        except Exception as e:
            logger.warning(f"加载 ΔFC 矩阵失败: {e}")

    if report_path.exists():
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                data["text_report"] = f.read()
        except Exception as e:
            logger.warning(f"读取文本报告失败: {e}")

    return data


def summarize_results(results_json: Dict) -> str:
    """将 JSON 结果压缩成适合放入 LLM 提示的摘要文本"""
    try:
        gm = results_json.get('global_metrics', {})
        node_cent = results_json.get('node_centrality', {})
        subnets = results_json.get('anomalous_subnetworks', [])
        comm = results_json.get('community_analysis', {})
        node_names = results_json.get('node_names', [])

        # 全局指标
        g_t = gm.get('correct', {})
        g_h = gm.get('hallucination', {})
        g_d = gm.get('delta', {})
        global_summary_lines = [
            f"MeanStrength: C={g_t.get('mean_strength',0):.4f}, H={g_h.get('mean_strength',0):.4f}, Δ={g_d.get('mean_strength',0):.4f}",
            f"Density:      C={g_t.get('density',0):.4f}, H={g_h.get('density',0):.4f}, Δ={g_d.get('density',0):.4f}",
            f"Clustering:   C={g_t.get('clustering',0):.4f}, H={g_h.get('clustering',0):.4f}, Δ={g_d.get('clustering',0):.4f}",
            f"Efficiency:   C={g_t.get('global_efficiency',0):.4f}, H={g_h.get('global_efficiency',0):.4f}, Δ={g_d.get('global_efficiency',0):.4f}",
            f"Modularity:   C={g_t.get('modularity',0):.4f}, H={g_h.get('modularity',0):.4f}, Δ={g_d.get('modularity',0):.4f}",
            f"Modules:      C={int(g_t.get('num_modules',0))}, H={int(g_h.get('num_modules',0))}, Δ={int(g_h.get('num_modules',0))-int(g_t.get('num_modules',0))}"
        ]

        # 中心性差异（取Top 10）
        cent_delta_lines = []
        for key, label in [("degree","Degree"),("betweenness","Betweenness"),("eigenvector","Eigenvector"),("participation","Participation")]:
            try:
                c_t = np.array(node_cent.get('correct',{}).get(key, []), dtype=float)
                c_h = np.array(node_cent.get('hallucination',{}).get(key, []), dtype=float)
                if c_t.size == 0 or c_h.size == 0:
                    continue
                delta = c_h - c_t
                idx = np.argsort(np.abs(delta))[-10:][::-1]
                lines = []
                for i in idx:
                    name = node_names[i] if i < len(node_names) else f"Node_{i}"
                    lines.append(f"{name}: Δ{label}={delta[i]:+.4f}")
                cent_delta_lines.append(f"Top Δ{label}:\n- " + "\n- ".join(lines))
            except Exception:
                continue

        # 异常子网（前5个）
        subnet_lines = []
        for s in subnets[:5]:
            subnet_lines.append(
                f"Subnet#{s.get('subnet_id','?')}: Nodes={s.get('num_nodes',0)}, Edges={s.get('num_edges',0)}, AvgΔ={s.get('avg_difference',0):+.4f}, AvgZ={s.get('avg_z_score',0):+.2f}"
            )

        # 社区结构
        comm_delta_q = comm.get('delta_modularity', 0)
        comm_summary = [
            f"Correct: Q={comm.get('correct',{}).get('modularity',0):.4f}, Modules={len(comm.get('correct',{}).get('communities',[]))}",
            f"Hallucination: Q={comm.get('hallucination',{}).get('modularity',0):.4f}, Modules={len(comm.get('hallucination',{}).get('communities',[]))}",
            f"ΔQ={comm_delta_q:+.4f} (Q_h - Q_t)"
        ]

        summary = [
            "## Global Metrics",
            "\n".join(global_summary_lines),
            "",
            "## Node Centrality Differences",
            "\n\n".join(cent_delta_lines) if cent_delta_lines else "(no centrality deltas)",
            "",
            "## Anomalous Subnetworks (pseudo-NBS)",
            "\n".join(subnet_lines) if subnet_lines else "(no subnetworks)",
            "",
            "## Community Structure",
            "\n".join(comm_summary)
        ]
        return "\n".join(summary)
    except Exception as e:
        logger.warning(f"摘要生成失败: {e}")
        return "(Failed to summarize results)"


def build_prompt(dataset_name: str, summary_text: str) -> str:
    """构建给 gpt-oss 的提示，聚焦 Truthful vs 幻觉 差异解释"""
    prompt = f"""
Please write a concise, Nature-style analytical note comparing functional connectivity patterns between Correct (Truthful) and Hallucination conditions for dataset: {dataset_name}.

Provide:
1) Key global-network differences (Δ of density, efficiency, clustering, modularity)
2) Node-level differences (top Δ degree/betweenness/eigenvector/participation) and their cognitive implications
3) Abnormal subnetworks (qualitative NBS): what they suggest about failure modes
4) Community structure shift (ΔQ) and its interpretation (integration vs segregation)
5) A brief mechanistic explanation linking these network changes to hallucination vs truthful behavior

Use scientific, precise language. Limit to 600-900 words. Avoid generic filler.

Analysis summary (data-derived facts):
{summary_text}
"""
    return prompt


def call_vllm_api(prompt: str, vllm_url: str = DEFAULT_VLLM_URL, api_key: str = DEFAULT_API_KEY,
                  model: str = DEFAULT_MODEL_PATH, max_tokens: int = 2000,
                  temperature: float = 0.1, timeout: int = 180) -> str:
    """调用 vLLM API (与 analysis_llm_summary.py 一致风格)"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a neuroscientific analysis assistant specializing in functional connectivity interpretation for LLM internal activations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        resp = requests.post(f"{vllm_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data and "choices" in data and data["choices"] and "message" in data["choices"][0]:
            return (data["choices"][0]["message"].get("content", "") or "").strip()
        raise RuntimeError(f"Unexpected response: {data}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"vLLM request failed: {e}")


def save_markdown(report_dir: Path, dataset_name: str, prompt: str, content: str) -> Path:
    """保存为 Markdown 报告"""
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"llm_connection_report_{dataset_name}.md"
    header = (
        f"# Functional Connectivity Analysis Report\n\n"
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Dataset: {dataset_name}\n"
        f"- Engine: gpt-oss (vLLM)\n\n"
        f"---\n\n"
        f"<details><summary>Prompt</summary>\n\n\n{prompt}\n\n</details>\n\n"
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(header + content + "\n")
    return out_path


def run_for_dataset(base_output_root: Path, dataset_name: str,
                    vllm_url: str, api_key: str, model: str) -> Optional[Path]:
    """对单个数据集运行 LLM 解释流程"""
    try:
        results_dir = base_output_root / dataset_name / "parcel_level" / "connection_plus"
        logger.info(f"数据集: {dataset_name} -> 结果目录: {results_dir}")
        data = load_connection_results(results_dir)
        summary = summarize_results(data.get("results", {}))
        prompt = build_prompt(dataset_name, summary)
        content = call_vllm_api(prompt, vllm_url=vllm_url, api_key=api_key, model=model)
        out_path = save_markdown(results_dir, dataset_name, prompt, content)
        logger.info(f"报告已保存: {out_path}")
        return out_path
    except Exception as e:
        logger.error(f"[{dataset_name}] 生成报告失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='LLM 连接组结果解释 (基于 gpt-oss)')
    parser.add_argument('--base_output_root', type=str, required=True,
                        help='analysis_output 根目录（包含各数据集子目录）')
    parser.add_argument('--datasets', type=str, nargs='*', default=None,
                        help='要处理的数据集列表（目录名）；为空则根据根目录自动发现')
    parser.add_argument('--vllm_url', type=str, default=DEFAULT_VLLM_URL,
                        help=f'vLLM API 地址 (默认: {DEFAULT_VLLM_URL})')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY,
                        help='API 密钥 (默认: abcabc)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='gpt-oss 模型路径')
    args = parser.parse_args()

    base_root = Path(args.base_output_root)
    if not base_root.exists():
        logger.error(f"根目录不存在: {base_root}")
        sys.exit(1)

    # 自动发现数据集
    datasets: List[str]
    if args.datasets:
        datasets = args.datasets
    else:
        # 遍历根目录下的子目录，若存在 connection_plus 结果则认为是合法数据集
        candidates = []
        for child in base_root.iterdir():
            if not child.is_dir():
                continue
            cp_dir = child / "parcel_level" / "connection_plus" / "connection_analysis_results.json"
            if cp_dir.exists():
                candidates.append(child.name)
        datasets = sorted(candidates)
        if not datasets:
            logger.error("未在根目录下发现任何可用数据集 (缺少 connection_plus 结果)")
            sys.exit(1)

    logger.info(f"将处理数据集: {datasets}")

    success = 0
    for ds in datasets:
        out = run_for_dataset(base_root, ds, args.vllm_url, args.api_key, args.model)
        if out is not None:
            success += 1

    logger.info(f"完成。成功生成 {success}/{len(datasets)} 个报告。")


if __name__ == "__main__":
    main()
