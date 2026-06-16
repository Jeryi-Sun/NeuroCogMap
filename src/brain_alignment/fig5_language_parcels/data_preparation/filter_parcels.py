#!/usr/bin/env python3
"""
依据 filter.md 中的规则，对 A/S 矩阵进行筛选。

核心思路：
    - Human parcels：基于字符串规则的筛选（Schaefer100 7Networks 左半球故事相关 ROI）
    - LLM parcels：用预测矩阵 A 的列最大值和功能相似度矩阵 S 的列最大值衡量，满足阈值则保留
    - 输出筛选后的子矩阵

输出（默认写入 data4draw/）：
    {tag}_prediction.csv
    {tag}_semantic.csv
    {tag}_summary.json
    {tag}_parcel_id_to_function_name.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests

story_name = "smoke_story"

DATA_ROOT = Path(
    "/path/to/project_root/"
    "Human_LLM_align/litcoder_core/data_analysis/draw_graphs"
).resolve()
DATA_DIR = DATA_ROOT / "data4draw" / story_name
DEFAULT_OUTPUT_DIR = DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据规则和阈值筛选 human/LLM parcels，可选 LLM 语义筛选。")
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=DATA_DIR / "prediction_matrix_gemma2_2b.csv",
        help="预测准确性矩阵 CSV 路径",
    )
    parser.add_argument(
        "--semantic-path",
        type=Path,
        default=DATA_DIR / "semantic_matrix_gemma2_2b.csv",
        help="功能相似度矩阵 CSV 路径",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=DATA_DIR / "gemma2_2b_parcel_id_to_function_name.json",
        help="parcel id -> 功能描述 JSON 路径",
    )
    parser.add_argument(
        "--parcel-descriptions-path",
        type=Path,
        default=Path(
            "/path/to/project_root/"
            "Human_LLM_align/litcoder_core/dataset/brain_parcel_description/parcel_descriptions.json"
        ),
        help="parcel_descriptions.json 文件路径，用于根据 parcel_id 获取 parcel_name",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="gemma2_2b_filtered",
        help="输出文件名前缀",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录（默认 data4draw）",
    )
    parser.add_argument("--tau-llm-acc", type=float, default=0.08, help="LLM parcel 预测阈值 τ_L。")
    parser.add_argument("--tau-llm-sim", type=float, default=0.20, help="LLM parcel 功能相似度阈值。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定则覆盖已存在的输出文件。",
    )
    parser.add_argument(
        "--skip-mapping",
        action="store_true",
        help="若指定则不更新 parcel 映射文件。",
    )
    parser.add_argument(
        "--zscore-columns",
        action="store_true",
        help="若指定则对筛选后的矩阵按列（LLM parcels）进行 z-score 归一化。",
    )
    parser.add_argument(
        "--zscore-rows-human",
        action="store_true",
        help="若指定则对筛选后的矩阵按行（Human parcels）进行 z-score 归一化。",
    )
    parser.add_argument(
        "--use-llm-filter",
        action="store_true",
        help="若指定，则对 human parcels 额外使用 LLM（基于 LeBel 任务描述）进行语义筛选，仅保留与自然叙事理解高度相关的脑区。",
    )
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default="http://0.0.0.0:8000/v1",
        help="vLLM / OpenAI 兼容接口的基础 URL（默认：http://0.0.0.0:8000/v1）。",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="abcabc",
        help="调用 LLM 时使用的 API Key（默认：abcabc）。",
    )
    parser.add_argument(
        "--llm-filter-cache",
        type=Path,
        default=DATA_DIR / "lebel_human_llm_filter.json",
        help="LLM 筛选结果缓存文件路径，用于避免重复调用模型（默认：data4draw/lebel_human_llm_filter.json）。",
    )
    return parser.parse_args()


def validate_inputs(acc_path: Path, sim_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not acc_path.exists():
        raise FileNotFoundError(f"找不到预测矩阵：{acc_path}")
    if not sim_path.exists():
        raise FileNotFoundError(f"找不到语义矩阵：{sim_path}")
    acc = pd.read_csv(acc_path, index_col=0)
    sim = pd.read_csv(sim_path, index_col=0)
    if acc.shape != sim.shape:
        raise ValueError(f"A 与 S 形状不一致：{acc.shape} vs {sim.shape}")
    return acc, sim


def load_parcel_descriptions(parcel_descriptions_path: Path) -> dict[int, str]:
    """
    加载 parcel_descriptions.json 文件，构建 parcel_id -> parcel_name 的映射。
    
    Args:
        parcel_descriptions_path: parcel_descriptions.json 文件路径
    
    Returns:
        dict: {parcel_id: parcel_name}
    """
    if not parcel_descriptions_path.exists():
        raise FileNotFoundError(f"找不到 parcel descriptions 文件：{parcel_descriptions_path}")
    
    with parcel_descriptions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"parcel_descriptions.json 应该是列表格式，但得到 {type(data)}")
    
    parcel_id_to_name = {}
    for item in data:
        parcel_id = item.get("parcel_id")
        parcel_name = item.get("parcel_name", "")
        if parcel_id is not None:
            # 确保 parcel_id 始终为整数类型，以匹配 extract_parcel_id_from_label 的返回类型
            try:
                parcel_id_int = int(parcel_id)
                parcel_id_to_name[parcel_id_int] = parcel_name
            except (ValueError, TypeError) as e:
                print(f"[Warning] 无法将 parcel_id={parcel_id} 转换为整数，跳过: {e}")
    
    print(f"[Info] 已加载 {len(parcel_id_to_name)} 个 parcel 描述")
    return parcel_id_to_name


def load_parcel_description_records(parcel_descriptions_path: Path) -> dict[int, dict[str, Any]]:
    """
    加载 parcel_descriptions.json，返回 parcel_id -> 完整记录 的映射。
    该函数用于后续 LLM 语义筛选（需要访问 function_description / role_in_human_brain / reasoning）。
    """
    if not parcel_descriptions_path.exists():
        raise FileNotFoundError(f"找不到 parcel descriptions 文件：{parcel_descriptions_path}")

    with parcel_descriptions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"parcel_descriptions.json 应该是列表格式，但得到 {type(data)}")

    records: dict[int, dict[str, Any]] = {}
    for item in data:
        parcel_id = item.get("parcel_id")
        if parcel_id is None:
            continue
        try:
            parcel_id_int = int(parcel_id)
        except (ValueError, TypeError):
            continue
        records[parcel_id_int] = item
    return records


def extract_parcel_id_from_label(label: str) -> int | None:
    """
    从 human parcel 标签中提取 parcel_id。
    
    支持的格式：
    - "Human_Parcel_{id}" -> {id}
    - "{id}" -> {id} (如果直接是数字)
    
    Args:
        label: human parcel 标签字符串
    
    Returns:
        parcel_id (int) 或 None（如果无法提取）
    """
    label_str = str(label).strip()
    
    # 尝试匹配 "Human_Parcel_{id}" 格式
    if label_str.startswith("Human_Parcel_"):
        try:
            return int(label_str.replace("Human_Parcel_", ""))
        except ValueError:
            pass
    
    # 尝试直接解析为整数
    try:
        return int(label_str)
    except ValueError:
        pass
    
    return None


LEBEL_TASK_DESCRIPTION = (
    "LeBel 是一个用于自然语言／叙事理解的公开 fMRI 数据集。"
    "该数据集记录了受试者在被动聆听完整、自然叙事故事时的大脑 BOLD 反应，"
    "非常适合研究语音、词汇、语义、叙事整合等多层次语言处理。"
)


def call_vllm_api_for_filter(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str = "/path/to/local_models/gpt-oss-20b",
    max_tokens: int = 10000,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    调用 vLLM / OpenAI 兼容接口，请求模型返回严格 JSON。

    预期返回格式：
    {
      "keep": true/false,
      "reason": "简要说明"
    }
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a cognitive neuroscience expert. Always reply with STRICT JSON only.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(f"{api_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    content = ""
    try:
        content = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LLM 响应解析失败: {data}") from exc

    # 从响应中提取 JSON
    try:
        m = re.search(r"\{[\s\S]*\}", content)
        if not m:
            raise ValueError(f"未在响应中找到 JSON 片段: {content[:200]}")
        return json.loads(m.group(0))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"解析 LLM JSON 响应失败: {exc}; 原始内容: {content[:400]}") from exc


def build_llm_filter_prompt(parcel_id: int, record: dict[str, Any]) -> str:
    """
    构造用于 LLM 筛选的 prompt。

    目标：判断该 human parcel 是否与 LeBel 任务（自然叙事理解、语言/语义处理）高度相关。
    """
    parcel_name = record.get("parcel_name", "")
    function_name = record.get("function_name", "")
    function_description = record.get("function_description", "")
    role_in_human_brain = record.get("role_in_human_brain", "")
    reasoning = record.get("reasoning", "")

    return (
        f"数据集背景：\n{LEBEL_TASK_DESCRIPTION}\n\n"
        "下面是一个大脑皮层 parcel 的功能注释，请根据这些信息判断："
        "该 parcel 是否在上述自然叙事理解任务中起到核心或重要作用"
        "（例如：语音处理、词汇/句法、语义理解、叙事整合、记忆-语言交互等）。\n\n"
        f"Parcel ID: {parcel_id}\n"
        f"Parcel name: {parcel_name}\n"
        f"Function name: {function_name}\n\n"
        f"Function description:\n{function_description}\n\n"
        f"Role in human brain:\n{role_in_human_brain}\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        "请你返回严格 JSON，格式如下（不要包含任何多余文字）：\n"
        "{\n"
        '  \"keep\": true 或 false,\n'
        '  \"reason\": \"用简短中文解释为什么该 parcel 适合/不适合作为 LeBel 语音/语义/叙事理解任务相关区域\"\n'
        "}\n"
        "判定标准：\n"
        "- keep=true：该 parcel 在语言理解、语义/叙事处理、听觉语音等方面有明确、核心的功能；\n"
        "- keep=false：该 parcel 主要与纯视觉、纯运动、情绪、痛觉、奖励/惩罚、控制/执行等非语言主导功能相关。"
    )


def llm_filter_human_parcels(
    labels: pd.Index,
    parcel_descriptions_path: Path,
    api_url: str,
    api_key: str,
    cache_path: Path,
) -> pd.Series:
    """
    使用 LLM 基于 parcel_descriptions.json 对 human parcels 进行语义筛选。

    返回布尔 Series，True 表示该 parcel 被 LLM 判定为与 LeBel 任务高度相关。
    """
    records = load_parcel_description_records(parcel_descriptions_path)

    # 读取/初始化缓存
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            print(f"[Warning] 无法读取 LLM 筛选缓存文件 {cache_path}，将重新创建。")
            cache = {}

    mask = pd.Series(False, index=labels)
    updated = False

    for label in labels:
        parcel_id = extract_parcel_id_from_label(label)
        if parcel_id is None:
            print(f"[Warning] LLM 筛选：无法从标签 '{label}' 中提取 parcel_id，跳过（标记为 False）")
            continue

        record = records.get(parcel_id)
        if record is None:
            print(f"[Warning] LLM 筛选：parcel_id={parcel_id} 在 parcel_descriptions.json 中不存在，跳过（标记为 False）")
            continue

        key = str(parcel_id)
        if key in cache:
            keep = bool(cache[key].get("keep", False))
            mask[label] = keep
            continue

        prompt = build_llm_filter_prompt(parcel_id, record)
        try:
            resp = call_vllm_api_for_filter(prompt, api_url=api_url, api_key=api_key)
            keep = bool(resp.get("keep", False))
            reason = str(resp.get("reason", "") or "")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] LLM 筛选调用失败（parcel_id={parcel_id}）：{exc}，该 parcel 将标记为 False")
            keep = False
            reason = f"LLM 调用失败：{exc}"

        cache[key] = {"keep": keep, "reason": reason}
        mask[label] = keep
        updated = True

    if updated:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Info] 已更新 LLM 筛选缓存：{cache_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] 无法写入 LLM 筛选缓存文件 {cache_path}：{exc}")

    kept = int(mask.sum())
    print(f"[Info] LLM 语义筛选（LeBel 任务相关性）：保留 {kept}/{len(mask)} 个 human parcels")
    return mask


def filter_human_parcels_by_rules(
    labels: pd.Index,
    parcel_descriptions_path: Path,
) -> pd.Series:
    """
    基于字符串规则筛选 human parcels（Schaefer100 7Networks 左半球故事相关 ROI）。
    
    流程：
        1. 从 labels 中提取 parcel_id
        2. 在 parcel_descriptions.json 中根据 parcel_id 匹配，获取 parcel_name
        3. 对 parcel_name 应用字符串规则筛选
    
    规则：
        1. 核心语义 + 叙事 Default 区：
           - 7Networks_LH_Default_Temp_* (颞叶语义区)
           - 7Networks_LH_Default_Par_* (AG/IPL 顶叶默认网络区)
           - 7Networks_LH_Default_pCunPCC_* (后扣带回/楔前叶)
        
        2. 左前额叶 Default（语义/叙事控制）：
           - 7Networks_LH_Default_PFC_*
        
        3. Control Network（故事理解的控制区）：
           - 7Networks_LH_Cont_PFCl_*
           - 7Networks_LH_Cont_Par_*
           - 7Networks_LH_Cont_pCun_*
        
        4. 左侧 SomMot 中的听觉邻近区（前期输入层）：
           - 7Networks_LH_SomMot_* 且包含 _4, _5, _6 后缀
    
    Args:
        labels: human parcel 标签的索引
        parcel_descriptions_path: parcel_descriptions.json 文件路径
    
    Returns:
        布尔 Series，True 表示保留该 parcel
    """
    # 加载 parcel descriptions
    parcel_id_to_name = load_parcel_descriptions(parcel_descriptions_path)
    
    mask = pd.Series(False, index=labels)
    matched_count = 0
    unmatched_labels = []
    
    # 用于统计各类 ROI 的数量
    core_rois_count = 0
    pfc_rois_count = 0
    control_rois_count = 0
    auditory_rois_count = 0
    
    for label in labels:
        # 从 label 中提取 parcel_id
        parcel_id = extract_parcel_id_from_label(label)
        
        if parcel_id is None:
            print(f"[Warning] 无法从标签 '{label}' 中提取 parcel_id，跳过")
            unmatched_labels.append(label)
            continue
        
        # 在 parcel_descriptions.json 中查找对应的 parcel_name
        parcel_name = parcel_id_to_name.get(parcel_id)
        
        if parcel_name is None:
            print(f"[Warning] 未找到 parcel_id={parcel_id} 对应的 parcel_name，跳过")
            unmatched_labels.append(label)
            continue
        
        parcel_name_str = str(parcel_name)
        matched_count += 1
        
        # -------------------------
        # 1. 核心语义 + 叙事 Default 区
        # -------------------------
        if parcel_name_str.startswith("7Networks_LH_Default_Temp_"):
            mask[label] = True
            core_rois_count += 1
            continue
        
        if parcel_name_str.startswith("7Networks_LH_Default_Par_"):
            mask[label] = True
            core_rois_count += 1
            continue
        
        if parcel_name_str.startswith("7Networks_LH_Default_pCunPCC_"):
            mask[label] = True
            core_rois_count += 1
            continue
        
        # -------------------------
        # 2. 左前额叶 Default（语义/叙事控制）
        # -------------------------
        if parcel_name_str.startswith("7Networks_LH_Default_PFC_"):
            mask[label] = True
            pfc_rois_count += 1
            continue
        
        # -------------------------
        # 3. Control Network（故事理解的控制区）
        # -------------------------
        if parcel_name_str.startswith("7Networks_LH_Cont_PFCl_"):
            mask[label] = True
            control_rois_count += 1
            continue
        
        if parcel_name_str.startswith("7Networks_LH_Cont_Par_"):
            mask[label] = True
            control_rois_count += 1
            continue
        
        if parcel_name_str.startswith("7Networks_LH_Cont_pCun_"):
            mask[label] = True
            control_rois_count += 1
            continue
        
        # -------------------------
        # 4. 左侧 SomMot 中的听觉邻近区（前期输入层）
        # -------------------------
        if parcel_name_str.startswith("7Networks_LH_SomMot_"):
            # 通常 SomMot_4,5,6 靠近听觉皮层，按后缀判断
            if any(suffix in parcel_name_str for suffix in ["_4", "_5", "_6"]):
                mask[label] = True
                auditory_rois_count += 1
                continue
    
    print(f"[Info] 成功匹配 {matched_count}/{len(labels)} 个 parcel")
    print(f"[Info] 筛选结果统计：")
    print(f"      - 核心语义 + 叙事 Default 区: {core_rois_count} 个")
    print(f"      - 左前额叶 Default: {pfc_rois_count} 个")
    print(f"      - Control Network: {control_rois_count} 个")
    print(f"      - 左侧 SomMot 听觉邻近区: {auditory_rois_count} 个")
    print(f"      - 总计筛选出: {mask.sum()} 个符合条件的 parcel")
    if unmatched_labels:
        print(f"[Warning] 有 {len(unmatched_labels)} 个标签无法匹配，已跳过")
    
    return mask


def apply_thresholds(
    acc: pd.DataFrame,
    sim: pd.DataFrame,
    tau_l_acc: float,
    tau_l_sim: float,
    parcel_descriptions_path: Path,
) -> dict:
    """
    应用筛选规则：
    - Human parcels: 默认不过滤（全部保留），如需筛选请在 main 中启用 LLM 语义筛选
    - LLM parcels: 基于阈值的筛选
    """
    # Human parcels：默认全部保留；如需语义筛选，在 main() 中通过 LLM 掩码进一步收缩
    human_mask = pd.Series(True, index=acc.index)
    
    # LLM parcels: 使用阈值筛选
    llm_max_acc = acc.max(axis=0)
    llm_max_sim = sim.max(axis=0)
    llm_mask = (llm_max_acc >= tau_l_acc) & (llm_max_sim >= tau_l_sim)
    
    # 计算统计信息（用于 summary）
    human_max_acc = acc.max(axis=1)
    human_max_sim = sim.max(axis=1)

    result = {
        "human_mask": human_mask,
        "llm_mask": llm_mask,
        "human_max_acc": human_max_acc,
        "human_max_sim": human_max_sim,
        "llm_max_acc": llm_max_acc,
        "llm_max_sim": llm_max_sim,
    }
    return result


def save_outputs(
    acc: pd.DataFrame,
    sim: pd.DataFrame,
    masks: dict,
    tag: str,
    overwrite: bool,
    thresholds: dict,
    output_dir: Path,
    zscore_columns: bool = False,
    zscore_rows_human: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_out = output_dir / f"{tag}_prediction.csv"
    sim_out = output_dir / f"{tag}_semantic.csv"
    summary_out = output_dir / f"{tag}_summary.json"

    if not overwrite and (pred_out.exists() or sim_out.exists() or summary_out.exists()):
        raise FileExistsError(
            f"{tag} 对应输出已存在（{output_dir}），如需覆盖请使用 --overwrite。"
        )

    filtered_acc = acc.loc[masks["human_mask"], masks["llm_mask"]]
    filtered_sim = sim.loc[masks["human_mask"], masks["llm_mask"]]
    
    # 应用归一化（可以同时应用列和行归一化）
    if zscore_columns:
        filtered_acc = zscore_by_column(filtered_acc)
        filtered_sim = zscore_by_column(filtered_sim)
        print("[Info] 已对筛选后的矩阵按列（LLM parcels）进行 z-score 归一化")
    
    if zscore_rows_human:
        filtered_acc = zscore_by_row(filtered_acc)
        filtered_sim = zscore_by_row(filtered_sim)
        print("[Info] 已对筛选后的矩阵按行（Human parcels）进行 z-score 归一化")
    
    filtered_acc.to_csv(pred_out)
    filtered_sim.to_csv(sim_out)

    summary = {
        "tag": tag,
        "filter_method": {
            "human": "rule_based",
            "llm": "threshold_based",
        },
        "thresholds": thresholds,
        "zscore_applied": {
            "columns_llm": zscore_columns,
            "rows_human": zscore_rows_human,
        },
        "input_shape": list(acc.shape),
        "filtered_shape": list(filtered_acc.shape),
        "human": {
            "kept": int(masks["human_mask"].sum()),
            "removed": int((~masks["human_mask"]).sum()),
            "kept_labels": acc.index[masks["human_mask"]].tolist(),
        },
        "llm": {
            "kept": int(masks["llm_mask"].sum()),
            "removed": int((~masks["llm_mask"]).sum()),
        },
        "outputs": {
            "prediction": str(pred_out),
            "semantic": str(sim_out),
            "summary": str(summary_out),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[Saved] {pred_out}")
    print(f"[Saved] {sim_out}")
    print(f"[Saved] {summary_out}")
    return {
        "prediction": pred_out,
        "semantic": sim_out,
        "summary": summary_out,
    }


def zscore_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个列（LLM parcel）执行 z-score 归一化。
    
    作用：
        - 消除不同 LLM parcels 之间的尺度差异
        - 使得每个 parcel 的数值分布标准化为均值 0、标准差 1
        - 有助于后续分析中公平比较不同 parcels
    
    注意：
        - 会丢失绝对数值信息（如预测准确度的绝对强度）
        - 适合用于相对比较，不适合用于绝对阈值判断
    """
    values = df.to_numpy(dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, ddof=0, keepdims=True)
    # 避免除零：如果某列标准差为 0，保持原值
    std[std == 0] = 1.0
    normalized = (values - mean) / std
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def zscore_by_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个行（Human parcel）执行 z-score 归一化。
    
    作用：
        - 消除不同 Human parcels 之间的尺度差异
        - 使得每个脑区的数值分布标准化为均值 0、标准差 1
        - 有助于后续分析中公平比较不同脑区
    
    注意：
        - 会丢失绝对数值信息（如预测准确度的绝对强度）
        - 适合用于相对比较，不适合用于绝对阈值判断
        - 与按列归一化可以同时使用（先列后行或先行后列）
    """
    values = df.to_numpy(dtype=float)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, ddof=0, keepdims=True)
    # 避免除零：如果某行标准差为 0，保持原值
    std[std == 0] = 1.0
    normalized = (values - mean) / std
    return pd.DataFrame(normalized, index=df.index, columns=df.columns)


def extract_numeric_id(label: str, expected_prefix: str) -> str:
    lower = label.lower()
    prefix = expected_prefix.lower()
    if prefix not in lower:
        return label
    try:
        return str(int(label.split("_")[-1]))
    except ValueError:
        return label


def update_mapping_file(
    mapping_path: Path,
    tag: str,
    human_labels: list[str],
    llm_labels: list[str],
    output_dir: Path,
    overwrite: bool,
) -> Path:
    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到映射文件：{mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    human_map = mapping.get("human_parcels", {})
    llm_map = mapping.get("llm_parcels", {})

    human_selected = {}
    for label in human_labels:
        key = extract_numeric_id(label, "human_parcel")
        if key in human_map:
            human_selected[key] = human_map[key]
        else:
            human_selected[key] = label
    llm_selected = {}
    for label in llm_labels:
        key = extract_numeric_id(label, "llm_parcel")
        if key in llm_map:
            llm_selected[key] = llm_map[key]
        else:
            llm_selected[key] = label

    new_mapping = {
        "human_parcels": human_selected,
        "llm_parcels": llm_selected,
    }
    mapping_out = output_dir / f"{tag}_parcel_id_to_function_name.json"
    if mapping_out.exists() and not overwrite:
        raise FileExistsError(f"{mapping_out} 已存在，使用 --overwrite 以更新。")
    mapping_out.write_text(json.dumps(new_mapping, indent=2, ensure_ascii=False))
    print(f"[Saved] {mapping_out}")
    return mapping_out


def main() -> None:
    args = parse_args()
    acc, sim = validate_inputs(args.prediction_path, args.semantic_path)
    masks = apply_thresholds(
        acc,
        sim,
        tau_l_acc=args.tau_llm_acc,
        tau_l_sim=args.tau_llm_sim,
        parcel_descriptions_path=args.parcel_descriptions_path,
    )
    # 可选：使用 LLM（基于 LeBel 任务描述）进一步筛选 human parcels
    if args.use_llm_filter:
        print("[Info] 启用 LLM 语义筛选（基于 LeBel 自然叙事任务）...")
        llm_mask = llm_filter_human_parcels(
            acc.index,
            parcel_descriptions_path=args.parcel_descriptions_path,
            api_url=args.llm_api_url,
            api_key=args.llm_api_key,
            cache_path=args.llm_filter_cache,
        )
        masks["human_mask"] = masks["human_mask"] & llm_mask
    
    # 打印筛选信息
    human_kept = masks["human_mask"].sum()
    human_total = len(masks["human_mask"])
    if args.use_llm_filter:
        print(f"[Info] Human parcels: {human_kept}/{human_total} 保留（规则 + LLM 语义联合筛选）")
    else:
        print(f"[Info] Human parcels: {human_kept}/{human_total} 保留（基于规则筛选）")
    
    llm_kept = masks["llm_mask"].sum()
    llm_total = len(masks["llm_mask"])
    print(f"[Info] LLM parcels: {llm_kept}/{llm_total} 保留（基于阈值筛选，τ_acc={args.tau_llm_acc}, τ_sim={args.tau_llm_sim}）")
    
    outputs = save_outputs(
        acc,
        sim,
        masks,
        args.tag,
        args.overwrite,
        thresholds={
            "tau_l_acc": float(args.tau_llm_acc),
            "tau_l_sim": float(args.tau_llm_sim),
        },
        output_dir=args.output_dir,
        zscore_columns=args.zscore_columns,
        zscore_rows_human=args.zscore_rows_human,
    )
    if not args.skip_mapping:
        human_labels = acc.index[masks["human_mask"]].tolist()
        llm_labels = acc.columns[masks["llm_mask"]].tolist()
        mapping_out = update_mapping_file(
            args.mapping_path,
            args.tag,
            human_labels,
            llm_labels,
            args.output_dir,
            args.overwrite,
        )
        summary_path = outputs["summary"]
        summary_data = json.loads(summary_path.read_text())
        summary_data["outputs"]["mapping"] = str(mapping_out)
        summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

