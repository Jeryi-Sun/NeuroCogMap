#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jailbreak 干预系统

基于幻觉干预系统的实现，对参数进行适配，便于在 jailbreak 数据集上复用。

主要改动：
- 复用 `HallucinationIntervention` 主类
- 调整默认目录（analysis/output、results、dataset 等）
- 保持与幻觉版本一致的接口
"""

import os
import sys
from pathlib import Path

# 构建 hallucination_intervention 模块的绝对路径
CURRENT_DIR = Path(__file__).resolve().parent
# 从 jailbreak/code/intervention 向上三级到 safety_explanation，然后进入 hallucination/code/intervention
HALLUCINATION_INTERVENTION_DIR = (
    CURRENT_DIR.parent.parent.parent / "hallucination" / "code" / "intervention"
).resolve()

# 确保路径存在
if not HALLUCINATION_INTERVENTION_DIR.exists():
    raise ImportError(
        f"无法找到 hallucination_intervention 目录: {HALLUCINATION_INTERVENTION_DIR}\n"
        f"当前文件位置: {CURRENT_DIR}\n"
        f"请检查目录结构是否正确"
    )

# 添加到 sys.path
hallucination_dir_str = str(HALLUCINATION_INTERVENTION_DIR)
if hallucination_dir_str not in sys.path:
    sys.path.insert(0, hallucination_dir_str)

# 验证模块文件是否存在
hallucination_module_file = HALLUCINATION_INTERVENTION_DIR / "hallucination_intervention.py"
if not hallucination_module_file.exists():
    raise ImportError(
        f"无法找到 hallucination_intervention.py 文件: {hallucination_module_file}\n"
        f"请确认文件路径是否正确"
    )

from hallucination_intervention import HallucinationIntervention  # type: ignore


class JailbreakIntervention(HallucinationIntervention):
    """Jailbreak 场景专用干预系统（复用幻觉版本的实现）"""

    def __init__(
        self,
        model_name: str = "google/gemma-2-2b",
        sae_release: str = "gemma-scope-2b-pt-res",
        sae_local_base_dir: str = "/path/to/local_models/gemma-scope-2b-pt-res",
        parcel_json_path: str = "/path/to/project_root/safety_explanation/jailbreak/results/analysis_output/JBB-Behaviors_gemma-2-2b/parcel_level/top_anomalous_parcels.json",
        latent_parcel_assignments_path: str = "/path/to/project_root/neural_area/divide_area_by_sae_act/cluster_output_2b_pt/clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/latent_parcel_assignments.json",
        max_activation_dir: str = "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation",
        results_dir: str = "/path/to/project_root/safety_explanation/jailbreak/results/intervention",
        is_instruct: bool = False,
        lambda_scale: float = 0.3,
        smooth: float = 80.0,
        min_scale: float = -1.0,
        max_scale: float = 1.0,
        strength: float = 0.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            sae_release=sae_release,
            sae_local_base_dir=sae_local_base_dir,
            parcel_json_path=parcel_json_path,
            latent_parcel_assignments_path=latent_parcel_assignments_path,
            max_activation_dir=max_activation_dir,
            results_dir=results_dir,
            is_instruct=is_instruct,
            lambda_scale=lambda_scale,
            smooth=smooth,
            min_scale=min_scale,
            max_scale=max_scale,
            strength=strength,
        )


