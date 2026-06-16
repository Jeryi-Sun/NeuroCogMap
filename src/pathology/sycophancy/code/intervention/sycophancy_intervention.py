#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sycophancy 干预系统

思路：
- 直接复用幻觉干预系统的核心实现 (`HallucinationIntervention`)，只修改默认路径：
  - 使用谄媚(sycophancy) 的 parcel-level 分析结果
  - 使用谄媚专用的结果目录
  - 其它 SAE / 最大激活 / latent-parcel 分配与公平偏见 / 幻觉复用同一套资源

注意：
- 这里只是一个薄封装类，具体实验逻辑由 `run_intervention.py` 控制。
"""

import sys
from pathlib import Path

# 当前目录：.../safety_explanation/sycophancy/code/intervention
CURRENT_DIR = Path(__file__).resolve().parent

# hallucination 的 intervention 目录：
HALLUCINATION_INTERVENTION_DIR = (
    CURRENT_DIR.parent.parent.parent / "hallucination" / "code" / "intervention"
).resolve()

if not HALLUCINATION_INTERVENTION_DIR.exists():
    raise ImportError(
        f"无法找到 hallucination_intervention 目录: {HALLUCINATION_INTERVENTION_DIR}\n"
        f"当前文件位置: {CURRENT_DIR}\n"
        f"请检查目录结构是否正确"
    )

hallucination_dir_str = str(HALLUCINATION_INTERVENTION_DIR)
if hallucination_dir_str not in sys.path:
    sys.path.insert(0, hallucination_dir_str)

hallucination_module_file = HALLUCINATION_INTERVENTION_DIR / "hallucination_intervention.py"
if not hallucination_module_file.exists():
    raise ImportError(
        f"无法找到 hallucination_intervention.py 文件: {hallucination_module_file}\n"
        f"请确认文件路径是否正确"
    )

from hallucination_intervention import HallucinationIntervention  # type: ignore


class SycophancyIntervention(HallucinationIntervention):
    """Sycophancy 场景专用干预系统（复用幻觉版本的实现，只改默认路径）"""

    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        sae_release: str = "gemma-scope-9b-it-res",
        sae_local_base_dir: str = "/path/to/local_models/gemma-scope-9b-it-res",
        parcel_json_path: str = (
            "/path/to/project_root/"
            "safety_explanation/sycophancy/results/analysis_output/"
            "answer_gemma-2-9b-it/parcel_level/top_anomalous_parcels.json"
        ),
        latent_parcel_assignments_path: str = (
            "/path/to/project_root/"
            "neural_area/divide_area_by_sae_act/cluster_output_9b_it/"
            "clustering_results_sentence_prep0.03_0.8_svdvar0p80_parcels20_iter50_spatial0.01_nparcels270/"
            "latent_parcel_assignments.json"
        ),
        max_activation_dir: str = (
            "/path/to/project_root/"
            "neural_area/connect_cap_parcel/results/steer_activation"
        ),
        results_dir: str = (
            "/path/to/project_root/"
            "safety_explanation/sycophancy/results/intervention"
        ),
        is_instruct: bool = True,
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

    # 这里显式重写评价接口，避免日后不小心复用幻觉场景下的“正确率型”评价逻辑。
    # 当前版本仅负责生成基线/干预输出，本身不做谄媚打分；
    # 真正的 sycophancy 评估请复用 `sycophancy_eval.py` 中的反馈评估逻辑，
    # 其中已经实现了“正反两个方向都判为谄媚才记为谄媚”的规则。
    def evaluate_intervention(  # type: ignore[override]
        self,
        test_data,
        parcel_ids,
        intervention_strength: float = 1.0,
        use_incontext: bool = False,
    ):
        """
        对给定 parcel 与干预强度，生成基线与干预后的回答并做简单统计。

        注意：
        - 本方法**不做谄媚判定**，只负责把生成结果整理出来；
        - 谄媚标签应通过 `sycophancy_eval.py` 的 feedback 评估脚本来计算，
          且该脚本已经按照你的要求：只有正反两个方向都判为谄媚时才记为 sycophancy = 1。
        """
        # 直接调用父类的生成与长度统计逻辑，返回结构不变，后续由专门的评估脚本处理谄媚标签。
        return super().evaluate_intervention(
            test_data=test_data,
            parcel_ids=parcel_ids,
            intervention_strength=intervention_strength,
            use_incontext=use_incontext,
        )

