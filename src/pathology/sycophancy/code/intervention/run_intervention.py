#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sycophancy 干预运行脚本

整体结构参考 fairness_bias 的 run_intervention.py：
- 提供单实验 (--mode single) 与批量实验 (--mode batch) 两种模式；
- 配置由 JSON 文件提供（model / SAE / parcel 路径等）；
- 干预核心由 `SycophancyIntervention` 负责；
- 数据加载由 `SycophancyDataLoader` 负责；
- 评估器暂未实现（enable_evaluation 默认为 false）。
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# 优先从当前目录导入
CURRENT_DIR = Path(__file__).resolve().parent
current_dir_str = str(CURRENT_DIR)
if current_dir_str not in sys.path:
    sys.path.insert(0, current_dir_str)
elif sys.path.index(current_dir_str) != 0:
    sys.path.remove(current_dir_str)
    sys.path.insert(0, current_dir_str)

from data_loader import SycophancyDataLoader  # noqa: E402
from sycophancy_intervention import SycophancyIntervention  # noqa: E402


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("sycophancy_intervention")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_experiment(
    config: Dict[str, Any],
    dataset_path: str,
    parcel_ids: List[int],
    intervention_strength: float = 1.0,
    max_samples: int = 0,
    enable_evaluation: bool = False,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    logger = logging.getLogger("sycophancy_intervention")

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    model_name = config.get("model_name", "google/gemma-2-9b-it")

    # sycophancy 目前 parcel_json_path_template 与 dataset_name 关系较弱：
    # 默认直接使用 config_9b.json 里给出的路径（特定于 answer_gemma-2-9b-it）。
    parcel_json_path = config.get("parcel_json_path_template") or config.get("parcel_json_path")
    logger.info(f"使用 parcel_json_path: {parcel_json_path}")

    intervention = SycophancyIntervention(
        model_name=model_name,
        sae_release=config.get("sae_release", "gemma-scope-9b-it-res"),
        sae_local_base_dir=config.get(
            "sae_local_base_dir",
            "/path/to/local_models/gemma-scope-9b-it-res",
        ),
        parcel_json_path=parcel_json_path,
        latent_parcel_assignments_path=config.get("latent_parcel_assignments_path"),
        max_activation_dir=config.get(
            "max_activation_dir",
            "/path/to/project_root/neural_area/connect_cap_parcel/results/steer_activation",
        ),
        results_dir=config.get(
            "results_dir",
            "/path/to/project_root/safety_explanation/sycophancy/results/intervention",
        ),
        is_instruct=config.get("is_instruct", True),
        lambda_scale=config.get("lambda_scale", 0.3),
        smooth=config.get("smooth", 80.0),
        min_scale=config.get("min_scale", -1.0),
        max_scale=config.get("max_scale", 1.0),
        strength=intervention_strength,
    )

    logger.info("加载模型和 SAE...")
    intervention.load_model_and_tokenizer()
    intervention.load_saes(
        config.get("sae_paths", []),
        max_saes=config.get("max_saes", 10),
    )

    logger.info(f"加载测试数据: {dataset_path}")
    data_loader = SycophancyDataLoader(logger)
    test_data = data_loader.load_dataset(dataset_path, max_samples)

    use_incontext = config.get("use_incontext", False)

    logger.info(f"开始干预: parcels={parcel_ids}, strength={intervention_strength}")
    results = intervention.run_intervention_experiment(
        test_data,
        parcel_ids,
        save_results=False,
        use_incontext=use_incontext,
    )

    if enable_evaluation:
        logger.warning("SycophancyEvaluator 尚未实现，当前版本不支持自动评估谄媚率变化。")

    results["dataset_info"] = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "model_name": model_name,
        "parcel_ids": parcel_ids,
        "intervention_strength": intervention_strength,
        "max_samples": max_samples,
        "enable_evaluation": enable_evaluation,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"实验结果已保存至: {output_path}")
    else:
        print("\n" + "=" * 50)
        print("EXPERIMENT_RESULTS_JSON_START")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        print("EXPERIMENT_RESULTS_JSON_END")
        print("=" * 50)

    return results


def run_batch_experiments(
    config: Dict[str, Any],
    dataset_paths: List[str],
    parcel_id_groups: List[List[int]],
    intervention_strengths: Optional[List[float]] = None,
) -> Dict[str, Any]:
    logger = logging.getLogger("sycophancy_intervention")

    if intervention_strengths is None:
        intervention_strengths = [0.3, 0.5, 1.0]

    all_results: Dict[str, Any] = {
        "batch_info": {
            "num_datasets": len(dataset_paths),
            "num_parcel_groups": len(parcel_id_groups),
            "intervention_strengths": intervention_strengths,
        },
        "experiments": {},
    }

    for dataset_path in dataset_paths:
        dataset_name = Path(dataset_path).stem
        logger.info(f"处理数据集: {dataset_name}")

        dataset_results: Dict[str, Any] = {}

        for parcel_ids in parcel_id_groups:
            parcel_key = "_".join(map(str, parcel_ids))
            dataset_results[parcel_key] = {}

            for strength in intervention_strengths:
                result = run_single_experiment(
                    config,
                    dataset_path,
                    parcel_ids,
                    strength,
                    max_samples=config.get("max_samples", 0),
                    enable_evaluation=config.get("enable_evaluation", False),
                )
                dataset_results[parcel_key][str(strength)] = result

        all_results["experiments"][dataset_name] = dataset_results

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Sycophancy 干预实验运行脚本")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--dataset", type=str, help="单个数据集 JSONL 路径")
    parser.add_argument("--parcel_ids", nargs="*", type=int, help="Parcel ID 列表")
    parser.add_argument("--intervention_strength", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--enable_evaluation", action="store_true")
    parser.add_argument("--output_file", type=str, help="结果文件保存路径（JSON 格式）")
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    config = load_config(args.config)

    if args.mode == "single":
        if not args.dataset:
            raise ValueError("单实验模式必须提供 --dataset")
        if not args.parcel_ids:
            raise ValueError("单实验模式必须提供 --parcel_ids")

        run_single_experiment(
            config,
            args.dataset,
            args.parcel_ids,
            intervention_strength=args.intervention_strength,
            max_samples=args.max_samples or config.get("max_samples", 0),
            enable_evaluation=args.enable_evaluation or config.get("enable_evaluation", False),
            output_file=args.output_file,
        )
    else:
        dataset_dir = config.get("dataset_dir")
        if not dataset_dir or not os.path.isdir(dataset_dir):
            raise ValueError("批量模式需要在配置文件中提供 dataset_dir（包含 JSONL 文件）")

        dataset_paths = sorted(str(p) for p in Path(dataset_dir).glob("*.jsonl"))
        if not dataset_paths:
            raise ValueError(f"未在 {dataset_dir} 找到 JSONL 数据集")

        parcel_groups = config.get("parcel_groups", [])
        if not parcel_groups:
            raise ValueError("批量模式需要在配置中提供 parcel_groups")

        strengths = config.get("intervention_strengths", [args.intervention_strength])

        results = run_batch_experiments(
            config,
            dataset_paths,
            parcel_groups,
            strengths,
        )

        results_path = Path(config.get("results_dir", ".")) / "batch_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"批量实验结果已保存到: {results_path}")


if __name__ == "__main__":
    main()

