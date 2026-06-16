#!/usr/bin/env python3
import argparse
import os
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging

from encoding.assembly.assembly_loader import load_assembly
from encoding.features.factory import FeatureExtractorFactory
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer


def _parse_parcel_ids(parcel_ids_str: str) -> list[int]:
    parcel_ids: list[int] = []
    if not parcel_ids_str.strip():
        return parcel_ids
    for token in parcel_ids_str.split(","):
        token = token.strip()
        if not token:
            continue
        parcel_ids.append(int(token))
    return sorted(set(parcel_ids))


def _load_all_parcel_ids_from_mapping(parcel_mapping_path: str) -> list[int]:
    with open(parcel_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    parcel_to_latents = mapping_data.get("parcel_to_latents")
    if not isinstance(parcel_to_latents, dict) or not parcel_to_latents:
        raise ValueError(
            "parcel_mapping_path 中缺少 parcel_to_latents 或内容为空，无法构建全 Parcel 列表"
        )
    parcel_ids: list[int] = []
    for parcel_name in parcel_to_latents.keys():
        if isinstance(parcel_name, str) and parcel_name.startswith("parcel_"):
            parcel_ids.append(int(parcel_name.split("_")[-1]))
        else:
            raise ValueError(f"无法解析 parcel 名称: {parcel_name}")
    if not parcel_ids:
        raise ValueError("未在映射文件中解析到任何 parcel_id")
    return sorted(set(parcel_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE activation-based model on LeBel dataset")
    parser.add_argument("--assembly_path", type=str, default="./")
    parser.add_argument("--parcel_id", type=int, default=61)
    parser.add_argument("--parcel_mapping_path", type=str, required=True)
    parser.add_argument("--sae_release", type=str, default="gemma-scope-2b-pt-res")
    parser.add_argument("--sae_local_base_dir", type=str, required=True)
    parser.add_argument("--sae_paths", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="cache_saeact_model")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--wandb_project", type=str, default="lebel-saeact-model")
    parser.add_argument("--logger_backend", type=str, default="wandb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reverse_order", action="store_true", default=False)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--parcel_level", action="store_true", default=False)
    parser.add_argument("--roi_level", action="store_true", default=False)
    parser.add_argument("--seven_net_level", action="store_true", default=False)
    parser.add_argument("--all_parcels", dest="all_parcels", action="store_true")
    parser.add_argument("--single_parcel", dest="all_parcels", action="store_false")
    parser.set_defaults(all_parcels=True)
    parser.add_argument(
        "--parcel_ids",
        type=str,
        default="",
        help="逗号分隔的 parcel id 列表。all_parcels 模式下优先使用该列表，否则从 mapping 自动读取全部 parcel。",
    )
    parser.add_argument(
        "--test_story_from_end",
        type=int,
        default=1,
        help="倒数第 N 个故事作为测试集 (默认 1，即最后一个)",
    )
    parser.add_argument(
        "--eval_story_from_end",
        type=int,
        default=None,
        help="从训练故事中再留出倒数第 N 个故事作为 evaluation 集；不设置则不创建 evaluation 集",
    )
    parser.add_argument("--use_train_test_split", action="store_true", default=False)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--dataset_type", type=str, default="narratives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Loading assembly from {args.assembly_path}")
    assembly = load_assembly(args.assembly_path)

    sae_paths = [p.strip() for p in args.sae_paths.split(",") if p.strip()]
    configured_parcel_ids = _parse_parcel_ids(args.parcel_ids)
    use_multi_parcels = args.all_parcels
    if use_multi_parcels:
        selected_parcel_ids = (
            configured_parcel_ids
            if configured_parcel_ids
            else _load_all_parcel_ids_from_mapping(args.parcel_mapping_path)
        )
        logger.info(
            "Using multi-parcel mode with %d parcels. Preview: %s",
            len(selected_parcel_ids),
            selected_parcel_ids[:10],
        )
    else:
        selected_parcel_ids = [args.parcel_id]
        logger.info("Using single-parcel mode with parcel_id=%d", args.parcel_id)

    extractor_config = {
        "parcel_mapping_path": args.parcel_mapping_path,
        "sae_release": args.sae_release,
        "sae_local_base_dir": args.sae_local_base_dir,
        "sae_paths": sae_paths,
        "last_token": True,
    }
    if use_multi_parcels:
        extractor_config["parcel_ids"] = selected_parcel_ids
    else:
        extractor_config["parcel_id"] = args.parcel_id

    extractor = FeatureExtractorFactory.create_extractor(
        modality="saeact_model",
        model_name=args.model_name,
        config=extractor_config,
        cache_dir=args.cache_dir,
    )

    downsampler = Downsampler()
    model = NestedCVModel(model_name="ridge_regression")

    fir_delays = [1, 2, 3, 4]
    trimming_config = {
        "train_features_start": 10,
        "train_features_end": -5,
        "train_targets_start": 0,
        "train_targets_end": None,
        "test_features_start": 50,
        "test_features_end": -5,
        "test_targets_start": 40,
        "test_targets_end": None,
    }
    parcel_cfg = {
        "lh_annot_path": "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/annotation/lh.Schaefer2018_100Parcels_7Networks_order.annot",
        "rh_annot_path": "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/annotation/rh.Schaefer2018_100Parcels_7Networks_order.annot",
    }
    seven_net_cfg = {
        "lh_annot_path": parcel_cfg["lh_annot_path"],
        "rh_annot_path": parcel_cfg["rh_annot_path"],
    }
    roi_cfg = {
    "mask_left_path": "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/roi_masks/language/language_mask_left.npy",
    "mask_right_path": "/path/to/project_root/Human_LLM_align/litcoder_core/dataset/roi_masks/language/language_mask_right.npy",
    "combine_hemispheres": True,  # True: 返回单个值，False: 返回两个值
    "use_nanmean": True,  # 是否使用 nanmean
}
    trainer = AbstractTrainer(
        assembly=assembly,
        feature_extractors=[extractor],
        downsampler=downsampler,
        model=model,
        fir_delays=fir_delays,
        trimming_config=trimming_config,
        use_train_test_split=args.use_train_test_split,
        logger_backend=args.logger_backend,
        wandb_project_name=args.wandb_project,
        dataset_type=args.dataset_type,
        results_dir=args.results_dir,
        reverse_order=args.reverse_order,
        layer_idx=-1 if use_multi_parcels else args.parcel_id,
        lookback=args.lookback,
        forward_fill_zeros=True,
        parcel_projection_config=parcel_cfg if args.parcel_level else None,
        seven_network_projection_config=seven_net_cfg if args.seven_net_level else None,
        roi_projection_config=roi_cfg if args.roi_level else None,
        test_story_from_end=args.test_story_from_end,
        eval_story_from_end=args.eval_story_from_end,
    )

    logger.info("Starting training with SAE activation-based parcel features...")
    metrics = trainer.train()

    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")
    if "n_significant" in metrics:
        logger.info(f"Significant voxels: {metrics['n_significant']}")


if __name__ == "__main__":
    main()

