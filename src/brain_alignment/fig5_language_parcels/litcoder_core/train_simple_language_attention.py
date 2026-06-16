#!/usr/bin/env python3
import argparse
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging

from encoding.assembly.assembly_loader import load_assembly
from encoding.features.factory import FeatureExtractorFactory
from encoding.downsample.downsampling import Downsampler
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train language model on LeBel dataset (extracting Attention hidden state)")
    parser.add_argument("--assembly_path", type=str, default="./")
    parser.add_argument("--layer_idx", type=int, default=13, help="Layer to extract features from")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--cache_dir", type=str, default="cache_language_model_attention")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--wandb_project", type=str, default="lebel-language-model-attention")
    parser.add_argument("--logger_backend", type=str, default="wandb")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--reverse_order", action="store_true", default=False)
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--parcel_level", action="store_true", default=False)
    parser.add_argument("--roi_level", action="store_true", default=False)
    parser.add_argument("--seven_net_level", action="store_true", default=False)
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
    parser.add_argument("--dataset_type", type=str, default="narratives")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Loading assembly from {args.assembly_path}")
    assembly = load_assembly(args.assembly_path)

    extractor = FeatureExtractorFactory.create_extractor(
        modality="language_model",
        model_name=args.model_name,
        config={
            "model_name": args.model_name,
            "layer_idx": args.layer_idx,
            "hook_type": "hook_attn_out",  # 使用 Attention 输出而不是 hidden state
            "last_token": True,
            "lookback": args.lookback,
            "context_type": "fullcontext",
        },
        cache_dir=args.cache_dir,
    )

    downsampler = Downsampler()
    model = NestedCVModel(model_name="ridge_regression")

    fir_delays = [1, 2]
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
        layer_idx=args.layer_idx,
        lookback=args.lookback,
        forward_fill_zeros=True,
        parcel_projection_config=parcel_cfg if args.parcel_level else None,
        seven_network_projection_config=seven_net_cfg if args.seven_net_level else None,
        roi_projection_config=roi_cfg if args.roi_level else None,
        test_story_from_end=args.test_story_from_end,
        eval_story_from_end=args.eval_story_from_end,
    )

    logger.info(f"Starting training with language model layer {args.layer_idx} (Attention hidden state)...")
    metrics = trainer.train()

    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")
    if "n_significant" in metrics:
        logger.info(f"Significant voxels: {metrics['n_significant']}")


if __name__ == "__main__":
    main()
