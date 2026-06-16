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
    parser = argparse.ArgumentParser(description="Train embedding model (word2vec/glove) on LeBel dataset")
    parser.add_argument("--assembly_path", type=str, default="./")
    parser.add_argument("--vector_path", type=str, default="/path/to/local_models/word2vec/nlwiki_20180420_300d.txt", help="Path to embedding vectors file")
    parser.add_argument("--embedding_model", type=str, default="word2vec", help="Embedding model type: word2vec, glove, etc.")
    parser.add_argument("--cache_dir", type=str, default="cache_embeddings")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--wandb_project", type=str, default="lebel-embeddings")
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
        modality="embeddings",
        model_name=args.embedding_model,
        config={
            "vector_path": args.vector_path,
            "binary": False,  # Set to True for .bin files, False for .txt files
            "lowercase": False,  # Set to True if your embeddings expect lowercase tokens
            "oov_handling": "copy_prev",  # How to handle out-of-vocabulary words
            "use_tqdm": True,  # Show progress bar
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
        lookback=args.lookback,
        forward_fill_zeros=True,
        parcel_projection_config=parcel_cfg if args.parcel_level else None,
        seven_network_projection_config=seven_net_cfg if args.seven_net_level else None,
        roi_projection_config=roi_cfg if args.roi_level else None,
        test_story_from_end=args.test_story_from_end,
    )


    logger.info(f"Starting training with embeddings...")
    metrics = trainer.train()

    logger.info("\n=== Final Results ===")
    logger.info(f"Median correlation: {metrics.get('median_score', float('nan')):.4f}")
    if "n_significant" in metrics:
        logger.info(f"Significant voxels: {metrics['n_significant']}")


if __name__ == "__main__":
    main()