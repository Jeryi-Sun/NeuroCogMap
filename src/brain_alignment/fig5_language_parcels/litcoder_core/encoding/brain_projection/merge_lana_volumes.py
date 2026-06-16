#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 LanA 左/右半球概率体积合并为单个 NIfTI 文件。

合并策略：对两个体积取逐元素最大值（max），保持仿射矩阵一致。
"""

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib


def merge_volumes(left_path: Path, right_path: Path, output_path: Path) -> None:
    left_img = nib.load(str(left_path))
    right_img = nib.load(str(right_path))

    if not np.allclose(left_img.affine, right_img.affine):
        raise ValueError("左、右半球 LanA 体积的仿射矩阵不一致，无法直接合并")

    left_data = left_img.get_fdata()
    right_data = right_img.get_fdata()
    merged_data = np.maximum(left_data, right_data)

    merged_img = nib.Nifti1Image(merged_data, left_img.affine, left_img.header)
    merged_img.to_filename(str(output_path))
    print(f"[INFO] 合并完成，输出文件: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并 LanA 左/右半球概率体积"
    )
    parser.add_argument("--left", type=str, required=True, help="左半球 NIfTI 路径")
    parser.add_argument("--right", type=str, required=True, help="右半球 NIfTI 路径")
    parser.add_argument("--output", type=str, required=True, help="输出 NIfTI 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_volumes(Path(args.left), Path(args.right), Path(args.output))


if __name__ == "__main__":
    main()

