#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parcel Activation Scaling Intervention for Jailbreak 场景

该实现与幻觉场景共用同一套逻辑：根据 `top_anomalous_parcels.json`
中的 `activation_diff` 计算 parcel 的增强/抑制系数，并在干预阶段对
激活进行缩放。
"""

import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os


class ParcelScaler:
    """Parcel 激活调节器"""

    def __init__(self,
                 parcel_json_path: str,
                 lambda_scale: float = 0.3,
                 smooth: float = 80.0,
                 min_scale: float = -1.0,
                 max_scale: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        self.parcel_json_path = parcel_json_path
        self.lambda_scale = lambda_scale
        self.smooth = smooth
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.logger = logger or logging.getLogger(__name__)

        self.scaling_table = self._load_scaling_factors()
        self.stats = self._compute_stats()

        self.logger.info(f"ParcelScaler 初始化完成，共加载 {len(self.scaling_table)} 个 parcel")
        self.logger.info(
            f"增强: {self.stats['enhance_count']} | 抑制: {self.stats['suppress_count']} | 保持: {self.stats['keep_count']}"
        )

    def _load_scaling_factors(self) -> Dict[int, float]:
        if not os.path.exists(self.parcel_json_path):
            raise FileNotFoundError(f"未找到 parcel 文件: {self.parcel_json_path}")

        with open(self.parcel_json_path, 'r', encoding='utf-8') as f:
            parcels = json.load(f)

        scaling_table = {}

        for parcel in parcels:
            if not parcel.get("is_significant", False):
                continue

            parcel_id = parcel["parcel_id"]
            activation_diff = parcel["activation_diff"]
            function_name = parcel.get("function_name", "Unknown")

            if activation_diff < 0:
                alpha = self.lambda_scale * np.tanh(-activation_diff / self.smooth)
                action = "增强"
            elif activation_diff > 0:
                alpha = -self.lambda_scale * np.tanh(activation_diff / self.smooth)
                action = "抑制"
            else:
                alpha = 0.0
                action = "保持"

            alpha = float(np.clip(alpha, self.min_scale, self.max_scale))
            scaling_table[parcel_id] = alpha

            self.logger.debug(
                f"Parcel {parcel_id} ({function_name}): activation_diff={activation_diff:.3f}, "
                f"action={action}, alpha={alpha:.3f}"
            )

        return scaling_table

    def _compute_stats(self) -> Dict[str, Any]:
        enhance_count = 0
        suppress_count = 0
        keep_count = 0

        for alpha in self.scaling_table.values():
            if alpha > 0.0:
                enhance_count += 1
            elif alpha < 0.0:
                suppress_count += 1
            else:
                keep_count += 1

        return {
            "total_parcels": len(self.scaling_table),
            "enhance_count": enhance_count,
            "suppress_count": suppress_count,
            "keep_count": keep_count,
            "avg_scale": np.mean(list(self.scaling_table.values())) if self.scaling_table else 0.0,
            "min_scale": min(self.scaling_table.values()) if self.scaling_table else 0.0,
            "max_scale": max(self.scaling_table.values()) if self.scaling_table else 0.0
        }

    def apply_scaling(self, parcel_activations: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        scaled = {}
        for parcel_id, activation in parcel_activations.items():
            scale = self.scaling_table.get(parcel_id, 1.0)
            scaled[parcel_id] = activation * scale
        return scaled

    def get_scaling_factor(self, parcel_id: int) -> float:
        return self.scaling_table.get(parcel_id, 1.0)

    def get_scaling_table(self) -> Dict[int, float]:
        return self.scaling_table.copy()

    def visualize_scaling(self) -> str:
        if not self.scaling_table:
            return "没有需要调节的 parcels"

        lines = ["Parcel Scaling 系数表:", "-" * 60, f"{'Parcel ID':<10} {'Scaling':<10} {'Action':<10}", "-" * 60]

        for parcel_id, scale in sorted(self.scaling_table.items()):
            if scale > 0.0:
                action = "增强"
            elif scale < 0.0:
                action = "抑制"
            else:
                action = "保持"
            lines.append(f"{parcel_id:<10} {scale:<10.3f} {action:<10}")

        lines.append("-" * 60)
        lines.append(
            f"统计: 总计{self.stats['total_parcels']}个, 增强{self.stats['enhance_count']}个, "
            f"抑制{self.stats['suppress_count']}个, 保持{self.stats['keep_count']}个"
        )

        return "\n".join(lines)

    def save_table(self, output_path: str):
        import pandas as pd

        data = []
        for parcel_id, scale in self.scaling_table.items():
            if scale > 0.0:
                action = "增强"
            elif scale < 0.0:
                action = "抑制"
            else:
                action = "保持"

            data.append({
                "parcel_id": parcel_id,
                "scaling_factor": scale,
                "action": action
            })

        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')


