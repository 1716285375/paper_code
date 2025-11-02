# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : advantage.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:02
@Update Date    :
@Description    : 优势归一化器
将优势值标准化为零均值、单位方差的分布（无状态归一化）
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Union

import numpy as np
import torch

from .base import Normalizer


class AdvantageNormalizer(Normalizer):
    """
    优势归一化器

    将优势值标准化为零均值、单位方差的分布，并使用裁剪防止异常值。
    这是一个无状态的归一化器，每次归一化都使用当前批次的统计信息。
    """

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        """
        初始化优势归一化器

        Args:
            clip: 裁剪范围，归一化后的值将被限制在[-clip, clip]内
            eps: 数值稳定性常数，防止除零
        """
        self.clip = float(clip)
        self.eps = float(eps)

    def reset(self) -> None:
        """
        重置归一化器（无状态，无需操作）
        """
        return

    def update(self, x) -> None:
        """
        更新统计信息（无状态归一化器，无需操作）

        Args:
            x: 数据（未使用）
        """
        return

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        归一化优势值

        Args:
            x: 待归一化的优势值数组或张量

        Returns:
            归一化后的优势值，公式为 (x - mean(x)) / (std(x) + eps)，
            并裁剪到[-clip, clip]
        """
        if isinstance(x, torch.Tensor):
            mean = x.mean()
            std = x.std()
            y = (x - mean) / (std + self.eps)  # 标准化
            return torch.clamp(y, -self.clip, self.clip)  # 裁剪

        x_np = np.asarray(x)
        y = (x_np - x_np.mean()) / (x_np.std() + self.eps)
        return np.clip(y, -self.clip, self.clip)
