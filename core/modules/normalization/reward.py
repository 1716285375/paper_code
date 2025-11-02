# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : reward.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:05
@Update Date    :
@Description    : 奖励归一化器
使用运行时的标准差对奖励进行归一化（仅缩放，不减去均值）
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from .base import Normalizer
from .running_mean_std import RunningMeanStd


class RewardNormalizer(Normalizer):
    """
    奖励归一化器

    使用运行时的标准差对奖励进行归一化（只除以标准差，不减去均值）。
    有助于稳定训练，特别是在奖励尺度变化较大的环境中。
    """

    def __init__(self, clip: float = 10.0) -> None:
        """
        初始化奖励归一化器

        Args:
            clip: 裁剪范围，归一化后的值将被限制在[-clip, clip]内
        """
        self.rms = RunningMeanStd(())  # 标量统计
        self.clip = float(clip)

    def reset(self) -> None:
        """重置归一化器状态"""
        self.rms.reset()

    def update(self, r: Union[np.ndarray, torch.Tensor, float]) -> None:
        """
        更新归一化器的统计信息

        Args:
            r: 新的奖励值（可以是标量、数组或张量）
        """
        if isinstance(r, torch.Tensor):
            r_np = r.detach().cpu().numpy()
        else:
            r_np = np.asarray(r)
        self.rms.update(r_np)

    def normalize(
        self, r: Union[np.ndarray, torch.Tensor, float]
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """
        归一化奖励

        Args:
            r: 待归一化的奖励值

        Returns:
            归一化后的奖励，公式为 r / std，并裁剪到[-clip, clip]
            注意：只除以标准差，不减去均值（保持奖励的相对大小）
        """
        std = self.rms.std
        if isinstance(r, torch.Tensor):
            std_t = torch.as_tensor(std, device=r.device, dtype=r.dtype)
            return torch.clamp(r / std_t, -self.clip, self.clip)
        r_np = np.asarray(r)
        return np.clip(r_np / std, -self.clip, self.clip)
