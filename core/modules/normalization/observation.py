# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : observation.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:02
@Update Date    :
@Description    : 观测归一化器
使用运行时的均值和标准差对观测进行归一化
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from .base import Normalizer
from .running_mean_std import RunningMeanStd


class ObservationNormalizer(Normalizer):
    """
    观测归一化器

    使用运行时的均值和标准差对观测进行归一化（标准化）。
    有助于稳定训练，特别适用于观测值范围变化较大的环境。
    """

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0) -> None:
        """
        初始化观测归一化器

        Args:
            shape: 观测数据的形状（用于多维观测）
            clip: 裁剪范围，归一化后的值将被限制在[-clip, clip]内
        """
        self.rms = RunningMeanStd(shape)  # 运行时统计计算器
        self.clip = float(clip)

    def reset(self) -> None:
        """重置归一化器状态"""
        self.rms.reset()

    def update(self, x: Union[np.ndarray, torch.Tensor]) -> None:
        """
        更新归一化器的统计信息

        Args:
            x: 新的观测数据（批次或单个观测）
        """
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)
        self.rms.update(x_np)

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        归一化观测数据

        Args:
            x: 待归一化的观测数据

        Returns:
            归一化后的观测数据，公式为 (x - mean) / std，并裁剪到[-clip, clip]
        """
        mean = self.rms.mean
        std = self.rms.std

        if isinstance(x, torch.Tensor):
            mean_t = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std_t = torch.as_tensor(std, device=x.device, dtype=x.dtype)
            y = (x - mean_t) / std_t
            return torch.clamp(y, -self.clip, self.clip)

        x_np = (np.asarray(x) - mean) / std
        return np.clip(x_np, -self.clip, self.clip)
