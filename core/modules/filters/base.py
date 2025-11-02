# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 过滤模块基类
"""
# ------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseFilter(nn.Module, ABC):
    """
    过滤模块基类
    用于在中央价值函数输入上做特征过滤（Hadamard乘）
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        应用过滤掩码

        Args:
            features: 输入特征，形状为 (B, feature_dim)

        Returns:
            过滤后的特征，形状与输入相同
        """
        raise NotImplementedError


