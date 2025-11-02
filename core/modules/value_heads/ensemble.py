# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : ensemble.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:13
@Update Date    :
@Description    : 集成价值头
使用多个价值头的平均值来减少方差
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .base import BaseValueHead


class EnsembleValueHead(BaseValueHead):
    """
    集成价值头

    使用多个价值头的集合，通过平均它们的输出来减少价值估计的方差。
    集成方法可以提高价值估计的稳定性和准确性。
    """

    def __init__(self, heads: List[BaseValueHead]) -> None:
        """
        初始化集成价值头

        Args:
            heads: 价值头列表，将对这些价值头的输出进行平均
        """
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回多个价值头的平均值

        Args:
            features: 输入特征，形状 (batch_size, feature_dim)

        Returns:
            平均后的价值估计，形状 (batch_size,)
        """
        values = [head(features) for head in self.heads]  # 计算所有价值头的输出
        return torch.stack(values, dim=-1).mean(dim=-1)  # 堆叠并求平均
