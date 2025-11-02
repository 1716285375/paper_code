# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : value.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:13
@Update Date    :
@Description    : 价值头实现
提供线性和MLP两种价值头实现
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.networks.mlp import MLPEncoder

from .base import BaseValueHead


class LinearValueHead(BaseValueHead):
    """
    线性价值头

    使用单个线性层直接将特征映射为价值估计。
    简单高效，适用于特征已经足够表达的情况。
    """

    def __init__(self, in_dim: int) -> None:
        """
        初始化线性价值头

        Args:
            in_dim: 输入特征维度
        """
        super().__init__()
        self.out = nn.Linear(in_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出状态价值

        Args:
            features: 输入特征，形状为 (batch_size, in_dim)

        Returns:
            状态价值估计，形状为 (batch_size,)
        """
        return self.out(features).squeeze(-1)


class MLPValueHead(BaseValueHead):
    """
    MLP价值头

    使用多层感知机将特征映射为价值估计。
    可以学习更复杂的价值函数，适用于复杂环境。
    """

    def __init__(
        self, in_dim: int, hidden_dims: Optional[List[int]] = None, activation=nn.ReLU
    ) -> None:
        """
        初始化MLP价值头

        Args:
            in_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，如果为None或空列表，则直接使用线性层
            activation: 激活函数，默认ReLU
        """
        super().__init__()
        hidden_dims = hidden_dims or []
        self.mlp = MLPEncoder(in_dim=in_dim, hidden_dims=hidden_dims, activation=activation)
        self.out = nn.Linear(self.mlp.output_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出状态价值

        Args:
            features: 输入特征，形状为 (batch_size, in_dim)

        Returns:
            状态价值估计，形状为 (batch_size,)
        """
        h = self.mlp(features)  # 通过MLP提取特征
        return self.out(h).squeeze(-1)  # 映射为标量价值
