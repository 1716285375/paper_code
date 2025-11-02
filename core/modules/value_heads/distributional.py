# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : distributional.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:13
@Update Date    :
@Description    : 分布价值头
实现C51风格的分类分布价值估计
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.networks.mlp import MLPEncoder

from .base import BaseValueHead


class CategoricalValueHead(BaseValueHead):
    """
    分类分布价值头（C51风格）

    将价值估计建模为分类分布，而不是单一标量。
    使用固定的支撑点（support points），输出每个支撑点的概率，
    最终返回期望价值。这种方法可以减少价值估计的方差。
    """

    def __init__(
        self,
        in_dim: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        hidden_dims: Optional[list[int]] = None,
    ) -> None:
        """
        初始化分类价值头

        Args:
            in_dim: 输入特征维度
            num_atoms: 支撑点数量（分类分布的类别数），默认51
            v_min: 价值范围下界
            v_max: 价值范围上界
            hidden_dims: 隐藏层维度列表
        """
        super().__init__()
        hidden_dims = hidden_dims or []
        self.mlp = MLPEncoder(in_dim=in_dim, hidden_dims=hidden_dims, activation=nn.ReLU)
        self.logits = nn.Linear(self.mlp.output_dim, num_atoms)  # 输出每个支撑点的logits
        # 注册支撑点（固定的价值范围）
        self.register_buffer("supports", torch.linspace(v_min, v_max, num_atoms))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回期望价值

        Args:
            features: 输入特征，形状 (batch_size, in_dim)

        Returns:
            期望价值，形状 (batch_size,)
            计算公式：E[V] = sum(probs * supports)
        """
        h = self.mlp(features)
        logits = self.logits(h)  # 计算每个支撑点的logits
        probs = torch.softmax(logits, dim=-1)  # 转换为概率分布
        # 返回期望价值：概率加权平均
        return torch.sum(probs * self.supports, dim=-1)
