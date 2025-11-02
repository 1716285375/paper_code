# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : continuous.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 16:30
@Update Date    :
@Description    : 连续动作策略头
将特征映射为连续动作的高斯分布
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from core.networks.mlp import MLPEncoder

from .base import BasePolicyHead


class DiagGaussianPolicyHead(BasePolicyHead):
    """
    对角高斯策略头

    用于连续动作空间，输出对角高斯分布（均值-方差形式）。
    支持可选的tanh压缩，将动作限制在[-1, 1]范围内。
    """

    def __init__(
        self,
        in_dim: int,
        action_dim: int,
        hidden_dims: Optional[List[int]] = None,
        log_std_init: float = -0.5,
        squashed: bool = False,
    ) -> None:
        """
        初始化对角高斯策略头

        Args:
            in_dim: 输入特征维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表
            log_std_init: 对数标准差的初始值
            squashed: 是否使用tanh压缩动作到[-1, 1]
        """
        super().__init__()
        hidden_dims = hidden_dims or []
        self.mlp = MLPEncoder(in_dim=in_dim, hidden_dims=hidden_dims, activation=nn.ReLU)
        self.mean = nn.Linear(self.mlp.output_dim, action_dim)  # 均值网络
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)  # 可学习的对数标准差
        self.squashed = squashed

    def forward(self, features: torch.Tensor) -> Distribution:
        """
        前向传播，输出连续动作分布

        Args:
            features: 输入特征，形状为 (batch_size, in_dim)

        Returns:
            连续动作分布：
                - 如果squashed=True：经过tanh变换的分布（动作在[-1, 1]）
                - 如果squashed=False：标准的高斯分布
        """
        h = self.mlp(features)  # 通过MLP提取特征
        mean = self.mean(h)  # 计算均值
        std = torch.exp(self.log_std).clamp_min(1e-6)  # 计算标准差（确保为正）
        base = Normal(mean, std)  # 基础高斯分布

        # 如果使用tanh压缩，则应用变换
        if self.squashed:
            return TransformedDistribution(base, [TanhTransform(cache_size=1)])
        return base
