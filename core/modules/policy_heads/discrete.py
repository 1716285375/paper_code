# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : discrete.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 16:31
@Update Date    :
@Description    : 离散动作策略头
将特征映射为离散动作的Categorical分布
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch.distributions import Categorical

from core.networks.mlp import MLPEncoder

from .base import BasePolicyHead


class DiscretePolicyHead(BasePolicyHead):
    """
    离散动作策略头

    使用MLP将特征映射为离散动作的logits，然后构造Categorical分布。
    适用于离散动作空间（如游戏中的移动方向、技能选择等）。
    """

    def __init__(
        self, in_dim: int, action_dim: int, hidden_dims: Optional[List[int]] = None
    ) -> None:
        """
        初始化离散策略头

        Args:
            in_dim: 输入特征维度
            action_dim: 动作空间大小（离散动作的数量）
            hidden_dims: 隐藏层维度列表，如果为None或空列表，则直接使用线性层映射
        """
        super().__init__()
        hidden_dims = hidden_dims or []
        self.mlp = MLPEncoder(in_dim=in_dim, hidden_dims=hidden_dims, activation=nn.ReLU)
        self.out = nn.Linear(self.mlp.output_dim, action_dim)

    def forward(self, features: torch.Tensor) -> Categorical:
        """
        前向传播，输出离散动作分布

        Args:
            features: 输入特征，形状为 (batch_size, in_dim)

        Returns:
            Categorical分布对象，可以用于采样动作或计算概率
        """
        h = self.mlp(features)  # 通过MLP提取特征
        logits = self.out(h)  # 映射为动作logits
        return Categorical(logits=logits)  # 构造Categorical分布
