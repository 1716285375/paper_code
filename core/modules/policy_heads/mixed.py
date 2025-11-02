# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : mixed.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:10
@Update Date    :
@Description    : 混合动作空间策略头
组合多个策略头，用于处理混合动作空间（同时包含离散和连续动作）
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BasePolicyHead


class MixedPolicyHead(BasePolicyHead):
    """
    混合动作空间策略头

    组合多个策略头，用于处理混合动作空间（例如同时包含离散和连续动作）。
    每个动作维度都有独立的策略头，返回动作分布字典。

    Args:
        heads: 策略头字典，格式为 {动作名称: BasePolicyHead实例}
            例如：{"discrete": DiscretePolicyHead(...), "continuous": DiagGaussianPolicyHead(...)}

    forward返回: {动作名称: 动作分布} 字典
    """

    def __init__(self, heads: Dict[str, BasePolicyHead]) -> None:
        """
        初始化混合策略头

        Args:
            heads: 策略头字典，每个键对应一个动作维度的策略头
        """
        super().__init__()
        self.heads = nn.ModuleDict(heads)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.distributions.Distribution]:
        """
        前向传播，输出多个动作分布

        Args:
            features: 输入特征，形状 (batch_size, feature_dim)

        Returns:
            动作分布字典，格式为 {动作名称: 动作分布}
        """
        return {name: head(features) for name, head in self.heads.items()}
