# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 21:42
@Update Date    :
@Description    : 策略头（Policy Head）抽象基类
定义了策略头的标准接口，用于将特征表示映射为动作分布
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BasePolicyHead(nn.Module, ABC):
    """
    策略头的抽象基类

    策略头负责将编码后的特征表示映射为动作分布（如离散动作的Categorical分布或连续动作的Gaussian分布）。
    所有具体的策略头实现（离散、连续、混合等）都需要继承此类。
    """

    def __init__(self) -> None:
        """
        初始化策略头
        """
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.distributions.Distribution:
        """
        根据特征输出动作分布

        Args:
            features: 编码后的特征张量，形状为 (batch_size, feature_dim)

        Returns:
            torch.distributions.Distribution 对象，表示动作的概率分布
            例如：Categorical（离散动作）或 Normal（连续动作）
        """
        ...
