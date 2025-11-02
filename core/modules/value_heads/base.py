# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 21:45
@Update Date    :
@Description    : 价值头（Value Head）抽象基类
定义了价值头的标准接口，用于估计状态价值
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseValueHead(nn.Module, ABC):
    """
    价值头的抽象基类

    价值头负责将编码后的特征表示映射为状态价值估计（标量值）。
    所有具体的价值头实现（线性、MLP、分布价值等）都需要继承此类。
    """

    def __init__(self) -> None:
        """
        初始化价值头
        """
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        根据特征估计状态价值

        Args:
            features: 编码后的特征张量，形状为 (batch_size, feature_dim)

        Returns:
            状态价值估计，形状为 (batch_size,) 或 (batch_size, 1)
            表示每个状态的价值（标量）
        """
        ...
