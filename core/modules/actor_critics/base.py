# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 10:52
@Update Date    :
@Description    : Actor-Critic网络基类接口
"""
# ------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Distribution


class BaseActorCritic(nn.Module, ABC):
    """
    Actor-Critic网络的基类接口
    定义统一的接口，支持不同的共享策略
    """

    @abstractmethod
    def forward(
        self, obs: torch.Tensor, hidden: Optional[Any] = None
    ) -> Tuple[Distribution, torch.Tensor, Optional[Any]]:
        """
        前向传播，同时输出策略分布和价值估计

        Args:
            obs: 观测张量
            hidden: 可选的隐藏状态（用于RNN/LSTM）

        Returns:
            distribution: 动作分布
            value: 状态价值估计
            hidden: 更新后的隐藏状态（如果有）
        """
        ...

    @abstractmethod
    def get_actor_parameters(self):
        """返回Actor相关参数（用于优化器）"""
        ...

    @abstractmethod
    def get_critic_parameters(self):
        """返回Critic相关参数（用于优化器）"""
        ...

    @abstractmethod
    def get_shared_parameters(self):
        """返回共享参数（如果有）"""
        ...


class BasePolicyHead(nn.Module, ABC):
    """
    策略头接口
    从特征中输出动作分布
    """

    @abstractmethod
    def forward(self, features: torch.Tensor) -> Distribution:
        """从特征输出动作分布"""
        ...


class BaseValueHead(nn.Module, ABC):
    """
    价值头接口
    从特征中输出状态价值
    """

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """从特征输出状态价值"""
        ...
