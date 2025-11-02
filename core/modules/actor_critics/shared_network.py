# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : shared_network.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 10:52
@Update Date    :
@Description    : Actor-Critic网络实现
提供多种Actor-Critic共享策略的实现：完全共享、完全独立、部分共享
"""
# ------------------------------------------------------------

from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from core.modules.actor_critics.base import BaseActorCritic
from core.modules.policy_heads import DiscretePolicyHead
from core.modules.value_heads import MLPValueHead


class SharedActorCritic(BaseActorCritic):
    """
    共享编码器的Actor-Critic网络

    架构:
        Encoder (共享) -> Shared Layers (共享) -> [Actor Head, Critic Head]
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        act_dim: int,
        shared_hidden_dims: List[int] = [128, 64],
        actor_hidden_dims: List[int] = [32],
        critic_hidden_dims: List[int] = [32],
    ):
        """
        Args:
            encoder: 编码器网络（可以是CNN+LSTM等）
            feature_dim: 编码器输出的特征维度
            act_dim: 动作空间维度
            shared_hidden_dims: 共享层的隐藏维度列表
            actor_hidden_dims: Actor头部的隐藏维度列表
            critic_hidden_dims: Critic头部的隐藏维度列表
        """
        super().__init__()

        self.encoder = encoder

        # 共享层
        shared_layers = []
        input_dim = feature_dim
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim

        self.shared_net = nn.Sequential(*shared_layers)
        self.shared_output_dim = input_dim

        # Actor和Critic头部（使用专用模块中的实现）
        self.policy_head = DiscretePolicyHead(
            in_dim=self.shared_output_dim, action_dim=act_dim, hidden_dims=actor_hidden_dims
        )

        self.value_head = MLPValueHead(
            in_dim=self.shared_output_dim, hidden_dims=critic_hidden_dims
        )

    def forward(
        self, obs: torch.Tensor, hidden: Optional[Any] = None
    ) -> Tuple[Categorical, torch.Tensor, Optional[Any]]:
        """
        前向传播

        Returns:
            distribution: 动作分布
            value: 状态价值
            hidden: 更新后的隐藏状态
        """
        # 编码观测
        if hasattr(self.encoder, "forward"):
            if hidden is not None:
                features, hidden = self.encoder(obs, hidden)
            else:
                features, hidden = self.encoder(obs)
        else:
            features = self.encoder(obs)
            hidden = None

        # 共享层处理
        shared_features = self.shared_net(features)

        # 分别通过Actor和Critic头部
        distribution = self.policy_head(shared_features)
        value = self.value_head(shared_features)

        return distribution, value, hidden

    def get_actor_parameters(self):
        """返回Actor相关参数"""
        return list(self.policy_head.parameters())

    def get_critic_parameters(self):
        """返回Critic相关参数"""
        return list(self.value_head.parameters())

    def get_shared_parameters(self):
        """返回共享参数（编码器+共享层）"""
        encoder_params = list(self.encoder.parameters())
        shared_params = list(self.shared_net.parameters())
        return encoder_params + shared_params


class SeparateActorCritic(BaseActorCritic):
    """
    完全独立的Actor-Critic网络（不共享任何层）

    架构:
        Encoder (共享) -> [Actor Net, Critic Net] -> [Actor Head, Critic Head]
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        act_dim: int,
        actor_hidden_dims: List[int] = [128, 64],
        critic_hidden_dims: List[int] = [128, 64],
    ):
        """
        Args:
            encoder: 编码器网络
            feature_dim: 编码器输出的特征维度
            act_dim: 动作空间维度
            actor_hidden_dims: Actor网络的隐藏维度列表
            critic_hidden_dims: Critic网络的隐藏维度列表
        """
        super().__init__()

        self.encoder = encoder

        # 独立的Actor网络
        actor_layers = []
        input_dim = feature_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.actor_net = nn.Sequential(*actor_layers)
        self.actor_output_dim = input_dim

        # 独立的Critic网络
        critic_layers = []
        input_dim = feature_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.critic_net = nn.Sequential(*critic_layers)
        self.critic_output_dim = input_dim

        # 头部
        self.policy_head = nn.Linear(self.actor_output_dim, act_dim)
        self.value_head = nn.Linear(self.critic_output_dim, 1)

    def forward(
        self, obs: torch.Tensor, hidden: Optional[Any] = None
    ) -> Tuple[Categorical, torch.Tensor, Optional[Any]]:
        """前向传播"""
        # 编码观测
        if hasattr(self.encoder, "forward"):
            if hidden is not None:
                features, hidden = self.encoder(obs, hidden)
            else:
                features, hidden = self.encoder(obs)
        else:
            features = self.encoder(obs)
            hidden = None

        # 独立的前向传播
        actor_features = self.actor_net(features)
        critic_features = self.critic_net(features)

        # 头部输出
        logits = self.policy_head(actor_features)
        value = self.value_head(critic_features).squeeze(-1)

        distribution = Categorical(logits=logits)

        return distribution, value, hidden

    def get_actor_parameters(self):
        """返回Actor相关参数（编码器+Actor网络+头部）"""
        encoder_params = list(self.encoder.parameters())
        actor_params = list(self.actor_net.parameters()) + list(self.policy_head.parameters())
        return encoder_params + actor_params

    def get_critic_parameters(self):
        """返回Critic相关参数（编码器+Critic网络+头部）"""
        encoder_params = list(self.encoder.parameters())
        critic_params = list(self.critic_net.parameters()) + list(self.value_head.parameters())
        return encoder_params + critic_params

    def get_shared_parameters(self):
        """返回共享参数（仅编码器）"""
        return list(self.encoder.parameters())


class PartialSharedActorCritic(BaseActorCritic):
    """
    部分共享的Actor-Critic网络

    架构:
        Encoder (共享) -> Shared Layer 1 -> [Actor Branch, Critic Branch] -> Heads
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        act_dim: int,
        shared_hidden_dims: List[int] = [128],
        actor_hidden_dims: List[int] = [64, 32],
        critic_hidden_dims: List[int] = [64, 32],
    ):
        """
        Args:
            encoder: 编码器网络
            feature_dim: 编码器输出的特征维度
            act_dim: 动作空间维度
            shared_hidden_dims: 第一层共享网络的隐藏维度
            actor_hidden_dims: Actor分支的隐藏维度
            critic_hidden_dims: Critic分支的隐藏维度
        """
        super().__init__()

        self.encoder = encoder

        # 第一层共享网络
        shared_layers = []
        input_dim = feature_dim
        for hidden_dim in shared_hidden_dims:
            shared_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*shared_layers)
        shared_output_dim = input_dim

        # Actor分支
        actor_layers = []
        input_dim = shared_output_dim
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.actor_branch = nn.Sequential(*actor_layers)
        actor_output_dim = input_dim

        # Critic分支
        critic_layers = []
        input_dim = shared_output_dim
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        self.critic_branch = nn.Sequential(*critic_layers)
        critic_output_dim = input_dim

        # 头部
        self.policy_head = nn.Linear(actor_output_dim, act_dim)
        self.value_head = nn.Linear(critic_output_dim, 1)

    def forward(
        self, obs: torch.Tensor, hidden: Optional[Any] = None
    ) -> Tuple[Categorical, torch.Tensor, Optional[Any]]:
        """前向传播"""
        # 编码观测
        if hasattr(self.encoder, "forward"):
            if hidden is not None:
                features, hidden = self.encoder(obs, hidden)
            else:
                features, hidden = self.encoder(obs)
        else:
            features = self.encoder(obs)
            hidden = None

        # 共享层
        shared_features = self.shared_net(features)

        # 分支处理
        actor_features = self.actor_branch(shared_features)
        critic_features = self.critic_branch(shared_features)

        # 头部输出
        logits = self.policy_head(actor_features)
        value = self.value_head(critic_features).squeeze(-1)

        distribution = Categorical(logits=logits)

        return distribution, value, hidden

    def get_actor_parameters(self):
        """返回Actor相关参数"""
        return list(self.actor_branch.parameters()) + list(self.policy_head.parameters())

    def get_critic_parameters(self):
        """返回Critic相关参数"""
        return list(self.critic_branch.parameters()) + list(self.value_head.parameters())

    def get_shared_parameters(self):
        """返回共享参数（编码器+共享层）"""
        encoder_params = list(self.encoder.parameters())
        shared_params = list(self.shared_net.parameters())
        return encoder_params + shared_params
