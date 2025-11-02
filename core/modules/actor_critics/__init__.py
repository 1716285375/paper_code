# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:29
@Update Date    :
@Description    : Actor-Critic网络模块集合
"""
# ------------------------------------------------------------

from __future__ import annotations

import torch

from core.modules.actor_critics.base import (
    BaseActorCritic,
    BasePolicyHead,
    BaseValueHead,
)
from core.modules.actor_critics.shared_network import (
    PartialSharedActorCritic,
    SeparateActorCritic,
    SharedActorCritic,
)

# Actor-Critic网络注册表
ACTOR_CRITIC_REGISTRY = {
    "shared": SharedActorCritic,
    "separate": SeparateActorCritic,
    "partial_shared": PartialSharedActorCritic,
}


def create_actor_critic(
    network_type: str, encoder: torch.nn.Module, feature_dim: int, act_dim: int, **kwargs
):
    """
    工厂函数：根据类型创建Actor-Critic网络

    Args:
        network_type: 网络类型 ('shared', 'separate', 'partial_shared')
        encoder: 编码器网络
        feature_dim: 特征维度
        act_dim: 动作空间维度
        **kwargs: 其他参数（hidden_dims等）

    Returns:
        ActorCritic网络实例
    """
    if network_type not in ACTOR_CRITIC_REGISTRY:
        raise ValueError(
            f"Unknown actor_critic type: {network_type}. "
            f"Available: {list(ACTOR_CRITIC_REGISTRY.keys())}"
        )

    actor_critic_class = ACTOR_CRITIC_REGISTRY[network_type]
    return actor_critic_class(encoder=encoder, feature_dim=feature_dim, act_dim=act_dim, **kwargs)


__all__ = [
    "BaseActorCritic",
    "BasePolicyHead",
    "BaseValueHead",
    "SharedActorCritic",
    "SeparateActorCritic",
    "PartialSharedActorCritic",
    "ACTOR_CRITIC_REGISTRY",
    "create_actor_critic",
]
