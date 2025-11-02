# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-29 22:25
@Update Date    :
@Description    : 核心模块集合
提供PPO算法所需的各种可热插拔模块，包括Actor-Critic网络、策略头、价值头、
探索策略、归一化、优势估计器、优化器等。

所有模块都支持通过配置文件和注册机制进行动态加载。
"""
# ------------------------------------------------------------

from __future__ import annotations

# Actor-Critic网络
from core.modules.actor_critics import (
    BaseActorCritic,
    PartialSharedActorCritic,
    SeparateActorCritic,
    SharedActorCritic,
    create_actor_critic,
)

# 探索策略
from core.modules.exploration import (
    BaseScheduler,
    EpsilonGreedy,
    ExplorationStrategy,
    GaussianNoise,
    LinearSchedule,
    TemperatureScaling,
)

# 策略头
from core.modules.policy_heads import (
    BasePolicyHead,
    DiagGaussianPolicyHead,
    DiscretePolicyHead,
    MixedPolicyHead,
)

# 价值头
from core.modules.value_heads import (
    BaseValueHead,
    CategoricalValueHead,
    EnsembleValueHead,
    LinearValueHead,
    MLPValueHead,
)

# 导出各子模块的常用接口，提供统一的导入方式
# 用户可以直接从 core.modules 导入，也可以从子模块导入






# 归一化模块（延迟导入，避免循环依赖）
def _import_normalization():
    """延迟导入归一化模块"""
    from core.modules.normalization import (
        AdvantageNormalizer,
        ObservationNormalizer,
        RewardNormalizer,
    )

    return ObservationNormalizer, RewardNormalizer, AdvantageNormalizer


# 优势估计器（延迟导入）
def _import_advantage_estimators():
    """延迟导入优势估计器"""
    from core.modules.advantage_estimators import GAE

    return GAE


# 优化器（延迟导入，避免循环依赖）
def _import_optimizers():
    """延迟导入优化器"""
    from core.modules.optimizers import (
        AdamBuilder,
        AdamWBuilder,
        AdaptiveOptimizer,
        OptimizerBuilder,
        OptimizerConfig,
        RMSPropBuilder,
    )

    return (
        OptimizerConfig,
        OptimizerBuilder,
        AdamBuilder,
        AdamWBuilder,
        RMSPropBuilder,
        AdaptiveOptimizer,
    )


__all__ = [
    # Actor-Critic
    "BaseActorCritic",
    "SharedActorCritic",
    "SeparateActorCritic",
    "PartialSharedActorCritic",
    "create_actor_critic",
    # 策略头
    "BasePolicyHead",
    "DiscretePolicyHead",
    "DiagGaussianPolicyHead",
    "MixedPolicyHead",
    # 价值头
    "BaseValueHead",
    "LinearValueHead",
    "MLPValueHead",
    "CategoricalValueHead",
    "EnsembleValueHead",
    # 探索策略
    "ExplorationStrategy",
    "BaseScheduler",
    "LinearSchedule",
    "EpsilonGreedy",
    "TemperatureScaling",
    "GaussianNoise",
]

# 注意：
# - 归一化模块和优化器模块使用延迟导入，避免循环依赖
# - 如需使用，请直接从子模块导入：
#   from core.modules.normalization import ObservationNormalizer
#   from core.modules.optimizers import OptimizerConfig
#   from core.modules.advantage_estimators import GAE
