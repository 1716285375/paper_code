# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : registry.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:29
@Update Date    :
@Description    : 模块注册表
提供模块的注册和创建功能，支持通过字符串名称创建模块实例
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Any, Callable, Dict, Type


class Registry:
    """
    模块注册表

    使用字符串名称注册模块构造函数，支持通过名称创建模块实例。
    用于实现模块的热插拔和动态加载。
    """

    def __init__(self) -> None:
        """
        初始化注册表
        """
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """
        注册模块构造函数

        Args:
            name: 模块名称（如 "networks/mlp"）
            constructor: 模块的构造函数（类或函数）

        Raises:
            ValueError: 如果模块名称已注册
        """
        if name in self._registry:
            raise ValueError(f"模块已注册: {name}")
        self._registry[name] = constructor

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        通过名称创建模块实例

        Args:
            name: 模块名称
            *args: 传递给构造函数的位置参数
            **kwargs: 传递给构造函数的关键字参数

        Returns:
            创建的模块实例

        Raises:
            KeyError: 如果模块名称未注册
        """
        if name not in self._registry:
            raise KeyError(f"未知模块: {name}")
        return self._registry[name](*args, **kwargs)

    def __contains__(self, name: str) -> bool:
        """
        检查模块是否已注册

        Args:
            name: 模块名称

        Returns:
            如果模块已注册返回True，否则返回False
        """
        return name in self._registry

    def keys(self):
        """
        获取所有已注册的模块名称

        Returns:
            已注册模块名称的视图对象
        """
        return self._registry.keys()


# 全局模块注册表实例
MODULES = Registry()


# ============ 自动注册所有模块 ============
# 从各模块目录导入并注册所有可用的模块实现

# 注册探索策略模块
from core.modules.exploration import EpsilonGreedy, GaussianNoise, TemperatureScaling

MODULES.register("exploration/epsilon_greedy", EpsilonGreedy)
MODULES.register("exploration/temperature", TemperatureScaling)
MODULES.register("exploration/gaussian_noise", GaussianNoise)

# 注册归一化模块
from core.modules.normalization import AdvantageNormalizer, ObservationNormalizer, RewardNormalizer

MODULES.register("normalization/observation", ObservationNormalizer)
MODULES.register("normalization/reward", RewardNormalizer)
MODULES.register("normalization/advantage", AdvantageNormalizer)

# 注册优势估计器
from core.modules.advantage_estimators import GAE

MODULES.register("advantage_estimators/gae", GAE)

# 注册网络编码器
from core.networks import CNNEncoder, LSTMEncoder, MLPEncoder, TransformerEncoder

MODULES.register("networks/mlp", MLPEncoder)
MODULES.register("networks/cnn", CNNEncoder)
MODULES.register("networks/lstm", LSTMEncoder)
MODULES.register("networks/transformer", TransformerEncoder)

# 注册策略头
from core.modules.policy_heads import DiagGaussianPolicyHead, DiscretePolicyHead

MODULES.register("policy_heads/discrete", DiscretePolicyHead)
MODULES.register("policy_heads/diag_gaussian", DiagGaussianPolicyHead)

# 注册价值头
from core.modules.value_heads import (
    CategoricalValueHead,
    EnsembleValueHead,
    LinearValueHead,
    MLPValueHead,
)

MODULES.register("value_heads/linear", LinearValueHead)
MODULES.register("value_heads/mlp", MLPValueHead)
MODULES.register("value_heads/categorical", CategoricalValueHead)
MODULES.register("value_heads/ensemble", EnsembleValueHead)

# 注册优化器（在ConfigurablePPOAgent中直接处理，不在此注册）
# from modules.optimizers import OptimizerConfig, AdamBuilder, AdamWBuilder, RMSPropBuilder


# 注册Agent（用于未来扩展）
# 延迟导入以避免循环导入
def _register_agents():
    """延迟注册Agent，避免循环导入"""
    try:
        from core.agent.utils import ConfigurablePPOAgent

        MODULES.register("agents/ppo", ConfigurablePPOAgent)
    except ImportError:
        pass  # 如果ConfigurablePPOAgent还未加载，跳过注册


# 不立即注册，让导入者决定何时注册
# _register_agents()
