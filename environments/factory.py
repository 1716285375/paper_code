# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : factory.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:31
@Update Date    :
@Description    : 环境工厂函数
根据环境ID创建对应的环境实例，支持注册机制以便扩展
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Callable, Dict

from core.base.environment import Env

# 环境注册表：{env_id: factory_function}
_ENV_REGISTRY: Dict[str, Callable[..., Env]] = {}


def register_env(env_id: str, factory: Callable[..., Env]) -> None:
    """
    注册环境工厂函数

    Args:
        env_id: 环境标识符（如 "magent2:battle_v4"）
        factory: 环境工厂函数，接受**kwargs并返回Env实例
    """
    _ENV_REGISTRY[env_id] = factory


def make_env(env_id: str, **kwargs: Any) -> Env:
    """
    环境工厂函数：根据环境ID创建环境实例

    支持的环境ID格式：
        - "magent2:battle_v4": MAgent2的battle_v4环境（团队对抗）
        - "magent2:adversarial_pursuit_v4": 对抗追击环境（捕食者-被捕食者）
        - "magent2:battlefield_v4": 战场环境（团队对抗）
        - "magent2:combined_arms_v6": 联合兵种环境（多兵种协同）
        - "magent2:gather_v4": 收集环境（资源收集）
        - "magent2:tiger_deer_v3": 虎鹿环境（生态系统模拟）

    也可以通过register_env()注册自定义环境。

    Args:
        env_id: 环境标识符，格式为 "backend:env_name"
        **kwargs: 传递给环境构造函数的额外参数

    Returns:
        环境实例

    Raises:
        ValueError: 如果环境ID不被支持

    示例:
        >>> env = make_env("magent2:battle_v4", map_size=45, minimap_mode=False)
        >>> # 注册自定义环境
        >>> register_env("custom:my_env", lambda **kw: MyCustomEnv(**kw))
        >>> env = make_env("custom:my_env", param1=value1)
    """
    # 首先检查注册表
    if env_id in _ENV_REGISTRY:
        return _ENV_REGISTRY[env_id](**kwargs)

    # 向后兼容：直接解析环境ID
    if env_id.startswith("magent2:"):
        name = env_id.split(":", 1)[1]
        if name == "battle_v4":
            from .magent2.wrapper import Magent2BattleV4Parallel
            return Magent2BattleV4Parallel(**kwargs)
        elif name == "adversarial_pursuit_v4":
            from .magent2.wrapper import Magent2AdversarialPursuitV4Parallel
            return
        raise ValueError(f"不支持的magent2环境名称: {name}。支持的环境: ['battle_v4']")

    raise ValueError(
        f"不支持的环境ID: {env_id}。\n"
        f"已注册的环境: {list(_ENV_REGISTRY.keys())}\n"
        f"或使用 register_env() 注册自定义环境。"
    )


def list_environments() -> list[str]:
    """
    列出所有已注册的环境ID

    Returns:
        环境ID列表
    """
    return list(_ENV_REGISTRY.keys())


# 自动注册MAgent2环境
def _register_magent2_envs():
    """自动注册所有MAgent2环境"""
    try:
        from .magent2.wrapper import (
            Magent2AdversarialPursuitV4Parallel,
            Magent2BattlefieldV4Parallel,
            Magent2BattleV4Parallel,
            Magent2CombinedArmsV6Parallel,
            Magent2GatherV4Parallel,
            Magent2TigerDeerV3Parallel,
        )

        # 注册所有环境
        register_env("magent2:battle_v4", Magent2BattleV4Parallel)
        register_env("magent2:adversarial_pursuit_v4", Magent2AdversarialPursuitV4Parallel)
        register_env("magent2:battlefield_v4", Magent2BattlefieldV4Parallel)
        register_env("magent2:combined_arms_v6", Magent2CombinedArmsV6Parallel)
        register_env("magent2:gather_v4", Magent2GatherV4Parallel)
        register_env("magent2:tiger_deer_v3", Magent2TigerDeerV3Parallel)
    except ImportError:
        pass  # MAgent2未安装，跳过注册


# 在模块导入时自动注册
_register_magent2_envs()
