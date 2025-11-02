# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : factory.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:29
@Update Date    :
@Description    : Agent工厂函数
从配置文件创建Agent实例
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

from core.base.agent import Agent

# 延迟导入 ConfigurablePPOAgent 以避免循环导入
# from core.agent.utils import ConfigurablePPOAgent


def build_module_from_config(config: Optional[Dict[str, Any]]) -> Optional[Any]:
    """
    从配置字典构建模块实例

    使用模块注册表 (registry) 根据配置中的 type 字段创建模块实例。

    Args:
        config: 配置字典，包含 "type" 和 "params" 字段
               格式: {"type": "networks/mlp", "params": {...}}
               如果为 None，返回 None

    Returns:
        创建的模块实例，如果 config 为 None 则返回 None

    Raises:
        KeyError: 如果模块类型未注册
        ValueError: 如果配置格式不正确
    """
    if config is None:
        return None

    if not isinstance(config, dict):
        raise ValueError(f"配置必须是字典，得到: {type(config)}")

    module_type = config.get("type") or config.get("name")
    if not module_type:
        return None

    module_params = config.get("params", {}) or config.get("kwargs", {})

    # 使用注册表创建模块
    from core.agent.registry import MODULES

    if module_type not in MODULES:
        # 模块未注册，返回 None 让调用者处理
        return None

    return MODULES.create(module_type, **module_params)


def build_agent_from_config(
    obs_dim: int,
    action_dim: int,
    config: Dict[str, Any],
    device: str = "cpu",
) -> Agent:
    """
    从配置字典构建Agent

    配置格式示例:
        {
            "type": "ppo",  # Agent类型
            "obs_dim": 845,
            "action_dim": 21,
            "encoder": {
                "type": "networks/mlp",
                "params": {"in_dim": 845, "hidden_dims": [128, 128]}
            },
            "policy_head": {
                "type": "policy_heads/discrete",
                "params": {"hidden_dims": [64]}
            },
            "value_head": {
                "type": "value_heads/mlp",
                "params": {"hidden_dims": [64]}
            },
            "optimizer": {
                "type": "optimizers/adam",
                "params": {"lr": 3e-4}
            }
        }

    Args:
        obs_dim: 观测维度
        action_dim: 动作空间维度
        config: 完整的配置字典，包含Agent类型和组件配置
        device: 设备

    Returns:
        Agent实例
    """
    agent_type = config.get("type", "ppo").lower()

    # 提取Agent特定的配置（去掉type）
    agent_config = {k: v for k, v in config.items() if k != "type"}

    if agent_type == "ppo":
        # 延迟导入以避免循环导入
        from core.agent.utils import ConfigurablePPOAgent

        return ConfigurablePPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=agent_config,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: ['ppo']")
