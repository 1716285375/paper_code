# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 14:47
@Update Date    :
@Description    : Agent模块集合
提供可配置的Agent实现，支持通过YAML配置不同的网络结构和组件
"""
# ------------------------------------------------------------

from core.agent.factory import build_agent_from_config
from core.agent.manager import AgentManager, create_agent_manager_from_config
from core.agent.pool import AgentPool, AgentPoolContext
from core.agent.utils import ConfigurablePPOAgent
from core.agent.utils_extras import (
    aggregate_metrics,
    batch_act,
    batch_learn,
    clone_agent_state,
    get_agent_statistics,
    prepare_batch_for_agent,
)
from core.base.agent import Agent


def create_agent(
    agent_type: str,
    obs_dim: int,
    action_dim: int,
    config: dict,
    device: str = "cpu",
) -> Agent:
    """
    工厂函数：根据类型创建Agent

    Args:
        agent_type: Agent类型（目前支持 'ppo'）
        obs_dim: 观测维度
        action_dim: 动作空间维度
        config: Agent配置字典
        device: 设备（'cpu' 或 'cuda'）

    Returns:
        Agent实例

    Raises:
        ValueError: 如果agent_type未知
    """
    if agent_type.lower() == "ppo":
        return ConfigurablePPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: ['ppo']")


__all__ = [
    # 基类和实现
    "Agent",
    "ConfigurablePPOAgent",
    # 管理器
    "AgentManager",
    "create_agent_manager_from_config",
    # 对象池
    "AgentPool",
    "AgentPoolContext",
    # 工厂函数
    "create_agent",
    "build_agent_from_config",
    # 工具函数
    "batch_act",
    "batch_learn",
    "get_agent_statistics",
    "clone_agent_state",
    "aggregate_metrics",
    "prepare_batch_for_agent",
]
