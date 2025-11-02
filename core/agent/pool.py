# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : pool.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:53
@Update Date    :
@Description    : Agent池
用于管理和复用Agent实例，适用于需要大量Agent的场景
"""
# ------------------------------------------------------------

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from core.agent.factory import build_agent_from_config
from core.base.agent import Agent


class AgentPool:
    """
    Agent对象池

    用于管理Agent实例的复用，减少创建和销毁的开销。
    适用于需要大量相同配置Agent的场景。
    """

    def __init__(
        self,
        pool_size: int,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
    ):
        """
        初始化Agent池

        Args:
            pool_size: 池大小（最多保留的Agent数量）
            obs_dim: 观测维度
            action_dim: 动作空间维度
            config: Agent配置
            device: 设备
        """
        self.pool_size = pool_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device

        # 使用队列管理可用的Agent
        self._available_agents: deque[Agent] = deque()
        self._all_agents: List[Agent] = []

        # 预创建一些Agent
        self._precreate_agents(min(5, pool_size))

    def acquire(self) -> Agent:
        """
        从池中获取一个Agent实例

        Returns:
            Agent实例
        """
        if len(self._available_agents) > 0:
            return self._available_agents.popleft()
        else:
            # 池为空，创建新的Agent
            return self._create_agent()

    def release(self, agent: Agent) -> None:
        """
        将Agent实例归还到池中

        Args:
            agent: Agent实例
        """
        # 重置Agent状态（可选）
        agent.to_eval_mode()

        if len(self._available_agents) < self.pool_size:
            self._available_agents.append(agent)

    def _create_agent(self) -> Agent:
        """创建新的Agent实例"""
        agent = build_agent_from_config(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            config=self.config,
            device=self.device,
        )
        self._all_agents.append(agent)
        return agent

    def _precreate_agents(self, count: int) -> None:
        """预创建Agent实例"""
        for _ in range(count):
            agent = self._create_agent()
            self._available_agents.append(agent)

    def clear(self) -> None:
        """清空池"""
        self._available_agents.clear()
        self._all_agents.clear()

    def size(self) -> int:
        """返回池中可用的Agent数量"""
        return len(self._available_agents)

    def total_size(self) -> int:
        """返回池中创建的Agent总数"""
        return len(self._all_agents)


class AgentPoolContext:
    """
    Agent池的上下文管理器

    自动获取和释放Agent
    """

    def __init__(self, pool: AgentPool):
        self.pool = pool
        self.agent: Optional[Agent] = None

    def __enter__(self) -> Agent:
        self.agent = self.pool.acquire()
        return self.agent

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.agent is not None:
            self.pool.release(self.agent)
            self.agent = None
