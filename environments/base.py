# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:31
@Update Date    :
@Description    : 环境包装器基类
提供单Agent和多Agent环境的抽象基类实现
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Any, Dict, Tuple

from core.base.environment import Env


class AgentEnv(Env):
    """
    单Agent环境包装器抽象类

    用于包装单Agent环境（如Gym环境），提供统一的环境接口
    """

    def reset(self) -> Any:
        """
        重置环境

        Returns:
            初始观测值
        """
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        执行一步环境交互

        Args:
            action: Agent选择的动作

        Returns:
            (observation, reward, done, info) 元组
        """
        raise NotImplementedError

    def close(self) -> None:
        """关闭环境，释放资源"""
        pass

    def render(self, mode: str = "human") -> Any:
        """
        渲染环境（可视化）

        Args:
            mode: 渲染模式
                - "human": 显示窗口（默认）
                - "rgb_array": 返回RGB数组，形状为(H, W, 3)，数据类型为uint8

        Returns:
            如果mode="rgb_array"，返回RGB数组（numpy数组）；否则返回None
        """
        pass

    def observation_space(self, agent_id: str) -> Any:
        """
        获取观测空间

        Args:
            agent_id: Agent标识符（对于单Agent环境可忽略）

        Returns:
            观测空间描述
        """
        raise NotImplementedError

    def action_space(self, agent_id: str) -> Any:
        """
        获取动作空间

        Args:
            agent_id: Agent标识符（对于单Agent环境可忽略）

        Returns:
            动作空间描述
        """
        raise NotImplementedError


class AgentParrelEnv(Env):
    """
    并行多Agent环境包装器抽象类

    用于包装多Agent环境（如PettingZoo、MAgent等），支持多个Agent并行交互
    """

    def reset(self) -> Any:
        """
        重置环境，开始新的episode

        Returns:
            所有Agent的初始观测值（字典形式，key为agent_id）
        """
        raise NotImplementedError

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        执行一步环境交互（所有Agent同时执行动作）

        Args:
            action: Agent动作字典，格式为 {agent_id: action}

        Returns:
            (observations, rewards, dones, infos) 元组，均为字典格式
        """
        raise NotImplementedError

    def close(self) -> None:
        """关闭环境，释放资源"""
        pass

    def render(self, mode: str = "human") -> Any:
        """
        渲染环境（可视化）

        Args:
            mode: 渲染模式
                - "human": 显示窗口（默认）
                - "rgb_array": 返回RGB数组，形状为(H, W, 3)，数据类型为uint8

        Returns:
            如果mode="rgb_array"，返回RGB数组（numpy数组）；否则返回None
        """
        pass

    def observation_space(self, agent_id: str) -> Any:
        """
        获取指定Agent的观测空间

        Args:
            agent_id: Agent的标识符

        Returns:
            该Agent的观测空间描述
        """
        raise NotImplementedError

    def action_space(self, agent_id: str) -> Any:
        """
        获取指定Agent的动作空间

        Args:
            agent_id: Agent的标识符

        Returns:
            该Agent的动作空间描述
        """
        raise NotImplementedError
