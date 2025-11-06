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
参考MultiAgentEnv设计，提供完整的多Agent环境接口支持
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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

    def render(self, mode: str = "human") -> Optional[Any]:
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

    def observation_space(self, agent_id: str = "agent_0") -> Any:
        """
        获取观测空间

        Args:
            agent_id: Agent标识符（对于单Agent环境可忽略，默认使用"agent_0"）

        Returns:
            观测空间描述
        """
        raise NotImplementedError

    def action_space(self, agent_id: str = "agent_0") -> Any:
        """
        获取动作空间

        Args:
            agent_id: Agent标识符（对于单Agent环境可忽略，默认使用"agent_0"）

        Returns:
            动作空间描述
        """
        raise NotImplementedError

    def get_obs(self) -> np.ndarray:
        """
        获取当前观测值（单Agent环境）

        Returns:
            当前观测值，形状为 (obs_dim,)
        """
        raise NotImplementedError

    def get_obs_size(self) -> int:
        """
        获取观测维度大小

        Returns:
            观测空间的维度大小
        """
        obs_space = self.observation_space("agent_0")
        if hasattr(obs_space, "shape"):
            return int(np.prod(obs_space.shape))
        elif hasattr(obs_space, "n"):
            return obs_space.n
        else:
            raise NotImplementedError(f"Cannot determine obs size from {obs_space}")

    def get_state(self) -> np.ndarray:
        """
        获取全局状态（单Agent环境中通常等于观测）

        Returns:
            全局状态，形状为 (state_dim,)
        """
        return self.get_obs()

    def get_state_size(self) -> int:
        """
        获取全局状态维度大小

        Returns:
            全局状态的维度大小
        """
        return self.get_obs_size()

    def get_total_actions(self) -> int:
        """
        获取动作空间大小

        Returns:
            动作空间的总动作数（适用于离散动作空间）
        """
        action_space = self.action_space("agent_0")
        if hasattr(action_space, "n"):
            return action_space.n
        elif hasattr(action_space, "shape"):
            # 连续动作空间
            return int(np.prod(action_space.shape))
        else:
            raise NotImplementedError(f"Cannot determine action size from {action_space}")

    def get_avail_actions(self) -> List[int]:
        """
        获取可用动作列表（单Agent环境，通常所有动作都可用）

        Returns:
            可用动作的列表，默认返回所有动作
        """
        total_actions = self.get_total_actions()
        return list(range(total_actions))

    def get_avail_agent_actions(self, agent_id: str = "agent_0") -> List[int]:
        """
        获取指定Agent的可用动作列表

        Args:
            agent_id: Agent标识符（单Agent环境可忽略）

        Returns:
            可用动作的列表
        """
        return self.get_avail_actions()

    def get_env_info(self) -> Dict[str, Any]:
        """
        获取环境信息字典

        Returns:
            包含环境信息的字典：
                - state_shape: 状态维度
                - obs_shape: 观测维度
                - n_actions: 动作空间大小
                - n_agents: Agent数量（单Agent环境为1）
                - episode_limit: Episode最大步数（如果支持）
        """
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": 1,
            "episode_limit": getattr(self, "episode_limit", None),
        }


class AgentParrelEnv(Env):
    """
    并行多Agent环境包装器抽象类

    用于包装多Agent环境（如PettingZoo、MAgent等），支持多个Agent并行交互
    参考MultiAgentEnv设计，提供完整的多Agent环境接口支持
    """

    def reset(self) -> Dict[str, Any]:
        """
        重置环境，开始新的episode

        Returns:
            所有Agent的初始观测值（字典形式，key为agent_id）
        """
        raise NotImplementedError

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        执行一步环境交互（所有Agent同时执行动作）

        Args:
            actions: Agent动作字典，格式为 {agent_id: action}

        Returns:
            (observations, rewards, dones, infos) 元组，均为字典格式：
                - observations: {agent_id: observation}
                - rewards: {agent_id: reward}
                - dones: {agent_id: done}
                - infos: {agent_id: info}
        """
        raise NotImplementedError

    def close(self) -> None:
        """关闭环境，释放资源"""
        pass

    def render(self, mode: str = "human") -> Optional[Any]:
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

    @property
    def agents(self) -> List[str]:
        """
        获取当前活跃的Agent列表

        Returns:
            Agent ID列表
        """
        raise NotImplementedError

    @property
    def n_agents(self) -> int:
        """
        获取Agent数量

        Returns:
            Agent数量
        """
        return len(self.agents)

    @property
    def episode_limit(self) -> Optional[int]:
        """
        获取Episode最大步数限制

        Returns:
            Episode最大步数，如果无限制则返回None
        """
        return getattr(self, "_episode_limit", None)

    def get_obs(self) -> List[np.ndarray]:
        """
        获取所有Agent的观测值列表

        Returns:
            所有Agent的观测值列表，按agents顺序排列
        """
        raise NotImplementedError

    def get_obs_agent(self, agent_id: str) -> np.ndarray:
        """
        获取指定Agent的观测值

        Args:
            agent_id: Agent标识符

        Returns:
            该Agent的观测值，形状为 (obs_dim,)
        """
        raise NotImplementedError

    def get_obs_size(self) -> Union[int, Tuple[int, ...]]:
        """
        获取观测维度大小

        Returns:
            观测空间的维度大小，可能是标量或元组（对于多Agent环境，通常所有Agent的观测维度相同）
        """
        if not self.agents:
            raise ValueError("No agents available")
        obs_space = self.observation_space(self.agents[0])
        if hasattr(obs_space, "shape"):
            return tuple(obs_space.shape) if len(obs_space.shape) > 1 else int(obs_space.shape[0])
        elif hasattr(obs_space, "n"):
            return obs_space.n
        else:
            raise NotImplementedError(f"Cannot determine obs size from {obs_space}")

    def get_state(self) -> np.ndarray:
        """
        获取全局状态（所有Agent的联合状态）

        Returns:
            全局状态，形状为 (state_dim,)
        """
        raise NotImplementedError

    def get_state_size(self) -> Union[int, Tuple[int, ...]]:
        """
        获取全局状态维度大小

        Returns:
            全局状态的维度大小，可能是标量或元组
        """
        raise NotImplementedError

    def get_total_actions(self) -> int:
        """
        获取动作空间大小（所有Agent的动作空间通常相同）

        Returns:
            动作空间的总动作数（适用于离散动作空间）
        """
        if not self.agents:
            raise ValueError("No agents available")
        action_space = self.action_space(self.agents[0])
        if hasattr(action_space, "n"):
            return action_space.n
        elif hasattr(action_space, "shape"):
            # 连续动作空间
            return int(np.prod(action_space.shape))
        else:
            raise NotImplementedError(f"Cannot determine action size from {action_space}")

    def get_avail_actions(self) -> List[List[int]]:
        """
        获取所有Agent的可用动作列表

        Returns:
            所有Agent的可用动作列表，按agents顺序排列
            每个元素是一个可用动作的列表
        """
        avail_actions = []
        for agent_id in self.agents:
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_avail_agent_actions(self, agent_id: str) -> List[int]:
        """
        获取指定Agent的可用动作列表

        Args:
            agent_id: Agent标识符

        Returns:
            可用动作的列表，默认返回所有动作
        """
        total_actions = self.get_total_actions()
        return list(range(total_actions))

    def get_env_info(self) -> Dict[str, Any]:
        """
        获取环境信息字典

        Returns:
            包含环境信息的字典：
                - state_shape: 状态维度
                - obs_shape: 观测维度
                - n_actions: 动作空间大小
                - n_agents: Agent数量
                - episode_limit: Episode最大步数（如果支持）
        """
        obs_size = self.get_obs_size()
        if isinstance(obs_size, tuple):
            obs_shape = obs_size
        else:
            obs_shape = (obs_size,)

        state_size = self.get_state_size()
        if isinstance(state_size, tuple):
            state_shape = state_size
        else:
            state_shape = (state_size,)

        return {
            "state_shape": state_shape,
            "obs_shape": obs_shape,
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }
