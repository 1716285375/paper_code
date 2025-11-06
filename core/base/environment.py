# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : environment.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:28
@Update Date    :
@Description    : Environment（环境）抽象基类接口
定义了强化学习环境的标准接口规范，统一不同环境后端（如Gym、PettingZoo等）的API
参考MultiAgentEnv设计，提供完整的多Agent环境接口支持
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union


class Env(ABC):
    """
    环境包装器接口

    统一环境API接口，适配不同环境后端（如Gym、PettingZoo、MAgent等）。
    所有环境实现都需要继承此类并提供标准的reset、step等方法。
    参考MultiAgentEnv设计，提供完整的多Agent环境接口支持。
    """

    @abstractmethod
    def reset(self) -> Any:
        """
        重置环境，开始新的episode

        Returns:
            初始观测值，类型取决于具体环境（可能是单个值或字典）
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        执行一步环境交互

        Args:
            action: Agent选择的动作

        Returns:
            (next_observation, reward, done, info) 元组：
                - next_observation: 下一步的观测值
                - reward: 获得的奖励
                - done: 是否结束（episode终止）
                - info: 额外的信息字典
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        关闭环境，释放资源
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> Optional[Any]:
        """
        渲染环境（可视化）

        Args:
            mode: 渲染模式
                - "human": 显示窗口（默认）
                - "rgb_array": 返回RGB数组

        Returns:
            如果mode="rgb_array"，返回RGB数组；否则返回None
        """
        pass

    @abstractmethod
    def observation_space(self, agent_id: str) -> Any:
        """
        获取指定agent的观测空间

        Args:
            agent_id: Agent的标识符（对于多Agent环境）

        Returns:
            观测空间的描述（如gym.Space对象）
        """
        pass

    @abstractmethod
    def action_space(self, agent_id: str) -> Any:
        """
        获取指定agent的动作空间

        Args:
            agent_id: Agent的标识符（对于多Agent环境）

        Returns:
            动作空间的描述（如gym.Space对象）
        """
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        """
        设置随机种子

        Args:
            seed: 随机种子值，如果为None则不设置
        """
        pass

    def save_replay(self, save_path: str) -> bool:
        """
        保存环境回放（如果支持）

        Args:
            save_path: 保存路径

        Returns:
            是否成功保存
        """
        return False
