# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : agent.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:27
@Update Date    :
@Description    : Agent（智能体）抽象基类接口
定义了强化学习智能体的标准接口规范，所有具体的Agent实现都需要继承此类
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class Agent(ABC):
    """
    抽象智能体接口

    定义了强化学习智能体的核心接口，包括：
    - 动作选择（act）
    - 学习更新（learn）
    - 状态保存和加载（state_dict/load_state_dict）
    - 训练/评估模式切换（to_training_mode/to_eval_mode）
    """

    @abstractmethod
    def act(self, observation: Any, deterministic: bool = False) -> Any:
        """
        根据观测选择动作

        Args:
            observation: 当前环境的观测值
            deterministic: 是否使用确定性策略（True：选择最优动作，False：采样动作）

        Returns:
            选中的动作，返回值类型取决于具体的实现
        """
        pass

    @abstractmethod
    def learn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        从批次数据中学习，更新策略

        Args:
            batch: 训练批次数据字典，通常包含：
                - obs: 观测值
                - actions: 动作
                - rewards: 奖励
                - advantages: 优势值
                - returns: 回报值
                等字段

        Returns:
            训练指标字典，通常包含损失值、熵等信息
        """
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        获取Agent的状态字典（用于保存模型）

        Returns:
            包含Agent所有可训练参数的状态字典
        """
        pass

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        加载Agent的状态字典（用于恢复模型）

        Args:
            state: 之前保存的状态字典
        """
        pass

    @abstractmethod
    def to_training_mode(self) -> None:
        """
        切换到训练模式
        通常会启用dropout、batch normalization的训练模式等
        """
        pass

    @abstractmethod
    def to_eval_mode(self) -> None:
        """
        切换到评估模式
        通常会禁用dropout、使用batch normalization的推理模式等
        """
        pass
