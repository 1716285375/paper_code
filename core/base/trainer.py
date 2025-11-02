# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:28
@Update Date    :
@Description    : Trainer（训练器）抽象基类接口
定义了强化学习训练器的标准接口规范，所有具体的训练器实现都需要继承此类
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class Trainer(ABC):
    """
    抽象训练器接口

    定义了强化学习训练器的核心接口，包括：
    - 训练循环（train）
    - 性能评估（evaluates）
    - 保存和加载（save/load）
    """

    @abstractmethod
    def train(self, num_updates: int) -> None:
        """
        执行训练循环

        Args:
            num_updates: 训练更新的次数
        """
        pass

    @abstractmethod
    def evaluates(self, num_episodes: int) -> Dict[str, Any]:
        """
        评估Agent的性能

        Args:
            num_episodes: 评估的episode数量

        Returns:
            评估指标字典，通常包含平均奖励、奖励标准差等
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存训练器状态（包括Agent模型）到文件

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        从文件加载训练器状态（包括Agent模型）

        Args:
            path: 加载路径
        """
        pass
