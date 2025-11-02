# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:06
@Update Date    :
@Description    : 探索策略（Exploration Strategy）抽象基类
定义了探索策略的标准接口，用于在强化学习中增加探索性
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch


class BaseScheduler(ABC):
    """
    调度器抽象基类

    用于动态调整参数值（如epsilon、temperature等），支持线性、余弦等调度策略。
    """

    def __init__(self, initial: float) -> None:
        """
        初始化调度器

        Args:
            initial: 初始值
        """
        self._value = float(initial)

    @abstractmethod
    def step(self) -> float:
        """
        执行一步调度，更新并返回当前值

        Returns:
            更新后的值
        """
        ...

    def get(self) -> float:
        """
        获取当前值

        Returns:
            当前值
        """
        return float(self._value)


class LinearSchedule(BaseScheduler):
    """
    线性调度器

    在指定的步数内，从初始值线性变化到最终值。
    常用于epsilon衰减等场景。
    """

    def __init__(self, initial: float, final: float, total_steps: int) -> None:
        """
        初始化线性调度器

        Args:
            initial: 初始值
            final: 最终值
            total_steps: 总步数（从初始值变化到最终值所需的步数）
        """
        super().__init__(initial)
        self.initial = float(initial)
        self.final = float(final)
        self.total_steps = max(1, int(total_steps))
        self._t = 0  # 当前步数

    def step(self) -> float:
        """
        执行一步调度

        Returns:
            当前步对应的值（线性插值）
        """
        self._t = min(self._t + 1, self.total_steps)
        frac = self._t / self.total_steps  # 完成比例
        self._value = self.initial + (self.final - self.initial) * frac  # 线性插值
        return self._value


class ExplorationStrategy(ABC):
    """
    探索策略抽象基类

    探索策略用于在强化学习中平衡探索（exploration）和利用（exploitation）。
    常见的策略包括epsilon-greedy、temperature scaling、高斯噪声等。
    """

    @abstractmethod
    def reset(self) -> None:
        """
        重置探索策略状态
        """
        ...

    def on_step(self, info: Optional[Dict[str, Any]] = None) -> None:
        """
        每步调用，用于更新探索策略（如epsilon衰减）

        Args:
            info: 可选的额外信息字典
        """
        return

    @abstractmethod
    def apply(
        self,
        *,
        logits: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        应用探索策略

        Args:
            logits: 动作logits（离散动作）
            action: 动作值（连续动作）
            mask: 动作掩码（标记哪些动作有效）
            low: 动作空间下界（连续动作）
            high: 动作空间上界（连续动作）

        Returns:
            修改后的参数字典，例如 {"logits": modified_logits} 或 {"action": modified_action}
        """
        ...
