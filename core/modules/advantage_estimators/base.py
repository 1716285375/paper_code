# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 10:52
@Update Date    :
@Description    : 优势估计器（Advantage Estimator）抽象基类
定义了优势估计的标准接口，用于计算策略梯度算法中的优势值
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class AdvantageEstimator(ABC):
    """
    优势估计函数的基类接口

    优势估计器用于计算动作的优势值（advantage），即动作值相对于状态值的偏差。
    常用的实现包括GAE（Generalized Advantage Estimation）等。
    """

    @abstractmethod
    def compute(
        self,
        rewards: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        dones: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算轨迹的优势值和回报值

        Args:
            rewards: 奖励列表或数组，形状为 (T, )，T为轨迹长度
            values: 状态价值估计列表或数组，形状为 (T, )
            dones: 完成标志列表或数组，形状为 (T, )，1表示episode结束，0表示继续

        Returns:
            (returns, advantages) 元组：
                - returns: 计算得到的回报值数组，形状 (T, )
                - advantages: 计算得到的优势值数组，形状 (T, )

        """
        ...

    def __call__(
        self,
        rewards: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        dones: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        允许估计器作为函数调用（语法糖）

        Args:
            rewards: 奖励数组
            values: 价值估计数组
            dones: 完成标志数组

        Returns:
            (returns, advantages) 元组
        """
        return self.compute(rewards, values, dones)
