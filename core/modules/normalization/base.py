# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 23:03
@Update Date    :
@Description    : 归一化器（Normalizer）抽象基类
定义了归一化器的标准接口，用于对数据进行归一化处理
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor, float]  # 支持的数组类型


class Normalizer(ABC):
    """
    归一化器抽象基类

    归一化器用于对数据进行归一化处理，常见用途包括：
    - 观测归一化：将观测值标准化，加速训练
    - 优势归一化：将优势值标准化，提高训练稳定性
    - 奖励归一化：将奖励标准化
    """

    @abstractmethod
    def reset(self) -> None:
        """
        重置归一化器状态

        清除累积的统计信息，重新开始统计。
        """
        ...

    @abstractmethod
    def update(self, x: ArrayLike) -> None:
        """
        更新归一化器的统计信息

        Args:
            x: 新的数据点（或批次），用于更新均值、方差等统计量
        """
        ...

    @abstractmethod
    def normalize(self, x: ArrayLike) -> ArrayLike:
        """
        归一化数据

        Args:
            x: 需要归一化的数据

        Returns:
            归一化后的数据，通常通过 (x - mean) / std 计算
        """
        ...

    @staticmethod
    def denormalize(x: ArrayLike) -> ArrayLike:
        """
        反归一化数据（默认实现为恒等映射）

        Args:
            x: 归一化后的数据

        Returns:
            反归一化后的数据
        """
        return x
