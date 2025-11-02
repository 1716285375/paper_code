# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : running_mean_std.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:05
@Update Date    :
@Description    : 运行时的均值和标准差计算
使用Welford算法在线计算均值和方差，适用于实时归一化
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class RunningMeanStd:
    """
    运行时均值和标准差计算器

    使用Welford在线算法实时更新均值和方差，无需存储所有历史数据。
    适用于需要实时归一化的场景，如观测归一化、奖励归一化等。
    """

    def __init__(self, shape: Tuple[int, ...] = ()) -> None:
        """
        初始化运行统计计算器

        Args:
            shape: 数据的形状（用于多维数据），标量数据使用空元组()
        """
        self._mean = np.zeros(shape, dtype=np.float64)  # 当前均值
        self._var = np.ones(shape, dtype=np.float64)  # 当前方差（初始化为1）
        self._count = 1e-8  # 样本计数（初始化为很小的值以避免除零）

    def reset(self) -> None:
        """
        重置统计信息

        将所有统计量重置为初始状态。
        """
        self._mean[...] = 0.0
        self._var[...] = 1.0
        self._count = 1e-8

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        """
        将输入转换为numpy数组

        Args:
            x: 输入数据（可能是torch.Tensor、numpy数组或标量）

        Returns:
            numpy数组
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def update(self, x) -> None:
        """
        更新运行统计（使用Welford算法）

        Args:
            x: 新的数据批次或单个数据点
               可以是numpy数组、torch.Tensor或标量
        """
        x_np = self._to_numpy(x).astype(np.float64)
        batch_mean = x_np.mean(axis=0)  # 批次均值
        batch_var = x_np.var(axis=0)  # 批次方差
        batch_count = x_np.shape[0] if x_np.ndim > 0 else 1.0  # 批次大小

        # Welford算法更新
        delta = batch_mean - self._mean  # 均值差
        tot_count = self._count + batch_count  # 总样本数

        # 更新均值
        new_mean = self._mean + delta * (batch_count / tot_count)

        # 更新方差（使用Welford公式）
        m_a = self._var * self._count  # 旧方差加权
        m_b = batch_var * batch_count  # 新方差加权
        M2 = m_a + m_b + delta**2 * self._count * batch_count / tot_count
        new_var = M2 / tot_count

        # 保存更新后的值
        self._mean = new_mean
        self._var = new_var
        self._count = tot_count

    @property
    def mean(self) -> np.ndarray:
        """
        获取当前均值

        Returns:
            当前均值数组
        """
        return self._mean

    @property
    def var(self) -> np.ndarray:
        """
        获取当前方差

        Returns:
            当前方差数组
        """
        return self._var

    @property
    def std(self) -> np.ndarray:
        """
        获取当前标准差

        Returns:
            当前标准差数组，计算公式为 sqrt(var + 1e-8)
            添加小常数1e-8以避免数值不稳定
        """
        return np.sqrt(self._var + 1e-8)
