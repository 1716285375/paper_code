# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : gae.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 11:07
@Update Date    :
@Description    : GAE（广义优势估计）实现
实现GAE(λ)算法，用于策略梯度算法中的优势估计
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

from .base import AdvantageEstimator


class GAE(AdvantageEstimator):
    """
    广义优势估计（Generalized Advantage Estimation, GAE）

    GAE(λ)是一种用于策略梯度算法中的优势估计方法，通过指数加权平均来平衡偏差和方差。

    GAE(λ) 优势计算公式（基于 TD-error）:
        δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)  # TD误差
        A_t = δ_t + (γ * λ) * (1 - done_t) * A_{t+1}        # GAE优势

    回报值计算: R_t = A_t + V(s_t)

    Args:
        gamma: 折扣因子（折现因子），取值范围 [0, 1]，默认 0.99
        lam: GAE lambda 参数，取值范围 [0, 1]，默认 0.95
            - λ=0: TD(0)，单步回报（低方差，可能有偏差）
            - λ=1: 蒙特卡洛回报（无偏差，高方差）
            - 0<λ<1: 指数加权平均（在偏差和方差之间平衡）
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma should be in [0, 1], got {gamma}")
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam should be in [0, 1], got {lam}")
        self.gamma = float(gamma)
        self.lam = float(lam)

    def compute(
        self,
        rewards: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        dones: Union[List[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算GAE优势和回报值

        通过反向遍历轨迹，累积计算每个时间步的优势值。

        Args:
            rewards: 奖励序列，形状 (T, )，T为轨迹长度
            values: 状态价值估计序列，形状 (T, )
            dones: 完成标志序列，形状 (T, )，True/1表示episode结束

        Returns:
            (returns, advantages) 元组：
                - returns: 回报值数组，形状 (T, )，计算公式 R_t = A_t + V(s_t)
                - advantages: 优势值数组，形状 (T, )，使用GAE(λ)计算

        """
        # 转换为numpy数组
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        dones = np.array(dones, dtype=np.bool_)

        n = len(rewards)
        if n == 0:
            # 空轨迹返回空数组
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0  # 上一个时间步的GAE值

        # 反向遍历轨迹计算GAE（从最后一个时间步开始）
        for t in reversed(range(n)):
            # 下一个状态的价值（如果是最后一步或已终止，则为0）
            next_value = 0.0 if t == n - 1 else values[t + 1]
            # 下一个状态是否非终止（如果当前步终止，则为0）
            next_non_terminal = 0.0 if t == n - 1 or dones[t] else 1.0

            # 计算TD误差
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]

            # 计算GAE优势（递归公式）
            lastgaelam = delta + self.gamma * self.lam * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        # 回报值 = 优势值 + 状态价值
        returns = advantages + values

        return returns, advantages
