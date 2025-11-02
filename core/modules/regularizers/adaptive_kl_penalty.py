# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : adaptive_kl_penalty.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 自适应KL惩罚系数调整器
用于PPO-Penalty变体中，根据KL散度自动调整惩罚系数beta
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch

from .kl_penalty import KLPenalty


class AdaptiveKLCoefficient:
    """
    自适应KL惩罚系数调整器

    根据实际KL散度与目标KL散度的关系，动态调整KL惩罚系数beta：
    - 如果KL散度太大（超过目标的1.5倍），增加beta
    - 如果KL散度太小（小于目标的1.5倍），减少beta

    算法：
        if kl > target_kl * 1.5:
            beta *= 2.0
        elif kl < target_kl / 1.5:
            beta /= 2.0

    这样可以让策略更新保持在合理范围内。
    """

    def __init__(
        self,
        initial_beta: float = 1.0,
        target_kl: float = 0.01,
        min_beta: float = 1e-6,
        max_beta: float = 100.0,
        adjustment_rate: float = 2.0,
    ) -> None:
        """
        初始化自适应KL系数调整器

        Args:
            initial_beta: 初始beta值
            target_kl: 目标KL散度值（通常0.01-0.05）
            min_beta: beta的最小值（防止过小）
            max_beta: beta的最大值（防止过大）
            adjustment_rate: 调整速率（默认2.0，即每次调整2倍）
        """
        self.initial_beta = float(initial_beta)
        self.beta = float(initial_beta)
        self.target_kl = float(target_kl)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.adjustment_rate = float(adjustment_rate)

    def update(self, kl_value: float) -> float:
        """
        根据KL散度值更新beta系数

        Args:
            kl_value: 当前的平均KL散度值

        Returns:
            更新后的beta值
        """
        # 根据KL散度与目标值的关系调整beta
        if kl_value > self.target_kl * 1.5:
            # KL散度太大，增加beta以加强约束
            self.beta = min(self.beta * self.adjustment_rate, self.max_beta)
        elif kl_value < self.target_kl / 1.5:
            # KL散度太小，减少beta以放松约束
            self.beta = max(self.beta / self.adjustment_rate, self.min_beta)

        return self.beta

    def get_beta(self) -> float:
        """获取当前的beta值"""
        return self.beta

    def reset(self, beta: Optional[float] = None) -> None:
        """
        重置beta值

        Args:
            beta: 新的beta值（如果为None则使用初始值）
        """
        if beta is None:
            self.beta = float(self.initial_beta)  # 使用初始值
        else:
            self.beta = float(beta)
        self.beta = max(self.min_beta, min(self.beta, self.max_beta))


class AdaptiveKLPenalty(KLPenalty):
    """
    自适应KL惩罚正则化器

    在KLPenalty的基础上，支持动态调整惩罚系数。
    用于PPO-Penalty变体，可以自动适应KL散度变化。
    """

    def __init__(
        self,
        adaptive_coefficient: AdaptiveKLCoefficient,
        use_adaptive: bool = True,
    ) -> None:
        """
        初始化自适应KL惩罚正则化器

        Args:
            adaptive_coefficient: 自适应KL系数调整器
            use_adaptive: 是否使用自适应调整（True时使用adaptive_coefficient的beta，False时使用固定coef）
        """
        # 使用adaptive_coefficient的初始beta作为基础coef
        super().__init__(coef=adaptive_coefficient.get_beta())
        self.adaptive_coefficient = adaptive_coefficient
        self.use_adaptive = use_adaptive

    def compute_penalty(self, **kwargs) -> torch.Tensor:
        """
        计算KL散度惩罚项（支持自适应系数）

        Args:
            **kwargs: 必须包含：
                - dist_new: 新策略分布
                - dist_old: 旧策略分布

        Returns:
            KL散度的平均值（标量张量）
        """
        kl = super().compute_penalty(**kwargs)

        # 如果使用自适应，更新系数
        if self.use_adaptive:
            kl_value = float(kl.detach().cpu().item())
            new_beta = self.adaptive_coefficient.update(kl_value)
            self.coef = new_beta

        return kl

    def get_beta(self) -> float:
        """获取当前的beta值（coef）"""
        return self.coef

