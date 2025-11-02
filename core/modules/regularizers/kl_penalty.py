# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : kl_penalty.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:11
@Update Date    :
@Description    : KL散度惩罚正则化器
计算新旧策略分布之间的KL散度作为惩罚项
"""
# ------------------------------------------------------------


from __future__ import annotations

import torch
from torch.distributions import Distribution, kl_divergence

from .base import LossRegularizer


class KLPenalty(LossRegularizer):
    """
    KL散度惩罚正则化器

    用于PPO-penalty变体中，计算新旧策略分布之间的KL散度作为惩罚项。
    约束策略更新幅度，防止策略变化过大。

    compute_penalty期望参数中包含 dist_new 和 dist_old（torch.distributions对象），
    返回平均KL散度。
    """

    def __init__(self, coef: float = 1.0) -> None:
        """
        初始化KL惩罚正则化器

        Args:
            coef: 正则化系数，控制KL惩罚的强度
        """
        super().__init__(coef)

    def compute_penalty(self, **kwargs) -> torch.Tensor:
        """
        计算KL散度惩罚项

        Args:
            **kwargs: 必须包含：
                - dist_new: 新策略分布（torch.distributions.Distribution）
                - dist_old: 旧策略分布（torch.distributions.Distribution）

        Returns:
            KL散度的平均值（标量张量）
        """
        dist_new: Distribution = kwargs["dist_new"]
        dist_old: Distribution = kwargs["dist_old"]
        kl = kl_divergence(dist_new, dist_old)  # 计算KL散度
        return kl.mean()  # 返回平均值
