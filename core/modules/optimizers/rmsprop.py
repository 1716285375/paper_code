# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : rmsprop.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 22:59
@Update Date    :
@Description    : RMSProp优化器构建器
提供RMSProp优化器的工厂实现
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Iterable

import torch

from .base import OptimizerBuilder, OptimizerConfig


class RMSPropBuilder(OptimizerBuilder):
    """
    RMSProp优化器构建器

    RMSProp（Root Mean Square Propagation）是一种自适应学习率优化算法，
    通过维护梯度平方的移动平均来调整学习率。
    """

    def build(
        self, params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """
        构建RMSProp优化器

        Args:
            params: 需要优化的参数
            cfg: 优化器配置

        Returns:
            RMSProp优化器实例
        """
        return torch.optim.RMSprop(
            params,
            lr=cfg.lr,
            alpha=cfg.alpha,  # 平滑系数
            eps=cfg.eps,
            momentum=cfg.momentum,  # 动量项
            centered=cfg.centered,  # 是否使用centered RMSProp
            weight_decay=cfg.weight_decay,
        )
