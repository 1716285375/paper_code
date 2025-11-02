# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : adam.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 22:51
@Update Date    :
@Description    : Adam和AdamW优化器构建器
提供Adam和AdamW优化器的工厂实现
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Iterable, Optional

import torch

from .base import OptimizerBuilder, OptimizerConfig


class AdamBuilder(OptimizerBuilder):
    """
    Adam优化器构建器

    Adam（Adaptive Moment Estimation）是一种自适应学习率的优化算法。
    """

    def build(
        self, params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """
        构建Adam优化器

        Args:
            params: 需要优化的参数
            cfg: 优化器配置

        Returns:
            Adam优化器实例
        """
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )


class AdamWBuilder(OptimizerBuilder):
    """
    AdamW优化器构建器

    AdamW是Adam的改进版本，使用解耦的权重衰减（decoupled weight decay）。
    通常比Adam有更好的泛化性能。
    """

    def build(
        self, params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """
        构建AdamW优化器

        Args:
            params: 需要优化的参数
            cfg: 优化器配置

        Returns:
            AdamW优化器实例
        """
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
