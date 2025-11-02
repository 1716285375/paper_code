# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : adaptive.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 22:54
@Update Date    :
@Description    : 自适应优化器包装器
封装优化器并提供梯度裁剪、学习率调度等功能
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Iterable, Optional

import torch

from .base import OptimizerBuilder, OptimizerConfig, build_scheduler


class AdaptiveOptimizer:
    """
    自适应优化器包装器

    封装PyTorch优化器，提供额外的功能：
    - 梯度裁剪（防止梯度爆炸）
    - 学习率调度（动态调整学习率）
    """

    def __init__(self, optimizer: torch.optim.Optimizer, cfg: OptimizerConfig) -> None:
        """
        初始化自适应优化器

        Args:
            optimizer: PyTorch优化器实例
            cfg: 优化器配置（用于学习率调度器和梯度裁剪）
        """
        self.optimizer = optimizer
        self.cfg = cfg
        self.scheduler = build_scheduler(
            optimizer,
            name=cfg.scheduler,
            warmup_steps=cfg.warmup_steps,
            total_steps=cfg.total_steps,
        )

    def zero_grad(self) -> None:
        """清零梯度"""
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, params: Iterable[torch.nn.Parameter]) -> None:
        """
        执行优化步骤

        包括：
        1. 梯度裁剪（如果配置了max_grad_norm）
        2. 优化器更新
        3. 学习率调度器更新（如果存在）

        Args:
            params: 需要优化的参数（用于梯度裁剪）
        """
        # 梯度裁剪
        if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)

        # 优化器更新
        self.optimizer.step()

        # 学习率调度器更新
        if self.scheduler is not None:
            self.scheduler.step()
