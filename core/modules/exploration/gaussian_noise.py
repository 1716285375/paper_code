# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : gaussian_noise.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:08
@Update Date    :
@Description    : 高斯噪声探索策略
通过向连续动作添加高斯噪声来增加探索性
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional

import torch

from .base import BaseScheduler, ExplorationStrategy


class GaussianNoise(ExplorationStrategy):
    """
    高斯噪声探索策略

    向连续动作添加高斯噪声，用于连续动作空间的探索。
    适用于需要平滑探索的场景（如机器人控制）。
    """

    def __init__(self, sigma: float = 0.1, scheduler: Optional[BaseScheduler] = None) -> None:
        """
        初始化高斯噪声策略

        Args:
            sigma: 噪声标准差，控制噪声强度
            scheduler: 可选的调度器，用于动态调整噪声强度
        """
        self._sigma = float(sigma)
        self._scheduler = scheduler

    def reset(self) -> None:
        """重置策略状态"""
        return

    def on_step(self, info: Optional[dict] = None) -> None:
        """
        每步更新（用于噪声衰减）

        Args:
            info: 可选的额外信息
        """
        if self._scheduler is not None:
            self._sigma = max(0.0, float(self._scheduler.step()))

    def apply(
        self,
        *,
        logits: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        """
        应用高斯噪声到动作

        Args:
            logits: 动作logits（不使用）
            action: 连续动作张量，形状 (batch_size, action_dim)
            mask: 动作掩码（不使用）
            low: 动作空间下界
            high: 动作空间上界

        Returns:
            添加噪声后的动作字典

        Raises:
            AssertionError: 如果action为None
        """
        assert action is not None, "GaussianNoise需要连续动作张量"

        # 生成高斯噪声并添加到动作
        noise = torch.randn_like(action) * self._sigma
        new_action = action + noise

        # 如果提供了边界，将动作裁剪到有效范围内
        if low is not None and high is not None:
            new_action = torch.max(torch.min(new_action, high), low)

        return {"action": new_action}
