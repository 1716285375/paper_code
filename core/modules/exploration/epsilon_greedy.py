# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : epsilon_greedy.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:07
@Update Date    :
@Description    : Epsilon-Greedy探索策略
以epsilon概率随机选择动作，否则选择最优动作
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional

import torch

from .base import BaseScheduler, ExplorationStrategy, LinearSchedule


class EpsilonGreedy(ExplorationStrategy):
    """
    Epsilon-Greedy探索策略

    以epsilon概率随机选择动作（探索），否则选择最优动作（利用）。
    支持epsilon的动态衰减（通过scheduler）。
    """

    def __init__(self, epsilon: float = 0.1, scheduler: Optional[BaseScheduler] = None) -> None:
        """
        初始化Epsilon-Greedy策略

        Args:
            epsilon: 探索概率，取值范围 [0, 1]
            scheduler: 可选的调度器，用于动态调整epsilon值
        """
        self._epsilon = float(epsilon)
        self._scheduler = scheduler

    def reset(self) -> None:
        """重置策略状态"""
        return

    def on_step(self, info: Optional[dict] = None) -> None:
        """
        每步更新（用于epsilon衰减）

        Args:
            info: 可选的额外信息
        """
        if self._scheduler is not None:
            self._epsilon = self._scheduler.step()

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
        应用epsilon-greedy探索

        Args:
            logits: 动作logits，形状 (batch_size, num_actions)
            action: 动作值（epsilon-greedy不使用）
            mask: 动作掩码，标记哪些动作有效
            low: 动作下界（不使用）
            high: 动作上界（不使用）

        Returns:
            修改后的logits字典

        Raises:
            AssertionError: 如果logits为None
        """
        assert logits is not None, "EpsilonGreedy需要logits用于离散动作"
        device = logits.device
        num_actions = logits.shape[-1]

        # 如果有掩码，将无效动作的logits设为极小值，避免被argmax选中
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min / 2)

        # 以epsilon概率随机探索
        if torch.rand((), device=device).item() < self._epsilon:
            # 随机选择：创建均匀分布
            if mask is not None:
                # 在有效动作上创建均匀分布
                valid_counts = mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                probs = mask.float() / valid_counts
                # 将logits映射为：无效动作为-inf，有效动作为0（softmax后变为均匀分布）
                new_logits = torch.full_like(logits, float("-inf"))
                new_logits = torch.where(mask.bool(), torch.zeros_like(new_logits), new_logits)
            else:
                # 所有动作均匀分布：零logits即可
                new_logits = torch.zeros_like(logits)
        else:
            # 利用：使用原始logits（选择最优动作）
            new_logits = logits

        return {"logits": new_logits}
