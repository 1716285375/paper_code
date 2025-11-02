# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : temperature.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:08
@Update Date    :
@Description    : 温度缩放探索策略
通过缩放logits来控制策略分布的平滑程度
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional

import torch

from .base import BaseScheduler, ExplorationStrategy, LinearSchedule


class TemperatureScaling(ExplorationStrategy):
    """
    温度缩放探索策略

    通过将logits除以温度参数来控制策略分布的平滑程度。
    温度越高，分布越平滑（探索性越强）；温度越低，分布越尖锐（利用性越强）。
    """

    def __init__(self, temperature: float = 1.0, scheduler: Optional[BaseScheduler] = None) -> None:
        """
        初始化温度缩放策略

        Args:
            temperature: 温度参数，通常 >= 1.0，值越大探索性越强
            scheduler: 可选的调度器，用于动态调整温度值
        """
        self._temperature = max(1e-6, float(temperature))  # 确保温度为正数
        self._scheduler = scheduler

    def reset(self) -> None:
        """重置策略状态"""
        return

    def on_step(self, info: Optional[dict] = None) -> None:
        """
        每步更新（用于温度衰减）

        Args:
            info: 可选的额外信息
        """
        if self._scheduler is not None:
            self._temperature = max(1e-6, self._scheduler.step())

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
        应用温度缩放

        Args:
            logits: 动作logits，形状 (batch_size, num_actions)
            action: 动作值（不使用）
            mask: 动作掩码，标记哪些动作有效
            low: 动作下界（不使用）
            high: 动作上界（不使用）

        Returns:
            缩放后的logits字典

        Raises:
            AssertionError: 如果logits为None
        """
        assert logits is not None, "TemperatureScaling需要logits（离散动作）"

        # 将温度转换为张量并缩放logits
        t = torch.as_tensor(self._temperature, device=logits.device, dtype=logits.dtype)
        scaled = logits / t  # 温度缩放：logits / temperature

        # 如果有掩码，将无效动作的logits设为极小值
        if mask is not None:
            scaled = scaled.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min / 2)

        return {"logits": scaled}
