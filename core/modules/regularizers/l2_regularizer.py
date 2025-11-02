# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : l2_regularizer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:11
@Update Date    :
@Description    : L2正则化器
在损失函数中添加L2权重正则化项
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Iterable, Optional

import torch

from .base import LossRegularizer


class L2Regularizer(LossRegularizer):
    """
    L2权重正则化器

    在损失函数中添加L2正则化项（权重平方和），用于防止过拟合。
    注意：如果使用AdamW优化器的weight_decay，通常不需要此正则化器。
    此正则化器可用于针对特定的参数子集进行正则化。
    """

    def __init__(self, coef: float = 1e-4, include_bias: bool = False) -> None:
        """
        初始化L2正则化器

        Args:
            coef: 正则化系数
            include_bias: 是否对偏置项也进行正则化
        """
        super().__init__(coef)
        self.include_bias = include_bias

    def compute_penalty(self, **kwargs) -> torch.Tensor:
        """
        计算L2正则化惩罚项

        Args:
            **kwargs: 必须包含 "params"，为参数的可迭代对象

        Returns:
            L2正则化惩罚项（所有参数平方和）
        """
        params: Iterable[torch.nn.Parameter] = kwargs["params"]
        total = torch.zeros((), device=next(iter(params)).device)

        for p in params:
            if p.requires_grad:
                # 如果include_bias=False，跳过一维参数（通常是bias）
                if not self.include_bias and p.ndim == 1:
                    continue
                total = total + (p * p).sum()  # 累加参数平方

        return total
