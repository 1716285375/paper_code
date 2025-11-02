# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : dropout.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:11
@Update Date    :
@Description    : Dropout特征正则化器
在特征层面应用Dropout来防止过拟合
"""
# ------------------------------------------------------------


from __future__ import annotations

import torch
import torch.nn as nn

from .base import FeatureRegularizer


class FeatureDropout(FeatureRegularizer):
    """
    Dropout特征正则化器

    在训练时随机丢弃部分特征，用于防止过拟合。
    这是特征层面的正则化，不改变损失函数。
    """

    def __init__(self, p: float = 0.1) -> None:
        """
        初始化Dropout正则化器

        Args:
            p: Dropout概率，即每个特征被丢弃的概率
        """
        self.dropout = nn.Dropout(p)

    def apply_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        对特征应用Dropout

        Args:
            x: 输入特征张量

        Returns:
            应用Dropout后的特征张量
        """
        return self.dropout(x)
