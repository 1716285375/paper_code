# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:10
@Update Date    :
@Description    : 正则化器抽象基类
定义了损失正则化和特征正则化的标准接口
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, Tuple

import torch


class LossRegularizer(ABC):
    """
    损失正则化器抽象基类

    在损失函数中添加正则化项（惩罚项），用于防止过拟合或约束模型行为。
    例如L2正则化、KL散度惩罚等。
    """

    def __init__(self, coef: float) -> None:
        """
        初始化正则化器

        Args:
            coef: 正则化系数，控制正则化项的强度
        """
        self.coef = float(coef)

    @abstractmethod
    def compute_penalty(self, **kwargs) -> torch.Tensor:
        """
        计算正则化惩罚项

        Args:
            **kwargs: 计算惩罚所需的参数

        Returns:
            惩罚项的张量（标量）
        """
        ...

    def apply(self, base_loss: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        应用正则化到损失函数

        Args:
            base_loss: 基础损失值
            **kwargs: 传递给compute_penalty的参数

        Returns:
            (正则化后的损失, 指标字典) 元组
        """
        penalty = self.compute_penalty(**kwargs)
        loss = base_loss + self.coef * penalty
        return loss, {self.__class__.__name__: float(penalty.detach().cpu().item())}


class FeatureRegularizer(ABC):
    """
    特征正则化器抽象基类

    对中间特征表示进行变换或处理，例如Dropout、BatchNorm等。
    与损失正则化不同，这类正则化在特征层面操作。
    """

    @abstractmethod
    def apply_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        对特征应用正则化变换

        Args:
            x: 输入特征张量

        Returns:
            变换后的特征张量
        """
        ...
