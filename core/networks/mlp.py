# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : mlp.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : MLP（多层感知机）编码器
使用全连接层（MLP）将输入向量编码为特征表示
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from .base import BaseEncoder


class MLPEncoder(BaseEncoder):
    """
    多层感知机（MLP）编码器

    使用一系列全连接层将输入向量编码为固定维度的特征表示。
    支持LayerNorm和Dropout等正则化技术。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        activation: nn.Module = nn.ReLU,
        dropout: Optional[float] = None,
        layer_norm: bool = False,
    ) -> None:
        """
        初始化MLP编码器

        Args:
            in_dim: 输入维度
            hidden_dims: 隐藏层维度列表，例如 [128, 64] 表示两层，每层维度分别为128和64
            activation: 激活函数，默认ReLU
            dropout: Dropout概率，如果为None或0则不使用Dropout
            layer_norm: 是否使用LayerNorm
        """
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = in_dim

        # 构建网络层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))  # 线性层
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))  # LayerNorm（可选）
            layers.append(activation())  # 激活函数
            if dropout is not None and dropout > 0.0:
                layers.append(nn.Dropout(dropout))  # Dropout（可选）
            input_dim = hidden_dim

        # 组合所有层
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.output_dim = input_dim  # 输出维度等于最后一层的隐藏维度

    def forward(self, x: torch.Tensor, hidden: Optional[Any] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, in_dim)
            hidden: 隐藏状态（MLP不使用，为兼容接口保留）

        Returns:
            编码后的特征，形状为 (batch_size, output_dim)
        """
        return self.net(x)
