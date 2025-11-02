# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : transformer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : Transformer编码器
使用Transformer网络处理序列观测，提取时序特征
"""
# ------------------------------------------------------------


from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .base import BaseEncoder


class PositionalEncoding(nn.Module):
    """
    位置编码

    为序列输入添加位置信息，使用正弦和余弦函数编码位置。
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        初始化位置编码

        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 交替使用sin和cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状变为 (1, L, D)
        self.register_buffer("pe", pe)  # 注册为buffer，不参与梯度更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码

        Args:
            x: 输入序列，形状 (B, T, D)

        Returns:
            添加位置编码后的序列，形状 (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerEncoder(BaseEncoder):
    """
    Transformer编码器

    使用Transformer网络处理序列输入，适用于处理时序观测数据。
    输入形状为 (B, T, D)，返回最后一个token的特征表示。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        初始化Transformer编码器

        Args:
            d_model: 模型维度（特征维度）
            nhead: 多头注意力的头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout概率
        """
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列，形状 (B, T, D)
            hidden: 隐藏状态（Transformer不使用，为兼容接口保留）

        Returns:
            最后一个时间步的特征，形状 (B, d_model)
        """
        y = self.encoder(self.pos_enc(x))  # 编码序列
        feats = y[:, -1, :]  # 提取最后一个时间步的特征
        return feats
