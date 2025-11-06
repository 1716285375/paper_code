# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : gru.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : GRU（门控循环单元）编码器
使用GRU处理序列观测，提取时序特征
GRU是LSTM的简化版本，参数更少，计算更快，但性能通常相当
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseEncoder


class GRUEncoder(BaseEncoder):
    """
    序列编码器，使用GRU网络

    适用于处理序列观测数据，能够捕捉时序依赖关系。
    输入期望形状为 (B, T, D)，其中B为批次大小，T为序列长度，D为特征维度（如果batch_first=True）。
    返回最后一个时间步的隐藏状态作为特征表示。

    GRU相比LSTM的优势：
    - 参数更少，计算更快
    - 内存占用更小
    - 性能通常与LSTM相当

    Args:
        input_dim: 输入特征维度
        hidden_size: GRU隐藏层大小，默认256
        num_layers: GRU层数，默认1
        batch_first: 如果为True，输入/输出形状为 (batch, seq, feature)，否则为 (seq, batch, feature)
        bidirectional: 是否使用双向GRU，默认False
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = False,
    ) -> None:
        """
        初始化GRU编码器

        Args:
            input_dim: 输入特征维度
            hidden_size: GRU隐藏层大小
            num_layers: GRU层数
            batch_first: 是否批次维度在前
            bidirectional: 是否双向
        """
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_dirs = 2 if bidirectional else 1  # 双向GRU的输出维度会翻倍
        self.output_dim = hidden_size * self.num_dirs

    def forward(
        self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列，形状为 (B, T, D) 或 (T, B, D)（取决于batch_first）
            hidden: 可选的初始隐藏状态 h_0，形状为 (num_layers * num_dirs, B, hidden_size)
                   如果为None，则使用零初始化

        Returns:
            (features, new_hidden) 元组：
                - features: 最后一个时间步的隐藏状态，形状为 (B, hidden_size * num_dirs)
                - new_hidden: 新的隐藏状态 h_n，形状为 (num_layers * num_dirs, B, hidden_size)
                            可用于下次前向传播
        """
        y, new_hidden = self.gru(x, hidden)
        # 提取最后一个时间步的特征
        feats = y[:, -1, :] if self.batch_first else y[-1, :, :]
        return feats, new_hidden

