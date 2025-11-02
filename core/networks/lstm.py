# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : lstm.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : LSTM（长短期记忆网络）编码器
使用LSTM处理序列观测，提取时序特征
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseEncoder


class LSTMEncoder(BaseEncoder):
    """
    序列编码器，使用LSTM网络

    适用于处理序列观测数据，能够捕捉时序依赖关系。
    输入期望形状为 (B, T, D)，其中B为批次大小，T为序列长度，D为特征维度（如果batch_first=True）。
    返回最后一个时间步的隐藏状态作为特征表示。

    Args:
        input_dim: 输入特征维度
        hidden_size: LSTM隐藏层大小，默认256
        num_layers: LSTM层数，默认1
        batch_first: 如果为True，输入/输出形状为 (batch, seq, feature)，否则为 (seq, batch, feature)
        bidirectional: 是否使用双向LSTM，默认False
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
        初始化LSTM编码器

        Args:
            input_dim: 输入特征维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            batch_first: 是否批次维度在前
            bidirectional: 是否双向
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_dirs = 2 if bidirectional else 1  # 双向LSTM的输出维度会翻倍
        self.output_dim = hidden_size * self.num_dirs

    def forward(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            x: 输入序列，形状为 (B, T, D) 或 (T, B, D)（取决于batch_first）
            hidden: 可选的初始隐藏状态 (h_0, c_0)

        Returns:
            (features, new_hidden) 元组：
                - features: 最后一个时间步的隐藏状态，形状为 (B, hidden_size * num_dirs)
                - new_hidden: 新的隐藏状态 (h_n, c_n)，可用于下次前向传播
        """
        y, new_hidden = self.lstm(x, hidden)
        # 提取最后一个时间步的特征
        feats = y[:, -1, :] if self.batch_first else y[-1, :, :]
        return feats, new_hidden
