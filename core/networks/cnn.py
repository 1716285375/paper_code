# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : cnn.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : CNN（卷积神经网络）编码器
使用卷积层处理图像观测，提取空间特征
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .base import BaseEncoder


class CNNEncoder(BaseEncoder):
    """
    简单卷积层编码器

    使用一系列卷积层处理图像输入，提取空间特征，最后展平为向量。
    适用于处理2D图像观测（如游戏画面、传感器图像等）。
    """

    def __init__(self, in_channels: int, conv_channels: List[int] = [32, 64, 128]) -> None:
        """
        初始化CNN编码器

        Args:
            in_channels: 输入图像的通道数（如RGB为3，灰度图为1）
            conv_channels: 各层卷积通道数列表，例如 [32, 64, 128] 表示三层，通道数分别为32、64、128
        """
        super().__init__()
        layers: List[nn.Module] = []
        c = in_channels

        # 构建卷积层
        for out_c in conv_channels:
            # 卷积层：kernel_size=3, stride=2, padding=1 使得每层尺寸减半
            layers.append(nn.Conv2d(c, out_c, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU())  # ReLU激活
            c = out_c  # 更新通道数

        self.conv = nn.Sequential(*layers) if layers else nn.Identity()
        self.flatten = nn.Flatten()  # 展平层，将特征图展平为向量
        self._out_channels = c
        self.output_dim = None  # 延迟确定，需要根据输入尺寸计算

    def forward(self, x: torch.Tensor, hidden=None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, in_channels, height, width)
            hidden: 隐藏状态（CNN不使用，为兼容接口保留）

        Returns:
            编码后的特征向量，形状为 (batch_size, output_dim)
            输出维度会在第一次前向传播时自动确定
        """
        y = self.conv(x)  # 卷积特征提取
        y = self.flatten(y)  # 展平为向量

        # 在第一次前向传播时确定输出维度
        if self.output_dim is None:
            self.output_dim = y.shape[-1]

        return y
