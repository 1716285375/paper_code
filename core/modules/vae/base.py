# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : VAE基类定义
"""
# ------------------------------------------------------------

from __future__ import annotations

import torch.nn as nn


class BaseVAE(nn.Module):
    """
    VAE基类
    定义VAE的标准接口
    """

    def __init__(self) -> None:
        super().__init__()

    def encode(self, *args, **kwargs):
        """编码观测到潜在空间"""
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        """解码潜在变量到观测和动作"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """前向传播"""
        raise NotImplementedError

