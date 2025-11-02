# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : 编码器（Encoder）抽象基类
定义了编码器的标准接口，用于将原始观测编码为特征表示
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    编码层的抽象基类接口

    编码器负责将原始观测（如图像、向量等）编码为固定维度的特征表示。
    所有具体的编码器（MLP、CNN、LSTM等）都需要继承此类。
    """

    def __init__(self) -> None:
        """
        初始化编码器
        """
        super().__init__()
        self.output_dim: Optional[int] = None  # 输出特征维度，某些编码器可能需要延迟确定

    @abstractmethod
    def forward(self, x: torch.Tensor, hidden: Optional[Any] = None) -> Any:
        """
        前向传播，将输入编码为特征

        Args:
            x: 输入张量（观测值）
            hidden: 可选的隐藏状态（用于RNN/LSTM等序列模型）

        Returns:
            编码后的特征表示，类型取决于具体实现
        """
        ...
