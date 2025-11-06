# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : state_filter.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 状态过滤器（参考 smpe-main）
用于对状态的其他部分进行加权，过滤冗余信息
输入：obs + actions_onehot + rewards
输出：状态其他部分的权重（state_dim - obs_dim）
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules.filters.base import BaseFilter


class StateFilter(BaseFilter):
    """
    状态过滤器（参考 smpe-main）
    用于对状态的其他部分进行加权，过滤冗余信息
    
    输入：obs + actions_onehot + rewards（可选）
    输出：状态其他部分的权重（state_dim - obs_dim）
    """

    def __init__(
        self,
        input_shape: int,  # obs_dim + actions_dim + rewards_dim
        embedding_shape: int,  # state_dim - obs_dim（输出维度）
        use_gumbel: bool = False,
        use_2layer_filter: bool = True,
        use_clip_weights: bool = False,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
        device: str = "cpu",
    ) -> None:
        """
        初始化状态过滤器

        Args:
            input_shape: 输入维度（obs_dim + actions_dim + rewards_dim）
            embedding_shape: 输出维度（state_dim - obs_dim）
            use_gumbel: 是否使用Gumbel-Softmax（否则使用Sigmoid）
            use_2layer_filter: 是否使用两层网络（否则使用单层）
            use_clip_weights: 是否裁剪权重
            clip_min: 权重最小值
            clip_max: 权重最大值
            device: 设备
        """
        super().__init__()
        self.device = torch.device(device)
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape
        self.use_gumbel = use_gumbel
        self.use_2layer_filter = use_2layer_filter
        self.use_clip_weights = use_clip_weights
        self.clip_min = clip_min
        self.clip_max = clip_max

        # 构建过滤网络
        if self.use_gumbel:
            # Gumbel-Softmax: 输出 (embedding_shape, 2) 然后取 argmax
            self.fc1 = nn.Linear(self.input_shape, 2 * self.embedding_shape).to(self.device)
        else:
            # Sigmoid: 输出 embedding_shape
            self.fc1 = nn.Linear(self.input_shape, self.embedding_shape).to(self.device)
            if self.use_2layer_filter:
                self.fc2 = nn.Linear(self.embedding_shape, self.embedding_shape).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（参考 smpe-main）

        Args:
            x: 输入，形状为 (B, input_shape)
                包含：obs + actions_onehot + rewards

        Returns:
            权重，形状为 (B, embedding_shape)
                用于对状态的其他部分进行加权
        """
        if self.use_gumbel:
            # Gumbel-Softmax: 输出离散的权重
            x = F.gumbel_softmax(self.fc1(x).view(-1, self.embedding_shape, 2), hard=True, dim=-1)
            x = torch.argmax(x, dim=-1)
        else:
            # Sigmoid: 输出连续的权重
            if self.use_2layer_filter:
                x = F.relu(self.fc1(x))
                x = torch.sigmoid(self.fc2(x))
            else:
                x = torch.sigmoid(self.fc1(x))
            
            # 裁剪权重（可选）
            if self.use_clip_weights:
                x = torch.clamp(x, min=self.clip_min, max=self.clip_max)
        
        return x

    def to(self, device: str) -> "StateFilter":
        """移动到指定设备"""
        self.device = torch.device(device)
        if hasattr(self, 'fc1'):
            self.fc1 = self.fc1.to(self.device)
        if hasattr(self, 'fc2'):
            self.fc2 = self.fc2.to(self.device)
        return self

