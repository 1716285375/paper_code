# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : role_filter.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 角色/特征过滤器
使用Sigmoid或Gumbel-Softmax生成掩码，在中央价值函数输入上做Hadamard乘
支持目标网络软更新
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.modules.filters.base import BaseFilter


class RoleFilter(BaseFilter):
    """
    角色/特征过滤器

    对潜在变量z或特征进行过滤，生成K维掩码（Sigmoid/Gumbel），
    在中央价值函数输入上做Hadamard乘，用于联合状态剪裁。
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 8,  # K维掩码
        use_gumbel: bool = False,
        temperature: float = 0.5,  # Gumbel温度或Sigmoid的软度
        tau: float = 0.01,  # 目标网络软更新系数
        device: str = "cpu",
    ) -> None:
        """
        初始化角色过滤器

        Args:
            input_dim: 输入特征维度（通常是潜在变量z的维度）
            num_filters: 过滤器数量（掩码维度K）
            use_gumbel: 是否使用Gumbel-Softmax（否则使用Sigmoid）
            temperature: Gumbel温度或Sigmoid的软度
            tau: 目标网络软更新系数
            device: 设备
        """
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        self.tau = tau

        # 主网络：生成掩码
        self.filter_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_filters),
        ).to(self.device)

        # 目标网络（用于软更新）
        self.target_filter_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_filters),
        ).to(self.device)

        # 初始化目标网络与主网络相同
        self.update_target(tau=1.0)

        # 优化器
        self.optimizer = torch.optim.Adam(self.filter_net.parameters(), lr=1e-3)

    def forward(self, features: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """
        应用过滤掩码

        Args:
            features: 输入特征，形状为 (B, input_dim)
            use_target: 是否使用目标网络（用于训练时的稳定性）

        Returns:
            过滤后的特征，形状为 (B, input_dim)
        """
        net = self.target_filter_net if use_target else self.filter_net

        # 生成掩码logits
        mask_logits = net(features)  # (B, num_filters)

        # 应用Sigmoid或Gumbel-Softmax
        if self.use_gumbel:
            # Gumbel-Softmax（离散但可微）
            mask = F.gumbel_softmax(mask_logits, tau=self.temperature, hard=False, dim=-1)  # (B, num_filters)
            # 对每个过滤器维度求和（或加权求和）来生成单一掩码
            mask = mask.sum(dim=-1, keepdim=True)  # (B, 1) - 简化版本
            mask = mask.expand_as(features)  # (B, input_dim)
        else:
            # Sigmoid（连续掩码）
            mask = torch.sigmoid(mask_logits)  # (B, num_filters)
            # 将多维度掩码加权求和或平均应用到特征上
            # 这里简化处理：取平均后扩展到特征维度
            mask = mask.mean(dim=-1, keepdim=True)  # (B, 1)
            mask = mask.expand_as(features)  # (B, input_dim)

        # Hadamard乘（逐元素相乘）
        filtered_features = features * mask

        return filtered_features

    def compute_target(self, features: torch.Tensor) -> torch.Tensor:
        """
        使用目标网络计算过滤目标（用于训练）

        Args:
            features: 输入特征

        Returns:
            目标过滤后的特征
        """
        return self.forward(features, use_target=True)

    def update_target(self, tau: Optional[float] = None) -> None:
        """
        软更新目标网络

        Args:
            tau: 更新系数（如果None则使用self.tau）
        """
        if tau is None:
            tau = self.tau

        # 软更新：target = tau * main + (1 - tau) * target
        for target_param, main_param in zip(self.target_filter_net.parameters(), self.filter_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)

    def update(
        self,
        features: torch.Tensor,
        filtered_target: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """
        更新过滤器参数

        Args:
            features: 输入特征
            filtered_target: 目标过滤结果（如果None则使用目标网络生成）

        Returns:
            损失字典
        """
        self.train()

        # 生成过滤结果
        filtered = self.forward(features, use_target=False)

        # 计算目标（如果未提供）
        if filtered_target is None:
            filtered_target = self.compute_target(features)

        # 损失：让主网络的输出接近目标网络
        loss = F.mse_loss(filtered, filtered_target)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.filter_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 软更新目标网络
        self.update_target()

        return {"filter_loss": float(loss.item())}

    def to(self, device: str) -> "RoleFilter":
        """移动到指定设备"""
        self.device = torch.device(device)
        self.filter_net = self.filter_net.to(self.device)
        self.target_filter_net = self.target_filter_net.to(self.device)
        return self

