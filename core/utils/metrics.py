# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : metrics.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : 训练指标计算工具
包含价值函数解释方差、KL散度等指标的计算
"""
# ------------------------------------------------------------

from __future__ import annotations

import torch
import numpy as np


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    计算解释方差（Explained Variance）
    
    解释方差衡量预测值对真实值方差的解释程度，范围通常在[0, 1]。
    R² = 1 表示完美预测，R² = 0 表示预测不比均值好。
    
    Args:
        y_pred: 预测值（价值函数输出）
        y_true: 真实值（回报）
    
    Returns:
        解释方差（R²），范围通常在[0, 1]，可能为负值（表示预测很差）
    
    Formula:
        R² = 1 - (Var(y_true - y_pred) / Var(y_true))
        = 1 - (sum((y_true - y_pred)²) / sum((y_true - mean(y_true))²))
    """
    if len(y_pred) == 0 or len(y_true) == 0:
        return 0.0
    
    # 确保是tensor
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
    if isinstance(y_true, np.ndarray):
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
    
    # 移动到同一设备
    if y_pred.device != y_true.device:
        y_true = y_true.to(y_pred.device)
    
    # 计算方差
    y_true_var = torch.var(y_true, unbiased=False)
    
    # 如果真实值没有方差（所有值相同），无法计算解释方差
    if y_true_var < 1e-8:
        return 0.0
    
    # 计算残差方差
    residual_var = torch.var(y_true - y_pred, unbiased=False)
    
    # 计算解释方差
    ev = 1.0 - (residual_var / y_true_var)
    
    return float(ev.item())


def compute_kl_divergence(
    old_dist: torch.distributions.Distribution,
    new_dist: torch.distributions.Distribution,
) -> torch.Tensor:
    """
    计算KL散度
    
    Args:
        old_dist: 旧分布
        new_dist: 新分布
    
    Returns:
        KL散度
    """
    return torch.distributions.kl.kl_divergence(old_dist, new_dist)


def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    计算熵
    
    Args:
        probs: 概率分布
        dim: 计算熵的维度
    
    Returns:
        熵值
    """
    # 避免log(0)
    probs = torch.clamp(probs, min=1e-8)
    entropy = -(probs * torch.log(probs)).sum(dim=dim)
    return entropy

