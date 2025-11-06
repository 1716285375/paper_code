# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : core.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : PPO核心算法实现
PPO算法的核心计算函数和工具函数
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def clip_ratio(ratio: torch.Tensor, clip_coef: float) -> torch.Tensor:
    """
    裁剪重要性采样比率
    
    Args:
        ratio: 重要性采样比率 (new_prob / old_prob)
        clip_coef: 裁剪系数
    
    Returns:
        裁剪后的比率
    """
    return torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)


def compute_ppo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算PPO策略损失
    
    Args:
        logprobs: 当前策略的对数概率
        old_logprobs: 旧策略的对数概率
        advantages: 优势估计
        clip_coef: PPO裁剪系数
    
    Returns:
        (policy_loss, clip_fraction): 策略损失和裁剪比例
    """
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = clip_ratio(ratio, clip_coef)
    
    # 计算损失（取最小值）
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(loss1, loss2).mean()
    
    # 计算裁剪比例（用于监控）
    clip_fraction = ((ratio - clipped_ratio).abs() > 1e-6).float().mean()
    
    return policy_loss, clip_fraction


def compute_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    """
    计算价值函数损失（MSE）
    
    Args:
        values: 价值函数预测
        returns: 回报（目标）
    
    Returns:
        价值损失
    """
    return ((values - returns) ** 2).mean()


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    计算策略熵
    
    Args:
        probs: 动作概率分布
    
    Returns:
        熵值
    """
    return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    标准化优势估计
    
    Args:
        advantages: 优势估计
        eps: 数值稳定性常数
    
    Returns:
        标准化后的优势
    """
    mean = advantages.mean()
    std = advantages.std()
    return (advantages - mean) / (std + eps)

