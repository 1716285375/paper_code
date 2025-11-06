# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : core.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HAPPO核心算法实现
HAPPO算法的核心计算函数
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Tuple
import torch


def update_marginal_advantage(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    current_advantage: torch.Tensor,
) -> torch.Tensor:
    """
    更新边际优势（Marginal Advantage）
    
    HAPPO的核心：每次更新一个agent后，使用新的logprob ratio更新advantage
    
    Args:
        new_logprobs: 新策略的对数概率
        old_logprobs: 旧策略的对数概率
        current_advantage: 当前优势估计
    
    Returns:
        更新后的边际优势
    """
    ratio = torch.exp(new_logprobs - old_logprobs)
    # 边际优势 = ratio * 当前优势
    marginal_advantage = ratio * current_advantage
    return marginal_advantage


def compute_happo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_coef: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算HAPPO策略损失（使用PPO裁剪）
    
    Args:
        logprobs: 当前策略的对数概率
        old_logprobs: 旧策略的对数概率
        advantages: 优势估计（可能是边际优势）
        clip_coef: PPO裁剪系数
    
    Returns:
        (policy_loss, clip_fraction)
    """
    ratio = torch.exp(logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    
    loss1 = ratio * advantages
    loss2 = clipped_ratio * advantages
    policy_loss = -torch.min(loss1, loss2).mean()
    
    clip_fraction = ((ratio - clipped_ratio).abs() > 1e-6).float().mean()
    
    return policy_loss, clip_fraction

