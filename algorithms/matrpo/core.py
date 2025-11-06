# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : core.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MATRPO核心算法实现
MATRPO算法的核心计算函数
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Tuple
import torch


def compute_trpo_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """
    计算TRPO策略损失（不使用裁剪，仅计算ratio * advantages）
    
    Args:
        logprobs: 当前策略的对数概率
        old_logprobs: 旧策略的对数概率
        advantages: 优势估计
    
    Returns:
        策略损失
    """
    ratio = torch.exp(logprobs - old_logprobs)
    # TRPO不使用裁剪，直接使用ratio
    policy_loss = -(ratio * advantages).mean()
    return policy_loss


def compute_trpo_kl(
    old_dist,
    new_dist,
) -> torch.Tensor:
    """
    计算KL散度
    
    Args:
        old_dist: 旧策略分布
        new_dist: 新策略分布
    
    Returns:
        KL散度
    """
    return old_dist.kl(new_dist).mean()

