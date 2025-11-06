# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : sequence_mask.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : RNN序列掩码工具
用于处理变长序列的padding掩码
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional, Union

import torch
import numpy as np


def sequence_mask(
    lengths: Union[torch.Tensor, np.ndarray, list],
    max_length: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    创建序列掩码（Sequence Mask）
    
    用于处理变长序列，创建一个掩码，其中：
    - 1 表示有效时间步
    - 0 表示padding时间步
    
    Args:
        lengths: 每个序列的实际长度，形状为 (B,)
        max_length: 最大序列长度（如果不提供，使用lengths的最大值）
        dtype: 掩码的数据类型
        device: 掩码的设备
    
    Returns:
        掩码张量，形状为 (B, max_length)，其中：
        - mask[i, j] = 1 如果 j < lengths[i]
        - mask[i, j] = 0 如果 j >= lengths[i]
    
    Example:
        >>> lengths = torch.tensor([3, 5, 2])
        >>> mask = sequence_mask(lengths, max_length=5)
        >>> # mask = [[1, 1, 1, 0, 0],
        >>> #         [1, 1, 1, 1, 1],
        >>> #         [1, 1, 0, 0, 0]]
    """
    # 转换为tensor
    if isinstance(lengths, (list, np.ndarray)):
        lengths = torch.as_tensor(lengths, dtype=torch.long)
    else:
        lengths = lengths.long()
    
    # 确定设备
    if device is None:
        device = lengths.device
    
    # 移动到设备
    lengths = lengths.to(device)
    
    # 确定最大长度
    if max_length is None:
        max_length = int(lengths.max().item())
    else:
        max_length = int(max_length)
    
    # 创建范围张量: [0, 1, 2, ..., max_length-1]
    range_tensor = torch.arange(max_length, device=device, dtype=torch.long)
    
    # 扩展维度以支持广播
    # lengths: (B,) -> (B, 1)
    # range_tensor: (max_length,) -> (1, max_length)
    lengths_expanded = lengths.unsqueeze(1)  # (B, 1)
    range_expanded = range_tensor.unsqueeze(0)  # (1, max_length)
    
    # 创建掩码: mask[i, j] = 1 if j < lengths[i], else 0
    mask = (range_expanded < lengths_expanded).to(dtype=dtype)
    
    return mask


def apply_sequence_mask(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = 1,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """
    应用序列掩码到张量
    
    Args:
        tensor: 要掩码的张量，形状为 (B, T, ...) 或 (B, T)
        mask: 掩码，形状为 (B, T)
        dim: 时间维度（通常是1）
        fill_value: 填充值（用于掩码位置）
    
    Returns:
        掩码后的张量
    """
    # 确保掩码维度匹配
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(-1)
    
    # 扩展掩码以匹配tensor的形状
    mask = mask.expand_as(tensor)
    
    # 应用掩码
    masked_tensor = tensor * mask + (1 - mask) * fill_value
    
    return masked_tensor


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    计算掩码后的均值
    
    Args:
        tensor: 输入张量
        mask: 掩码（1表示有效，0表示无效）
        dim: 计算均值的维度（如果为None，计算所有维度）
    
    Returns:
        掩码后的均值
    """
    # 应用掩码
    masked_tensor = tensor * mask
    
    # 计算有效元素的总和
    sum_masked = masked_tensor.sum(dim=dim) if dim is not None else masked_tensor.sum()
    
    # 计算有效元素的数量
    count = mask.sum(dim=dim) if dim is not None else mask.sum()
    
    # 避免除零
    count = torch.clamp(count, min=1.0)
    
    return sum_masked / count


def masked_sum(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    计算掩码后的总和
    
    Args:
        tensor: 输入张量
        mask: 掩码（1表示有效，0表示无效）
        dim: 计算总和的维度（如果为None，计算所有维度）
    
    Returns:
        掩码后的总和
    """
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) if dim is not None else masked_tensor.sum()

