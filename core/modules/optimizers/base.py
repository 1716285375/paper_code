# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 22:42
@Update Date    :
@Description    : 优化器（Optimizer）抽象基类和配置
定义了优化器的构建接口和学习率调度器
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch


@dataclass
class OptimizerConfig:
    """
    优化器配置类

    包含优化器的所有配置参数，支持多种优化器（Adam、AdamW、RMSProp等）。
    """

    lr: float = 0.01  # 学习率
    weight_decay: float = 0.0  # 权重衰减（L2正则化）
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam的beta参数
    eps: float = 1e-8  # 数值稳定性常数
    momentum: float = 0.9  # 动量（SGD、RMSProp使用）
    alpha: float = 0.99  # RMSProp的alpha参数
    centered: bool = False  # RMSProp是否使用centered模式
    max_grad_norm: Optional[float] = 0.5  # 梯度裁剪的最大范数
    scheduler: Optional[str] = None  # 学习率调度器类型（'linear', 'cosine'等）
    warmup_steps: int = 0  # 预热步数
    total_steps: int = 0  # 总训练步数（用于调度器）


class OptimizerBuilder(ABC):
    """
    优化器构建器抽象基类

    用于构建不同类型的优化器（Adam、AdamW、RMSProp等）。
    """

    @abstractmethod
    def build(
        self, params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig
    ) -> torch.optim.Optimizer:
        """
        构建优化器

        Args:
            params: 需要优化的参数列表
            cfg: 优化器配置

        Returns:
            PyTorch优化器实例
        """
        ...


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_norm: Optional[float]
) -> Optional[float]:
    """
    梯度裁剪

    将梯度的范数裁剪到指定范围内，防止梯度爆炸。

    Args:
        parameters: 模型参数
        max_norm: 最大梯度范数，如果为None则不进行裁剪

    Returns:
        裁剪前的梯度范数，如果未裁剪则返回None
    """
    if max_norm is None or max_norm < 0.0:
        return None
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()


# 可序列化的学习率lambda函数类
class LinearLRScheduler:
    """
    可序列化的线性学习率调度器
    
    用于替代局部函数，使得LambdaLR可以序列化
    """
    
    def __init__(self, warmup_steps: int, total_steps: int):
        """
        初始化线性学习率调度器
        
        Args:
            warmup_steps: 预热步数
            total_steps: 总训练步数
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def __call__(self, step: int) -> float:
        """
        计算学习率缩放因子
        
        Args:
            step: 当前步数
        
        Returns:
            学习率缩放因子
        """
        if self.total_steps <= 0:
            return 1.0
        # 预热阶段：线性增加
        if step < self.warmup_steps and self.warmup_steps > 0:
            return float(step) / float(max(1, self.warmup_steps))
        # 衰减阶段：线性减少
        return max(0.0, float(self.total_steps - step) / float(max(1, self.total_steps - self.warmup_steps)))


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: Optional[str],
    warmup_steps: int,
    total_steps: int,
):
    """
    构建学习率调度器

    支持的学习率调度策略：
        - 'linear': 线性衰减（带预热）- 使用可序列化的类
        - 'cosine': 余弦退火

    Args:
        optimizer: 优化器实例
        name: 调度器名称（'linear', 'cosine'等），None表示不使用调度器
        warmup_steps: 预热步数（学习率从0线性增加到初始学习率）
        total_steps: 总训练步数

    Returns:
        学习率调度器实例，如果name为None则返回None
    """
    if name is None:
        return None

    name = name.lower()

    if name == "cosine":
        # 余弦退火调度器
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    if name == "linear":
        # 线性衰减调度器（带预热）
        # 使用可序列化的类而不是局部函数
        lr_lambda = LinearLRScheduler(warmup_steps, total_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return None
