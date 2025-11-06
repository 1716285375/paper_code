# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:28
@Update Date    :
@Description    : PPO算法模块
包含PPO算法的训练器实现，包括标准训练和自博弈训练
"""
# ------------------------------------------------------------

from algorithms.ppo.trainers.base_trainer import PPOTrainer
from algorithms.ppo.trainers.self_play_trainer import SelfPlayPPOTrainer
from algorithms.ppo.core import (
    clip_ratio,
    compute_ppo_loss,
    compute_value_loss,
    compute_entropy,
    normalize_advantages,
)
from algorithms.ppo.config import PPOConfig

__all__ = [
    # Trainers
    "PPOTrainer",
    "SelfPlayPPOTrainer",
    # Core functions
    "clip_ratio",
    "compute_ppo_loss",
    "compute_value_loss",
    "compute_entropy",
    "normalize_advantages",
    # Config
    "PPOConfig",
]
