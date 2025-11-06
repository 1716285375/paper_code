# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : PPO训练器模块
包含所有PPO训练器实现
"""
# ------------------------------------------------------------

from algorithms.ppo.trainers.base_trainer import PPOTrainer
from algorithms.ppo.trainers.self_play_trainer import SelfPlayPPOTrainer

__all__ = [
    "PPOTrainer",
    "SelfPlayPPOTrainer",
]

