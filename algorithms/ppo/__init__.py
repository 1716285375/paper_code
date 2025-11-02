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

from algorithms.ppo.self_play_trainer import PolicyPool, SelfPlayPPOTrainer
from algorithms.ppo.trainer import PPOTrainer

__all__ = [
    "PPOTrainer",
    "SelfPlayPPOTrainer",
    "PolicyPool",
]
