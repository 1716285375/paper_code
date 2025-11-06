# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MAPPO算法模块
集中训练-分散执行（CTDE）的多智能体PPO实现
"""
# ------------------------------------------------------------

from algorithms.mappo.trainers.base_trainer import MAPPOTrainer
from algorithms.mappo.trainers.self_play_trainer import SelfPlayMAPPOTrainer
from algorithms.mappo.config import MAPPOConfig

__all__ = [
    "MAPPOTrainer",
    "SelfPlayMAPPOTrainer",
    "MAPPOConfig",
]

