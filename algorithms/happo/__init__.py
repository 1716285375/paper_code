# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HAPPO算法模块
异质智能体PPO（Heterogeneous Agent PPO）
"""
# ------------------------------------------------------------

from algorithms.happo.trainers.base_trainer import HAPPOTrainer
from algorithms.happo.trainers.self_play_trainer import SelfPlayHAPPOTrainer
from algorithms.happo.config import HAPPOConfig

__all__ = [
    "HAPPOTrainer",
    "SelfPlayHAPPOTrainer",
    "HAPPOConfig",
]

