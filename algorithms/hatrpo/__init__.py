# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HATRPO算法模块
异质智能体TRPO（Heterogeneous Agent TRPO）
"""
# ------------------------------------------------------------

from algorithms.hatrpo.trainers.base_trainer import HATRPOTrainer
from algorithms.hatrpo.trainers.self_play_trainer import SelfPlayHATRPOTrainer
from algorithms.hatrpo.config import HATRPOConfig

__all__ = [
    "HATRPOTrainer",
    "SelfPlayHATRPOTrainer",
    "HATRPOConfig",
]

