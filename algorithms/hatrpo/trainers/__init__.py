# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HATRPO训练器模块
"""
# ------------------------------------------------------------

from algorithms.hatrpo.trainers.base_trainer import HATRPOTrainer
from algorithms.hatrpo.trainers.self_play_trainer import SelfPlayHATRPOTrainer

__all__ = [
    "HATRPOTrainer",
    "SelfPlayHATRPOTrainer",
]

