# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MAPPO训练器模块
包含MAPPO训练器实现
"""
# ------------------------------------------------------------

from algorithms.mappo.trainers.base_trainer import MAPPOTrainer
from algorithms.mappo.trainers.self_play_trainer import SelfPlayMAPPOTrainer

__all__ = [
    "MAPPOTrainer",
    "SelfPlayMAPPOTrainer",
]
