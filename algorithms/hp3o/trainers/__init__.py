# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-06 00:00
@Update Date    :
@Description    : HP3O训练器模块
"""
# ------------------------------------------------------------

from algorithms.hp3o.trainers.base_trainer import HP3OTrainer
from algorithms.hp3o.trainers.self_play_trainer import SelfPlayHP3OTrainer

__all__ = [
    "HP3OTrainer",
    "SelfPlayHP3OTrainer",
]

