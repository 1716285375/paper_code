# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HAPPO训练器模块
"""
# ------------------------------------------------------------

from algorithms.happo.trainers.base_trainer import HAPPOTrainer
from algorithms.happo.trainers.self_play_trainer import SelfPlayHAPPOTrainer

__all__ = [
    "HAPPOTrainer",
    "SelfPlayHAPPOTrainer",
]

