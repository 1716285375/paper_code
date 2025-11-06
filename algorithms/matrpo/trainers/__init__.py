# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MATRPO训练器模块
"""
# ------------------------------------------------------------

from algorithms.matrpo.trainers.base_trainer import MATRPOTrainer
from algorithms.matrpo.trainers.self_play_trainer import SelfPlayMATRPOTrainer

__all__ = [
    "MATRPOTrainer",
    "SelfPlayMATRPOTrainer",
]

