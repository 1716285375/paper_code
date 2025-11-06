# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE训练器模块
包含SMPE训练器实现
"""
# ------------------------------------------------------------

from algorithms.smpe.trainers.base_trainer import SMPETrainer
from algorithms.smpe.trainers.self_play_trainer import SMPESelfPlayTrainer

__all__ = [
    "SMPETrainer",
    "SMPESelfPlayTrainer",
]

