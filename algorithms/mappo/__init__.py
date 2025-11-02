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

from algorithms.mappo.trainer import MAPPOTrainer
from algorithms.mappo.smpe_self_play_trainer import SMPESelfPlayTrainer

__all__ = [
    "MAPPOTrainer",
    "SMPESelfPlayTrainer",
]

