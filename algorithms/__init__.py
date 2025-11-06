# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 22:07
@Update Date    :
@Description    : 算法模块集合
"""
# ------------------------------------------------------------

from algorithms.ppo import PPOTrainer, PPOConfig
from algorithms.mappo import MAPPOTrainer, SelfPlayMAPPOTrainer, MAPPOConfig
from algorithms.smpe import SMPETrainer, SMPESelfPlayTrainer, SMPEConfig
from algorithms.matrpo import MATRPOTrainer, SelfPlayMATRPOTrainer, MATRPOConfig
from algorithms.happo import HAPPOTrainer, SelfPlayHAPPOTrainer, HAPPOConfig
from algorithms.hatrpo import HATRPOTrainer, SelfPlayHATRPOTrainer, HATRPOConfig

__all__ = [
    "PPOTrainer",
    "PPOConfig",
    "MAPPOTrainer",
    "SelfPlayMAPPOTrainer",
    "MAPPOConfig",
    "SMPETrainer",
    "SMPESelfPlayTrainer",
    "SMPEConfig",
    "MATRPOTrainer",
    "SelfPlayMATRPOTrainer",
    "MATRPOConfig",
    "HAPPOTrainer",
    "SelfPlayHAPPOTrainer",
    "HAPPOConfig",
    "HATRPOTrainer",
    "SelfPlayHATRPOTrainer",
    "HATRPOConfig",
]
