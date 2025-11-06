# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE算法模块
SMPE (State Modeling and Predictive Exploration) 算法实现
包含Agent、Trainer等组件
"""
# ------------------------------------------------------------

from algorithms.smpe.policy_agent import SMPEPolicyAgent
from algorithms.smpe.trainers import SMPETrainer, SMPESelfPlayTrainer
from algorithms.smpe.config import SMPEConfig
from algorithms.smpe.core import (
    compute_combined_reward,
    compute_warmup_factor,
    estimate_state_from_observations,
    prepare_actions_onehot_others,
)

__all__ = [
    # Agent
    "SMPEPolicyAgent",
    # Trainers
    "SMPETrainer",
    "SMPESelfPlayTrainer",
    # Core utilities
    "SMPEConfig",
    "compute_combined_reward",
    "compute_warmup_factor",
    "estimate_state_from_observations",
    "prepare_actions_onehot_others",
]

