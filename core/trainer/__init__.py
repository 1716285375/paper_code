# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 14:48
@Update Date    :
@Description    : Trainer通用组件模块
提供通用的训练器组件（rollout收集器、评估器等），不包含具体算法实现
"""
# ------------------------------------------------------------

from core.trainer.evaluator import Evaluator
from core.trainer.multi_agent_evaluator import MultiAgentEvaluator
from core.trainer.rollout_collector import RolloutCollector

__all__ = [
    "RolloutCollector",
    "Evaluator",
    "MultiAgentEvaluator",
]
