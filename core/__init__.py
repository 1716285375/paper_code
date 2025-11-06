# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:27
@Update Date    : 2025-11-05
@Description    : Core模块主入口
提供核心功能模块的统一导出
"""
# ------------------------------------------------------------

# Agent相关
from core.agent import (
    Agent,
    ConfigurablePPOAgent,
    AgentManager,
    build_agent_from_config,
    create_agent_manager_from_config,
)

# Trainer相关
from core.trainer import (
    RolloutCollector,
    Evaluator,
    MultiAgentEvaluator,
)

# 工具函数
from core.utils import (
    explained_variance,
    sequence_mask,
    masked_mean,
    masked_sum,
)

__all__ = [
    # Agent
    "Agent",
    "ConfigurablePPOAgent",
    "AgentManager",
    "build_agent_from_config",
    "create_agent_manager_from_config",
    # Trainer
    "RolloutCollector",
    "Evaluator",
    "MultiAgentEvaluator",
    # Utils
    "explained_variance",
    "sequence_mask",
    "masked_mean",
    "masked_sum",
]
