# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 算法共享组件
包含各算法模块共享的组件和工具
"""
# ------------------------------------------------------------

# 自博弈组件（使用 core 中的实现）
from algorithms.common.self_play import OpponentPool, EloRating

# 训练器混入
from algorithms.common.trainers.mixins import SelfPlayMixin

# TRPO核心工具
from algorithms.common.core_trpo import TrustRegionUpdater

__all__ = [
    # 自博弈组件（来自 core.agent.opponent_pool）
    "OpponentPool",
    "EloRating",
    # 训练器混入
    "SelfPlayMixin",
    # TRPO核心工具
    "TrustRegionUpdater",
]

