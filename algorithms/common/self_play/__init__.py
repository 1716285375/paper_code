# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 自博弈共享组件
包含策略池、对手采样等通用组件
"""
# ------------------------------------------------------------

# 使用 core 中的 OpponentPool，避免重复实现
from core.agent.opponent_pool import OpponentPool, EloRating

__all__ = [
    "OpponentPool",
    "EloRating",
]

