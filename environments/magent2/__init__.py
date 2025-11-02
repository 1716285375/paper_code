# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:31
@Update Date    :
@Description    : MAgent2环境模块
提供MAgent2环境的包装器实现
"""
# ------------------------------------------------------------

from environments.magent2.wrapper import (
    Magent2AdversarialPursuitV4Parallel,
    Magent2BattlefieldV4Parallel,
    Magent2BattleV4Parallel,
    Magent2CombinedArmsV6Parallel,
    Magent2GatherV4Parallel,
    Magent2ParallelBase,
    Magent2TigerDeerV3Parallel,
)

__all__ = [
    # 基类
    "Magent2ParallelBase",
    # 具体环境
    "Magent2BattleV4Parallel",
    "Magent2AdversarialPursuitV4Parallel",
    "Magent2BattlefieldV4Parallel",
    "Magent2CombinedArmsV6Parallel",
    "Magent2GatherV4Parallel",
    "Magent2TigerDeerV3Parallel",
    # 子模块
    "wrapper",
]
