# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-29 22:24
@Update Date    :
@Description    : 环境模块
提供统一的环境接口，支持多种环境后端（MAgent2、Gym等）
"""
# ------------------------------------------------------------

from environments.base import AgentEnv, AgentParrelEnv
from environments.factory import list_environments, make_env, register_env

__all__ = [
    # 基类
    "AgentEnv",
    "AgentParrelEnv",
    # 工厂函数
    "make_env",
    "register_env",
    "list_environments",
    # 子模块（按需导入）
    "base",
    "factory",
    "magent2",
]
