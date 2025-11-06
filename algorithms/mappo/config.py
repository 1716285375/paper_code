# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : config.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MAPPO算法配置
MAPPO算法的配置常量和默认值
"""
# ------------------------------------------------------------

from algorithms.ppo.config import PPOConfig


class MAPPOConfig(PPOConfig):
    """MAPPO算法配置常量（继承自PPOConfig）"""
    
    # MAPPO特定配置
    DEFAULT_USE_CENTRALIZED_CRITIC = False  # 是否使用集中式Critic
    DEFAULT_GLOBAL_OBS_DIM = None  # 全局观测维度（如果使用集中式Critic）
    DEFAULT_OPP_ACTION_IN_CC = False  # 是否在集中式Critic中使用对手动作

