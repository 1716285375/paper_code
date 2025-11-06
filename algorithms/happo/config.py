# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : config.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HAPPO算法配置
HAPPO算法的配置常量和默认值
"""
# ------------------------------------------------------------

from algorithms.mappo.config import MAPPOConfig


class HAPPOConfig(MAPPOConfig):
    """HAPPO算法配置常量（继承自MAPPOConfig）"""
    
    # HAPPO特定配置
    DEFAULT_UPDATE_ORDER = "random"  # 更新顺序：random, sequential
    DEFAULT_USE_MARGINAL_ADVANTAGE = True  # 是否使用边际优势（marginal advantage）

