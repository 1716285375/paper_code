# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : config.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MATRPO算法配置
MATRPO算法的配置常量和默认值
"""
# ------------------------------------------------------------

from algorithms.mappo.config import MAPPOConfig


class MATRPOConfig(MAPPOConfig):
    """MATRPO算法配置常量（继承自MAPPOConfig）"""
    
    # TRPO特定配置
    DEFAULT_KL_THRESHOLD = 0.01  # KL散度阈值
    DEFAULT_MAX_LINE_SEARCH_STEPS = 15  # 最大线搜索步数
    DEFAULT_ACCEPT_RATIO = 0.1  # 接受比率
    DEFAULT_BACK_RATIO = 0.8  # 回退比率
    DEFAULT_CG_DAMPING = 0.1  # 共轭梯度阻尼系数
    DEFAULT_CG_MAX_ITERS = 10  # 共轭梯度最大迭代次数

