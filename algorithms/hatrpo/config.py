# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : config.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HATRPO算法配置
HATRPO算法的配置常量和默认值
"""
# ------------------------------------------------------------

from algorithms.happo.config import HAPPOConfig
from algorithms.matrpo.config import MATRPOConfig


class HATRPOConfig(HAPPOConfig, MATRPOConfig):
    """HATRPO算法配置常量（继承自HAPPOConfig和MATRPOConfig）"""
    pass

