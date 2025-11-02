# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:30
@Update Date    :
@Description    : 正则化器模块
包含各种损失正则化和特征正则化实现
"""
# ------------------------------------------------------------

from core.modules.regularizers.adaptive_kl_penalty import (
    AdaptiveKLCoefficient,
    AdaptiveKLPenalty,
)
from core.modules.regularizers.base import FeatureRegularizer, LossRegularizer
from core.modules.regularizers.dropout import FeatureDropout
from core.modules.regularizers.kl_penalty import KLPenalty
from core.modules.regularizers.l2_regularizer import L2Regularizer

__all__ = [
    # 基类
    "LossRegularizer",
    "FeatureRegularizer",
    # KL惩罚
    "KLPenalty",
    "AdaptiveKLCoefficient",
    "AdaptiveKLPenalty",
    # 其他正则化
    "L2Regularizer",
    "FeatureDropout",
]
