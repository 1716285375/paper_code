# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : 工具函数模块
"""
# ------------------------------------------------------------

from .metrics import explained_variance, compute_kl_divergence, compute_entropy
from .sequence_mask import (
    sequence_mask,
    apply_sequence_mask,
    masked_mean,
    masked_sum,
)

__all__ = [
    "explained_variance",
    "compute_kl_divergence",
    "compute_entropy",
    "sequence_mask",
    "apply_sequence_mask",
    "masked_mean",
    "masked_sum",
]

