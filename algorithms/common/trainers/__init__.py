# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 训练器共享组件
包含可复用的训练器基类和混入类
"""
# ------------------------------------------------------------

# 训练器基类
from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer

# 训练器混入类
from algorithms.common.trainers.mixins import SelfPlayMixin

__all__ = [
    "BaseAlgorithmTrainer",
    "SelfPlayMixin",
]
