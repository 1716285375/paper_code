# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-29 20:48
@Update Date    :
@Description    : 通用工具模块
提供日志、检查点、指标收集、可视化等工具
"""
# ------------------------------------------------------------

# 可视化函数（从 plot 模块导入）
from common.plot import (
    plot_comparison,
    plot_distribution,
    plot_training_curves,
)

# Buffer 模块
from common.utils.buffer import (
    Buffer,
    DictBuffer,
    FIFOBuffer,
)
from common.utils.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    save_checkpoint,
)
from common.utils.data_manager import TrainingDataManager
from common.utils.logging import LoggerManager
from common.utils.metrics import (
    MetricsCollector,
    compute_statistics,
)

__all__ = [
    # 日志
    "LoggerManager",
    # 检查点
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    # 指标
    "MetricsCollector",
    "compute_statistics",
    # 可视化（从 plot 模块导入）
    "plot_training_curves",
    "plot_comparison",
    "plot_distribution",
    # Buffer
    "Buffer",
    "FIFOBuffer",
    "DictBuffer",
    # 数据管理
    "TrainingDataManager",
]
