# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-29 20:48
@Update Date    :
@Description    : Common工具模块
提供配置管理、日志记录、指标收集、可视化等通用工具
"""
# ------------------------------------------------------------

# 配置模块
from common.config import ConfigValidator, PPOConfig, load_config, validate_config

# 绘图模块
from common.plot import (
    Plotter,
    plot_evaluation_results,
    plot_learning_curves,
    plot_training_metrics,
)

# 实验跟踪模块
from common.tracking import (
    ExperimentTracker,
    TensorBoardTracker,
    WandBTracker,
)

# 工具模块
from common.utils import (
    Buffer,
    CheckpointManager,
    DictBuffer,
    FIFOBuffer,
    LoggerManager,
    MetricsCollector,
    TrainingDataManager,
    compute_statistics,
    load_checkpoint,
    plot_comparison,
    plot_distribution,
    plot_training_curves,
    save_checkpoint,
)

# 视频模块
from common.video import (
    VideoRecorder,
    record_episode,
)

__all__ = [
    # 配置
    "load_config",
    "PPOConfig",
    "validate_config",
    "ConfigValidator",
    # 工具
    "LoggerManager",
    "CheckpointManager",
    "save_checkpoint",
    "load_checkpoint",
    "MetricsCollector",
    "compute_statistics",
    # 可视化（向后兼容）
    "plot_training_curves",
    "plot_comparison",
    "plot_distribution",
    # Buffer
    "Buffer",
    "FIFOBuffer",
    "DictBuffer",
    # 数据管理
    "TrainingDataManager",
    # 绘图
    "Plotter",
    "plot_training_metrics",
    "plot_evaluation_results",
    "plot_learning_curves",
    # 视频
    "VideoRecorder",
    "record_episode",
    # 实验跟踪
    "ExperimentTracker",
    "WandBTracker",
    "TensorBoardTracker",
]
