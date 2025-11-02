# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : __init__.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 14:49
@Update Date    :
@Description    : 视频录制工具模块
提供环境交互过程的视频录制功能
"""
# ------------------------------------------------------------

from common.video.recorder import VideoRecorder, record_episode

__all__ = [
    "VideoRecorder",
    "record_episode",
]
