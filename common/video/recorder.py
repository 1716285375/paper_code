# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : recorder.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 视频录制工具
提供环境交互过程的视频录制功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# 尝试导入imageio用于视频录制
try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    imageio = None


class VideoRecorder:
    """
    视频录制器

    用于录制Agent在环境中的交互过程，生成视频文件。
    """

    def __init__(
        self,
        output_dir: str = "videos",
        fps: int = 30,
        codec: str = "libx264",
        quality: str = "medium",
    ):
        """
        初始化视频录制器

        Args:
            output_dir: 输出目录
            fps: 帧率
            codec: 视频编解码器
            quality: 视频质量（"low", "medium", "high"）
        """
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio is required for video recording")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.codec = codec
        self.quality = quality

        self.frames: list[np.ndarray] = []
        self.is_recording = False

    def start_recording(self) -> None:
        """开始录制"""
        self.frames.clear()
        self.is_recording = True

    def add_frame(self, frame: np.ndarray) -> None:
        """
        添加一帧

        Args:
            frame: 图像帧，形状为 (H, W, C) 或 (H, W)，数据类型为uint8
        """
        if not self.is_recording:
            return

        # 确保是numpy数组
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # 确保数据类型正确
        if frame.dtype != np.uint8:
            # 如果是浮点数 [0, 1]，转换为uint8
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # 如果是灰度图，确保是2D
        if len(frame.shape) == 2:
            frame = frame[..., np.newaxis]

        # 确保是3D（H, W, C）
        if len(frame.shape) != 3:
            raise ValueError(f"Frame must be 2D (H, W) or 3D (H, W, C), got shape {frame.shape}")

        self.frames.append(frame.copy())

    def stop_recording(self, filename: Optional[str] = None) -> Path:
        """
        停止录制并保存视频

        Args:
            filename: 文件名，如果不指定则自动生成

        Returns:
            保存的视频文件路径
        """
        if not self.is_recording or len(self.frames) == 0:
            raise ValueError("No frames recorded")

        if filename is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"episode_{timestamp}.mp4"

        output_path = self.output_dir / filename

        # 写入视频
        writer = imageio.get_writer(
            str(output_path),
            fps=self.fps,
            codec=self.codec,
            quality=self.quality,
        )

        for frame in self.frames:
            writer.append_data(frame)

        writer.close()

        self.is_recording = False
        self.frames.clear()

        return output_path

    def record_episode(
        self,
        env: Any,
        agent: Any,
        max_steps: int = 1000,
        filename: Optional[str] = None,
        deterministic: bool = True,
    ) -> Path:
        """
        录制一个完整的episode

        Args:
            env: 环境实例（需要支持render方法）
            agent: Agent实例
            max_steps: 最大步数
            filename: 输出文件名
            deterministic: 是否使用确定性策略

        Returns:
            保存的视频文件路径
        """
        self.start_recording()

        obs = env.reset()
        done = False
        step = 0

        while not done and step < max_steps:
            # 渲染当前帧
            frame = env.render(mode="rgb_array")
            if frame is not None:
                self.add_frame(frame)

            # Agent选择动作
            if hasattr(agent, "act"):
                action = agent.act(obs, deterministic=deterministic)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = env.action_space.sample()

            # 环境步进
            obs, reward, done, info = env.step(action)
            step += 1

        # 记录最后一帧
        frame = env.render(mode="rgb_array")
        if frame is not None:
            self.add_frame(frame)

        return self.stop_recording(filename)


def record_episode(
    env: Any,
    agent: Any,
    output_path: str,
    max_steps: int = 1000,
    deterministic: bool = True,
    fps: int = 30,
) -> Path:
    """
    便捷函数：录制一个episode

    Args:
        env: 环境实例
        agent: Agent实例
        output_path: 输出路径
        max_steps: 最大步数
        deterministic: 是否使用确定性策略
        fps: 帧率

    Returns:
        保存的视频文件路径
    """
    recorder = VideoRecorder(fps=fps)
    return recorder.record_episode(env, agent, max_steps, output_path, deterministic)
