# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : tracker.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-01-XX XX:XX
@Update Date    :
@Description    : 实验跟踪器基类和统一接口
定义了实验跟踪器的抽象接口和组合器实现
"""
# ------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseTracker(ABC):
    """
    实验跟踪器基类

    定义了实验跟踪器的标准接口，所有具体的跟踪器实现（wandb、tensorboard等）
    都需要继承此类并实现抽象方法。
    """

    @abstractmethod
    def init(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        初始化跟踪器

        Args:
            project: 项目名称
            name: 实验名称（可选）
            config: 实验配置字典（可选）
            **kwargs: 其他初始化参数
        """
        pass

    @abstractmethod
    def log(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """
        记录指标

        Args:
            metrics: 指标字典，键为指标名，值为指标值
            step: 当前步数（可选，如果不提供则自动递增）
        """
        pass

    @abstractmethod
    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录字典形式的指标（支持嵌套）

        Args:
            metrics: 指标字典，可以包含嵌套结构
            step: 当前步数（可选）
        """
        pass

    @abstractmethod
    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        """
        记录图像

        Args:
            tag: 图像标签/名称
            image: 图像数据（numpy数组、PIL图像等）
            step: 当前步数（可选）
        """
        pass

    @abstractmethod
    def log_video(self, tag: str, video: Any, step: Optional[int] = None, fps: int = 4) -> None:
        """
        记录视频

        Args:
            tag: 视频标签/名称
            video: 视频数据（numpy数组或列表）
            step: 当前步数（可选）
            fps: 视频帧率
        """
        pass

    @abstractmethod
    def log_histogram(
        self, tag: str, values: Any, step: Optional[int] = None, bins: Optional[int] = None
    ) -> None:
        """
        记录直方图

        Args:
            tag: 直方图标签/名称
            values: 数值数组或列表
            step: 当前步数（可选）
            bins: 直方图bin数量（可选）
        """
        pass

    @abstractmethod
    def set_property(self, key: str, value: Any) -> None:
        """
        设置实验属性（如标签、注释等）

        Args:
            key: 属性键
            value: 属性值
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """完成跟踪，清理资源"""
        pass

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """检查跟踪器是否已初始化"""
        pass


class ExperimentTracker:
    """
    实验跟踪器组合器

    可以同时使用多个跟踪器（如wandb和tensorboard），提供统一的接口。
    所有操作会自动分发到所有已注册的跟踪器。
    """

    def __init__(self, trackers: Optional[List[BaseTracker]] = None):
        """
        初始化实验跟踪器

        Args:
            trackers: 跟踪器列表，如果为None则创建空列表
        """
        self.trackers: List[BaseTracker] = trackers if trackers is not None else []

    def add_tracker(self, tracker: BaseTracker) -> None:
        """
        添加跟踪器

        Args:
            tracker: 要添加的跟踪器实例
        """
        if not isinstance(tracker, BaseTracker):
            raise TypeError(f"tracker must be an instance of BaseTracker, got {type(tracker)}")
        self.trackers.append(tracker)

    def init(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        初始化所有跟踪器

        Args:
            project: 项目名称
            name: 实验名称（可选）
            config: 实验配置字典（可选）
            **kwargs: 其他初始化参数（会传递给所有跟踪器）
        """
        for tracker in self.trackers:
            try:
                tracker.init(project=project, name=name, config=config, **kwargs)
            except Exception as e:
                print(f"Warning: Failed to initialize tracker {type(tracker).__name__}: {e}")

    def log(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """
        记录指标到所有跟踪器

        Args:
            metrics: 指标字典
            step: 当前步数（可选）
        """
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log(metrics=metrics, step=step)
                except Exception as e:
                    print(f"Warning: Failed to log to tracker {type(tracker).__name__}: {e}")

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录字典形式的指标到所有跟踪器

        Args:
            metrics: 指标字典（可嵌套）
            step: 当前步数（可选）
        """
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_dict(metrics=metrics, step=step)
                except Exception as e:
                    print(f"Warning: Failed to log_dict to tracker {type(tracker).__name__}: {e}")

    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        """
        记录图像到所有跟踪器

        Args:
            tag: 图像标签
            image: 图像数据
            step: 当前步数（可选）
        """
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_image(tag=tag, image=image, step=step)
                except Exception as e:
                    print(f"Warning: Failed to log_image to tracker {type(tracker).__name__}: {e}")

    def log_video(self, tag: str, video: Any, step: Optional[int] = None, fps: int = 4) -> None:
        """
        记录视频到所有跟踪器

        Args:
            tag: 视频标签
            video: 视频数据
            step: 当前步数（可选）
            fps: 视频帧率
        """
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_video(tag=tag, video=video, step=step, fps=fps)
                except Exception as e:
                    print(f"Warning: Failed to log_video to tracker {type(tracker).__name__}: {e}")

    def log_histogram(
        self, tag: str, values: Any, step: Optional[int] = None, bins: Optional[int] = None
    ) -> None:
        """
        记录直方图到所有跟踪器

        Args:
            tag: 直方图标签
            values: 数值数组
            step: 当前步数（可选）
            bins: bin数量（可选）
        """
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.log_histogram(tag=tag, values=values, step=step, bins=bins)
                except Exception as e:
                    print(
                        f"Warning: Failed to log_histogram to tracker {type(tracker).__name__}: {e}"
                    )

    def set_property(self, key: str, value: Any) -> None:
        """
        设置属性到所有跟踪器

        Args:
            key: 属性键
            value: 属性值
        """
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.set_property(key=key, value=value)
                except Exception as e:
                    print(
                        f"Warning: Failed to set_property on tracker {type(tracker).__name__}: {e}"
                    )

    def finish(self) -> None:
        """完成所有跟踪器，清理资源"""
        for tracker in self.trackers:
            if tracker.is_initialized:
                try:
                    tracker.finish()
                except Exception as e:
                    print(f"Warning: Failed to finish tracker {type(tracker).__name__}: {e}")

    @property
    def is_initialized(self) -> bool:
        """检查是否有已初始化的跟踪器"""
        return any(tracker.is_initialized for tracker in self.trackers)
