# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : tensorboard_tracker.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-01-XX XX:XX
@Update Date    :
@Description    : TensorBoard 跟踪器实现
提供TensorBoard的实验跟踪和可视化功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from common.tracking.tracker import BaseTracker

# 可选导入tensorboard
# 注意：Python 3.12+ 移除了 distutils，可能导致 tensorboard 导入失败
TENSORBOARD_AVAILABLE = False
SummaryWriter = None

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except (ImportError, AttributeError) as e:
    # AttributeError 可能由 distutils 问题引起（Python 3.12+）
    try:
        from tensorboardX import SummaryWriter

        TENSORBOARD_AVAILABLE = True
    except (ImportError, AttributeError):
        # 如果两者都失败，tensorboard 不可用
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None


class TensorBoardTracker(BaseTracker):
    """
    TensorBoard 跟踪器

    使用TensorBoard进行实验跟踪和可视化。
    如果未安装tensorboard，所有操作将静默失败。
    """

    def __init__(self, log_dir: Optional[Union[str, Path]] = None):
        """
        初始化TensorBoard跟踪器

        Args:
            log_dir: TensorBoard日志目录（可选，默认使用项目名）
        """
        if not TENSORBOARD_AVAILABLE:
            error_msg = (
                "tensorboard is not available. "
                "This may be due to:\n"
                "1. tensorboard not installed: pip install tensorboard\n"
                "2. Python 3.12+ compatibility issue (distutils removed). "
                "Try: pip install tensorboard --upgrade\n"
                "Or use tensorboardX as alternative: pip install tensorboardX"
            )
            raise ImportError(error_msg)
        self._initialized = False
        self._writer: Optional[SummaryWriter] = None
        self._log_dir: Optional[Path] = None
        if log_dir is not None:
            self._log_dir = Path(log_dir)
        self._current_step = 0

    def init(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        初始化TensorBoard SummaryWriter

        Args:
            project: 项目名称（用作日志目录的一部分）
            name: 实验名称（可选，用作日志目录的一部分）
            config: 配置字典（可选，会写入到TensorBoard）
            **kwargs: 其他参数，如：
                - log_dir: 直接指定日志目录（覆盖默认）
        """
        if not TENSORBOARD_AVAILABLE:
            print("Warning: tensorboard is not available, skipping initialization")
            return

        # 确定日志目录
        if "log_dir" in kwargs:
            # 如果kwargs中指定了log_dir，使用它，并在其下创建project/name结构
            base_log_dir = Path(kwargs["log_dir"])
            log_dir = base_log_dir / project
            if name is not None:
                log_dir = log_dir / name
        elif self._log_dir is not None:
            # 如果初始化时指定了log_dir，在其下创建project/name结构
            base_log_dir = self._log_dir
            log_dir = base_log_dir / project
            if name is not None:
                log_dir = log_dir / name
        else:
            # 构建默认日志目录：runs/{project}/{name}
            log_dir = Path("runs") / project
            if name is not None:
                log_dir = log_dir / name

        log_dir.mkdir(parents=True, exist_ok=True)

        # 创建SummaryWriter
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._initialized = True

        # 写入配置（如果有）
        if config is not None:
            self._write_config(config)

    def log(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """
        记录指标到TensorBoard

        Args:
            metrics: 指标字典
            step: 当前步数（可选，如果不提供则自动递增）
        """
        if not self._initialized or not TENSORBOARD_AVAILABLE or self._writer is None:
            return

        if step is None:
            step = self._current_step
            self._current_step += 1

        for key, value in metrics.items():
            self._writer.add_scalar(key, float(value), step)

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录字典形式的指标到TensorBoard

        支持嵌套字典，使用斜杠分隔键（TensorBoard风格）

        Args:
            metrics: 指标字典（可嵌套）
            step: 当前步数（可选）
        """
        if not self._initialized or not TENSORBOARD_AVAILABLE or self._writer is None:
            return

        if step is None:
            step = self._current_step
            self._current_step += 1

        # 展平嵌套字典并记录
        flattened = self._flatten_dict(metrics)
        for key, value in flattened.items():
            if isinstance(value, (int, float, np.number)):
                self._writer.add_scalar(key, float(value), step)
            elif isinstance(value, (list, np.ndarray)):
                # 如果是数组，尝试记录为直方图
                try:
                    values = np.array(value).flatten()
                    if len(values) > 0:
                        self._writer.add_histogram(key, values, step)
                except:
                    pass

    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        """
        记录图像到TensorBoard

        Args:
            tag: 图像标签
            image: 图像数据（numpy数组，shape为[H, W, C]或[C, H, W]）
            step: 当前步数（可选）
        """
        if not self._initialized or not TENSORBOARD_AVAILABLE or self._writer is None:
            return

        if step is None:
            step = self._current_step
            self._current_step += 1

        # 确保是numpy数组
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # TensorBoard期望的格式：[C, H, W]或[H, W, C]
        # 自动调整格式
        if image.ndim == 3:
            # 如果是[H, W, C]格式，转换为[C, H, W]
            if image.shape[-1] in [1, 3, 4]:  # 假设最后一个维度是通道
                image = image.transpose(2, 0, 1)

        # 归一化到[0, 1]范围
        if image.max() > 1.0:
            image = image / 255.0

        self._writer.add_image(tag, image, step)

    def log_video(self, tag: str, video: Any, step: Optional[int] = None, fps: int = 4) -> None:
        """
        记录视频到TensorBoard

        Args:
            tag: 视频标签
            video: 视频数据（numpy数组，shape为[T, H, W, C]或[T, C, H, W]）
            step: 当前步数（可选）
            fps: 视频帧率（TensorBoard会使用）
        """
        if not self._initialized or not TENSORBOARD_AVAILABLE or self._writer is None:
            return

        if step is None:
            step = self._current_step
            self._current_step += 1

        # 确保是numpy数组
        if not isinstance(video, np.ndarray):
            video = np.array(video)

        # TensorBoard期望的格式：[T, C, H, W]
        if video.ndim == 4:
            # 如果是[T, H, W, C]格式，转换为[T, C, H, W]
            if video.shape[-1] in [1, 3, 4]:  # 假设最后一个维度是通道
                video = video.transpose(0, 3, 1, 2)

        # 归一化到[0, 1]范围
        if video.max() > 1.0:
            video = video / 255.0

        self._writer.add_video(tag, video[np.newaxis, ...], step, fps=fps)

    def log_histogram(
        self, tag: str, values: Any, step: Optional[int] = None, bins: Optional[int] = None
    ) -> None:
        """
        记录直方图到TensorBoard

        Args:
            tag: 直方图标签
            values: 数值数组或列表
            step: 当前步数（可选）
            bins: bin数量（TensorBoard会自动选择，此参数忽略）
        """
        if not self._initialized or not TENSORBOARD_AVAILABLE or self._writer is None:
            return

        if step is None:
            step = self._current_step
            self._current_step += 1

        # 转换为numpy数组
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        self._writer.add_histogram(tag, values.flatten(), step)

    def set_property(self, key: str, value: Any) -> None:
        """
        设置TensorBoard属性

        Args:
            key: 属性键（TensorBoard主要支持通过文本记录）
            value: 属性值
        """
        if not self._initialized or not TENSORBOARD_AVAILABLE or self._writer is None:
            return

        # TensorBoard可以通过add_text记录文本信息
        if isinstance(value, str):
            self._writer.add_text(key, value, self._current_step)
        else:
            self._writer.add_text(key, str(value), self._current_step)

    def finish(self) -> None:
        """关闭TensorBoard SummaryWriter"""
        if self._initialized and self._writer is not None:
            self._writer.close()
            self._initialized = False
            self._writer = None

    @property
    def is_initialized(self) -> bool:
        """检查TensorBoard是否已初始化"""
        return self._initialized and TENSORBOARD_AVAILABLE and self._writer is not None

    def _write_config(self, config: Dict[str, Any]) -> None:
        """
        将配置写入TensorBoard（作为文本）

        Args:
            config: 配置字典
        """
        if self._writer is None:
            return

        # 将配置格式化为文本
        config_text = "\n".join([f"{k}: {v}" for k, v in self._flatten_dict(config).items()])
        self._writer.add_text("config", config_text, 0)

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
        """
        展平嵌套字典，使用斜杠分隔键（TensorBoard风格）

        Args:
            d: 要展平的字典
            parent_key: 父键前缀
            sep: 分隔符

        Returns:
            展平后的字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(TensorBoardTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
