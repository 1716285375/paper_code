# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : wandb_tracker.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-01-XX XX:XX
@Update Date    :
@Description    : Weights & Biases (wandb) 跟踪器实现
提供wandb的实验跟踪和可视化功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np

from common.tracking.tracker import BaseTracker

# 可选导入wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandBTracker(BaseTracker):
    """
    Weights & Biases (wandb) 跟踪器

    使用wandb进行实验跟踪和可视化。
    如果未安装wandb，所有操作将静默失败。
    """

    def __init__(self):
        """初始化wandb跟踪器"""
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not installed. Install it with: pip install wandb")
        self._initialized = False
        self._run = None

    def init(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """
        初始化wandb运行

        Args:
            project: 项目名称
            name: 运行名称（可选）
            config: 配置字典（可选）
            **kwargs: 其他wandb.init参数，如：
                - entity: wandb实体/团队名
                - tags: 标签列表
                - notes: 实验说明
                - mode: "online", "offline", "disabled"
                - dir: 保存目录
        """
        if not WANDB_AVAILABLE:
            print("Warning: wandb is not available, skipping initialization")
            return

        # 准备初始化参数
        init_kwargs = {"project": project, **kwargs}

        if name is not None:
            init_kwargs["name"] = name

        if config is not None:
            init_kwargs["config"] = config

        # 初始化wandb运行
        self._run = wandb.init(**init_kwargs)
        self._initialized = True

    def log(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None) -> None:
        """
        记录指标到wandb

        Args:
            metrics: 指标字典
            step: 当前步数（可选）
        """
        if not self._initialized or not WANDB_AVAILABLE:
            return

        # 确保所有值都是数字类型（wandb要求）
        log_dict = {}
        for k, v in metrics.items():
            # 跳过以下划线开头的特殊键（如_step）
            if k.startswith("_"):
                continue
            # 转换为float或int
            try:
                if isinstance(v, (int, float)):
                    log_dict[k] = (
                        float(v)
                        if isinstance(v, float) or isinstance(v, (np.floating, np.integer))
                        else int(v)
                    )
                elif isinstance(v, np.number):
                    log_dict[k] = float(v)
                else:
                    # 尝试转换
                    log_dict[k] = float(v)
            except (ValueError, TypeError):
                # 如果无法转换，跳过这个指标
                continue

        # 确保有数据才记录
        if not log_dict:
            return

        # 记录到wandb（不要添加_step到log_dict，只通过step参数传递）
        wandb.log(log_dict, step=step, commit=True)

    def log_dict(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        记录字典形式的指标到wandb（支持嵌套，使用点号分隔）

        Args:
            metrics: 指标字典（可嵌套）
            step: 当前步数（可选）
        """
        if not self._initialized or not WANDB_AVAILABLE:
            return

        # wandb支持嵌套字典，自动转换为点号分隔的键
        flattened = self._flatten_dict(metrics)
        if step is not None:
            flattened["_step"] = step

        wandb.log(flattened, step=step)

    def log_image(self, tag: str, image: Any, step: Optional[int] = None) -> None:
        """
        记录图像到wandb

        Args:
            tag: 图像标签
            image: 图像数据（numpy数组、PIL图像等）
            step: 当前步数（可选）
        """
        if not self._initialized or not WANDB_AVAILABLE:
            return

        # 转换numpy数组为wandb.Image
        if isinstance(image, np.ndarray):
            image = wandb.Image(image)

        wandb.log({tag: image}, step=step)

    def log_video(self, tag: str, video: Any, step: Optional[int] = None, fps: int = 4) -> None:
        """
        记录视频到wandb

        Args:
            tag: 视频标签
            video: 视频数据（numpy数组或列表，shape为[T, H, W, C]或[T, C, H, W]）
            step: 当前步数（可选）
            fps: 视频帧率
        """
        if not self._initialized or not WANDB_AVAILABLE:
            return

        # 转换numpy数组为wandb.Video
        if isinstance(video, np.ndarray):
            video = wandb.Video(video, fps=fps, format="mp4")
        elif isinstance(video, list):
            # 如果是列表，转换为numpy数组
            video_array = np.array(video)
            video = wandb.Video(video_array, fps=fps, format="mp4")

        wandb.log({tag: video}, step=step)

    def log_histogram(
        self, tag: str, values: Any, step: Optional[int] = None, bins: Optional[int] = None
    ) -> None:
        """
        记录直方图到wandb

        Args:
            tag: 直方图标签
            values: 数值数组或列表
            step: 当前步数（可选）
            bins: bin数量（可选，wandb会自动选择）
        """
        if not self._initialized or not WANDB_AVAILABLE:
            return

        # 转换为numpy数组
        if not isinstance(values, np.ndarray):
            values = np.array(values)

        wandb.log({tag: wandb.Histogram(values, num_bins=bins)}, step=step)

    def set_property(self, key: str, value: Any) -> None:
        """
        设置wandb运行属性

        Args:
            key: 属性键，支持的键包括：
                - "tags": 设置标签列表
                - "notes": 设置实验说明
                - "name": 设置运行名称
            value: 属性值
        """
        if not self._initialized or not WANDB_AVAILABLE:
            return

        if key == "tags" and isinstance(value, list):
            wandb.run.tags = value
        elif key == "notes":
            wandb.run.notes = str(value)
        elif key == "name":
            wandb.run.name = str(value)
        else:
            # 其他属性可以通过config设置
            if hasattr(wandb.run, "config"):
                wandb.run.config[key] = value

    def finish(self) -> None:
        """完成wandb运行"""
        if self._initialized and WANDB_AVAILABLE and self._run is not None:
            wandb.finish()
            self._initialized = False
            self._run = None

    @property
    def is_initialized(self) -> bool:
        """检查wandb是否已初始化"""
        return self._initialized and WANDB_AVAILABLE

    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        展平嵌套字典，使用点号分隔键

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
                items.extend(WandBTracker._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
