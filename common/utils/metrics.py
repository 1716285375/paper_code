# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : metrics.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : 指标收集工具
提供训练指标的收集、聚合和统计功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class MetricsCollector:
    """
    指标收集器

    用于收集、存储和聚合训练过程中的各种指标。
    支持滑动平均、历史记录等功能。
    """

    def __init__(
        self,
        window_size: int = 100,
        keep_history: bool = True,
    ):
        """
        初始化指标收集器

        Args:
            window_size: 滑动窗口大小（用于计算移动平均）
            keep_history: 是否保留完整历史记录
        """
        self.window_size = window_size
        self.keep_history = keep_history

        # 指标存储：{metric_name: [values...]}
        self.metrics: Dict[str, List[float]] = defaultdict(list)

        # 步数记录
        self.steps: List[int] = []

    def update(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        更新指标

        Args:
            metrics: 指标字典
            step: 当前步数（可选）
        """
        if step is not None:
            self.steps.append(step)

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics[name].append(float(value))

                # 限制历史记录大小
                if self.keep_history:
                    # 保留最多 window_size * 10 条记录
                    max_history = self.window_size * 10
                    if len(self.metrics[name]) > max_history:
                        self.metrics[name] = self.metrics[name][-max_history:]
                else:
                    # 只保留最近 window_size 条记录
                    if len(self.metrics[name]) > self.window_size:
                        self.metrics[name] = self.metrics[name][-self.window_size :]

    def get(self, name: str, reduce: str = "mean") -> Optional[float]:
        """
        获取指标值

        Args:
            name: 指标名称
            reduce: 聚合方式（"mean", "sum", "last", "max", "min"）

        Returns:
            聚合后的指标值，如果不存在返回None
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None

        values = self.metrics[name]

        if reduce == "mean":
            return float(np.mean(values))
        elif reduce == "sum":
            return float(np.sum(values))
        elif reduce == "last":
            return values[-1]
        elif reduce == "max":
            return float(np.max(values))
        elif reduce == "min":
            return float(np.min(values))
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")

    def get_smoothed(self, name: str, window_size: Optional[int] = None) -> Optional[float]:
        """
        获取平滑后的指标值（滑动平均）

        Args:
            name: 指标名称
            window_size: 窗口大小，如果为None则使用默认值

        Returns:
            平滑后的指标值
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return None

        values = self.metrics[name]
        window = window_size or self.window_size
        window = min(window, len(values))

        return float(np.mean(values[-window:]))

    def get_history(self, name: str) -> List[float]:
        """
        获取指标的完整历史记录

        Args:
            name: 指标名称

        Returns:
            指标值列表
        """
        return self.metrics[name].copy()

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有指标的摘要统计

        Returns:
            指标摘要字典，格式为 {metric_name: {mean, std, min, max, last}}
        """
        summary = {}

        for name in self.metrics:
            values = self.metrics[name]
            if len(values) > 0:
                summary[name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last": values[-1],
                    "count": len(values),
                }

        return summary

    def reset(self) -> None:
        """重置所有指标"""
        self.metrics.clear()
        self.steps.clear()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于保存）"""
        return {
            "metrics": dict(self.metrics),
            "steps": self.steps.copy(),
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """从字典加载（用于恢复）"""
        self.metrics = defaultdict(list, {k: v.copy() for k, v in data["metrics"].items()})
        self.steps = data["steps"].copy()


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    计算数值列表的统计信息

    Args:
        values: 数值列表

    Returns:
        统计信息字典（mean, std, min, max, median）
    """
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    values_array = np.array(values)

    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "median": float(np.median(values_array)),
    }
