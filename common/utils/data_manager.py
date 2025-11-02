# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : data_manager.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 训练数据管理器
统一管理训练过程中的指标数据和视频录制
"""
# ------------------------------------------------------------

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.utils.metrics import MetricsCollector
from common.video import VideoRecorder


class TrainingDataManager:
    """
    训练数据管理器

    统一管理训练过程中的：
    - 指标数据的收集和保存（JSON/CSV格式）
    - 环境视频的录制和保存
    - 数据文件的组织和管理
    """

    def __init__(
        self,
        output_dir: str = "training_data",
        save_format: str = "both",  # "json", "csv", "both"
        video_recorder: Optional[VideoRecorder] = None,
    ):
        """
        初始化训练数据管理器

        Args:
            output_dir: 输出目录
            save_format: 保存格式（"json", "csv", "both"）
            video_recorder: 视频录制器（可选）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_format = save_format
        self.video_recorder = video_recorder

        # 指标收集器
        self.metrics_collector = MetricsCollector(keep_history=True)

        # 数据存储
        self.all_metrics: List[Dict[str, float]] = []  # 所有指标记录
        self.steps: List[int] = []  # 对应的步数

        # 视频记录
        self.video_records: List[Dict[str, Any]] = []  # 视频记录元数据

    def update_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        更新指标

        Args:
            metrics: 指标字典
            step: 当前步数（可选）
        """
        # 更新指标收集器
        self.metrics_collector.update(metrics, step)

        # 保存完整记录
        self.all_metrics.append(metrics.copy())
        if step is not None:
            self.steps.append(step)
        else:
            # 如果没有提供step，使用历史记录的长度
            self.steps.append(len(self.steps))

    def save_metrics(
        self,
        filename: Optional[str] = None,
        format: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        保存指标数据到文件

        Args:
            filename: 文件名（不含扩展名），如果不指定则自动生成
            format: 保存格式（覆盖初始化时的设置）

        Returns:
            保存的文件路径字典 {"json": path, "csv": path}
        """
        if len(self.all_metrics) == 0:
            return {}

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_metrics_{timestamp}"

        format = format or self.save_format
        saved_paths = {}

        # 保存为JSON格式
        if format in ["json", "both"]:
            json_path = self.output_dir / f"{filename}.json"

            # 构建JSON数据
            json_data = {
                "metadata": {
                    "total_updates": len(self.all_metrics),
                    "metrics": list(self.all_metrics[0].keys()),
                },
                "data": [],
            }

            for i, metrics in enumerate(self.all_metrics):
                entry = {"step": self.steps[i] if i < len(self.steps) else i}
                entry.update(metrics)
                json_data["data"].append(entry)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            saved_paths["json"] = json_path

        # 保存为CSV格式
        if format in ["csv", "both"]:
            csv_path = self.output_dir / f"{filename}.csv"

            # 转换为DataFrame
            data_list = []
            for i, metrics in enumerate(self.all_metrics):
                entry = {"step": self.steps[i] if i < len(self.steps) else i}
                entry.update(metrics)
                data_list.append(entry)

            df = pd.DataFrame(data_list)
            df.to_csv(csv_path, index=False, encoding="utf-8")

            saved_paths["csv"] = csv_path

        return saved_paths

    def record_video(
        self,
        env: Any,
        agent: Any,
        episode_name: Optional[str] = None,
        max_steps: int = 1000,
        deterministic: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        录制一个episode的视频

        Args:
            env: 环境实例
            agent: Agent实例
            episode_name: Episode名称（用于文件命名）
            max_steps: 最大步数
            deterministic: 是否使用确定性策略
            metadata: 额外的元数据（如update_count等）

        Returns:
            保存的视频文件路径，如果录制失败则返回None
        """
        if self.video_recorder is None:
            return None

        try:
            if episode_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                episode_name = f"episode_{timestamp}"

            # 录制视频
            video_path = self.video_recorder.record_episode(
                env=env,
                agent=agent,
                max_steps=max_steps,
                filename=f"{episode_name}.mp4",
                deterministic=deterministic,
            )

            # 记录元数据
            self.video_records.append(
                {
                    "episode_name": episode_name,
                    "video_path": str(video_path),
                    "metadata": metadata or {},
                    "timestamp": datetime.now().isoformat(),
                }
            )

            return video_path
        except Exception as e:
            print(f"Warning: Failed to record video: {e}")
            return None

    def save_video_index(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        保存视频索引文件（记录所有录制的视频信息）

        Args:
            filename: 文件名

        Returns:
            保存的文件路径
        """
        if len(self.video_records) == 0:
            return None

        if filename is None:
            filename = "video_index.json"

        index_path = self.output_dir / filename

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.video_records, f, indent=2, ensure_ascii=False)

        return index_path

    def save_all(self, prefix: Optional[str] = None) -> Dict[str, Path]:
        """
        保存所有数据（指标和视频索引）

        Args:
            prefix: 文件名前缀

        Returns:
            保存的所有文件路径
        """
        saved_paths = {}

        # 保存指标
        metrics_paths = self.save_metrics(filename=prefix)
        saved_paths.update(metrics_paths)

        # 保存视频索引
        if len(self.video_records) > 0:
            index_path = self.save_video_index(
                filename=f"{prefix}_video_index.json" if prefix else None
            )
            if index_path:
                saved_paths["video_index"] = index_path

        return saved_paths

    def get_metrics_dataframe(self) -> Optional[pd.DataFrame]:
        """
        获取指标数据的DataFrame（用于直接绘图）

        Returns:
            pandas DataFrame，如果无数据则返回None
        """
        if len(self.all_metrics) == 0:
            return None

        data_list = []
        for i, metrics in enumerate(self.all_metrics):
            entry = {"step": self.steps[i] if i < len(self.steps) else i}
            entry.update(metrics)
            data_list.append(entry)

        return pd.DataFrame(data_list)

    def load_metrics(self, json_path: str) -> None:
        """
        从JSON文件加载指标数据

        Args:
            json_path: JSON文件路径
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.all_metrics = []
        self.steps = []

        for entry in data["data"]:
            step = entry.pop("step")
            self.steps.append(step)
            self.all_metrics.append(entry)

        # 更新指标收集器
        for metrics, step in zip(self.all_metrics, self.steps):
            self.metrics_collector.update(metrics, step)

    def reset(self) -> None:
        """重置所有数据"""
        self.metrics_collector.reset()
        self.all_metrics.clear()
        self.steps.clear()
        self.video_records.clear()
