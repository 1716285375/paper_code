# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : checkpoint.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:33
@Update Date    :
@Description    : 检查点管理工具
提供模型保存和加载的便捷函数
"""
# ------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    检查点管理器

    用于管理和自动保存/加载训练检查点，支持：
    - 自动保存最佳模型
    - 定期保存检查点
    - 自动管理检查点目录
    - 保留最近N个检查点
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        max_to_keep: int = 5,
        save_best: bool = True,
        best_metric: str = "eval_mean_reward",
        mode: str = "max",
    ):
        """
        初始化检查点管理器

        Args:
            save_dir: 检查点保存目录
            max_to_keep: 最多保留的检查点数量
            save_best: 是否保存最佳模型
            best_metric: 用于判断最佳模型的指标名称
            mode: "max" 或 "min"，表示指标越大越好还是越小越好
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_to_keep = max_to_keep
        self.save_best = save_best
        self.best_metric = best_metric
        self.mode = mode

        self.best_value: Optional[float] = None
        self.checkpoint_history: list[Path] = []

    def save(
        self,
        state: Dict[str, Any],
        step: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        prefix: str = "checkpoint",
    ) -> Path:
        """
        保存检查点

        Args:
            state: 要保存的状态字典（通常包含模型、优化器等）
            step: 训练步数
            metrics: 当前指标（用于判断是否是最佳模型）
            is_best: 是否标记为最佳模型
            prefix: 文件前缀

        Returns:
            保存的文件路径
        """
        # 判断是否是最佳模型
        if self.save_best and metrics:
            current_value = metrics.get(self.best_metric)
            if current_value is not None:
                is_best = self._is_better(current_value, self.best_value)
                if is_best:
                    self.best_value = current_value

        # 保存检查点
        checkpoint_path = self.save_dir / f"{prefix}_step_{step}.pt"
        torch.save(
            {
                "step": step,
                "state": state,
                "metrics": metrics or {},
                "is_best": is_best,
            },
            checkpoint_path,
        )

        # 记录历史
        self.checkpoint_history.append(checkpoint_path)

        # 保存最佳模型
        if is_best:
            best_path = self.save_dir / f"{prefix}_best.pt"
            torch.save(
                {
                    "step": step,
                    "state": state,
                    "metrics": metrics or {},
                    "is_best": True,
                },
                best_path,
            )

        # 清理旧检查点
        self._cleanup()

        return checkpoint_path

    def load(self, path: Optional[str] = None, load_best: bool = False) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            path: 检查点路径，如果为None则加载最新的
            load_best: 是否加载最佳模型

        Returns:
            加载的状态字典
        """
        if load_best:
            path = self.save_dir / "checkpoint_best.pt"
        elif path is None:
            # 加载最新的
            if self.checkpoint_history:
                path = self.checkpoint_history[-1]
            else:
                raise ValueError("No checkpoint found")

        if isinstance(path, Path):
            path = str(path)

        checkpoint = torch.load(path, map_location="cpu")

        # 更新最佳值
        if checkpoint.get("is_best") and checkpoint.get("metrics"):
            current_value = checkpoint["metrics"].get(self.best_metric)
            if current_value is not None:
                self.best_value = current_value

        return checkpoint

    def _is_better(self, current: float, best: Optional[float]) -> bool:
        """判断当前值是否更好"""
        if best is None:
            return True

        if self.mode == "max":
            return current > best
        else:
            return current < best

    def _cleanup(self) -> None:
        """清理旧的检查点"""
        if len(self.checkpoint_history) <= self.max_to_keep:
            return

        # 移除最旧的检查点（但保留最佳模型）
        num_to_remove = len(self.checkpoint_history) - self.max_to_keep
        for i in range(num_to_remove):
            old_checkpoint = self.checkpoint_history[i]
            # 如果是最佳模型，跳过
            if old_checkpoint.name == "checkpoint_best.pt":
                continue
            if old_checkpoint.exists():
                old_checkpoint.unlink()

        # 更新历史记录
        self.checkpoint_history = self.checkpoint_history[-self.max_to_keep :]

    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点路径"""
        if self.checkpoint_history:
            return self.checkpoint_history[-1]
        return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """获取最佳模型路径"""
        best_path = self.save_dir / "checkpoint_best.pt"
        if best_path.exists():
            return best_path
        return None


def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    **kwargs,
) -> None:
    """
    便捷函数：保存检查点

    Args:
        state: 要保存的状态字典
        path: 保存路径
        **kwargs: 额外要保存的数据
    """
    checkpoint = {
        "state": state,
        **kwargs,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    便捷函数：加载检查点

    Args:
        path: 检查点路径
        map_location: 加载位置（"cpu"或"cuda"）

    Returns:
        加载的状态字典
    """
    return torch.load(path, map_location=map_location)
