# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : plotter.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 高级绘图工具
提供完整的训练过程可视化和结果分析功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# 尝试导入matplotlib和seaborn
try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None


class Plotter:
    """
    高级绘图工具类

    提供统一的接口来绘制各种训练相关的图表。
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8",
        figsize: tuple = (10, 6),
        dpi: int = 100,
    ):
        """
        初始化绘图器

        Args:
            style: matplotlib样式
            figsize: 默认图表大小
            dpi: 图表分辨率
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib is required for plotting")

        self.style = style
        self.figsize = figsize
        self.dpi = dpi

        # 设置样式
        if SEABORN_AVAILABLE:
            sns.set_style("darkgrid")
        else:
            plt.style.use(style if style else "default")

    def plot_training_metrics(
        self,
        metrics: Dict[str, List[Union[int, float]]],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        smooth: bool = False,
        window_size: int = 10,
    ) -> None:
        """
        绘制训练指标

        Args:
            metrics: 指标字典，格式为 {metric_name: [values...]}
            save_path: 保存路径
            show: 是否显示
            smooth: 是否使用滑动平均平滑
            window_size: 平滑窗口大小
        """
        num_metrics = len(metrics)
        if num_metrics == 0:
            return

        ncols = min(3, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows)
        )

        if num_metrics == 1:
            axes = [axes]
        elif nrows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, (name, values) in enumerate(metrics.items()):
            ax = axes[idx]

            # 处理数据
            values_array = np.array(values)
            if smooth and len(values_array) > window_size:
                # 滑动平均
                smoothed = np.convolve(
                    values_array, np.ones(window_size) / window_size, mode="valid"
                )
                ax.plot(smoothed, label=f"{name} (smoothed)", alpha=0.7)
                ax.plot(values_array, label=name, alpha=0.3)
            else:
                ax.plot(values_array, label=name)

            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # 隐藏多余的子图
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis("off")

        plt.suptitle("Training Metrics", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_evaluation_results(
        self,
        results: Dict[str, Union[List[float], Dict[str, float]]],
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        绘制评估结果

        Args:
            results: 评估结果，可以是列表（多次评估）或字典（单次评估的多个指标）
            save_path: 保存路径
            show: 是否显示
        """
        fig, axes = plt.subplots(1, 1, figsize=self.figsize)

        # 处理不同格式的结果
        if isinstance(list(results.values())[0], list):
            # 多次评估的时间序列
            for name, values in results.items():
                axes.plot(values, label=name, marker="o", markersize=4)
        else:
            # 单次评估的多个指标（柱状图）
            names = list(results.keys())
            values = [float(v) if isinstance(v, (int, float)) else 0.0 for v in results.values()]
            axes.bar(names, values)
            axes.set_xticklabels(names, rotation=45, ha="right")

        axes.set_title("Evaluation Results")
        axes.set_xlabel("Episode")
        axes.set_ylabel("Reward")
        axes.grid(True, alpha=0.3)
        axes.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_curves(
        self,
        train_rewards: List[float],
        eval_rewards: Optional[List[float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
    ) -> None:
        """
        绘制学习曲线

        Args:
            train_rewards: 训练奖励序列
            eval_rewards: 评估奖励序列（可选）
            save_path: 保存路径
            show: 是否显示
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        ax.plot(train_rewards, label="Train Reward", alpha=0.7)

        if eval_rewards:
            # 评估点可能更稀疏，需要调整x轴
            eval_x = np.linspace(0, len(train_rewards) - 1, len(eval_rewards))
            ax.plot(eval_x, eval_rewards, label="Eval Reward", marker="o", markersize=4)

        ax.set_title("Learning Curves")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


# 便捷函数
def plot_training_metrics(
    metrics: Dict[str, List[Union[int, float]]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> None:
    """便捷函数：绘制训练指标"""
    plotter = Plotter(**kwargs)
    plotter.plot_training_metrics(metrics, save_path=save_path, show=show)


def plot_evaluation_results(
    results: Dict[str, Union[List[float], Dict[str, float]]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> None:
    """便捷函数：绘制评估结果"""
    plotter = Plotter(**kwargs)
    plotter.plot_evaluation_results(results, save_path=save_path, show=show)


def plot_learning_curves(
    train_rewards: List[float],
    eval_rewards: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    **kwargs,
) -> None:
    """便捷函数：绘制学习曲线"""
    plotter = Plotter(**kwargs)
    plotter.plot_learning_curves(train_rewards, eval_rewards, save_path=save_path, show=show)


# ============ 便捷函数 ============
# 提供简单的函数接口用于绘图


def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Training Curves",
    figsize: tuple = (12, 6),
) -> None:
    """
    绘制训练曲线（向后兼容函数）

    这是 plot_training_metrics 的别名，用于保持向后兼容性。

    Args:
        metrics: 指标字典，格式为 {metric_name: [values...]}
        save_path: 保存路径（可选）
        show: 是否显示图表
        title: 图表标题
        figsize: 图表大小
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return

    plotter = Plotter(style="default", figsize=figsize)
    plotter.plot_training_metrics(metrics, save_path=save_path, show=show)


def plot_comparison(
    data_list: List[Dict[str, List[float]]],
    labels: List[str],
    metric_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    绘制多条曲线对比（向后兼容函数）

    Args:
        data_list: 多个数据字典列表
        labels: 对应的标签列表
        metric_name: 要对比的指标名称
        save_path: 保存路径（可选）
        show: 是否显示图表
        title: 图表标题
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return

    plt.figure(figsize=(10, 6))

    for data, label in zip(data_list, labels):
        if metric_name in data:
            plt.plot(data[metric_name], label=label)

    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.title(title or f"Comparison: {metric_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_distribution(
    values: List[float],
    bins: int = 30,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Distribution",
) -> None:
    """
    绘制数值分布直方图（向后兼容函数）

    Args:
        values: 数值列表
        bins: 直方图bins数量
        save_path: 保存路径（可选）
        show: 是否显示图表
        title: 图表标题
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
