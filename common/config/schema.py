# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : schema.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:33
@Update Date    :
@Description    : 配置模式定义
定义PPO算法的配置数据结构
"""
# ------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    PPO算法配置类

    包含PPO训练所需的所有超参数配置。
    """

    seed: int  # 随机种子，用于保证实验可复现性
    env_id: str  # 环境标识符（如 "magent2:battle_v4"）
    num_updates: int  # 训练更新次数
    clip_coef: float  # PPO裁剪系数，通常为0.2
    value_coef: float  # 价值损失系数，平衡策略损失和价值损失
    entropy_coef: float  # 熵正则化系数，鼓励探索
    lr: float  # 学习率
    batch_size: int  # 批次大小（每次更新的样本数）
    minibatch_size: int  # 小批次大小（PPO内部更新时的小批次）
    epochs: int  # PPO更新轮数（对同一批数据更新多次）
