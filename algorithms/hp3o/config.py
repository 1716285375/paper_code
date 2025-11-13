# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : config.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-06 00:00
@Update Date    :
@Description    : HP3O算法配置
HP3O算法的配置常量和默认值
"""
# ------------------------------------------------------------

from typing import Optional


class HP3OConfig:
    """HP3O算法配置常量"""
    
    # HP3O核心参数（轨迹重放）
    DEFAULT_TRAJECTORY_BUFFER_SIZE = 10  # 轨迹缓冲区大小
    DEFAULT_TRAJECTORY_SAMPLE_SIZE = 3  # 每次采样轨迹数量
    DEFAULT_DATA_SAMPLE_SIZE = 256  # 从轨迹中采样数据量
    DEFAULT_THRESHOLD = 0.5  # 轨迹筛选阈值
    DEFAULT_USE_BEST_VALUE = False  # 是否使用最佳轨迹的价值函数
    
    # PPO核心参数（继承自PPO）
    DEFAULT_CLIP_COEF = 0.2  # PPO裁剪系数
    DEFAULT_VALUE_COEF = 0.5  # 价值损失权重
    DEFAULT_ENTROPY_COEF = 0.01  # 熵正则化权重
    
    # 训练参数
    DEFAULT_NUM_EPOCHS = 10  # PPO更新轮数
    DEFAULT_BATCH_SIZE = 64  # 批次大小
    DEFAULT_GAMMA = 0.99  # 折扣因子
    DEFAULT_GAE_LAMBDA = 0.95  # GAE lambda参数
    
    # 优化器参数
    DEFAULT_LR = 3e-4  # 学习率
    DEFAULT_MAX_GRAD_NORM = 0.5  # 梯度裁剪
    
    # 价值函数裁剪
    DEFAULT_VF_CLIP_PARAM = None  # 价值函数裁剪参数
    
    # 评估和保存
    DEFAULT_EVAL_FREQ = 50  # 评估频率（每N个更新）
    DEFAULT_SAVE_FREQ = 100  # 保存频率（每N个更新）
    DEFAULT_LOG_FREQ = 10  # 日志频率
    
    # 自博弈参数
    DEFAULT_SELF_PLAY_UPDATE_FREQ = 10  # 自博弈更新频率
    DEFAULT_POLICY_POOL_SIZE = 15  # 策略池大小

