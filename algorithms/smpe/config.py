# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : config.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE算法配置
SMPE算法的配置常量和默认值
"""
# ------------------------------------------------------------

class SMPEConfig:
    """SMPE算法配置常量"""
    
    # VAE更新配置
    DEFAULT_VAE_UPDATE_FREQ = 1024  # 每1024环境步更新VAE
    DEFAULT_VAE_EPOCHS = 3  # VAE训练轮数
    
    # Filter更新配置
    DEFAULT_FILTER_UPDATE_FREQ = 1  # Filter更新频率（每步）
    
    # 内在奖励配置
    DEFAULT_INTRINSIC_REWARD_BETA1 = 0.1  # SimHash内在奖励权重
    DEFAULT_INTRINSIC_REWARD_BETA2 = 0.05  # 自博弈奖励权重
    DEFAULT_INTRINSIC_WARMUP_STEPS = 20000  # 内在奖励warm-up步数
    
    # 对手池配置（使用 core 的 OpponentPool 默认值）
    DEFAULT_OPPONENT_POOL_SIZE = 15  # 对手池大小
    DEFAULT_OPPONENT_POOL_STRATEGY = "pfsp"  # 默认策略：PFSP
    DEFAULT_SNAPSHOT_FREQ = 50000  # 快照频率（每50000环境步）
    
    # 自博弈配置
    DEFAULT_SELF_PLAY_UPDATE_FREQ = 10  # 对手更新频率（每10个更新）

