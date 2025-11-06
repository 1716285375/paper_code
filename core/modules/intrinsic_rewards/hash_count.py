# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : hash_count.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HashCount 内在奖励（参考 smpe-main）
使用多个哈希桶和投影矩阵进行计数，生成内在奖励
"""
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


class HashCount:
    """
    Hash-based count bonus for exploration（参考 smpe-main）
    
    使用多个哈希桶和投影矩阵进行计数，生成内在奖励
    奖励公式：r_int = 1 / max(1, sqrt(count))
    """

    def __init__(
        self,
        obs_size: int,
        key_dim: int = 16,
        decay_factor: float = 1.0,
        bucket_sizes: Optional[list[int]] = None,
    ) -> None:
        """
        初始化 HashCount

        Args:
            obs_size: 观测维度
            key_dim: 哈希键维度
            decay_factor: 衰减因子
            bucket_sizes: 哈希桶大小列表（如果None则使用默认值）
        """
        self.obs_size = obs_size
        self.key_dim = key_dim
        self.decay_factor = decay_factor

        # 默认使用多个质数作为桶大小（参考 smpe-main）
        if bucket_sizes is None:
            self.bucket_sizes = [9931, 9953, 9959, 9961, 9979, 9983]
        else:
            self.bucket_sizes = bucket_sizes

        # 计算每个桶的模数列表
        mods_list = []
        for bucket_size in self.bucket_sizes:
            mod = 1
            mods = []
            for _ in range(self.key_dim):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        
        self.bucket_sizes = np.asarray(self.bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        
        # 初始化哈希表
        self.tables = np.zeros((len(self.bucket_sizes), np.max(self.bucket_sizes)))
        
        # 投影矩阵（固定随机生成，保证一致性）
        np.random.seed(42)  # 固定种子以保证可复现性
        self.projection_matrix = np.random.normal(size=(self.obs_size, self.key_dim))

    def compute_keys(self, obss: np.ndarray) -> np.ndarray:
        """
        计算哈希键

        Args:
            obss: 观测数组，形状为 (B, obs_size) 或 (obs_size,)

        Returns:
            哈希键数组，形状为 (B, len(bucket_sizes))
        """
        # 确保是2维数组
        if obss.ndim == 1:
            obss = obss.reshape(1, -1)
        
        # 计算二进制编码：sign(obss @ projection_matrix)
        binaries = np.sign(np.asarray(obss).dot(self.projection_matrix))
        
        # 计算哈希键：binaries @ mods_list % bucket_sizes
        keys = np.cast["int"](binaries.dot(self.mods_list)) % self.bucket_sizes
        
        return keys

    def inc_hash(self, obss: np.ndarray) -> None:
        """
        增加哈希计数

        Args:
            obss: 观测数组，形状为 (B, obs_size) 或 (obs_size,)
        """
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], self.decay_factor)

    def query_hash(self, obss: np.ndarray) -> np.ndarray:
        """
        查询哈希计数

        Args:
            obss: 观测数组，形状为 (B, obs_size) 或 (obs_size,)

        Returns:
            计数数组，形状为 (B,)，取所有桶的最小值
        """
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs: np.ndarray) -> None:
        """
        处理样本前先查询计数（用于训练时）

        Args:
            obs: 观测，形状为 (obs_size,) 或 (B, obs_size)
        """
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.query_hash(obss)
        self.inc_hash(obss)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        预测内在奖励

        Args:
            obs: 观测，形状为 (obs_size,) 或 (B, obs_size)

        Returns:
            内在奖励，形状为 (B,) 或标量
            奖励公式：1.0 / max(1.0, sqrt(count))
        """
        counts = self.query_hash(obs)
        prediction = 1.0 / np.maximum(1.0, np.sqrt(counts))
        return prediction

    def reset(self) -> None:
        """重置计数器"""
        self.tables = np.zeros((len(self.bucket_sizes), np.max(self.bucket_sizes)))

