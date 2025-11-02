# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : opponent_pool.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 对手池实现
支持PFSP（Policy-Space Response Oracles）、Elo匹配、Uniform采样
"""
# ------------------------------------------------------------

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class EloRating:
    """
    Elo评分系统
    用于评估和匹配对手策略
    """

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0) -> None:
        """
        初始化Elo评分

        Args:
            initial_rating: 初始评分
            k_factor: K因子（评分更新幅度）
        """
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[int, float] = {}  # {pool_index: rating}

    def get_rating(self, pool_index: int) -> float:
        """获取评分"""
        return self.ratings.get(pool_index, self.initial_rating)

    def update(self, pool_index_a: int, pool_index_b: int, score_a: float) -> None:
        """
        更新Elo评分

        Args:
            pool_index_a: 策略A在池中的索引
            pool_index_b: 策略B在池中的索引
            score_a: A相对于B的得分（1.0表示A胜，0.0表示A负，0.5表示平局）
        """
        rating_a = self.get_rating(pool_index_a)
        rating_b = self.get_rating(pool_index_b)

        # 期望得分
        expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
        expected_b = 1.0 - expected_a

        # 更新评分
        self.ratings[pool_index_a] = rating_a + self.k_factor * (score_a - expected_a)
        self.ratings[pool_index_b] = rating_b + self.k_factor * ((1.0 - score_a) - expected_b)


class OpponentPool:
    """
    对手策略池

    支持多种采样策略：
    - Uniform: 均匀采样
    - Elo: 按Elo评分分布采样（带温度）
    - PFSP: Policy-Space Response Oracles（按相对胜率采样）
    """

    def __init__(
        self,
        max_size: int = 15,
        strategy: str = "pfsp",  # "uniform", "elo", "pfsp"
        elo_temperature: float = 1.0,  # Elo采样温度
        pfsp_temperature: float = 1.0,  # PFSP采样温度
        device: str = "cpu",
    ) -> None:
        """
        初始化对手池

        Args:
            max_size: 池的最大大小
            strategy: 采样策略（"uniform", "elo", "pfsp"）
            elo_temperature: Elo采样温度（越大越均匀）
            pfsp_temperature: PFSP采样温度
            device: 设备
        """
        self.max_size = max_size
        self.strategy = strategy
        self.elo_temperature = elo_temperature
        self.pfsp_temperature = pfsp_temperature
        self.device = torch.device(device)

        # 策略池（存储state_dict）
        self.policies: deque = deque(maxlen=max_size)

        # Elo评分系统
        self.elo = EloRating()

        # 胜率统计（用于PFSP）：{pool_index: {opponent_index: (wins, total)}}
        self.win_rates: Dict[int, Dict[int, tuple[float, int]]] = {}

        # 策略索引到策略的映射
        self.policy_indices: Dict[int, Any] = {}

    def add_policy(self, policy_state: Dict[str, Any], method: str = "fifo") -> int:
        """
        添加策略到池中

        Args:
            policy_state: 策略状态字典（state_dict）
            method: 添加方法
                - "fifo": 先进先出（当池满时）
                - "kmeans": 使用K-Means保留代表（需要行为嵌入，暂不支持）

        Returns:
            策略在池中的索引
        """
        # 深拷贝状态
        policy_copy = copy.deepcopy(policy_state)

        # 添加到池
        if len(self.policies) >= self.max_size and method == "fifo":
            # FIFO：移除最旧的策略
            removed_index = 0
            # 更新索引映射
            new_indices = {}
            for i, (idx, policy) in enumerate(self.policy_indices.items()):
                if idx != removed_index:
                    new_indices[i] = policy
            self.policy_indices = new_indices
            self.policies.popleft()
            # 重新编号索引
            self.policy_indices = {i: self.policies[i] for i in range(len(self.policies))}
        else:
            self.policies.append(policy_copy)

        # 更新索引
        new_index = len(self.policies) - 1
        self.policy_indices[new_index] = policy_copy

        # 初始化Elo评分
        self.elo.ratings[new_index] = self.elo.initial_rating

        # 初始化胜率统计
        self.win_rates[new_index] = {}

        return new_index

    def sample_opponent(self) -> Optional[Dict[str, Any]]:
        """
        从池中采样对手策略

        Returns:
            策略状态字典，如果池为空则返回None
        """
        if len(self.policies) == 0:
            return None

        if self.strategy == "uniform":
            # 均匀采样
            index = np.random.randint(0, len(self.policies))
        elif self.strategy == "elo":
            # 按Elo评分分布采样（带温度）
            ratings = [self.elo.get_rating(i) for i in range(len(self.policies))]
            ratings_array = np.array(ratings)
            # Softmax采样（带温度）
            probs = np.exp(ratings_array / self.elo_temperature)
            probs = probs / probs.sum()
            index = np.random.choice(len(self.policies), p=probs)
        elif self.strategy == "pfsp":
            # PFSP：按相对胜率采样
            if len(self.policies) == 1:
                index = 0
            else:
                # 计算每个策略的平均胜率
                avg_win_rates = []
                for i in range(len(self.policies)):
                    if i in self.win_rates and len(self.win_rates[i]) > 0:
                        wins_total = [(w, t) for w, t in self.win_rates[i].values()]
                        total_wins = sum(w for w, _ in wins_total)
                        total_games = sum(t for _, t in wins_total)
                        avg_win_rate = total_wins / total_games if total_games > 0 else 0.5
                    else:
                        avg_win_rate = 0.5  # 默认胜率
                    avg_win_rates.append(avg_win_rate)

                # Softmax采样（温度越高越均匀）
                win_rates_array = np.array(avg_win_rates)
                probs = np.exp(win_rates_array / self.pfsp_temperature)
                probs = probs / probs.sum()
                index = np.random.choice(len(self.policies), p=probs)
        else:
            # 默认均匀采样
            index = np.random.randint(0, len(self.policies))

        return copy.deepcopy(self.policies[index])

    def update_win_rate(self, pool_index_a: int, pool_index_b: int, win_a: bool) -> None:
        """
        更新胜率统计（用于PFSP）

        Args:
            pool_index_a: 策略A的索引
            pool_index_b: 策略B的索引
            win_a: A是否获胜
        """
        if pool_index_a not in self.win_rates:
            self.win_rates[pool_index_a] = {}

        if pool_index_b not in self.win_rates[pool_index_a]:
            self.win_rates[pool_index_a][pool_index_b] = (0.0, 0)

        wins, total = self.win_rates[pool_index_a][pool_index_b]
        self.win_rates[pool_index_a][pool_index_b] = (wins + (1.0 if win_a else 0.0), total + 1)

    def update_elo(self, pool_index_a: int, pool_index_b: int, score_a: float) -> None:
        """
        更新Elo评分

        Args:
            pool_index_a: 策略A的索引
            pool_index_b: 策略B的索引
            score_a: A的得分（1.0=胜，0.0=负，0.5=平）
        """
        self.elo.update(pool_index_a, pool_index_b, score_a)

    def get_size(self) -> int:
        """获取池大小"""
        return len(self.policies)

    def clear(self) -> None:
        """清空池"""
        self.policies.clear()
        self.policy_indices.clear()
        self.elo.ratings.clear()
        self.win_rates.clear()

