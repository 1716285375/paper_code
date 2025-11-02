# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : simhash_reward.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SimHash内在奖励
使用SimHash对潜在变量z和原观测o进行计数，生成内在奖励
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class SimHashIntrinsicReward:
    """
    SimHash内在奖励生成器

    对潜在变量z和原观测o分别做SimHash，使用计数器生成内在奖励。
    奖励公式：r_int = 1/√count，归一化后裁剪到[0, r_max]
    """

    def __init__(
        self,
        hash_bits: int = 512,
        bucket_size: int = 2**16,  # 2^16 = 65536个桶
        r_max: float = 0.2,
        normalize: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        初始化SimHash内在奖励生成器

        Args:
            hash_bits: SimHash的哈希位数
            bucket_size: 哈希桶数量（稀疏计数）
            r_max: 奖励最大值
            normalize: 是否归一化奖励
            device: 设备
        """
        self.hash_bits = hash_bits
        self.bucket_size = bucket_size
        self.r_max = r_max
        self.normalize = normalize
        self.device = torch.device(device)

        # 计数器：分别对z和o计数
        self.z_counters: Dict[int, int] = {}  # {hash_value: count}
        self.o_counters: Dict[int, int] = {}  # {hash_value: count}

        # 统计信息
        self.total_z_visits = 0
        self.total_o_visits = 0

    def _simhash(self, data: np.ndarray) -> int:
        """
        计算SimHash值

        Args:
            data: 输入数据（可以是向量或张量），会被展平

        Returns:
            SimHash值（整数）
        """
        # 展平并转换为numpy数组
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data_flat = data.flatten().astype(np.float32)

        # 确保数据至少有一个元素
        if len(data_flat) == 0:
            data_flat = np.array([0.0])

        # 生成随机投影矩阵（固定种子以保证一致性）
        # 使用哈希来生成伪随机向量，避免存储大量随机向量
        seed = int(abs(np.sum(data_flat) * 1000)) % (2**31)
        np.random.seed(seed)
        
        # 扩展数据到hash_bits维度（如果需要）
        if len(data_flat) < self.hash_bits:
            # 重复数据直到达到hash_bits长度
            repeat_times = (self.hash_bits // len(data_flat)) + 1
            data_extended = np.tile(data_flat, repeat_times)[:self.hash_bits]
        elif len(data_flat) > self.hash_bits:
            # 截断到hash_bits长度
            data_extended = data_flat[:self.hash_bits]
        else:
            data_extended = data_flat

        # 生成随机投影矩阵：每行是一个投影向量
        # 形状为 (hash_bits, data_dim)
        random_matrix = np.random.randn(self.hash_bits, len(data_extended))

        # 计算投影：对每个哈希位，计算数据与对应投影向量的内积
        # 结果是一个长度为hash_bits的向量
        projection = np.dot(random_matrix, data_extended)

        # 符号函数：正为1，负为0，转换为整数数组
        hash_bits = (projection > 0).astype(int)

        # 转换为整数哈希值：将二进制位转换为整数
        # 但为了效率，我们可以使用位运算或直接计算
        # 方法1：转换为二进制字符串再转整数（适用于较小的hash_bits）
        if self.hash_bits <= 64:
            hash_value = int("".join(map(str, hash_bits)), 2) % self.bucket_size
        else:
            # 方法2：对于较大的hash_bits，使用位运算累加
            hash_value = 0
            for i, bit in enumerate(hash_bits):
                hash_value = (hash_value * 2 + bit) % self.bucket_size

        return hash_value

    def _compute_intrinsic_reward(self, hash_value: int, counter: Dict[int, int], visit_key: str) -> float:
        """
        计算内在奖励

        Args:
            hash_value: SimHash值
            counter: 计数器字典
            visit_key: 访问统计键（'z'或'o'）

        Returns:
            内在奖励值
        """
        # 获取或初始化计数
        count = counter.get(hash_value, 0)
        counter[hash_value] = count + 1

        # 更新访问统计
        if visit_key == "z":
            self.total_z_visits += 1
        else:
            self.total_o_visits += 1

        # 计算奖励：r_int = 1/√count
        if count == 0:
            reward = 1.0  # 首次访问给予最高奖励
        else:
            reward = 1.0 / np.sqrt(count + 1)  # +1避免除零

        return float(reward)

    def compute(
        self,
        z: Optional[torch.Tensor] = None,
        obs: Optional[torch.Tensor] = None,
        combine_mode: str = "average",  # "average", "sum", "max"
    ) -> torch.Tensor:
        """
        计算内在奖励

        Args:
            z: 潜在变量，形状为 (B, z_dim) 或 (z_dim,)
            obs: 原观测，形状为 (B, obs_dim) 或 (obs_dim,)
            combine_mode: 组合模式（"average": 平均, "sum": 求和, "max": 最大值）

        Returns:
            内在奖励，形状为 (B,) 或标量
        """
        rewards_z = None
        rewards_o = None

        # 处理z的奖励
        if z is not None:
            # 处理维度
            if isinstance(z, torch.Tensor):
                z_batch = z.detach().cpu().numpy()
            else:
                z_batch = np.asarray(z)
            
            if z_batch.ndim == 1:
                z_batch = z_batch.reshape(1, -1)
            elif z_batch.ndim == 0:
                z_batch = z_batch.reshape(1, 1)

            batch_size = z_batch.shape[0]

            rewards_z_list = []
            for i in range(batch_size):
                z_i = z_batch[i]
                hash_value = self._simhash(z_i)
                reward = self._compute_intrinsic_reward(hash_value, self.z_counters, "z")
                rewards_z_list.append(reward)

            rewards_z = torch.tensor(rewards_z_list, device=self.device, dtype=torch.float32)

        # 处理obs的奖励
        if obs is not None:
            # 处理维度
            if isinstance(obs, torch.Tensor):
                obs_batch = obs.detach().cpu().numpy()
            else:
                obs_batch = np.asarray(obs)
            
            if obs_batch.ndim == 1:
                obs_batch = obs_batch.reshape(1, -1)
            elif obs_batch.ndim == 0:
                obs_batch = obs_batch.reshape(1, 1)

            batch_size = obs_batch.shape[0]

            rewards_o_list = []
            for i in range(batch_size):
                obs_i = obs_batch[i]
                hash_value = self._simhash(obs_i)
                reward = self._compute_intrinsic_reward(hash_value, self.o_counters, "o")
                rewards_o_list.append(reward)

            rewards_o = torch.tensor(rewards_o_list, device=self.device, dtype=torch.float32)

        # 组合奖励
        if rewards_z is not None and rewards_o is not None:
            if combine_mode == "average":
                combined = (rewards_z + rewards_o) / 2.0
            elif combine_mode == "sum":
                combined = rewards_z + rewards_o
            elif combine_mode == "max":
                combined = torch.maximum(rewards_z, rewards_o)
            else:
                combined = (rewards_z + rewards_o) / 2.0  # 默认平均
        elif rewards_z is not None:
            combined = rewards_z
        elif rewards_o is not None:
            combined = rewards_o
        else:
            # 两者都未提供，返回零奖励
            combined = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # 归一化和裁剪
        if self.normalize and combined.numel() > 1:
            # 归一化到[0, 1]
            combined_min = combined.min()
            combined_max = combined.max()
            if combined_max > combined_min:
                combined = (combined - combined_min) / (combined_max - combined_min)
            else:
                combined = torch.zeros_like(combined)

        # 裁剪到[0, r_max]
        combined = torch.clamp(combined, 0.0, self.r_max)

        return combined

    def reset(self) -> None:
        """重置计数器"""
        self.z_counters.clear()
        self.o_counters.clear()
        self.total_z_visits = 0
        self.total_o_visits = 0

    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            "total_z_visits": self.total_z_visits,
            "total_o_visits": self.total_o_visits,
            "unique_z_states": len(self.z_counters),
            "unique_o_states": len(self.o_counters),
            "avg_z_count": np.mean(list(self.z_counters.values())) if self.z_counters else 0.0,
            "avg_o_count": np.mean(list(self.o_counters.values())) if self.o_counters else 0.0,
        }

