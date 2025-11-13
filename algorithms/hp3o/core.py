# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : core.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-06 00:00
@Update Date    :
@Description    : HP3O核心算法实现
HP3O算法的轨迹重放缓冲区和核心计算函数
"""
# ------------------------------------------------------------

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import uuid

import numpy as np
import torch


class Trajectory:
    """
    单个轨迹类
    
    存储一个完整的episode轨迹，包括观测、动作、奖励、价值等。
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        初始化轨迹
        
        Args:
            observation_shape: 观测空间形状
            action_shape: 动作空间形状
            device: 计算设备
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
        """
        self.id = uuid.uuid4()
        self.observations: Dict[int, np.ndarray] = {}
        self.actions: Dict[int, np.ndarray] = {}
        self.rewards: Dict[int, float] = {}
        self.dones: Dict[int, bool] = {}
        self.values: Dict[int, float] = {}
        self.log_probs: Dict[int, float] = {}
        self.advantages: Dict[int, float] = {}
        self.returns: Dict[int, float] = {}
        
        self.current_step = 0
        self.cumulative_reward = 0.0
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        添加一步数据到轨迹
        
        Args:
            obs: 观测
            action: 动作
            reward: 奖励
            done: 是否结束
            value: 价值函数预测
            log_prob: 动作对数概率
        """
        self.cumulative_reward += reward
        
        self.observations[self.current_step] = np.array(obs, dtype=np.float32)
        self.actions[self.current_step] = np.array(action, dtype=np.float32)
        self.rewards[self.current_step] = float(reward)
        self.dones[self.current_step] = bool(done)
        self.values[self.current_step] = float(value)
        self.log_probs[self.current_step] = float(log_prob)
        
        self.current_step += 1
    
    def compute_returns_and_advantage(
        self,
        best_trajectory: Optional[Trajectory] = None,
        states_array: Optional[np.ndarray] = None,
        use_best_value: bool = False,
        threshold: float = 1.0,
    ) -> None:
        """
        计算GAE优势和回报
        
        Args:
            best_trajectory: 最佳轨迹（用于价值函数增强）
            states_array: 最佳轨迹的状态数组
            use_best_value: 是否使用最佳轨迹的价值
            threshold: 状态距离阈值
        """
        steps = sorted(self.rewards.keys())
        horizon = len(steps)
        last_gae_lam = 0.0
        
        # 反向计算GAE
        for step in reversed(steps):
            next_non_terminal = 1.0 - (self.dones[step] if step < horizon - 1 else 0.0)
            if step == horizon - 1:
                next_values = 0.0
            else:
                next_values = self.values[step + 1]
            
            # 如果使用最佳轨迹的价值，尝试从最佳轨迹中查找
            current_value = self.values[step]
            if use_best_value and best_trajectory is not None and states_array is not None:
                if best_trajectory.id == self.id:
                    current_value = self.values[step]
                else:
                    current_obs = self.observations[step]
                    best_value, distance = self._search_state_in_trajectory(
                        current_obs, states_array, best_trajectory, return_distance=True
                    )
                    
                    if best_value is not None and self.values[step] < best_value and distance <= threshold:
                        current_value = best_value
            
            # 计算GAE
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - current_value
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # 计算回报
        for step in steps:
            self.returns[step] = self.advantages[step] + self.values[step]
    
    def _search_state_in_trajectory(
        self,
        current_state: np.ndarray,
        states_array: np.ndarray,
        trajectory: Trajectory,
        return_distance: bool = False,
    ) -> Tuple[Optional[float], float]:
        """
        在轨迹中搜索最接近的状态
        
        Args:
            current_state: 当前状态
            states_array: 状态数组
            trajectory: 目标轨迹
            return_distance: 是否返回距离
        
        Returns:
            (best_value, distance): 最佳价值和距离
        """
        if not isinstance(current_state, np.ndarray):
            current_state = np.array(current_state)
        
        if states_array.ndim > 1:
            distances = np.linalg.norm(states_array - current_state, axis=1)
        else:
            distances = np.abs(states_array - current_state)
        
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        
        # 获取对应的价值
        trajectory_steps = sorted(trajectory.values.keys())
        if min_idx < len(trajectory_steps):
            best_value = trajectory.values[trajectory_steps[min_idx]]
        else:
            best_value = trajectory.values[trajectory_steps[-1]]
        
        return (best_value, min_dist) if return_distance else (best_value, min_dist == 0)
    
    def get_data(self) -> Dict[str, np.ndarray]:
        """
        获取轨迹的所有数据
        
        Returns:
            包含所有轨迹数据的字典
        """
        num_steps = len(self.observations)
        steps = sorted(self.observations.keys())
        
        observations = np.array([self.observations[i] for i in steps])
        actions = np.array([self.actions[i] for i in steps])
        rewards = np.array([self.rewards[i] for i in steps])
        dones = np.array([self.dones[i] for i in steps])
        values = np.array([self.values[i] for i in steps])
        log_probs = np.array([self.log_probs[i] for i in steps])
        advantages = np.array([self.advantages.get(i, 0.0) for i in steps])
        returns = np.array([self.returns.get(i, 0.0) for i in steps])
        
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
            "advantages": advantages,
            "returns": returns,
            "cumulative_reward": self.cumulative_reward,
        }


class RolloutBufferSamples(NamedTuple):
    """轨迹缓冲区采样数据"""
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class TrajectoryBuffer:
    """
    轨迹缓冲区
    
    存储多个完整轨迹，支持轨迹采样和数据采样。
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        初始化轨迹缓冲区
        
        Args:
            buffer_size: 缓冲区大小（最多存储的轨迹数）
            observation_shape: 观测空间形状
            action_shape: 动作空间形状
            device: 计算设备
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
        """
        self.trajectories = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.current_trajectory: Optional[Trajectory] = None
        self.cache: List[uuid.UUID] = []  # 已计算GAE的轨迹ID
    
    def start_trajectory(self) -> None:
        """开始新的轨迹"""
        self.current_trajectory = Trajectory(
            self.observation_shape,
            self.action_shape,
            self.device,
            self.gamma,
            self.gae_lambda,
        )
    
    def add_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        添加一步数据
        
        Args:
            obs: 观测
            action: 动作
            reward: 奖励
            done: 是否结束
            value: 价值函数预测
            log_prob: 动作对数概率
        """
        if self.current_trajectory is None:
            self.start_trajectory()
        
        self.current_trajectory.add_step(obs, action, reward, done, value, log_prob)
        
        if done:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = None
            self.cache.clear()  # 清空缓存，因为新轨迹加入
    
    def best_trajectory(self) -> Tuple[int, Trajectory]:
        """
        获取奖励最高的轨迹
        
        Returns:
            (index, trajectory): 轨迹索引和轨迹对象
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories in buffer")
        
        max_cumulative_reward = -np.inf
        idx_of_max_reward = -1
        
        for idx, trajectory in enumerate(self.trajectories):
            if trajectory.cumulative_reward > max_cumulative_reward:
                max_cumulative_reward = trajectory.cumulative_reward
                idx_of_max_reward = idx
        
        return idx_of_max_reward, self.trajectories[idx_of_max_reward]
    
    def sample_trajectories(self, batch_size_trajectory: int) -> List[int]:
        """
        采样轨迹索引
        
        Args:
            batch_size_trajectory: 采样轨迹数量
        
        Returns:
            轨迹索引列表（总是包含最佳轨迹）
        """
        assert len(self.trajectories) >= batch_size_trajectory, \
            f"Not enough trajectories to sample: {len(self.trajectories)} < {batch_size_trajectory}"
        
        # 获取最佳轨迹索引
        idx_of_max_reward, _ = self.best_trajectory()
        
        # 准备其他轨迹索引
        indices_to_sample = list(range(len(self.trajectories)))
        indices_to_sample.remove(idx_of_max_reward)
        
        # 采样其他轨迹
        additional_samples_needed = batch_size_trajectory - 1
        if additional_samples_needed > 0:
            sampled_indices = np.random.choice(
                indices_to_sample,
                min(additional_samples_needed, len(indices_to_sample)),
                replace=False
            ).tolist()
        else:
            sampled_indices = []
        
        # 添加最佳轨迹
        sampled_indices.append(idx_of_max_reward)
        
        return sampled_indices
    
    def sample(
        self,
        sampled_trajectories_indices: List[int],
        buffer_size_sample: int,
        batch_size: Optional[int] = None,
        use_best_value: bool = False,
        threshold: float = 0.01,
    ):
        """
        从采样的轨迹中采样数据
        
        Args:
            sampled_trajectories_indices: 已采样的轨迹索引
            buffer_size_sample: 总采样数据量
            batch_size: 批次大小（如果为None，返回所有数据）
            use_best_value: 是否使用最佳轨迹的价值
            threshold: 状态距离阈值
        
        Yields:
            RolloutBufferSamples: 批次数据
        """
        num_sampled_trajectories = len(sampled_trajectories_indices)
        samples_per_trajectory = max(1, buffer_size_sample // num_sampled_trajectories)
        
        # 准备存储列表
        sampled_observations = []
        sampled_actions = []
        sampled_values = []
        sampled_log_probs = []
        sampled_advantages = []
        sampled_returns = []
        
        # 获取最佳轨迹（如果需要）
        if use_best_value:
            best_trajectory = self.trajectories[sampled_trajectories_indices[-1]]
            # 构建状态数组
            best_steps = sorted(best_trajectory.observations.keys())
            states_array = np.array([best_trajectory.observations[i] for i in best_steps])
        else:
            best_trajectory = None
            states_array = None
        
        # 从每个轨迹中采样数据
        for trajectory_index in sampled_trajectories_indices:
            trajectory = self.trajectories[trajectory_index]
            num_observations = len(trajectory.observations)
            num_samples = min(samples_per_trajectory, num_observations)
            
            # 采样索引
            if num_samples < num_observations:
                indices = np.random.choice(
                    list(trajectory.observations.keys()),
                    num_samples,
                    replace=False
                ).tolist()
            else:
                indices = sorted(trajectory.observations.keys())
            
            # 计算GAE（如果尚未计算）
            if trajectory.id not in self.cache:
                trajectory.compute_returns_and_advantage(
                    best_trajectory, states_array, use_best_value, threshold
                )
                self.cache.append(trajectory.id)
            
            # 添加采样数据
            for idx in indices:
                # 确保观测和动作都被展平为1D数组
                obs = trajectory.observations[idx]
                if isinstance(obs, np.ndarray):
                    # 展平多维数组
                    obs = obs.flatten()
                else:
                    # 转换为numpy数组并展平
                    obs = np.array(obs).flatten()
                
                action = trajectory.actions[idx]
                if isinstance(action, np.ndarray):
                    # 展平多维数组
                    action = action.flatten()
                else:
                    # 转换为numpy数组并展平
                    action = np.array(action).flatten()
                
                sampled_observations.append(obs)
                sampled_actions.append(action)
                sampled_values.append(trajectory.values[idx])
                sampled_log_probs.append(trajectory.log_probs[idx])
                sampled_advantages.append(trajectory.advantages[idx])
                sampled_returns.append(trajectory.returns[idx])
        
        # 转换为numpy数组，确保所有元素都有相同的形状
        # 首先找到最大维度
        if len(sampled_observations) > 0:
            max_obs_dim = max(obs.shape[0] if hasattr(obs, 'shape') else len(obs) for obs in sampled_observations)
            # 确保所有观测都有相同的维度（如果不同，进行填充或截断）
            normalized_obs = []
            for obs in sampled_observations:
                if isinstance(obs, np.ndarray):
                    obs_flat = obs.flatten()
                else:
                    obs_flat = np.array(obs).flatten()
                
                if len(obs_flat) < max_obs_dim:
                    # 填充零
                    obs_flat = np.pad(obs_flat, (0, max_obs_dim - len(obs_flat)), mode='constant', constant_values=0)
                elif len(obs_flat) > max_obs_dim:
                    # 截断
                    obs_flat = obs_flat[:max_obs_dim]
                normalized_obs.append(obs_flat)
            sampled_observations = np.array(normalized_obs)
        else:
            sampled_observations = np.array([])
        
        # 对动作做同样的处理
        if len(sampled_actions) > 0:
            max_action_dim = max(action.shape[0] if hasattr(action, 'shape') else len(action) for action in sampled_actions)
            normalized_actions = []
            for action in sampled_actions:
                if isinstance(action, np.ndarray):
                    action_flat = action.flatten()
                else:
                    action_flat = np.array(action).flatten()
                
                if len(action_flat) < max_action_dim:
                    # 填充零
                    action_flat = np.pad(action_flat, (0, max_action_dim - len(action_flat)), mode='constant', constant_values=0)
                elif len(action_flat) > max_action_dim:
                    # 截断
                    action_flat = action_flat[:max_action_dim]
                normalized_actions.append(action_flat)
            sampled_actions = np.array(normalized_actions)
        else:
            sampled_actions = np.array([])
        
        # 转换为torch tensor
        sampled_observations = torch.tensor(
            sampled_observations, device=self.device, dtype=torch.float32
        )
        sampled_actions = torch.tensor(
            sampled_actions, device=self.device, dtype=torch.float32
        )
        sampled_values = torch.tensor(
            np.array(sampled_values), device=self.device, dtype=torch.float32
        )
        sampled_log_probs = torch.tensor(
            np.array(sampled_log_probs), device=self.device, dtype=torch.float32
        )
        sampled_advantages = torch.tensor(
            np.array(sampled_advantages), device=self.device, dtype=torch.float32
        )
        sampled_returns = torch.tensor(
            np.array(sampled_returns), device=self.device, dtype=torch.float32
        )
        
        # 处理形状（确保是1D或2D）
        if sampled_observations.ndim > 2:
            sampled_observations = sampled_observations.reshape(sampled_observations.shape[0], -1)
        if sampled_actions.ndim > 2:
            sampled_actions = sampled_actions.reshape(sampled_actions.shape[0], -1)
        
        # 批量返回
        if batch_size is None:
            yield RolloutBufferSamples(
                sampled_observations,
                sampled_actions,
                sampled_values,
                sampled_log_probs,
                sampled_advantages,
                sampled_returns,
            )
        else:
            total_samples = sampled_observations.size(0)
            start_idx = 0
            while start_idx < total_samples:
                end_idx = min(start_idx + batch_size, total_samples)
                yield RolloutBufferSamples(
                    sampled_observations[start_idx:end_idx],
                    sampled_actions[start_idx:end_idx],
                    sampled_values[start_idx:end_idx],
                    sampled_log_probs[start_idx:end_idx],
                    sampled_advantages[start_idx:end_idx],
                    sampled_returns[start_idx:end_idx],
                )
                start_idx = end_idx
    
    def reset(self) -> None:
        """重置缓冲区"""
        self.trajectories.clear()
        self.current_trajectory = None
        self.cache.clear()

