# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : smpe.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE算法核心模块
SMPE (State Modeling and Predictive Exploration) 算法的核心实现
包含算法核心逻辑、工具函数、常量定义等
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


# SMPEConfig 已移动到 config.py
from algorithms.smpe.config import SMPEConfig


def compute_combined_reward(
    env_reward: np.ndarray,
    intrinsic_reward: Optional[torch.Tensor] = None,
    self_play_reward: Optional[np.ndarray] = None,
    beta1: float = 0.1,
    beta2: float = 0.05,
    warmup_factor: float = 1.0,
) -> np.ndarray:
    """
    计算SMPE组合奖励
    
    r_total = r_env + β1 * r_intrinsic + β2 * r_selfplay
    
    Args:
        env_reward: 环境奖励，形状为 (T,)
        intrinsic_reward: 内在奖励（SimHash），形状为 (T,) 或标量
        self_play_reward: 自博弈奖励，形状为 (T,)
        beta1: 内在奖励权重
        beta2: 自博弈奖励权重
        warmup_factor: Warm-up系数（0-1）
    
    Returns:
        组合奖励，形状为 (T,)
    """
    combined = env_reward.copy()
    
    # 应用warm-up
    beta1_warmed = beta1 * warmup_factor
    beta2_warmed = beta2 * warmup_factor
    
    # 添加内在奖励
    if intrinsic_reward is not None:
        if isinstance(intrinsic_reward, torch.Tensor):
            intrinsic_np = intrinsic_reward.detach().cpu().numpy()
        else:
            intrinsic_np = intrinsic_reward
        
        # 处理维度
        if intrinsic_np.ndim == 0:
            # 标量，扩展到所有步
            intrinsic_np = np.full(len(env_reward), intrinsic_np.item())
        elif len(intrinsic_np) != len(env_reward):
            # 长度不匹配，取前N个或填充
            if len(intrinsic_np) < len(env_reward):
                padding = np.zeros(len(env_reward) - len(intrinsic_np))
                intrinsic_np = np.concatenate([intrinsic_np, padding])
            else:
                intrinsic_np = intrinsic_np[:len(env_reward)]
        
        combined = combined + beta1_warmed * intrinsic_np
    
    # 添加自博弈奖励
    if self_play_reward is not None:
        if len(self_play_reward) != len(env_reward):
            if len(self_play_reward) < len(env_reward):
                padding = np.zeros(len(env_reward) - len(self_play_reward))
                self_play_reward = np.concatenate([self_play_reward, padding])
            else:
                self_play_reward = self_play_reward[:len(env_reward)]
        combined = combined + beta2_warmed * self_play_reward
    
    return combined


def compute_warmup_factor(total_steps: int, warmup_steps: int) -> float:
    """
    计算warm-up系数（线性增长）
    
    Args:
        total_steps: 总步数
        warmup_steps: Warm-up步数
    
    Returns:
        Warm-up系数（0-1）
    """
    return min(1.0, total_steps / warmup_steps) if warmup_steps > 0 else 1.0


def estimate_state_from_observations(
    observations: Dict[str, np.ndarray],
    state_dim: int,
) -> np.ndarray:
    """
    从多个智能体的观测估算全局状态
    
    Args:
        observations: 观测字典 {agent_id: obs_array}，每个obs_array形状为 (T, obs_dim) 或其他形状
        state_dim: 目标状态维度
    
    Returns:
        估算的全局状态，形状为 (T, state_dim)
    """
    if not observations:
        return np.zeros((0, state_dim))
    
    # 获取第一个观测并确定形状
    sample_obs = next(iter(observations.values()))
    sample_obs = np.asarray(sample_obs)
    
    # 处理不同维度的观测
    # 如果观测是多维的（如 (T, H, W, C)），需要展平
    if sample_obs.ndim > 2:
        # 多维观测，展平除第一维外的所有维度
        T = len(sample_obs)
        all_obs_list = []
        for agent_id in sorted(observations.keys()):
            obs = np.asarray(observations[agent_id])
            if len(obs) == T:
                # 展平：(T, ...) -> (T, -1)
                obs_flat = obs.reshape(T, -1)
                all_obs_list.append(obs_flat)
    else:
        # 2维观测，直接使用
        T = len(sample_obs)
        all_obs_list = []
        for agent_id in sorted(observations.keys()):
            obs = np.asarray(observations[agent_id])
            if len(obs) == T:
                # 确保是2维
                if obs.ndim == 1:
                    obs = obs.reshape(T, -1)
                all_obs_list.append(obs)
    
    if not all_obs_list:
        return np.zeros((T, state_dim))
    
    # 拼接所有观测（沿最后一个维度）
    if len(all_obs_list) == 1:
        states_array = all_obs_list[0]
    else:
        states_array = np.concatenate(all_obs_list, axis=-1)
    
    # 确保 states_array 是2维的
    if states_array.ndim > 2:
        T = len(states_array)
        states_array = states_array.reshape(T, -1)
    
    # 调整维度
    if states_array.shape[-1] > state_dim:
        states_array = states_array[:, :state_dim]
    elif states_array.shape[-1] < state_dim:
        # 创建与 states_array 相同维度的 padding
        T = len(states_array)
        padding_shape = (T, state_dim - states_array.shape[-1])
        padding = np.zeros(padding_shape, dtype=states_array.dtype)
        states_array = np.concatenate([states_array, padding], axis=-1)
    
    return states_array


def prepare_actions_onehot_others(
    actions: Dict[str, np.ndarray],
    current_agent_id: str,
    n_actions: int,
) -> np.ndarray:
    """
    准备其他智能体的动作one-hot编码
    
    Args:
        actions: 动作字典 {agent_id: action_array}，每个action_array形状为 (T,)
        current_agent_id: 当前智能体ID
        n_actions: 动作空间大小
    
    Returns:
        其他智能体的动作one-hot，形状为 (T, n_actions * (n_agents - 1))
    """
    if not actions:
        return np.zeros((0, 0))
    
    # 获取时间步数
    sample_action = next(iter(actions.values()))
    T = len(sample_action)
    
    # 收集其他智能体的动作
    other_actions_list = []
    for agent_id in sorted(actions.keys()):
        if agent_id != current_agent_id:
            action = actions[agent_id]
            if len(action) == T:
                other_actions_list.append(action)
    
    if not other_actions_list:
        # 如果没有其他智能体，返回零数组
        return np.zeros((T, 0))
    
    # 转换为one-hot并拼接
    import torch.nn.functional as F
    
    other_actions_tensor = torch.from_numpy(np.stack(other_actions_list, axis=1)).long()
    other_actions_onehot = F.one_hot(other_actions_tensor, num_classes=n_actions).float()
    
    # 重塑为 (T, n_actions * (n_agents - 1))
    other_actions_onehot = other_actions_onehot.view(T, -1)
    
    return other_actions_onehot.numpy()
