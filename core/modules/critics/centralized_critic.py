# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : centralized_critic.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : 集中式Critic（Centralized Critic）
用于MAPPO、MATRPO、HAPPO、HATRPO等算法的集中训练-分散执行（CTDE）
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from core.networks import MLPEncoder


class CentralizedCritic(nn.Module):
    """
    集中式Critic网络
    
    输入：
    - 全局状态（state）：环境的全局状态表示
    - 对手动作（可选）：其他agent的动作（用于COMA等算法）
    
    输出：
    - 价值估计：全局状态的价值
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [128, 64],
        opponent_action_dim: Optional[int] = None,
        n_opponents: Optional[int] = None,
        use_opponent_actions: bool = False,
    ):
        """
        初始化集中式Critic
        
        Args:
            state_dim: 全局状态维度
            hidden_dims: 隐藏层维度列表
            opponent_action_dim: 对手动作维度（如果使用对手动作）
            n_opponents: 对手数量（如果使用对手动作）
            use_opponent_actions: 是否使用对手动作输入
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.use_opponent_actions = use_opponent_actions
        self.opponent_action_dim = opponent_action_dim
        self.n_opponents = n_opponents
        
        # 计算输入维度
        input_dim = state_dim
        if use_opponent_actions and opponent_action_dim is not None and n_opponents is not None:
            # 如果使用对手动作，将对手动作展平后拼接
            input_dim = state_dim + opponent_action_dim * n_opponents
        
        # 构建价值网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 输出层（价值估计）
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        state: torch.Tensor,
        opponent_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 全局状态，形状为 (B, state_dim)
            opponent_actions: 对手动作，形状为 (B, n_opponents, action_dim) 或 (B, n_opponents * action_dim)
                            如果为None且use_opponent_actions=True，则不使用对手动作
        
        Returns:
            价值估计，形状为 (B,)
        """
        # 处理全局状态
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (state_dim,) -> (1, state_dim)
        
        # 处理对手动作
        if self.use_opponent_actions and opponent_actions is not None:
            # 展平对手动作
            if opponent_actions.dim() == 3:  # (B, n_opponents, action_dim)
                opponent_actions = opponent_actions.view(opponent_actions.shape[0], -1)  # (B, n_opponents * action_dim)
            elif opponent_actions.dim() == 2 and opponent_actions.shape[1] == self.n_opponents * self.opponent_action_dim:
                # 已经是展平状态
                pass
            else:
                raise ValueError(
                    f"Invalid opponent_actions shape: {opponent_actions.shape}. "
                    f"Expected (B, n_opponents, action_dim) or (B, n_opponents * action_dim)"
                )
            
            # 拼接状态和对手动作
            x = torch.cat([state, opponent_actions], dim=-1)
        else:
            x = state
        
        # 通过价值网络
        value = self.net(x).squeeze(-1)  # (B, 1) -> (B,)
        
        return value


class CentralizedValueMixin:
    """
    Mixin类，用于为Agent添加集中式Critic功能
    
    使用方法：
        在Agent类中继承此类，并初始化central_critic
    """
    
    def __init__(self):
        """初始化集中式Critic Mixin"""
        # 初始化时设置默认值，避免AttributeError
        if not hasattr(self, 'central_critic'):
            self.central_critic: Optional[CentralizedCritic] = None
        if not hasattr(self, 'compute_central_vf'):
            self.compute_central_vf = None
    
    def set_central_critic(self, central_critic: CentralizedCritic):
        """
        设置集中式Critic网络
        
        Args:
            central_critic: 集中式Critic网络实例
        """
        self.central_critic = central_critic
        self.compute_central_vf = self._compute_central_value
    
    def _compute_central_value(
        self,
        state: torch.Tensor,
        opponent_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算集中式价值函数
        
        Args:
            state: 全局状态
            opponent_actions: 对手动作（可选）
        
        Returns:
            价值估计
        """
        if self.central_critic is None:
            raise ValueError("Central critic not initialized. Call set_central_critic() first.")
        
        return self.central_critic(state, opponent_actions)
    
    def central_value_function(
        self,
        state: torch.Tensor,
        opponent_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        集中式价值函数（公共接口）
        
        Args:
            state: 全局状态
            opponent_actions: 对手动作（可选）
        
        Returns:
            价值估计
        """
        return self._compute_central_value(state, opponent_actions)

