# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : policy_agent.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE² Agent实现
结合VAE状态建模、Filter过滤、SimHash内在奖励的MAPPO Agent
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from core.agent.utils import ConfigurablePPOAgent
from core.modules.filters.role_filter import RoleFilter
from core.modules.intrinsic_rewards.simhash_reward import SimHashIntrinsicReward
from core.modules.vae.vae_controller import VAEController


class SMPEPolicyAgent(ConfigurablePPOAgent):
    """
    SMPE² Policy Agent

    在标准PPO Agent基础上，集成：
    - VAE状态建模（VAEController）
    - 角色/特征过滤（RoleFilter）
    - SimHash内在奖励（SimHashIntrinsicReward）

    用于MAPPO训练（集中训练-分散执行）
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
        agent_id: Optional[int] = None,
        agent_id_dim: int = 32,
        state_dim: Optional[int] = None,
        n_agents: Optional[int] = None,
    ) -> None:
        """
        初始化SMPE² Agent

        Args:
            obs_dim: 观测维度
            action_dim: 动作空间维度
            config: Agent配置字典，额外包含：
                - vae: VAE配置（可选）
                    - embedding_shape: 潜在变量维度（默认16，旧版用z_dim）
                    - use_actions: 是否预测动作（默认True）
                    - lambda_rec: 重构损失权重（默认1.0，旧版用rec_coef）
                    - lambda_kl_loss_obs: KL散度权重（默认1e-3，旧版用beta_kl）
                    - actions_loss_lambda: 动作损失权重（默认1.0，旧版用action_rec_coef）
                    - lr_agent_model: 学习率（默认1e-3）
                - filter: Filter配置（可选）
                    - num_filters: 过滤器数量（默认8）
                    - use_gumbel: 是否使用Gumbel（默认False）
                    - temperature: 温度（默认0.5）
                - intrinsic_reward: 内在奖励配置（可选）
                    - hash_bits: 哈希位数（默认512）
                    - bucket_size: 桶大小（默认2^16）
                    - r_max: 奖励最大值（默认0.2）
            device: 设备
            agent_id: Agent ID（用于生成agent_id one-hot）
            agent_id_dim: Agent ID嵌入维度
            state_dim: 全局状态维度（如果None，使用 obs_dim * n_agents 估算）
            n_agents: 智能体数量（如果None，从config中获取或使用默认值）
        """
        super().__init__(obs_dim, action_dim, config, device)

        self.agent_id = agent_id
        self.agent_id_dim = agent_id_dim

        # 获取智能体数量
        if n_agents is None:
            n_agents = config.get("n_agents", 2)  # 默认2（自博弈通常是2个团队）
        self.n_agents = n_agents

        # 获取全局状态维度（如果未提供，使用估算值）
        if state_dim is None:
            state_dim = config.get("state_dim", obs_dim * n_agents)  # 默认估算
        self.state_dim = state_dim

        # 生成agent_id one-hot（如果提供了agent_id）
        if agent_id is not None:
            self.agent_id_onehot = torch.zeros(agent_id_dim, device=self.device)
            self.agent_id_onehot[min(agent_id, agent_id_dim - 1)] = 1.0
        else:
            self.agent_id_onehot = torch.zeros(agent_id_dim, device=self.device)

        # VAE配置（新API）
        vae_config = config.get("vae", {})
        # 兼容旧配置：z_dim -> embedding_shape
        embedding_shape = vae_config.get("embedding_shape", vae_config.get("z_dim", 16))
        self.z_dim = embedding_shape  # 保持兼容性

        # 初始化VAE（如果启用）- 使用新API
        self.use_vae = config.get("use_vae", True)
        if self.use_vae:
            self.vae = VAEController(
                obs_dim=obs_dim,
                state_dim=state_dim,
                n_agents=n_agents,
                n_actions=action_dim,
                embedding_shape=embedding_shape,
                encoder_hidden_dims=vae_config.get("encoder_hidden_dims", None),
                decoder_hidden_dims=vae_config.get("decoder_hidden_dims", None),
                use_actions=vae_config.get("use_actions", True),
                lambda_rec=vae_config.get("lambda_rec", vae_config.get("rec_coef", 1.0)),
                lambda_kl_loss_obs=vae_config.get("lambda_kl_loss_obs", vae_config.get("beta_kl", 1e-3)),
                actions_loss_lambda=vae_config.get("actions_loss_lambda", vae_config.get("action_rec_coef", 1.0)),
                lr_agent_model=vae_config.get("lr_agent_model", 1e-3),
                device=device,
            )
        else:
            self.vae = None

        # Filter配置
        filter_config = config.get("filter", {})
        self.use_filter = config.get("use_filter", True)

        if self.use_filter:
            self.filter = RoleFilter(
                input_dim=self.z_dim if self.use_vae else obs_dim,
                num_filters=filter_config.get("num_filters", 8),
                use_gumbel=filter_config.get("use_gumbel", False),
                temperature=filter_config.get("temperature", 0.5),
                tau=filter_config.get("tau", 0.01),
                device=device,
            )
        else:
            self.filter = None

        # SimHash内在奖励配置
        intrinsic_config = config.get("intrinsic_reward", {})
        self.use_intrinsic = config.get("use_intrinsic", True)

        if self.use_intrinsic:
            self.intrinsic_reward = SimHashIntrinsicReward(
                hash_bits=intrinsic_config.get("hash_bits", 512),
                bucket_size=intrinsic_config.get("bucket_size", 2**16),
                r_max=intrinsic_config.get("r_max", 0.2),
                normalize=intrinsic_config.get("normalize", True),
                device=device,
            )
        else:
            self.intrinsic_reward = None

        # 存储上一次动作（用于VAE）
        self._last_action: Optional[int] = None

    def _forward(self, obs: torch.Tensor) -> Tuple[Distribution, torch.Tensor]:
        """
        前向传播（重写以集成VAE和Filter）

        Args:
            obs: 观测

        Returns:
            (dist, value): 策略分布和价值
        """
        # 1. 如果使用VAE，先编码到潜在空间（使用新API）
        if self.use_vae and self.vae is not None:
            # 确保obs是2维的
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            
            # 使用新API：forward(obs, agent_id, test_mode) -> z
            agent_id_int = self.agent_id if self.agent_id is not None else 0
            z = self.vae.forward(obs, agent_id_int, test_mode=False)

            # 2. 如果使用Filter，对潜在变量进行过滤
            if self.use_filter and self.filter is not None:
                z_filtered = self.filter(z)
                # 使用过滤后的z作为特征（简化：直接使用z作为输入到actor/critic）
                features = z_filtered
            else:
                features = z

            # 3. 使用features作为输入（需要扩展维度以匹配encoder的输入格式）
            # 这里简化处理：使用z作为额外特征，或者直接使用原始观测
            # 为了保持兼容性，我们仍然使用原始obs通过encoder，但可以考虑融合z
            obs_for_encoder = obs  # 可以改为 torch.cat([obs, z], dim=-1) 进行特征融合

        else:
            # 不使用VAE，直接使用原始观测
            obs_for_encoder = obs
            features = None

        # 调用父类的前向传播（使用encoder编码观测）
        dist, value = super()._forward(obs_for_encoder)

        # 如果使用VAE和Filter，可以考虑用filtered features来增强value估计
        # 这里先保持简单，直接使用原始的dist和value

        return dist, value

    def act(self, observation: Any, deterministic: bool = False) -> Any:
        """
        选择动作（重写以更新_last_action和计算内在奖励）

        Args:
            observation: 观测
            deterministic: 是否使用确定性策略

        Returns:
            (action, logprob, value): 动作、对数概率、价值
        """
        # 调用父类方法
        action, logprob, value = super().act(observation, deterministic=deterministic)

        # 更新_last_action
        self._last_action = action

        return action, logprob, value

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算内在奖励

        Args:
            obs: 观测
            z: 潜在变量（如果None且使用VAE，则从VAE获取）

        Returns:
            内在奖励
        """
        if not self.use_intrinsic or self.intrinsic_reward is None:
            return torch.tensor(0.0, device=self.device)

        # 如果z未提供但使用VAE，尝试从VAE获取（使用新API）
        z_tensor = None
        if z is None and self.use_vae and self.vae is not None:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # 使用新API：forward(obs, agent_id, test_mode) -> z
            agent_id_int = self.agent_id if self.agent_id is not None else 0
            z_tensor = self.vae.forward(obs_tensor, agent_id_int, test_mode=False)

        # 计算内在奖励
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        intrinsic_reward = self.intrinsic_reward.compute(z=z_tensor, obs=obs_tensor)

        return intrinsic_reward

    def update_vae(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        actions_onehot: torch.Tensor,
        actions_onehot_others: torch.Tensor,
        agent_id: Optional[int] = None,
        epochs: int = 1,
    ) -> Dict[str, float]:
        """
        更新VAE参数（使用新API）

        Args:
            obs: 观测，形状为 (B, obs_dim)
            states: 全局状态，形状为 (B, state_dim)
            actions_onehot: 当前agent的动作one-hot，形状为 (B, n_actions)
            actions_onehot_others: 其他agent的动作one-hot，形状为 (B, n_actions * n_agents)
            agent_id: 智能体ID（如果None，使用self.agent_id）
            epochs: 训练轮数

        Returns:
            平均损失字典
        """
        if not self.use_vae or self.vae is None:
            return {}

        # 确保 agent_id 是整数
        # 如果传入的 agent_id 是 None，使用 self.agent_id，但必须确保是整数
        if agent_id is not None:
            agent_id_int = int(agent_id) if isinstance(agent_id, (int, torch.Tensor, np.integer)) else 0
        else:
            if self.agent_id is not None:
                # 确保 self.agent_id 是整数
                if isinstance(self.agent_id, (int, torch.Tensor, np.integer)):
                    agent_id_int = int(self.agent_id)
                else:
                    agent_id_int = 0
            else:
                agent_id_int = 0
        
        return self.vae.update_agent_vae(
            obs=obs,
            states=states,
            actions_onehot=actions_onehot,
            actions_onehot_others=actions_onehot_others,
            agent_id=agent_id_int,
            epochs=epochs,
        )

    def update_filter(self, features: torch.Tensor) -> Dict[str, float]:
        """
        更新Filter参数

        Args:
            features: 输入特征（通常是潜在变量z）

        Returns:
            损失字典
        """
        if not self.use_filter or self.filter is None:
            return {}

        return self.filter.update(features)

    def reset(self) -> None:
        """重置Agent状态（包括内在奖励计数器）"""
        self._last_action = None
        if self.use_intrinsic and self.intrinsic_reward is not None:
            self.intrinsic_reward.reset()

    def get_vae_z(self, obs: torch.Tensor, last_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取VAE潜在变量z（用于特征提取，使用新API）

        Args:
            obs: 观测
            last_action: 上一次动作（新API中不再需要，保留以兼容旧代码）

        Returns:
            潜在变量z
        """
        if not self.use_vae or self.vae is None:
            if obs.dim() == 1:
                return torch.zeros(self.z_dim, device=self.device)
            else:
                return torch.zeros(obs.shape[0], self.z_dim, device=self.device)

        # 确保obs是2维的
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 使用新API：forward(obs, agent_id, test_mode) -> z
        agent_id_int = self.agent_id if self.agent_id is not None else 0
        z = self.vae.forward(obs, agent_id_int, test_mode=False)
        
        # 如果输入是1维的，返回1维的z
        if obs.shape[0] == 1 and z.shape[0] == 1:
            z = z.squeeze(0)

        return z

