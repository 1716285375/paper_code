# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : vae_controller.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : VAE状态建模控制器
参考 smpe-main 实现：为每个智能体配一个VAE，用于建模其他智能体的状态/动作
每个智能体的VAE从自己的观测推断其他智能体的状态信息
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from core.modules.vae.base import BaseVAE


def kl_distance(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    计算两个高斯分布之间的KL散度（参考 smpe-main）
    
    Args:
        mu1, sigma1: 第一个分布的均值和标准差
        mu2, sigma2: 第二个分布的均值和标准差
        
    Returns:
        KL散度（标量）
    """
    # Fully Factorized Gaussians
    numerator = (mu1 - mu2) ** 2 + (sigma1) ** 2
    denominator = 2 * (sigma2) ** 2 + 1e-8
    return torch.sum(numerator / denominator + torch.log(sigma2) - torch.log(sigma1) - 1 / 2)


class VariationalEncoder(nn.Module):
    """
    VAE编码器（参考 smpe-main）
    输入：观测 obs
    输出：潜在变量 z 的均值和方差
    """

    def __init__(
        self,
        input_shape: int,
        embedding_shape: int,
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """
        初始化VAE编码器

        Args:
            input_shape: 输入维度（观测维度）
            embedding_shape: 潜在变量维度
            hidden_dims: 隐藏层维度列表（可选，默认使用 smpe-main 的两层结构）
        """
        super().__init__()
        self.input_shape = input_shape
        self.embedding_shape = embedding_shape

        # 参考 smpe-main 的两层结构
        if hidden_dims is None:
            hidden_dims = [embedding_shape, embedding_shape]
        
        # 构建编码网络
        self.fc1 = nn.Linear(self.input_shape, hidden_dims[0])
        if len(hidden_dims) > 1:
            self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.use_fc2 = True
        else:
            self.use_fc2 = False
        
        # 输出层：分别输出均值和方差的对数
        final_dim = hidden_dims[-1] if len(hidden_dims) > 0 else hidden_dims[0]
        self.mu = nn.Linear(final_dim, self.embedding_shape)
        self.logvar = nn.Linear(final_dim, self.embedding_shape)

        # 用于重参数化的标准正态分布（延迟初始化，在forward中根据设备创建）
        self.N = None
        self.kl = torch.tensor(0.0)  # 存储KL散度

    def forward(self, x: torch.Tensor, test_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码输入到潜在空间（参考 smpe-main）

        Args:
            x: 观测，形状为 (B, input_shape)
            test_mode: 是否使用测试模式（使用均值而非采样）

        Returns:
            (z, mu, sigma): 潜在变量、均值和标准差
        """
        # 处理维度：确保输入是2维的
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (input_shape,) -> (1, input_shape)
        elif x.dim() > 2:
            # 多维观测，展平
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            if x.shape[-1] != self.input_shape:
                if x.shape[-1] > self.input_shape:
                    x = x[:, :self.input_shape]
                else:
                    padding = torch.zeros(
                        batch_size, self.input_shape - x.shape[-1], 
                        device=x.device, dtype=x.dtype
                    )
                    x = torch.cat([x, padding], dim=-1)

        # 编码网络
        x = F.relu(self.fc1(x))
        if self.use_fc2:
            x = F.relu(self.fc2(x))
        
        # 输出均值和方差
        mu = self.mu(x)
        sigma = torch.exp(0.5 * self.logvar(x))
        
        # 重参数化技巧
        if test_mode:
            z = mu
        else:
            # 创建或移动 N 到正确的设备
            if self.N is None or self.N.loc.device != x.device:
                self.N = Normal(0, 1)
                self.N.loc = self.N.loc.to(x.device)
                self.N.scale = self.N.scale.to(x.device)
            z = mu + sigma * self.N.sample(mu.shape)
        
        # 计算KL散度（相对于标准正态分布）
        self.kl = kl_distance(mu, sigma, torch.zeros_like(mu), torch.ones_like(sigma))
        
        return z, mu, sigma


class Decoder(nn.Module):
    """
    VAE解码器（参考 smpe-main）
    输入：潜在变量 z
    输出：其他智能体的状态信息（state - obs）
    """

    def __init__(
        self,
        embedding_shape: int,
        output_shape: int,
        hidden_dims: Optional[List[int]] = None,
        use_actions: bool = False,
        n_actions: int = 0,
        n_agents: int = 0,
    ) -> None:
        """
        初始化VAE解码器

        Args:
            embedding_shape: 潜在变量维度
            output_shape: 输出维度（其他agent的状态维度，通常是 state_dim - obs_dim）
            hidden_dims: 隐藏层维度列表（可选）
            use_actions: 是否预测动作
            n_actions: 动作空间大小
            n_agents: 智能体数量
        """
        super().__init__()
        self.embedding_shape = embedding_shape
        self.output_shape = output_shape
        self.use_actions = use_actions
        self.n_actions = n_actions
        self.n_agents = n_agents

        # 参考 smpe-main 的两层结构
        if hidden_dims is None:
            hidden_dims = [embedding_shape, embedding_shape]

        # 构建解码网络
        self.fc1 = nn.Linear(self.embedding_shape, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0])
        self.fc3 = nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], self.output_shape)

        # 如果需要预测动作
        if self.use_actions and self.n_actions > 0 and self.n_agents > 0:
            self.fc4 = nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], 
                                 self.n_actions * (self.n_agents - 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码潜在变量（输出其他agent的状态）

        Args:
            z: 潜在变量，形状为 (B, embedding_shape)

        Returns:
            predicted_states: 预测的其他agent的状态，形状为 (B, output_shape)
        """
        # 处理维度
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # 解码网络
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

    def forward_actions(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码潜在变量（输出其他agent的动作）

        Args:
            z: 潜在变量，形状为 (B, embedding_shape)

        Returns:
            predicted_actions: 预测的其他agent的动作，形状为 (B, n_actions * (n_agents - 1))
        """
        if not self.use_actions:
            raise ValueError("use_actions must be True to use forward_actions")
        
        # 处理维度
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # 解码网络
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        out = self.fc4(x)

        return out


class VAE(nn.Module):
    """
    单个VAE模型（参考 smpe-main）
    组合编码器和解码器
    """
    
    def __init__(
        self,
        input_shape: int,
        embedding_shape: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        use_actions: bool = False,
        n_actions: int = 0,
        n_agents: int = 0,
    ) -> None:
        """
        初始化VAE
        
        Args:
            input_shape: 输入维度（观测维度）
            embedding_shape: 潜在变量维度
            output_dim: 输出维度（其他agent的状态维度，state_dim - obs_dim）
            hidden_dims: 隐藏层维度列表
            use_actions: 是否预测动作
            n_actions: 动作空间大小
            n_agents: 智能体数量
        """
        super().__init__()
        self.encoder = VariationalEncoder(input_shape, embedding_shape, hidden_dims)
        self.decoder = Decoder(
            embedding_shape, output_dim, hidden_dims, 
            use_actions, n_actions, n_agents
        )
    
    def forward(self, x: torch.Tensor, test_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入观测
            test_mode: 是否使用测试模式
            
        Returns:
            (decoded, z, mu, sigma): 解码输出、潜在变量、均值、标准差
        """
        z, mu, sigma = self.encoder(x, test_mode)
        decoded = self.decoder(z)
        return decoded, z, mu, sigma


class VAEController(BaseVAE):
    """
    VAE控制器（参考 smpe-main）
    为每个智能体配一个VAE，用于建模其他智能体的状态/动作
    每个智能体的VAE从自己的观测推断其他智能体的状态信息
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        embedding_shape: int = 16,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        use_actions: bool = True,
        lambda_rec: float = 1.0,
        lambda_kl_loss_obs: float = 1e-3,
        actions_loss_lambda: float = 1.0,
        lr_agent_model: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        """
        初始化VAE控制器

        Args:
            obs_dim: 观测维度
            state_dim: 全局状态维度
            n_agents: 智能体数量
            n_actions: 动作空间大小
            embedding_shape: 潜在变量维度
            encoder_hidden_dims: 编码器隐藏层维度列表
            decoder_hidden_dims: 解码器隐藏层维度列表
            use_actions: 是否预测动作
            lambda_rec: 重构损失权重
            lambda_kl_loss_obs: KL散度损失权重
            actions_loss_lambda: 动作损失权重
            lr_agent_model: 学习率
            device: 设备
        """
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.embedding_shape = embedding_shape
        self.use_actions = use_actions
        self.lambda_rec = lambda_rec
        self.lambda_kl_loss_obs = lambda_kl_loss_obs
        self.actions_loss_lambda = actions_loss_lambda

        # 输出维度：其他agent的状态维度（state_dim - obs_dim）
        # 假设每个agent的状态维度是 state_dim // n_agents
        # 其他agent的状态维度 = (n_agents - 1) * (state_dim // n_agents)
        self.state_dim_per_agent = state_dim // n_agents if n_agents > 0 else state_dim
        self.output_dim = (n_agents - 1) * self.state_dim_per_agent if n_agents > 1 else state_dim - obs_dim

        # 为每个智能体创建独立的VAE
        self.agent_models = nn.ModuleList([
            VAE(
                input_shape=obs_dim,
                embedding_shape=embedding_shape,
                output_dim=self.output_dim,
                hidden_dims=encoder_hidden_dims,
                use_actions=use_actions,
                n_actions=n_actions,
                n_agents=n_agents,
            ).to(self.device)
            for _ in range(n_agents)
        ])

        # 为每个智能体创建独立的优化器
        self.agent_params = [list(model.parameters()) for model in self.agent_models]
        self.agent_optimizers = [
            torch.optim.RMSprop(params=param, lr=lr_agent_model)
            for param in self.agent_params
        ]

    def forward(self, obs: torch.Tensor, agent_id: int, test_mode: bool = False) -> torch.Tensor:
        """
        前向传播（参考 smpe-main）
        
        Args:
            obs: 观测，形状为 (B, obs_dim) 或 (obs_dim,)
            agent_id: 智能体ID（整数索引）
            test_mode: 是否使用测试模式
            
        Returns:
            z: 潜在变量，形状为 (B, embedding_shape)
        """
        # 确保 agent_id 是整数（用于索引）
        if isinstance(agent_id, torch.Tensor):
            agent_id = int(agent_id.item())
        elif not isinstance(agent_id, int):
            agent_id = int(agent_id) if agent_id is not None else 0
        
        # 确保 agent_id 在有效范围内
        agent_id = max(0, min(agent_id, len(self.agent_models) - 1))
        
        # 确保 obs 是2维的
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # 使用对应agent的VAE进行编码
        z, _, _ = self.agent_models[agent_id].encoder.forward(obs, test_mode)
        return z

    def compute_loss(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        actions_onehot: torch.Tensor,
        actions_onehot_others: torch.Tensor,
        agent_id: int,
    ) -> Dict[str, torch.Tensor]:
        """
        计算VAE损失（参考 smpe-main）

        Args:
            obs: 当前观测，形状为 (B, obs_dim)
            states: 全局状态，形状为 (B, state_dim)
            actions_onehot: 当前agent的动作one-hot，形状为 (B, n_actions)
            actions_onehot_others: 其他agent的动作one-hot，形状为 (B, n_actions * (n_agents - 1))
            agent_id: 智能体ID（整数索引）

        Returns:
            损失字典，包含：
                - total_loss: 总损失
                - recon_loss: 重构损失
                - action_loss: 动作损失（如果使用）
                - kl_loss: KL散度损失
        """
        # 确保 agent_id 是整数（用于索引）
        # 先转换为 Python int，避免 numpy int64 或 torch.Tensor 类型问题
        if isinstance(agent_id, torch.Tensor):
            # 如果是 tensor，先提取标量再转换
            if agent_id.numel() == 1:
                agent_idx = int(agent_id.item())
            else:
                agent_idx = 0
        elif isinstance(agent_id, (np.integer, np.floating)):
            agent_idx = int(agent_id)
        elif isinstance(agent_id, int):
            agent_idx = agent_id
        else:
            try:
                agent_idx = int(agent_id) if agent_id is not None else 0
            except (ValueError, TypeError):
                agent_idx = 0
        
        # 确保 agent_idx 在有效范围内
        agent_idx = max(0, min(agent_idx, len(self.agent_models) - 1))
        
        # 最终类型检查：确保是 Python int（不是 numpy.int64 等）
        # 使用 type() 检查并强制转换
        if type(agent_idx).__name__ != 'int':
            agent_idx = int(agent_idx)
        # 再次确保是纯 Python int
        agent_idx = int(agent_idx)
        
        # 验证：确保 agent_idx 是纯 Python int
        assert isinstance(agent_idx, int) and not isinstance(agent_idx, (np.integer, np.floating)), \
            f"agent_idx must be Python int, got {type(agent_idx)}"
        
        # 获取模型（使用明确的整数索引）
        # 使用 list() 转换为列表再索引，避免 ModuleList 的索引问题
        agent_model = list(self.agent_models)[agent_idx]
        
        # 编码
        z_others, _, _ = agent_model.encoder.forward(obs)
        
        # 解码：预测其他agent的状态
        predicted_states = agent_model.decoder.forward(z_others)
        
        # 处理状态维度匹配问题
        # 如果state是全局状态（如Magent2的2000维），不能简单地按agent分割
        # 需要根据predicted_states的实际维度来调整states_others
        
        pred_states_shape = predicted_states.shape[-1]
        states_shape = states.shape[-1]
        
        # 检查state是否是全局状态（不能整除n_agents，或者维度不匹配）
        if states_shape % self.n_agents != 0 or (states_shape // self.n_agents) * (self.n_agents - 1) != pred_states_shape:
            # 全局状态情况：直接使用整个state，或者使用预测的维度
            # 如果predicted_states的维度小于states，使用predicted_states的维度
            if pred_states_shape < states_shape:
                # 截取states的前pred_states_shape维
                states_others = states[:, :pred_states_shape]
            elif pred_states_shape == states_shape:
                # 维度相同，直接使用
                states_others = states
            else:
                # predicted_states维度更大，需要填充states
                padding = torch.zeros(
                    states.shape[0],
                    pred_states_shape - states_shape,
                    device=states.device,
                    dtype=states.dtype
                )
                states_others = torch.cat([states, padding], dim=-1)
        else:
            # 可以按agent分割的情况（传统方式）
            cut = self.state_dim_per_agent
            agent_idx_int = int(agent_idx)  # 再次确保是 int
            states_others = torch.cat([
                states[:, 0:agent_idx_int * cut],
                states[:, (agent_idx_int * cut + cut):]
            ], dim=-1)
            
            # 确保维度匹配
            if states_others.shape[-1] != pred_states_shape:
                min_shape = min(pred_states_shape, states_others.shape[-1])
                predicted_states = predicted_states[:, :min_shape]
                states_others = states_others[:, :min_shape]
        
        # 最终确保维度完全匹配
        if predicted_states.shape[-1] != states_others.shape[-1]:
            min_shape = min(predicted_states.shape[-1], states_others.shape[-1])
            predicted_states = predicted_states[:, :min_shape]
            states_others = states_others[:, :min_shape]
        
        # 重构损失（MSE）
        reconstruction_loss = ((predicted_states - states_others) ** 2).sum()
        reconstruction_loss = self.lambda_rec * reconstruction_loss
        
        # KL散度损失
        # 确保 encoder.kl 是 tensor，且 lambda_kl_loss_obs 是标量
        kl_value = agent_model.encoder.kl
        if not isinstance(kl_value, torch.Tensor):
            kl_value = torch.tensor(float(kl_value), device=self.device)
        
        # 确保 lambda_kl_loss_obs 是数值类型
        lambda_kl = float(self.lambda_kl_loss_obs) if isinstance(self.lambda_kl_loss_obs, str) else self.lambda_kl_loss_obs
        kl_loss = lambda_kl * kl_value
        
        # 动作损失（如果使用）
        action_loss = torch.tensor(0.0, device=self.device)
        if self.use_actions:
            predicted_actions = agent_model.decoder.forward_actions(z_others)
            
            # 检查 predicted_actions 的形状
            pred_actions_shape = predicted_actions.shape[-1]
            expected_shape = self.n_actions * (self.n_agents - 1)
            
            # 过滤掉当前agent的动作
            # actions_onehot_others 的形状可能是 (B, n_actions * n_agents) 或 (B, n_actions * (n_actions - 1))
            # 需要根据实际形状来调整过滤逻辑
            actions_onehot_others_shape = actions_onehot_others.shape[-1]
            
            if actions_onehot_others_shape == self.n_actions * self.n_agents:
                # 包含所有agent的动作，需要过滤掉当前agent的动作
                actions_onehot_others_filtered = torch.cat([
                    actions_onehot_others[:, 0:agent_idx_int * self.n_actions],
                    actions_onehot_others[:, (agent_idx_int * self.n_actions + self.n_actions):]
                ], dim=-1)
            elif actions_onehot_others_shape == expected_shape:
                # 已经是过滤后的形状（不包含当前agent的动作），直接使用
                actions_onehot_others_filtered = actions_onehot_others
            else:
                # 形状不匹配，尝试调整
                # 如果实际形状小于预期，可能需要截取或填充
                if actions_onehot_others_shape < pred_actions_shape:
                    # 实际形状小于预测形状，填充零
                    padding = torch.zeros(
                        actions_onehot_others.shape[0],
                        pred_actions_shape - actions_onehot_others_shape,
                        device=actions_onehot_others.device,
                        dtype=actions_onehot_others.dtype
                    )
                    actions_onehot_others_filtered = torch.cat([actions_onehot_others, padding], dim=-1)
                else:
                    # 实际形状大于预测形状，截取
                    actions_onehot_others_filtered = actions_onehot_others[:, :pred_actions_shape]
            
            # 确保两个tensor的形状匹配
            min_shape = min(pred_actions_shape, actions_onehot_others_filtered.shape[-1])
            predicted_actions = predicted_actions[:, :min_shape]
            actions_onehot_others_filtered = actions_onehot_others_filtered[:, :min_shape]
            
            action_loss = self.actions_loss_lambda * ((predicted_actions - actions_onehot_others_filtered) ** 2).sum()
        
        # 总损失
        total_loss = reconstruction_loss + kl_loss + action_loss
        
        return {
            "total_loss": total_loss,
            "recon_loss": reconstruction_loss,
            "action_loss": action_loss,
            "kl_loss": kl_loss,
        }

    def update_agent_vae(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        actions_onehot: torch.Tensor,
        actions_onehot_others: torch.Tensor,
        agent_id: int,
        epochs: int = 1,
    ) -> Dict[str, float]:
        """
        更新指定agent的VAE参数（参考 smpe-main）

        Args:
            obs: 观测，形状为 (B, obs_dim)
            states: 全局状态，形状为 (B, state_dim)
            actions_onehot: 当前agent的动作one-hot，形状为 (B, n_actions)
            actions_onehot_others: 其他agent的动作one-hot，形状为 (B, n_actions * n_agents)
            agent_id: 智能体ID（整数索引）
            epochs: 训练轮数

        Returns:
            平均损失字典
        """
        # 确保 agent_id 是整数（用于索引）
        # 先转换为 Python int，避免 numpy int64 或 torch.Tensor 类型问题
        if isinstance(agent_id, torch.Tensor):
            # 如果是 tensor，先提取标量再转换
            if agent_id.numel() == 1:
                agent_idx = int(agent_id.item())
            else:
                agent_idx = 0
        elif isinstance(agent_id, (np.integer, np.floating)):
            agent_idx = int(agent_id)
        elif isinstance(agent_id, int):
            agent_idx = agent_id
        else:
            try:
                agent_idx = int(agent_id) if agent_id is not None else 0
            except (ValueError, TypeError):
                agent_idx = 0
        
        # 确保 agent_idx 在有效范围内
        agent_idx = max(0, min(agent_idx, len(self.agent_models) - 1))
        
        # 最终类型检查：确保是 Python int（不是 numpy.int64 等）
        # 使用 type() 检查并强制转换
        if type(agent_idx).__name__ != 'int':
            agent_idx = int(agent_idx)
        # 再次确保是纯 Python int
        agent_idx = int(agent_idx)
        
        # 验证：确保 agent_idx 是纯 Python int
        assert isinstance(agent_idx, int) and not isinstance(agent_idx, (np.integer, np.floating)), \
            f"agent_idx must be Python int, got {type(agent_idx)}"
        
        # 使用 list() 转换避免 ModuleList 索引问题
        list_models = list(self.agent_models)
        list_models[agent_idx].train()
        total_losses = {"total_loss": 0.0, "recon_loss": 0.0, "action_loss": 0.0, "kl_loss": 0.0}

        for epoch in range(epochs):
            # 计算损失
            losses = self.compute_loss(
                obs=obs,
                states=states,
                actions_onehot=actions_onehot,
                actions_onehot_others=actions_onehot_others,
                agent_id=agent_idx,  # 传递转换后的整数索引
            )

            # 反向传播
            # 使用明确的整数索引访问优化器
            opt_idx = int(agent_idx)
            list_optimizers = list(self.agent_optimizers)
            list_optimizers[opt_idx].zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.agent_params)[opt_idx],
                max_norm=1.0,
            )
            list_optimizers[opt_idx].step()

            # 累加损失
            for key in total_losses:
                total_losses[key] += losses[key].item()

        # 平均损失
        for key in total_losses:
            total_losses[key] /= epochs

        return total_losses

    def encode(self, obs: torch.Tensor, agent_id: int, test_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码观测到潜在空间（实现BaseVAE接口）
        
        Args:
            obs: 观测
            agent_id: 智能体ID（整数索引）
            test_mode: 是否使用测试模式
            
        Returns:
            (z, mu): 潜在变量和均值
        """
        # 确保 agent_id 是整数（用于索引）
        if isinstance(agent_id, torch.Tensor):
            agent_id = int(agent_id.item())
        elif not isinstance(agent_id, int):
            agent_id = int(agent_id) if agent_id is not None else 0
        
        # 确保 agent_id 在有效范围内
        agent_id = max(0, min(agent_id, len(self.agent_models) - 1))
        
        z, mu, _ = self.agent_models[agent_id].encoder.forward(obs, test_mode)
        return z, mu

    def decode(self, z: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        解码潜在变量（实现BaseVAE接口）
        
        Args:
            z: 潜在变量
            agent_id: 智能体ID（整数索引）
            
        Returns:
            predicted_states: 预测的其他agent的状态
        """
        # 确保 agent_id 是整数（用于索引）
        if isinstance(agent_id, torch.Tensor):
            agent_id = int(agent_id.item())
        elif not isinstance(agent_id, int):
            agent_id = int(agent_id) if agent_id is not None else 0
        
        # 确保 agent_id 在有效范围内
        agent_id = max(0, min(agent_id, len(self.agent_models) - 1))
        
        return self.agent_models[agent_id].decoder.forward(z)

    def to(self, device: str) -> "VAEController":
        """移动到指定设备"""
        self.device = torch.device(device)
        for model in self.agent_models:
            model.to(self.device)
        return self

