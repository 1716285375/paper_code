# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : vae_controller.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : VAE状态建模控制器
为每个智能体配一个VAE，用于建模对手和队友的状态/动作
支持共享参数 + agent-id one-hot编码
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from core.modules.vae.base import BaseVAE


class VAEEncoder(nn.Module):
    """
    VAE编码器
    输入：观测 + 上一次动作 + agent_id one-hot
    输出：潜在变量 z 的均值和方差
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id_dim: int = 32,  # agent_id one-hot维度（通常是agent数量或固定的嵌入维度）
        hidden_dims: list[int] = [64, 32],
        z_dim: int = 16,
    ) -> None:
        """
        初始化VAE编码器

        Args:
            obs_dim: 观测维度
            action_dim: 动作空间维度
            agent_id_dim: agent_id嵌入维度
            hidden_dims: 隐藏层维度列表
            z_dim: 潜在变量维度
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_id_dim = agent_id_dim
        self.z_dim = z_dim

        # Agent ID嵌入层
        self.agent_id_embed = nn.Linear(agent_id_dim, 16)

        # 输入维度：obs + action_onehot + agent_id_embed
        input_dim = obs_dim + action_dim + 16

        # 构建编码网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # 输出层：分别输出均值和方差的对数
        self.fc_mean = nn.Linear(prev_dim, z_dim)
        self.fc_logvar = nn.Linear(prev_dim, z_dim)

    def forward(self, obs: torch.Tensor, last_action: torch.Tensor, agent_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码输入到潜在空间

        Args:
            obs: 观测，形状为 (B, obs_dim) 或 (obs_dim,)
            last_action: 上一次动作（整数索引），形状为 (B,) 或标量
            agent_id: agent_id one-hot，形状为 (B, agent_id_dim) 或 (agent_id_dim,)

        Returns:
            (z_mean, z_logvar): 潜在变量的均值和方差对数
        """
        # 处理维度：确保所有输入都是2维的 (B, features)
        # 展平观测（如果是多维的，如图像）
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)  # (obs_dim,) -> (1, obs_dim)
        elif obs.dim() > 2:
            # 多维观测（如 (B, H, W, C) 或 (B, C, H, W)），需要展平
            # 保持batch维度，展平其他所有维度
            batch_size = obs.shape[0]
            obs = obs.view(batch_size, -1)  # (B, H, W, C) -> (B, H*W*C)
            # 确保展平后的维度等于obs_dim
            if obs.shape[-1] != self.obs_dim:
                # 如果维度不匹配，可能需要截断或填充（通常应该匹配）
                if obs.shape[-1] > self.obs_dim:
                    obs = obs[:, :self.obs_dim]
                else:
                    # 如果小于obs_dim，用零填充
                    padding = torch.zeros(batch_size, self.obs_dim - obs.shape[-1], device=obs.device, dtype=obs.dtype)
                    obs = torch.cat([obs, padding], dim=-1)
        
        # 确保观测是2维的 (B, obs_dim)
        if obs.dim() != 2:
            raise ValueError(f"观测维度错误: 期望2维 (B, obs_dim)，得到 {obs.dim()}维，形状 {obs.shape}")
        
        # 处理last_action
        if not isinstance(last_action, torch.Tensor):
            last_action = torch.tensor([last_action], device=obs.device)
        if last_action.dim() == 0:
            last_action = last_action.unsqueeze(0)  # 标量 -> (1,)
        elif last_action.dim() > 1:
            # 如果是多维的，展平
            last_action = last_action.flatten()
        
        # 处理agent_id
        if agent_id.dim() == 1:
            # 如果batch_size=1，需要unsqueeze；如果batch_size>1，已经是正确的
            if obs.shape[0] == 1 and agent_id.shape[0] == self.agent_id_dim:
                agent_id = agent_id.unsqueeze(0)  # (agent_id_dim,) -> (1, agent_id_dim)
        elif agent_id.dim() == 0:
            agent_id = agent_id.unsqueeze(0).unsqueeze(0)  # 标量 -> (1, 1) -> 需要扩展到 (1, agent_id_dim)
            if agent_id.shape[-1] != self.agent_id_dim:
                raise ValueError(f"agent_id维度错误: 期望 {self.agent_id_dim}，得到 {agent_id.shape[-1]}")
        
        # 确保batch维度一致
        batch_size = obs.shape[0]
        if last_action.shape[0] != batch_size:
            if last_action.shape[0] == 1:
                last_action = last_action.expand(batch_size)
            else:
                raise ValueError(f"last_action的batch维度不匹配: obs {batch_size}, last_action {last_action.shape[0]}")
        
        if agent_id.shape[0] != batch_size:
            if agent_id.shape[0] == 1:
                agent_id = agent_id.expand(batch_size, -1)
            else:
                raise ValueError(f"agent_id的batch维度不匹配: obs {batch_size}, agent_id {agent_id.shape[0]}")

        # 将动作转换为one-hot
        if last_action.dtype != torch.long:
            last_action = last_action.long()
        action_onehot = F.one_hot(last_action, num_classes=self.action_dim).float()  # (B, action_dim)

        # Agent ID嵌入
        agent_id_embed = self.agent_id_embed(agent_id)  # (B, agent_id_dim) -> (B, 16)

        # 拼接输入（确保所有张量都是2维的）
        x = torch.cat([obs, action_onehot, agent_id_embed], dim=-1)  # (B, obs_dim + action_dim + 16)

        # 编码
        h = self.encoder(x)

        # 输出均值和方差
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)

        return z_mean, z_logvar


class VAEDecoder(nn.Module):
    """
    VAE解码器
    输入：潜在变量 z
    输出：重构的观测和动作
    """

    def __init__(
        self,
        z_dim: int,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = [32, 64],
    ) -> None:
        """
        初始化VAE解码器

        Args:
            z_dim: 潜在变量维度
            obs_dim: 观测维度
            action_dim: 动作空间维度
            hidden_dims: 隐藏层维度列表（反向顺序）
        """
        super().__init__()
        self.z_dim = z_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 构建解码网络
        layers = []
        prev_dim = z_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # 输出层：分别输出观测和动作logits
        self.fc_obs = nn.Linear(prev_dim, obs_dim)
        self.fc_action = nn.Linear(prev_dim, action_dim)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码潜在变量

        Args:
            z: 潜在变量，形状为 (B, z_dim) 或 (z_dim,)

        Returns:
            (obs_recon, action_logits): 重构的观测和动作logits
        """
        # 处理维度
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # 解码
        h = self.decoder(z)

        # 输出观测和动作
        obs_recon = self.fc_obs(h)
        action_logits = self.fc_action(h)

        return obs_recon, action_logits


class VAEController(BaseVAE):
    """
    VAE控制器
    为每个智能体配一个VAE，用于建模对手和队友的状态/动作
    支持共享参数 + agent-id one-hot编码
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id_dim: int = 32,
        z_dim: int = 16,
        encoder_hidden_dims: list[int] = [64, 32],
        decoder_hidden_dims: list[int] = [32, 64],
        beta_kl: float = 1e-3,
        rec_coef: float = 1.0,
        action_rec_coef: float = 1.5,
        device: str = "cpu",
    ) -> None:
        """
        初始化VAE控制器

        Args:
            obs_dim: 观测维度
            action_dim: 动作空间维度
            agent_id_dim: agent_id嵌入维度
            z_dim: 潜在变量维度
            encoder_hidden_dims: 编码器隐藏层维度
            decoder_hidden_dims: 解码器隐藏层维度（反向顺序）
            beta_kl: KL散度权重
            rec_coef: 观测重构权重
            action_rec_coef: 动作重构权重
            device: 设备
        """
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.z_dim = z_dim
        self.beta_kl = beta_kl
        self.rec_coef = rec_coef
        self.action_rec_coef = action_rec_coef

        # 创建编码器和解码器
        self.encoder = VAEEncoder(
            obs_dim=obs_dim,
            action_dim=action_dim,
            agent_id_dim=agent_id_dim,
            hidden_dims=encoder_hidden_dims,
            z_dim=z_dim,
        ).to(self.device)

        self.decoder = VAEDecoder(
            z_dim=z_dim,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=decoder_hidden_dims,
        ).to(self.device)

        # 优化器（单独优化VAE）
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3,
        )

    def encode(self, obs: torch.Tensor, last_action: torch.Tensor, agent_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码观测到潜在空间

        Args:
            obs: 观测
            last_action: 上一次动作（整数索引）
            agent_id: agent_id one-hot

        Returns:
            (z_mean, z_logvar): 潜在变量的均值和方差对数
        """
        return self.encoder(obs, last_action, agent_id)

    def reparameterize(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧采样潜在变量

        Args:
            z_mean: 均值
            z_logvar: 方差对数

        Returns:
            z: 采样的潜在变量
        """
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码潜在变量

        Args:
            z: 潜在变量

        Returns:
            (obs_recon, action_logits): 重构的观测和动作logits
        """
        return self.decoder(z)

    def forward(self, obs: torch.Tensor, last_action: torch.Tensor, agent_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观测
            last_action: 上一次动作
            agent_id: agent_id one-hot

        Returns:
            (z, obs_recon, action_logits, z_mean, z_logvar)
        """
        z_mean, z_logvar = self.encode(obs, last_action, agent_id)
        z = self.reparameterize(z_mean, z_logvar)
        obs_recon, action_logits = self.decode(z)
        return z, obs_recon, action_logits, z_mean, z_logvar

    def compute_loss(
        self,
        obs: torch.Tensor,
        last_action: torch.Tensor,
        agent_id: torch.Tensor,
        target_obs: Optional[torch.Tensor] = None,
        target_action: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算VAE损失

        Args:
            obs: 当前观测（用于编码）
            last_action: 上一次动作
            agent_id: agent_id one-hot
            target_obs: 目标观测（用于重构损失，如果None则使用obs）
            target_action: 目标动作（用于动作重构损失，如果None则使用last_action）

        Returns:
            损失字典，包含：
                - total_loss: 总损失
                - recon_loss: 重构损失
                - action_recon_loss: 动作重构损失
                - kl_loss: KL散度损失
        """
        if target_obs is None:
            target_obs = obs
        if target_action is None:
            target_action = last_action

        # 处理target_obs的维度：确保它与obs_recon的形状一致
        # obs_recon是展平的 (B, obs_dim)，所以target_obs也需要展平
        if target_obs.dim() > 2:
            # 多维观测（如 (B, H, W, C)），需要展平
            batch_size = target_obs.shape[0]
            target_obs_flat = target_obs.view(batch_size, -1)  # (B, H, W, C) -> (B, H*W*C)
            # 确保展平后的维度等于obs_dim
            if target_obs_flat.shape[-1] != self.obs_dim:
                if target_obs_flat.shape[-1] > self.obs_dim:
                    target_obs_flat = target_obs_flat[:, :self.obs_dim]
                else:
                    # 如果小于obs_dim，用零填充
                    padding = torch.zeros(
                        batch_size,
                        self.obs_dim - target_obs_flat.shape[-1],
                        device=target_obs.device,
                        dtype=target_obs.dtype
                    )
                    target_obs_flat = torch.cat([target_obs_flat, padding], dim=-1)
            target_obs = target_obs_flat
        elif target_obs.dim() == 1:
            # 如果是1维，添加batch维度
            target_obs = target_obs.unsqueeze(0)

        # 前向传播
        z, obs_recon, action_logits, z_mean, z_logvar = self.forward(obs, last_action, agent_id)

        # 观测重构损失（MSE）
        recon_loss = F.mse_loss(obs_recon, target_obs, reduction="mean")

        # 动作重构损失（交叉熵）
        # 确保target_action是正确的形状和类型
        if target_action.dim() > 1:
            # 如果是多维的，展平或squeeze
            target_action = target_action.squeeze()
        if target_action.dtype != torch.long:
            target_action = target_action.long()
        # 确保target_action是1维的（batch_size,）
        if target_action.dim() == 0:
            target_action = target_action.unsqueeze(0)
        action_recon_loss = F.cross_entropy(action_logits, target_action, reduction="mean")

        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1).mean()

        # 确保所有系数都是Python标量（float），而不是tensor
        # 处理多种可能的类型：int, float, numpy标量, torch标量tensor
        def to_float(value):
            """将值转换为Python float"""
            if isinstance(value, (int, float)):
                return float(value)
            # 处理numpy类型
            try:
                import numpy as np
                if isinstance(value, (np.integer, np.floating)):
                    return float(value.item() if hasattr(value, 'item') else value)
            except (ImportError, AttributeError):
                pass
            # 处理torch tensor（只处理标量tensor）
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    if value.numel() == 1:
                        return float(value.item())
                    else:
                        # 如果不是标量tensor，尝试取第一个元素（不应该发生）
                        raise ValueError(f"Coefficient must be a scalar, got tensor of shape {value.shape}")
            except (ImportError, AttributeError):
                pass
            # 如果无法转换，尝试直接转换为float（可能会失败）
            return float(value)
        
        rec_coef = to_float(self.rec_coef)
        action_rec_coef = to_float(self.action_rec_coef)
        beta_kl = to_float(self.beta_kl)

        # 总损失
        total_loss = (
            rec_coef * recon_loss
            + action_rec_coef * action_recon_loss
            + beta_kl * kl_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "action_recon_loss": action_recon_loss,
            "kl_loss": kl_loss,
        }

    def update(self, batch: Dict[str, torch.Tensor], epochs: int = 1) -> Dict[str, float]:
        """
        更新VAE参数

        Args:
            batch: 批次数据，包含：
                - obs: 观测
                - last_action: 上一次动作
                - agent_id: agent_id one-hot
                - target_obs: 目标观测（可选）
                - target_action: 目标动作（可选）
            epochs: 训练轮数

        Returns:
            平均损失字典
        """
        self.train()
        total_losses = {"total_loss": 0.0, "recon_loss": 0.0, "action_recon_loss": 0.0, "kl_loss": 0.0}

        for epoch in range(epochs):
            # 计算损失
            losses = self.compute_loss(
                obs=batch["obs"],
                last_action=batch["last_action"],
                agent_id=batch["agent_id"],
                target_obs=batch.get("target_obs"),
                target_action=batch.get("target_action"),
            )

            # 反向传播
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            # 累加损失
            for key in total_losses:
                total_losses[key] += losses[key].item()

        # 平均损失
        for key in total_losses:
            total_losses[key] /= epochs

        return total_losses

    def to(self, device: str) -> "VAEController":
        """移动到指定设备"""
        self.device = torch.device(device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        return self

