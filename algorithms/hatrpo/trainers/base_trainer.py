# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HATRPO训练器实现
异质智能体TRPO（Heterogeneous Agent TRPO）
顺序更新每个agent，使用TRPO的信任区域更新
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import torch
import random

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer
from algorithms.common.core_trpo import TrustRegionUpdater
from algorithms.matrpo.core import compute_trpo_loss
from algorithms.happo.core import update_marginal_advantage

# Logger是可选的
try:
    from common.utils.logging import LoggerManager
    Logger = LoggerManager
except ImportError:
    Logger = None

# Tracker是可选的
try:
    from common.tracking import ExperimentTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    ExperimentTracker = None


class HATRPOTrainer(BaseAlgorithmTrainer):
    """
    HATRPO训练器
    
    结合HAPPO和TRPO：
    - 顺序更新每个agent（HAPPO特性）
    - 使用TRPO的信任区域更新（TRPO特性）
    - 每次更新后重新计算边际优势
    """

    def __init__(
        self,
        agent,
        env,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
    ):
        """
        初始化HATRPO训练器

        Args:
            agent: AgentManager实例（多Agent）
            env: 环境实例
            config: 训练配置
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
        """
        super().__init__(agent, env, config, logger, tracker)
        
        # TRPO特定配置
        self.kl_threshold = config.get("kl_threshold", 0.01)
        self.max_line_search_steps = config.get("max_line_search_steps", 15)
        self.accept_ratio = config.get("accept_ratio", 0.1)
        self.back_ratio = config.get("back_ratio", 0.8)
        self.cg_damping = config.get("cg_damping", 0.1)
        self.cg_max_iters = config.get("cg_max_iters", 10)
        self.critic_lr = config.get("critic_lr", 5e-3)
        
        # 存储每个agent的TRPO更新器
        self.trpo_updaters: Dict[str, TrustRegionUpdater] = {}

    def _happo_update_agent(
        self,
        agent: Any,
        agent_id: str,
        batch: Dict[str, Any],
        marginal_advantages: Dict[str, torch.Tensor],
        processed_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        使用HATRPO更新单个agent（使用TRPO而不是PPO）

        Args:
            agent: Agent实例
            agent_id: Agent ID
            batch: 训练批次
            marginal_advantages: 边际优势字典
            processed_data: 所有agent的数据

        Returns:
            训练指标
        """
        # 准备数据
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=agent.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=agent.device)
        advantages = batch["advantages"]
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.as_tensor(advantages, dtype=torch.float32, device=agent.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=agent.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取当前策略分布
        dist, values = agent._forward(obs)
        new_logprobs = dist.log_prob(actions)
        
        # 计算TRPO策略损失（不使用裁剪）
        policy_loss = compute_trpo_loss(new_logprobs, old_logprobs, advantages)
        
        # 计算价值损失（需要分离，因为update_actor会消耗计算图）
        value_loss = ((values.detach() - returns) ** 2).mean()
        # 但我们需要可微分的value_loss用于critic更新，所以重新计算
        # 注意：我们需要在update_actor之后重新计算value_loss
        value_loss_for_critic = None  # 将在update_actor之后计算
        
        # 计算熵
        entropy = dist.entropy().mean()
        
        # 创建或获取TRPO更新器
        if agent_id not in self.trpo_updaters:
            from torch.distributions import Categorical
            dist_class = Categorical
            
            self.trpo_updaters[agent_id] = TrustRegionUpdater(
                model=agent,
                dist_class=dist_class,
                train_batch={
                    "obs": obs,
                    "actions": actions,
                    "logprobs": old_logprobs,
                    "action_dist_inputs": dist.probs.detach(),
                },
                advantages=advantages,
                kl_threshold=self.kl_threshold,
                max_line_search_steps=self.max_line_search_steps,
                accept_ratio=self.accept_ratio,
                back_ratio=self.back_ratio,
                critic_lr=self.critic_lr,
                cg_damping=self.cg_damping,
                cg_max_iters=self.cg_max_iters,
                device=agent.device,
            )
        
        updater = self.trpo_updaters[agent_id]
        
        # 更新train_batch
        updater.train_batch = {
            "obs": obs,
            "actions": actions,
            "logprobs": old_logprobs,
            "action_dist_inputs": dist.probs.detach(),
            "old_dist_inputs": dist.probs.detach(),
        }
        updater.advantages = advantages
        
        # 执行TRPO更新
        value_coef = float(batch.get("value_coef", self.value_coef))
        entropy_coef = float(batch.get("entropy_coef", self.entropy_coef))
        
        # 更新actor（使用信任区域方法）
        actor_updated = updater.update_actor(policy_loss)
        
        # 更新actor后，重新计算value_loss（因为模型参数可能已改变，且需要新的计算图）
        with torch.enable_grad():
            # 重新前向传播获取新的value
            dist_new, values_new = agent._forward(obs)
            # 计算新的value_loss（支持价值函数裁剪）
            vf_clip_param = batch.get("vf_clip_param", None)
            if vf_clip_param is not None and vf_clip_param > 0:
                old_values = torch.as_tensor(batch.get("old_values"), dtype=torch.float32, device=agent.device)
                vf_loss1 = (values_new - returns) ** 2
                vf_clipped = old_values + torch.clamp(values_new - old_values, -vf_clip_param, vf_clip_param)
                vf_loss2 = (vf_clipped - returns) ** 2
                value_loss_for_critic = torch.max(vf_loss1, vf_loss2).mean()
            else:
                value_loss_for_critic = ((values_new - returns) ** 2).mean()
        
        # 更新critic（使用标准优化器，避免计算图冲突）
        # 因为actor和critic可能共享编码器，TRPO更新actor时已经消耗了计算图
        # 所以使用标准优化器更新critic更安全
        agent.optimizer.zero_grad()
        (value_coef * value_loss_for_critic).backward()
        
        # 只更新critic相关的参数（如果有分离的参数）
        if hasattr(agent, 'value_head') or hasattr(agent, 'critic'):
            # 尝试获取critic参数
            try:
                if hasattr(agent, 'get_critic_parameters'):
                    critic_params = agent.get_critic_parameters()
                    # 只对critic参数进行梯度更新
                    critic_lr = self.config.get("critic_lr", 5e-3)
                    for param in critic_params:
                        if param.grad is not None:
                            param.data -= critic_lr * param.grad
                else:
                    # 如果没有分离的参数，更新所有参数
                    agent.optimizer.step(list(agent.parameters()))
            except Exception:
                # 回退到标准更新
                agent.optimizer.step(list(agent.parameters()))
        else:
            # 如果没有单独的critic，使用标准优化器更新
            agent.optimizer.step(list(agent.parameters()))
        
        # 计算KL散度（用于监控）- 使用更新后的分布
        with torch.no_grad():
            old_dist = torch.distributions.Categorical(logits=dist.logits.detach())
            new_dist_kl = dist_new  # 使用更新后的分布
            kl = torch.distributions.kl.kl_divergence(old_dist, new_dist_kl).mean()
        
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss_for_critic.item() if value_loss_for_critic is not None else value_loss.item()),
            "entropy": float(entropy.item()),
            "kl_divergence": float(kl.item()),
            "actor_updated": 1.0 if actor_updated else 0.0,
            "total_loss": float(policy_loss.item() + value_coef * (value_loss_for_critic.item() if value_loss_for_critic is not None else value_loss.item()) - entropy_coef * entropy.item()),
        }

