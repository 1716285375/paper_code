# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : ppo_penalty.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    : 2025-11-05
@Description    : PPO-Penalty Agent实现
PPO的Penalty变体，使用自适应KL惩罚而非clipping

注意：
    - 此Agent类主要用于需要独立Agent实例的场景
    - 对于Trainer，建议使用ConfigurablePPOAgent + use_penalty配置
    - 配置示例：
        agent:
          type: "ppo"
          use_penalty: true
          kl_penalty:
            initial_beta: 1.0
            target_kl: 0.01
    - 在trainer配置中设置 use_penalty: true 即可启用PPO-Penalty模式
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Distribution

from core.agent.utils import ConfigurablePPOAgent
from core.modules.regularizers.adaptive_kl_penalty import (
    AdaptiveKLCoefficient,
    AdaptiveKLPenalty,
)


class PPOPenaltyAgent(ConfigurablePPOAgent):
    """
    PPO-Penalty Agent

    PPO的Penalty变体，使用自适应KL惩罚而非clipping。
    与标准PPO的区别：
    - 标准PPO: policy_loss = -min(ratio * adv, clip(ratio) * adv)
    - PPO-Penalty: policy_loss = -(ratio * adv) + beta * KL(old, new)

    其中beta是自适应调整的KL惩罚系数。
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        """
        初始化PPO-Penalty Agent

        Args:
            obs_dim: 观测维度
            action_dim: 动作空间维度
            config: Agent配置字典，额外包含：
                - kl_penalty: KL惩罚配置（可选）
                    - initial_beta: 初始beta值（默认1.0）
                    - target_kl: 目标KL散度（默认0.01）
                    - min_beta: beta最小值（默认1e-6）
                    - max_beta: beta最大值（默认100.0）
                    - adjustment_rate: 调整速率（默认2.0）
            device: 设备
        """
        super().__init__(obs_dim, action_dim, config, device)

        # 初始化自适应KL惩罚
        kl_config = config.get("kl_penalty", {})
        initial_beta = kl_config.get("initial_beta", 1.0)
        target_kl = kl_config.get("target_kl", 0.01)
        min_beta = kl_config.get("min_beta", 1e-6)
        max_beta = kl_config.get("max_beta", 100.0)
        adjustment_rate = kl_config.get("adjustment_rate", 2.0)

        self.kl_coefficient = AdaptiveKLCoefficient(
            initial_beta=initial_beta,
            target_kl=target_kl,
            min_beta=min_beta,
            max_beta=max_beta,
            adjustment_rate=adjustment_rate,
        )

        self.kl_penalty = AdaptiveKLPenalty(
            adaptive_coefficient=self.kl_coefficient,
            use_adaptive=True,
        )

        # 用于存储旧策略分布（用于KL计算）
        self._old_dist_cache: Optional[Distribution] = None

    def learn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        PPO-Penalty学习步骤

        与标准PPO的区别：
        - 不使用clipping，而是使用KL惩罚
        - beta系数根据KL散度自适应调整

        Args:
            batch: 训练批次，必须包含：
                - obs: 观测
                - actions: 动作
                - logprobs: 旧策略的log概率
                - advantages: 优势
                - returns: 回报
                - value_coef: 价值损失系数
                - entropy_coef: 熵正则化系数

        Returns:
            训练指标字典，包含：
                - policy_loss: 策略损失
                - value_loss: 价值损失
                - entropy: 熵
                - kl_divergence: KL散度
                - beta: 当前beta值
                - total_loss: 总损失
        """
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        value_coef = float(batch.get("value_coef", 0.5))
        entropy_coef = float(batch.get("entropy_coef", 0.01))

        # Normalize advantages if configured
        if self.adv_normalizer is not None:
            advantages = torch.as_tensor(
                self.adv_normalizer.normalize(advantages.cpu().numpy()),
                device=self.device,
                dtype=advantages.dtype,
            )

        # 获取新策略分布和旧策略分布
        dist_new, values = self._forward(obs)

        # 从旧logprobs重建旧分布（用于KL计算）
        # 注意：这是一个近似，真正的KL需要完整的旧分布
        # 在实际实现中，我们使用logprobs的差异来近似KL
        # 对于离散动作空间：KL ≈ E[log(π_new(a|s)) - log(π_old(a|s))]
        logprobs_new = dist_new.log_prob(actions)
        ratio = torch.exp(logprobs_new - old_logprobs)

        # PPO-Penalty目标：使用ratio而非clipping
        policy_loss = -(ratio * advantages).mean()

        # 计算KL散度惩罚
        # 对于离散分布，我们可以直接计算KL散度
        # 但为了效率，我们使用logprobs差异的均值作为近似
        # 真正的KL散度需要完整的分布，这里使用近似
        kl_approx = (logprobs_new - old_logprobs).mean()

        # 如果有完整的旧分布缓存，使用真实的KL散度
        if self._old_dist_cache is not None:
            try:
                kl = self.kl_penalty.compute_penalty(
                    dist_new=dist_new, dist_old=self._old_dist_cache
                )
            except Exception:
                # 如果计算失败，使用近似
                kl = torch.abs(kl_approx)
        else:
            # 使用近似KL散度
            kl = torch.abs(kl_approx)

        # 更新beta（自适应KL惩罚）
        kl_value = float(kl.detach().cpu().item())
        beta = self.kl_coefficient.update(kl_value)

        # 添加KL惩罚项
        policy_loss_with_kl = policy_loss + beta * kl

        # 价值损失和熵正则化
        value_loss = F.mse_loss(values, returns)
        entropy = dist_new.entropy().mean()

        # 总损失
        loss = policy_loss_with_kl + value_coef * value_loss - entropy_coef * entropy

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(list(self.parameters()))

        # 缓存当前分布作为下次的旧分布
        # 注意：这里简化处理，实际应该保存完整的分布状态
        self._old_dist_cache = dist_new

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl_divergence": float(kl.item()),
            "beta": float(beta),
            "total_loss": float(loss.item()),
        }

    def reset_kl_coefficient(self, beta: Optional[float] = None) -> None:
        """
        重置KL系数

        Args:
            beta: 新的beta值（如果为None则使用初始值）
        """
        self.kl_coefficient.reset(beta)

