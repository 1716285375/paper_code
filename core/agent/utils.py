# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : utils.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:29
@Update Date    :
@Description    : 可配置的PPO Agent实现
支持通过YAML配置文件灵活配置不同的模块组件
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution

from core.agent.factory import build_module_from_config
from core.base.agent import Agent
from core.modules.optimizers import AdamBuilder, AdamWBuilder, AdaptiveOptimizer, OptimizerConfig
from core.modules.policy_heads import BasePolicyHead, DiscretePolicyHead
from core.modules.value_heads import BaseValueHead, LinearValueHead, MLPValueHead
from core.networks import CNNEncoder, LSTMEncoder, MLPEncoder


class ConfigurablePPOAgent(Agent):
    """
    可配置的PPO Agent，支持通过配置字典灵活组合不同的模块组件

    可通过配置文件或字典动态选择编码器、策略头、价值头、优化器等组件，
    实现模块化的热插拔设计。

    配置结构示例:
        encoder:
            type: "networks/mlp" | "networks/cnn" | "networks/lstm"
            params:
                in_dim: 845  # 对于CNN使用obs_shape
                hidden_dims: [128, 128]
        policy_head:
            type: "policy_heads/discrete"
            params:
                hidden_dims: [64]
        value_head:
            type: "value_heads/mlp" | "value_heads/linear"
            params:
                hidden_dims: [64]  # MLP可选参数
        optimizer:
            type: "optimizers/adam" | "optimizers/adamw"
            params:
                lr: 3e-4
                scheduler: "linear"  # 学习率调度器
                warmup_steps: 1000
                total_steps: 100000
        exploration:  # 可选，探索策略
            type: "exploration/epsilon_greedy"
            params:
                epsilon: 0.2
        obs_normalizer:  # 可选，观测归一化
            type: "normalization/observation"
            params:
                shape: [845]
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build encoder
        encoder_config = config.get(
            "encoder",
            {"type": "networks/mlp", "params": {"in_dim": obs_dim, "hidden_dims": [128, 128]}},
        )
        encoder_type = encoder_config.get("type") or encoder_config.get("name", "networks/mlp")
        encoder_params = encoder_config.get("params", {}) or encoder_config.get("kwargs", {})

        if encoder_type == "networks/mlp":
            self.encoder = MLPEncoder(
                in_dim=obs_dim, hidden_dims=encoder_params.get("hidden_dims", [128, 128])
            ).to(self.device)
            feature_dim = self.encoder.output_dim
        elif encoder_type == "networks/cnn":
            obs_shape = encoder_params.get("obs_shape", (3, 13, 13))
            conv_channels = encoder_params.get("conv_channels", [32, 64])
            self.encoder = CNNEncoder(in_channels=obs_shape[0], conv_channels=conv_channels).to(
                self.device
            )
            # Will be set after first forward
            feature_dim = None
        elif encoder_type == "networks/lstm":
            hidden_size = encoder_params.get("hidden_size", 128)
            self.encoder = LSTMEncoder(input_dim=obs_dim, hidden_size=hidden_size).to(self.device)
            feature_dim = self.encoder.output_dim
        else:
            # Try registry
            self.encoder = build_module_from_config(encoder_config)
            if self.encoder is None:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
            if hasattr(self.encoder, "output_dim"):
                feature_dim = self.encoder.output_dim
            else:
                feature_dim = None

        # Build policy head
        policy_config = config.get(
            "policy_head", {"type": "policy_heads/discrete", "params": {"hidden_dims": [64]}}
        )
        policy_type = policy_config.get("type") or policy_config.get(
            "name", "policy_heads/discrete"
        )
        policy_params = policy_config.get("params", {}) or policy_config.get("kwargs", {})

        if policy_type == "policy_heads/discrete":
            if feature_dim is None:
                # Delay initialization
                self.policy_head: Optional[BasePolicyHead] = None
                self._policy_config = (policy_type, policy_params)
            else:
                self.policy_head = DiscretePolicyHead(
                    in_dim=feature_dim,
                    action_dim=action_dim,
                    hidden_dims=policy_params.get("hidden_dims", [64]),
                ).to(self.device)
        else:
            self.policy_head = build_module_from_config(policy_config)
            if self.policy_head is None:
                raise ValueError(f"Unknown policy head type: {policy_type}")

        # Build value head
        value_config = config.get(
            "value_head", {"type": "value_heads/mlp", "params": {"hidden_dims": [64]}}
        )
        value_type = value_config.get("type") or value_config.get("name", "value_heads/mlp")
        value_params = value_config.get("params", {}) or value_config.get("kwargs", {})

        if value_type == "value_heads/mlp":
            if feature_dim is None:
                self.value_head: Optional[BaseValueHead] = None
                self._value_config = (value_type, value_params)
            else:
                self.value_head = MLPValueHead(
                    in_dim=feature_dim, hidden_dims=value_params.get("hidden_dims", [64])
                ).to(self.device)
        elif value_type == "value_heads/linear":
            if feature_dim is None:
                self.value_head = None
                self._value_config = (value_type, value_params)
            else:
                self.value_head = LinearValueHead(in_dim=feature_dim).to(self.device)
        else:
            self.value_head = build_module_from_config(value_config)

        # Build optimizer
        opt_config = config.get("optimizer", {"type": "optimizers/adam", "params": {"lr": 3e-4}})
        opt_type = opt_config.get("type") or opt_config.get("name", "optimizers/adam")
        opt_params = opt_config.get("params", {}) or opt_config.get("kwargs", {})

        if opt_type in ["optimizers/adam", "optimizers/adamw"]:
            # 确保数值参数被正确转换为浮点数（YAML可能将科学计数法读取为字符串）
            normalized_params = {}
            for key, value in opt_params.items():
                if key in ["lr", "weight_decay", "eps", "momentum", "alpha", "max_grad_norm"]:
                    # 数值参数：转换为float（支持字符串形式的科学计数法，如"3e-4"）
                    if value is not None:
                        normalized_params[key] = (
                            float(value) if not isinstance(value, float) else value
                        )
                    else:
                        normalized_params[key] = None
                elif key == "betas" and isinstance(value, (list, tuple)):
                    # betas元组：确保所有元素是float
                    normalized_params[key] = tuple(float(x) for x in value)
                elif key in ["warmup_steps", "total_steps"]:
                    # 整数参数：转换为int
                    normalized_params[key] = int(value) if value is not None else 0
                else:
                    # 其他参数：保持原样（如scheduler字符串、centered布尔值等）
                    normalized_params[key] = value

            opt_cfg = OptimizerConfig(**normalized_params)
            builder = AdamBuilder() if opt_type == "optimizers/adam" else AdamWBuilder()
            optimizer = builder.build(list(self.parameters()), opt_cfg)
            self.optimizer = AdaptiveOptimizer(optimizer, opt_cfg)
        else:
            # Try registry
            optimizer = build_module_from_config(opt_config)
            if optimizer is None:
                raise ValueError(f"Unknown optimizer type: {opt_type}")
            self.optimizer = optimizer

        # Optional modules
        self.exploration = build_module_from_config(config.get("exploration"))
        self.obs_normalizer = build_module_from_config(config.get("obs_normalizer"))
        self.adv_normalizer = build_module_from_config(config.get("adv_normalizer"))

        self._feature_dim = feature_dim

    def _ensure_heads(self, features: torch.Tensor) -> None:
        """Lazy initialization of heads if feature_dim was unknown."""
        if self.policy_head is None or self.value_head is None:
            feat_dim = features.shape[-1]
            policy_type, policy_params = getattr(
                self, "_policy_config", ("policy_heads/discrete", {})
            )
            value_type, value_params = getattr(self, "_value_config", ("value_heads/mlp", {}))

            if self.policy_head is None:
                if policy_type == "policy_heads/discrete":
                    self.policy_head = DiscretePolicyHead(
                        in_dim=feat_dim,
                        action_dim=self.action_dim,
                        hidden_dims=policy_params.get("hidden_dims", [64]),
                    ).to(self.device)
                else:
                    self.policy_head = build_module_from_config(
                        {"type": policy_type, "params": policy_params}
                    )

            if self.value_head is None:
                if value_type == "value_heads/mlp":
                    self.value_head = MLPValueHead(
                        in_dim=feat_dim, hidden_dims=value_params.get("hidden_dims", [64])
                    ).to(self.device)
                elif value_type == "value_heads/linear":
                    self.value_head = LinearValueHead(in_dim=feat_dim).to(self.device)
                else:
                    self.value_head = build_module_from_config(
                        {"type": value_type, "params": value_params}
                    )

    def _forward(self, obs: torch.Tensor) -> Tuple[Distribution, torch.Tensor]:
        # Normalize observation if configured (before encoding)
        if self.obs_normalizer is not None:
            # Flatten spatial dimensions for normalizer if needed
            original_shape = obs.shape
            obs_flat = obs.flatten(1) if obs.dim() > 2 else obs  # (B, ...) -> (B, D)
            # Normalize
            obs_np = obs_flat.cpu().numpy()
            obs_normalized_np = self.obs_normalizer.normalize(obs_np)
            obs_normalized = torch.as_tensor(obs_normalized_np, device=self.device, dtype=obs.dtype)
            # Reshape back if needed (but for MLPEncoder we keep flattened)
            if isinstance(self.encoder, MLPEncoder) or not isinstance(self.encoder, CNNEncoder):
                obs = obs_normalized  # Already flattened
            else:
                obs = obs_normalized.reshape(original_shape)

        # Encode
        if isinstance(self.encoder, LSTMEncoder):
            # LSTM expects (B, T, D) - fake sequence length 1
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            features, _ = self.encoder(obs)
        elif isinstance(self.encoder, MLPEncoder):
            # Ensure MLPEncoder gets flattened input
            if obs.dim() > 2:
                obs = obs.flatten(1)  # (B, H, W, C) -> (B, H*W*C)
            features = self.encoder(obs)
        elif isinstance(self.encoder, CNNEncoder):
            # CNN expects (B, C, H, W)
            if obs.dim() == 4 and obs.shape[-1] == 5:  # (B, H, W, C)
                obs = obs.permute(0, 3, 1, 2)  # (B, C, H, W)
            features = self.encoder(obs)
        else:
            features = self.encoder(obs)

        self._ensure_heads(features)

        # Get distribution and value
        dist = self.policy_head(features)
        value = self.value_head(features)
        return dist, value

    def act(self, observation: Any, deterministic: bool = False) -> Any:
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        # Handle different input shapes
        if x.dim() == 3:  # (H, W, C) or (C, H, W)
            # Check if it's image-like (spatial dimensions)
            if x.shape[0] == 13 or x.shape[2] == 5:  # battle_v4 shape (H, W, C) = (13, 13, 5)
                x = x.unsqueeze(0)  # Add batch dim: (1, 13, 13, 5)
                # For MLPEncoder, we need to flatten spatial dimensions
                if isinstance(self.encoder, MLPEncoder) or not isinstance(self.encoder, CNNEncoder):
                    x = x.flatten(1)  # (1, 845)
            else:
                x = x.unsqueeze(0)
        elif x.dim() == 1:
            x = x.unsqueeze(0)  # (1, D)
        elif x.dim() == 2:
            # Already (B, D) or (H, W) - check if needs batch dim
            if x.shape[0] > 20:  # Likely (H, W) spatial
                x = x.flatten().unsqueeze(0)
            else:
                x = x if x.shape[0] == 1 else x.unsqueeze(0)

        with torch.no_grad():
            dist, value = self._forward(x)

            # Apply exploration if configured
            if self.exploration is not None and not deterministic:
                logits = dist.logits if hasattr(dist, "logits") else None
                if logits is not None:
                    result = self.exploration.apply(logits=logits)
                    if "logits" in result:
                        dist = type(dist)(logits=result["logits"])

            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
            logprob = dist.log_prob(action)

        return int(action.item()), float(logprob.item()), float(value.item())

    def learn(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        clip_coef = float(batch.get("clip_coef", 0.2))
        value_coef = float(batch.get("value_coef", 0.5))
        entropy_coef = float(batch.get("entropy_coef", 0.01))

        # Normalize advantages if configured
        if self.adv_normalizer is not None:
            advantages = torch.as_tensor(
                self.adv_normalizer.normalize(advantages.cpu().numpy()),
                device=self.device,
                dtype=advantages.dtype,
            )

        dist, values = self._forward(obs)
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = torch.exp(logprobs - old_logprobs)
        policy_loss = -(
            torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages,
            )
        ).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(list(self.parameters()))

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "total_loss": float(loss.item()),
        }

    def parameters(self):
        for p in self.encoder.parameters():
            yield p
        if self.policy_head is not None:
            for p in self.policy_head.parameters():
                yield p
        if self.value_head is not None:
            for p in self.value_head.parameters():
                yield p

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "policy_head": self.policy_head.state_dict() if self.policy_head is not None else None,
            "value_head": self.value_head.state_dict() if self.value_head is not None else None,
            "optimizer": (
                self.optimizer.optimizer.state_dict()
                if hasattr(self.optimizer, "optimizer")
                else None
            ),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state["encoder"])
        if self.policy_head is not None and state.get("policy_head") is not None:
            self.policy_head.load_state_dict(state["policy_head"])
        if self.value_head is not None and state.get("value_head") is not None:
            self.value_head.load_state_dict(state["value_head"])

    def to_training_mode(self) -> None:
        self.encoder.train()
        if self.policy_head is not None:
            self.policy_head.train()
        if self.value_head is not None:
            self.value_head.train()

    def to_eval_mode(self) -> None:
        self.encoder.eval()
        if self.policy_head is not None:
            self.policy_head.eval()
        if self.value_head is not None:
            self.value_head.eval()
