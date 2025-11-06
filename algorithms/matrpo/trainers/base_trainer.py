# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MATRPO训练器实现
多智能体TRPO（Multi-Agent Trust Region Policy Optimization）
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer
from algorithms.common.core_trpo import TrustRegionUpdater
from algorithms.matrpo.core import compute_trpo_loss

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


class MATRPOTrainer(BaseAlgorithmTrainer):
    """
    MATRPO训练器
    
    在MAPPO基础上，使用TRPO的信任区域更新方法替代PPO的裁剪方法：
    - 使用共轭梯度法计算自然梯度方向
    - 使用线搜索确保KL散度约束
    - 支持集中训练-分散执行（CTDE）
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
        初始化MATRPO训练器

        Args:
            agent: AgentManager实例（多Agent）
            env: 环境实例
            config: 训练配置，包含：
                - TRPO相关：kl_threshold, max_line_search_steps等
                - MAPPO相关：use_centralized_critic, global_obs_dim等
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
        """
        # 调用基类初始化
        super().__init__(agent, env, config, logger, tracker)
        
        # PPO/MAPPO相关配置
        self.num_epochs = config.get("num_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
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
    
    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练（实现基类的抽象方法）
        
        Args:
            processed_data: 处理后的数据
        
        Returns:
            训练指标
        """
        # MATRPO只支持多Agent
        return self._train_step_multi_agent(processed_data)

    def _train_step_multi_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        多Agent训练步骤（MATRPO版本，使用TRPO更新）

        Args:
            processed_data: 处理后的数据（按agent_id组织）

        Returns:
            训练指标
        """
        all_metrics = []

        # 为每个agent准备批次数据
        batches = {}
        for agent_id, data in processed_data.items():
            obs = data["obs"]
            actions = data["actions"]
            old_logprobs = data["logprobs"]
            advantages = data["advantages"]
            returns = data["returns"]
            values = data.get("values", data.get("old_values", []))

            # 优势标准化（MATRPO中必须标准化）
            if isinstance(advantages, np.ndarray):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif isinstance(advantages, torch.Tensor):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 更新data中的advantages（标准化后）
            data["advantages"] = advantages

            num_samples = len(obs)
            batch_indices = np.random.permutation(num_samples)

            # TRPO多轮更新（通常比PPO少）
            for epoch in range(self.num_epochs):
                # 分批训练
                for i in range(0, num_samples, self.batch_size):
                    indices = batch_indices[i : i + self.batch_size]
                    
                    # 处理advantages（确保是numpy数组）
                    if isinstance(advantages, torch.Tensor):
                        batch_advantages = advantages[indices].cpu().numpy()
                    else:
                        batch_advantages = advantages[indices] if isinstance(advantages, np.ndarray) else np.array(advantages)[indices]

                    batch = {
                        "obs": obs[indices],
                        "actions": actions[indices],
                        "logprobs": old_logprobs[indices],
                        "advantages": batch_advantages,
                        "returns": returns[indices],
                        "old_values": data.get("old_values", values)[indices] if len(values) > 0 else returns[indices],
                        "value_coef": self.value_coef,
                        "entropy_coef": self.entropy_coef,
                        "vf_clip_param": self.config.get("vf_clip_param"),
                    }

                    # 如果使用集中式Critic，添加全局信息
                    if self.use_centralized_critic:
                        # 从processed_data获取全局状态（优先使用后处理后的state）
                        if "state" in data and len(data["state"]) > 0:
                            if len(indices) <= len(data["state"]):
                                batch["state"] = data["state"][indices]
                            else:
                                batch["state"] = data["state"]
                        else:
                            global_obs = self._get_global_obs(agent_id, indices, processed_data)
                            batch["state"] = global_obs
                        
                        batch["use_centralized_critic"] = True
                        
                        # 如果配置了对手动作输入，添加对手动作
                        if self.config.get("opp_action_in_cc", False):
                            opponent_actions = self._get_opponent_actions(agent_id, indices, processed_data)
                            if opponent_actions is not None:
                                batch["opponent_actions"] = opponent_actions

                    if agent_id not in batches:
                        batches[agent_id] = []
                    batches[agent_id].append(batch)

        # 使用TRPO更新每个agent
        for agent_id, agent_batches in batches.items():
            agent = self.agent.get_agent(agent_id)
            
            # 合并所有批次
            merged_batch = self._merge_batches(agent_batches)
            
            # 使用TRPO更新
            metrics = self._trpo_update_agent(agent, agent_id, merged_batch)
            all_metrics.append(metrics)

        return self._aggregate_metrics(all_metrics)
    
    def _get_global_obs(self, agent_id: str, indices: np.ndarray, processed_data: Dict[str, Any]) -> np.ndarray:
        """
        获取全局状态（用于集中式Critic）
        
        Args:
            agent_id: Agent ID
            indices: 批次索引
            processed_data: 所有agent的数据（应该包含state字段）
        
        Returns:
            全局状态
        """
        # 优先使用环境提供的全局状态
        if "state" in processed_data.get(agent_id, {}):
            states = processed_data[agent_id]["state"]
            if len(states) > 0:
                if len(indices) <= len(states):
                    return states[indices]
                else:
                    return states
        
        # 回退：拼接所有agent的观测
        global_obs_list = []
        for other_agent_id, other_data in processed_data.items():
            if len(other_data.get("obs", [])) > 0:
                other_obs = other_data["obs"][indices] if len(indices) <= len(other_data["obs"]) else other_data["obs"]
                global_obs_list.append(other_obs)

        if global_obs_list:
            global_obs = np.concatenate(global_obs_list, axis=-1)
        else:
            global_obs = processed_data[agent_id]["obs"][indices]

        return global_obs
    
    def _get_opponent_actions(
        self, agent_id: str, indices: np.ndarray, processed_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        获取对手动作（用于集中式Critic）
        
        Args:
            agent_id: 当前Agent ID
            indices: 批次索引
            processed_data: 所有agent的数据
        
        Returns:
            对手动作数组，形状为 (B, n_opponents, action_dim) 或 None
        """
        opponent_actions_list = []
        for other_agent_id, other_data in processed_data.items():
            if other_agent_id != agent_id and "actions" in other_data:
                actions = other_data["actions"]
                if len(indices) <= len(actions):
                    opponent_actions_list.append(actions[indices])
                else:
                    opponent_actions_list.append(actions)
        
        if opponent_actions_list:
            opponent_actions = np.stack(opponent_actions_list, axis=1)
            return opponent_actions
        return None

    def _trpo_update_agent(
        self,
        agent: Any,
        agent_id: str,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        使用TRPO更新单个agent

        Args:
            agent: Agent实例
            agent_id: Agent ID
            batch: 训练批次

        Returns:
            训练指标
        """
        # 准备数据
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=agent.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=agent.device)
        old_logprobs = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=agent.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=agent.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=agent.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 获取当前策略分布
        dist, values = agent._forward(obs)
        new_logprobs = dist.log_prob(actions)
        
        # 计算策略损失（TRPO不使用裁剪）
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
            # 需要获取dist_class，这里假设是Categorical
            from torch.distributions import Categorical
            dist_class = Categorical
            
            self.trpo_updaters[agent_id] = TrustRegionUpdater(
                model=agent,
                dist_class=dist_class,
                train_batch={
                    "obs": obs,
                    "actions": actions,
                    "logprobs": old_logprobs,
                    "action_dist_inputs": dist.probs.detach(),  # 保存旧分布参数
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
            "old_dist_inputs": dist.probs.detach(),  # 保存旧分布
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
            # 计算新的value_loss
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

    def _merge_batches(self, batches: list) -> Dict[str, Any]:
        """合并多个批次"""
        if len(batches) == 0:
            return {}
        if len(batches) == 1:
            return batches[0]
        
        merged = {}
        for key in batches[0].keys():
            values = [batch[key] for batch in batches]
            
            # 处理标量值（如 clip_coef, value_coef 等）
            if isinstance(values[0], (int, float, bool)) or (isinstance(values[0], np.generic) and values[0].ndim == 0):
                # 标量值：使用第一个批次的值
                merged[key] = values[0]
                continue
            
            # 处理tensor和array
            try:
                if isinstance(values[0], torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], np.ndarray):
                    # 确保不是0维数组
                    if values[0].ndim > 0:
                        merged[key] = np.concatenate(values, axis=0)
                    else:
                        # 0维数组（标量），使用第一个值
                        merged[key] = values[0]
                elif isinstance(values[0], list):
                    merged[key] = sum(values, [])
                else:
                    # 尝试转换为numpy数组
                    try:
                        arr_values = [np.array(v) for v in values]
                        if arr_values[0].ndim > 0:
                            merged[key] = np.concatenate(arr_values, axis=0)
                        else:
                            merged[key] = arr_values[0]
                    except Exception:
                        # 如果无法合并，使用第一个批次的值
                        merged[key] = values[0]
            except Exception as e:
                # 如果合并失败，使用第一个批次的值
                merged[key] = values[0]
        
        return merged
    
    def _aggregate_metrics(self, metrics_list: list) -> Dict[str, float]:
        """聚合指标列表"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = float(np.mean(values))
        
        return aggregated

