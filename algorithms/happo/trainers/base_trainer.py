# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : HAPPO训练器实现
异质智能体PPO（Heterogeneous Agent PPO）
顺序更新每个agent，每次更新后重新计算advantage
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional, List
import numpy as np
import torch
import random

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer
from algorithms.happo.core import update_marginal_advantage, compute_happo_loss

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


class HAPPOTrainer(BaseAlgorithmTrainer):
    """
    HAPPO训练器
    
    异质智能体PPO的核心思想：
    - 顺序更新每个agent（而不是并行）
    - 每次更新一个agent后，使用新的策略重新计算advantage（边际优势）
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
        初始化HAPPO训练器

        Args:
            agent: AgentManager实例（多Agent）
            env: 环境实例
            config: 训练配置
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
        
        # HAPPO特定配置
        self.update_order = config.get("update_order", "random")  # random or sequential
        self.use_marginal_advantage = config.get("use_marginal_advantage", True)
    
    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练（实现基类的抽象方法）
        
        Args:
            processed_data: 处理后的数据
        
        Returns:
            训练指标
        """
        # HAPPO只支持多Agent
        return self._train_step_multi_agent(processed_data)

    def _train_step_multi_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        多Agent训练步骤（HAPPO版本，顺序更新）

        Args:
            processed_data: 处理后的数据（按agent_id组织）

        Returns:
            训练指标
        """
        all_metrics = []
        
        # 获取所有agent ID
        agent_ids = list(processed_data.keys())
        
        # 确定更新顺序
        if self.update_order == "random":
            update_sequence = random.sample(agent_ids, len(agent_ids))
        else:
            update_sequence = agent_ids
        
        # 初始化边际优势（使用原始优势）
        marginal_advantages = {}
        for agent_id, data in processed_data.items():
            advantages = data["advantages"]
            if isinstance(advantages, np.ndarray):
                advantages = torch.as_tensor(advantages, dtype=torch.float32)
            marginal_advantages[agent_id] = advantages.clone()
        
        # 为每个agent准备批次数据
        batches = {}
        for agent_id, data in processed_data.items():
            obs = data["obs"]
            actions = data["actions"]
            old_logprobs = data["logprobs"]
            returns = data["returns"]
            # 保存原始advantages用于后续索引
            original_advantages = marginal_advantages[agent_id]
            
            num_samples = len(obs)
            batch_indices = np.random.permutation(num_samples)
            
            # 准备批次
            for epoch in range(self.num_epochs):
                for i in range(0, num_samples, self.batch_size):
                    indices = batch_indices[i : i + self.batch_size]
                    
                    # 从原始advantages中提取对应索引的值
                    if isinstance(original_advantages, torch.Tensor):
                        batch_advantages = original_advantages[indices]
                    elif isinstance(original_advantages, np.ndarray):
                        batch_advantages = original_advantages[indices]
                        batch_advantages = torch.as_tensor(batch_advantages, dtype=torch.float32)
                    else:
                        batch_advantages = torch.as_tensor(np.array(original_advantages)[indices], dtype=torch.float32)
                    
                    batch = {
                        "obs": obs[indices],
                        "actions": actions[indices],
                        "logprobs": old_logprobs[indices],
                        "returns": returns[indices],
                        "advantages": batch_advantages,  # 直接使用对应索引的advantages
                        "old_values": data.get("old_values", data.get("values", []))[indices],
                        "clip_coef": self.clip_coef,
                        "value_coef": self.value_coef,
                        "entropy_coef": self.entropy_coef,
                        "vf_clip_param": self.config.get("vf_clip_param"),
                    }
                    
                    if agent_id not in batches:
                        batches[agent_id] = []
                    batches[agent_id].append(batch)
        
        # 顺序更新每个agent
        for agent_id in update_sequence:
            agent = self.agent.get_agent(agent_id)
            agent_batches = batches[agent_id]
            
            # 合并批次（advantages已经在每个batch中，会自动合并）
            merged_batch = self._merge_batches(agent_batches)
            
            # 确保advantages在正确的设备上
            if "advantages" in merged_batch:
                if isinstance(merged_batch["advantages"], torch.Tensor):
                    merged_batch["advantages"] = merged_batch["advantages"].to(agent.device)
                else:
                    merged_batch["advantages"] = torch.as_tensor(
                        merged_batch["advantages"], 
                        dtype=torch.float32, 
                        device=agent.device
                    )
            
            # 执行PPO更新
            metrics = self._happo_update_agent(
                agent,
                agent_id,
                merged_batch,
                marginal_advantages,
                processed_data,
            )
            all_metrics.append(metrics)
            
            # 更新边际优势（使用新策略的logprob ratio）
            # 注意：需要使用原始数据，而不是合并后的批次
            if self.use_marginal_advantage and agent_id in update_sequence[:-1]:  # 最后一个agent不需要更新
                self._update_marginal_advantages(
                    agent,
                    agent_id,
                    processed_data[agent_id],  # 使用原始数据，而不是merged_batch
                    marginal_advantages,
                )
        
        return self._aggregate_metrics(all_metrics)

    def _happo_update_agent(
        self,
        agent: Any,
        agent_id: str,
        batch: Dict[str, Any],
        marginal_advantages: Dict[str, torch.Tensor],
        processed_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        使用HAPPO更新单个agent

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
        
        # 计算HAPPO损失（使用PPO裁剪）
        clip_coef = float(batch.get("clip_coef", self.clip_coef))
        policy_loss, clip_fraction = compute_happo_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            clip_coef,
        )
        
        # 计算价值损失（支持价值函数裁剪）
        vf_clip_param = batch.get("vf_clip_param", None)
        if vf_clip_param is not None and vf_clip_param > 0:
            old_values = torch.as_tensor(batch.get("old_values"), dtype=torch.float32, device=agent.device)
            vf_loss1 = (values - returns) ** 2
            vf_clipped = old_values + torch.clamp(values - old_values, -vf_clip_param, vf_clip_param)
            vf_loss2 = (vf_clipped - returns) ** 2
            value_loss = torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = ((values - returns) ** 2).mean()
        
        # 计算熵
        entropy = dist.entropy().mean()
        
        # 总损失
        value_coef = float(batch.get("value_coef", self.value_coef))
        entropy_coef = float(batch.get("entropy_coef", self.entropy_coef))
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        # 更新参数
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step(list(agent.parameters()))
        
        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "clip_fraction": float(clip_fraction.item()),
            "total_loss": float(loss.item()),
        }

    def _update_marginal_advantages(
        self,
        agent: Any,
        agent_id: str,
        data: Dict[str, Any],
        marginal_advantages: Dict[str, torch.Tensor],
    ):
        """
        更新边际优势（使用新策略的logprob ratio）
        
        Args:
            agent: 刚刚更新的agent
            agent_id: Agent ID
            data: 原始数据（不是合并后的批次）
            marginal_advantages: 边际优势字典
        """
        with torch.no_grad():
            obs = data["obs"]
            actions = data["actions"]
            old_logprobs = data["logprobs"]
            
            # 转换为tensor
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            else:
                obs = obs.to(agent.device)
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions, dtype=torch.long, device=agent.device)
            else:
                actions = actions.to(agent.device)
            if not isinstance(old_logprobs, torch.Tensor):
                old_logprobs = torch.as_tensor(old_logprobs, dtype=torch.float32, device=agent.device)
            else:
                old_logprobs = old_logprobs.to(agent.device)
            
            # 获取新策略分布（使用更新后的策略）
            dist, _ = agent._forward(obs)
            new_logprobs = dist.log_prob(actions)
            
            # 更新所有agent的边际优势（使用完整的原始数据）
            current_advantage = marginal_advantages[agent_id]
            
            # 确保current_advantage在正确的设备上
            if isinstance(current_advantage, torch.Tensor):
                current_advantage = current_advantage.to(agent.device)
            else:
                current_advantage = torch.as_tensor(current_advantage, dtype=torch.float32, device=agent.device)
            
            # 确保维度匹配
            if len(new_logprobs) != len(current_advantage):
                # 如果维度不匹配，使用较小的维度
                min_len = min(len(new_logprobs), len(current_advantage))
                new_logprobs = new_logprobs[:min_len]
                old_logprobs = old_logprobs[:min_len]
                current_advantage = current_advantage[:min_len]
            
            updated_advantage = update_marginal_advantage(
                new_logprobs,
                old_logprobs,
                current_advantage,
            )
            
            # 更新所有agent的边际优势（使用相同的更新）
            for aid in marginal_advantages.keys():
                # 确保维度匹配
                if len(updated_advantage) == len(marginal_advantages[aid]):
                    marginal_advantages[aid] = updated_advantage.clone()
                else:
                    # 如果维度不匹配，只更新对应的部分
                    min_len = min(len(updated_advantage), len(marginal_advantages[aid]))
                    marginal_advantages[aid][:min_len] = updated_advantage[:min_len].clone()

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

