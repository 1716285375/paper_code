# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE基础训练器
SMPE算法的基础训练器实现，集成VAE、Filter、内在奖励
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer
from algorithms.smpe.policy_agent import SMPEPolicyAgent
from algorithms.smpe.config import SMPEConfig
from algorithms.smpe.core import (
    compute_combined_reward,
    compute_warmup_factor,
    estimate_state_from_observations,
    prepare_actions_onehot_others,
)

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


class SMPETrainer(BaseAlgorithmTrainer):
    """
    SMPE基础训练器
    
    在MAPPO基础上，集成：
    - VAE状态建模（每N步更新VAE）
    - Filter过滤（软更新目标网络）
    - SimHash内在奖励（组合奖励）
    
    注意：不包含自博弈对手池，如需自博弈请使用SMPESelfPlayTrainer
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
        初始化SMPE训练器

        Args:
            agent: AgentManager实例（应该是SMPEPolicyAgent）
            env: 环境实例
            config: 训练配置，额外包含：
                - vae_update_freq: VAE更新频率（每N个训练步，默认每1024环境步）
                - vae_epochs: VAE训练轮数（默认3）
                - filter_update_freq: Filter更新频率（默认每步更新）
                - intrinsic_reward_beta1: 内在奖励权重（默认0.1-0.3）
                - intrinsic_warmup_steps: 内在奖励warm-up步数（默认20000）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
        """
        # 调用基类初始化
        super().__init__(agent, env, config, logger, tracker)
        
        # PPO/MAPPO相关配置（SMPE基于MAPPO）
        self.num_epochs = config.get("num_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)

        # SMPE²特定配置
        self.vae_update_freq = config.get("vae_update_freq", SMPEConfig.DEFAULT_VAE_UPDATE_FREQ)
        self.vae_epochs = config.get("vae_epochs", SMPEConfig.DEFAULT_VAE_EPOCHS)
        self.filter_update_freq = config.get("filter_update_freq", SMPEConfig.DEFAULT_FILTER_UPDATE_FREQ)
        self.intrinsic_reward_beta1 = config.get("intrinsic_reward_beta1", SMPEConfig.DEFAULT_INTRINSIC_REWARD_BETA1)
        self.intrinsic_warmup_steps = config.get("intrinsic_warmup_steps", SMPEConfig.DEFAULT_INTRINSIC_WARMUP_STEPS)
        
        # Checkpoint目录配置
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

        # 训练状态
        self.total_env_steps = 0
        self.vae_update_count = 0
        self.filter_update_count = 0

    def _compute_combined_reward(
        self,
        env_rewards: Dict[str, np.ndarray],
        rollout_data: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        计算组合奖励（使用smpe.py中的工具函数）

        Args:
            env_rewards: 环境奖励 {agent_id: rewards}
            rollout_data: Rollout数据

        Returns:
            组合奖励 {agent_id: combined_rewards}
        """
        combined_rewards = {}

        # Warm-up系数
        warmup_factor = compute_warmup_factor(self.total_env_steps, self.intrinsic_warmup_steps)

        for agent_id, env_reward in env_rewards.items():
            # 获取agent
            agent = self.agent.get_agent(agent_id)

            # 获取内在奖励
            intrinsic_reward = None
            if isinstance(agent, SMPEPolicyAgent) and agent.use_intrinsic:
                agent_data = rollout_data.get(agent_id, {})
                obs = agent_data.get("obs", [])
                if len(obs) > 0:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                    intrinsic_reward = agent.compute_intrinsic_reward(obs_tensor)

            # 使用工具函数计算组合奖励
            combined_reward = compute_combined_reward(
                env_reward=env_reward,
                intrinsic_reward=intrinsic_reward,
                self_play_reward=None,  # 基础训练器不使用自博弈奖励
                beta1=self.intrinsic_reward_beta1,
                beta2=0.0,
                warmup_factor=warmup_factor,
            )

            combined_rewards[agent_id] = combined_reward

        return combined_rewards

    def _update_vae(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """
        更新VAE参数（使用新API）

        Args:
            rollout_data: Rollout数据

        Returns:
            VAE损失字典
        """
        all_vae_losses = {}

        # 获取所有agent的数据
        all_agent_ids = list(rollout_data.keys())
        if len(all_agent_ids) == 0:
            return all_vae_losses

        # 准备批次数据
        batch_size = len(rollout_data[all_agent_ids[0]].get("obs", []))
        if batch_size == 0:
            return all_vae_losses

        # 获取第一个agent作为参考（用于获取维度）
        first_agent_id = all_agent_ids[0]
        first_agent = self.agent.get_agent(first_agent_id)
        
        if not isinstance(first_agent, SMPEPolicyAgent) or not first_agent.use_vae or first_agent.vae is None:
            return all_vae_losses

        # 获取状态维度
        state_dim = first_agent.state_dim
        n_agents = first_agent.n_agents
        n_actions = first_agent.action_dim

        # 准备观测和动作字典
        observations = {aid: np.asarray(rollout_data[aid].get("obs", [])) for aid in all_agent_ids}
        actions = {aid: np.asarray(rollout_data[aid].get("actions", [])) for aid in all_agent_ids}

        # 创建 agent_id 到整数索引的映射（用于 VAE）
        # VAE 需要整数索引（0 到 n_agents-1）来访问 agent_models
        agent_id_to_index = {aid: idx for idx, aid in enumerate(sorted(all_agent_ids))}

        for agent_id in all_agent_ids:
            agent = self.agent.get_agent(agent_id)

            if not isinstance(agent, SMPEPolicyAgent) or not agent.use_vae or agent.vae is None:
                continue

            # 准备VAE训练批次
            data = rollout_data[agent_id]
            obs = data.get("obs", [])
            actions_list = data.get("actions", [])
            if len(obs) == 0 or len(actions_list) == 0:
                continue
            
            # 构建批次
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
            actions_tensor = torch.as_tensor(actions_list, dtype=torch.long, device=agent.device)

            # 获取全局状态（使用工具函数估算）
            states = data.get("states", None)
            if states is None:
                states_array = estimate_state_from_observations(observations, state_dim)
                states_tensor = torch.as_tensor(states_array, dtype=torch.float32, device=agent.device)
            else:
                states_tensor = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
                if states_tensor.shape[-1] != state_dim:
                    if states_tensor.shape[-1] > state_dim:
                        states_tensor = states_tensor[:, :state_dim]
                    else:
                        padding = torch.zeros(
                            states_tensor.shape[0],
                            state_dim - states_tensor.shape[-1],
                            device=agent.device,
                            dtype=states_tensor.dtype
                        )
                        states_tensor = torch.cat([states_tensor, padding], dim=-1)

            # 动作one-hot
            actions_onehot = F.one_hot(actions_tensor, num_classes=n_actions).float()

            # 其他agent的动作one-hot（使用工具函数）
            actions_onehot_others_np = prepare_actions_onehot_others(actions, agent_id, n_actions)
            actions_onehot_others = torch.as_tensor(
                actions_onehot_others_np, 
                dtype=actions_onehot.dtype, 
                device=agent.device
            )

            # 获取 VAE 索引（将 agent_id 映射到整数索引）
            # VAE 需要整数索引（0 到 n_agents-1）来访问 agent_models
            vae_agent_index = agent_id_to_index.get(agent_id, 0)
            
            # 如果 agent.agent_id 是整数且在范围内，优先使用；否则使用映射的索引
            if agent.agent_id is not None and isinstance(agent.agent_id, int) and 0 <= agent.agent_id < n_agents:
                vae_agent_index = agent.agent_id
            else:
                vae_agent_index = agent_id_to_index.get(agent_id, 0)

            # 更新VAE（使用新API）
            vae_losses = agent.update_vae(
                obs=obs_tensor,
                states=states_tensor,
                actions_onehot=actions_onehot,
                actions_onehot_others=actions_onehot_others,
                agent_id=vae_agent_index,  # 使用整数索引
                epochs=self.vae_epochs,
            )
            all_vae_losses[f"{agent_id}_vae"] = vae_losses

        return all_vae_losses

    def _update_filter(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """
        更新Filter参数

        Args:
            rollout_data: Rollout数据

        Returns:
            Filter损失字典
        """
        all_filter_losses = {}

        for agent_id, data in rollout_data.items():
            agent = self.agent.get_agent(agent_id)

            if isinstance(agent, SMPEPolicyAgent) and agent.use_filter and agent.filter is not None:
                # 获取潜在变量z
                obs = data.get("obs", [])
                if len(obs) == 0:
                    continue

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)

                # 获取z（新API不需要last_action）
                z = agent.get_vae_z(obs_tensor)

                # 更新Filter
                filter_losses = agent.update_filter(z)
                all_filter_losses[f"{agent_id}_filter"] = filter_losses

        return all_filter_losses
    
    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练（实现基类的抽象方法）
        
        Args:
            processed_data: 处理后的数据（包含advantages和returns）
        
        Returns:
            训练指标
        """
        # SMPE使用MAPPO风格的训练（多Agent）
        all_metrics = []
        
        # 为每个agent准备批次数据
        batches = {}
        for agent_id, data in processed_data.items():
            obs = data["obs"]
            actions = data["actions"]
            old_logprobs = data["logprobs"]
            advantages = data["advantages"]
            returns = data["returns"]
            
            num_samples = len(obs)
            batch_indices = np.random.permutation(num_samples)
            
            # PPO多轮更新
            for epoch in range(self.num_epochs):
                # 分批训练
                for i in range(0, num_samples, self.batch_size):
                    indices = batch_indices[i : i + self.batch_size]
                    
                    batch = {
                        "obs": obs[indices],
                        "actions": actions[indices],
                        "logprobs": old_logprobs[indices],
                        "advantages": advantages[indices],
                        "returns": returns[indices],
                        "clip_coef": self.clip_coef,
                        "value_coef": self.value_coef,
                        "entropy_coef": self.entropy_coef,
                    }
                    
                    if agent_id not in batches:
                        batches[agent_id] = []
                    batches[agent_id].append(batch)
        
        # 批量学习（按agent分组）
        for agent_id, agent_batches in batches.items():
            for batch in agent_batches:
                metrics = self.agent.get_agent(agent_id).learn(batch)
                all_metrics.append(metrics)
        
        # 聚合指标
        return self._aggregate_metrics(all_metrics)
    
    def _aggregate_metrics(self, metrics_list: list) -> Dict[str, float]:
        """聚合指标列表"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = float(np.mean(values))
        
        return aggregated

    def train(self, num_updates: int) -> None:
        """
        执行SMPE训练

        Args:
            num_updates: 训练更新次数
        """
        self.agent.to_training_mode()

        for update in range(num_updates):
            # 1. 收集数据
            rollout_data = self.rollout_collector.collect()
            self.episode_count += 1

            # 更新环境步数
            for agent_id, data in rollout_data.items():
                self.total_env_steps += len(data.get("obs", []))

            # 2. 统一批后更新（SMPE²时序）
            # 2.1 更新VAE（每N步）
            if self.total_env_steps % self.vae_update_freq == 0:
                vae_losses = self._update_vae(rollout_data)
                self.vae_update_count += 1
                if self.logger and vae_losses:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(f"VAE updated: {vae_losses}")
                    elif hasattr(self.logger, "info"):
                        self.logger.info(f"VAE updated: {vae_losses}")

            # 2.2 更新Filter（每N步或每步）
            if self.filter_update_count % self.filter_update_freq == 0:
                filter_losses = self._update_filter(rollout_data)
                self.filter_update_count += 1

            # 2.3 计算组合奖励
            env_rewards = {
                agent_id: data.get("rewards", np.zeros(len(data.get("obs", []))))
                for agent_id, data in rollout_data.items()
            }
            combined_rewards = self._compute_combined_reward(env_rewards, rollout_data)

            # 更新rollout_data中的奖励
            for agent_id, combined_reward in combined_rewards.items():
                if agent_id in rollout_data:
                    rollout_data[agent_id]["rewards"] = combined_reward

            # 3. 计算优势和回报
            processed_data = self._process_rollout(rollout_data)

            # 4. 更新RL（Actor/Critic）
            metrics = self._train_step(processed_data)

            # 5. 更新计数
            self.update_count += 1

            # 6. 记录日志
            if self.update_count % self.log_freq == 0:
                # 合并VAE和Filter损失到metrics
                if self.vae_update_count > 0 and "vae_losses" in locals():
                    metrics.update(vae_losses)
                if self.filter_update_count > 0 and "filter_losses" in locals():
                    metrics.update(filter_losses)
                self._log_metrics(metrics, rollout_data)

            # 7. 评估
            if self.update_count % self.eval_freq == 0:
                eval_metrics = self.evaluator.evaluate(num_episodes=10)
                eval_log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
                eval_log_dict["update"] = self.update_count

                if self.logger:
                    eval_log_lines = [
                        "=" * 60,
                        f"Evaluation at Update {self.update_count}",
                        "-" * 60,
                    ]
                    for key, value in eval_metrics.items():
                        eval_log_lines.append(f"{key}: {value:.4f}")
                    eval_log_lines.append("=" * 60)

                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info("\n" + "\n".join(eval_log_lines))
                    elif hasattr(self.logger, "info"):
                        self.logger.info("\n" + "\n".join(eval_log_lines))

                # 记录到tracker
                if self.tracker and self.tracker.is_initialized:
                    try:
                        # 过滤掉非标量值（字典、列表等），只保留标量值
                        eval_metrics_for_tracker = {}
                        for k, v in eval_log_dict.items():
                            # 跳过元数据字段
                            if k == "update":
                                continue
                            # 只保留标量值（int, float, numpy标量）
                            if isinstance(v, (int, float, np.number)):
                                eval_metrics_for_tracker[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
                            elif isinstance(v, np.ndarray) and v.size == 1:
                                # 单元素数组，提取标量
                                eval_metrics_for_tracker[k] = float(v.item())
                            # 跳过字典、列表等非标量类型
                        
                        if eval_metrics_for_tracker:
                            self.tracker.log(eval_metrics_for_tracker, step=self.update_count)
                    except Exception as e:
                        if self.logger:
                            if hasattr(self.logger, "logger"):
                                self.logger.logger.warning(f"Failed to log eval metrics: {e}")
                            elif hasattr(self.logger, "warning"):
                                self.logger.warning(f"Failed to log eval metrics: {e}")

            # 8. 保存
            if self.update_count % self.save_freq == 0:
                checkpoint_path = f"{self.checkpoint_dir}/smpe_checkpoint_{self.update_count}.pt"
                self.save(checkpoint_path)

