# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : MAPPO训练器实现
集中训练-分散执行（CTDE）的多智能体PPO
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.ppo.trainer import PPOTrainer

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


class MAPPOTrainer(PPOTrainer):
    """
    MAPPO训练器

    在标准PPOTrainer基础上，实现集中训练-分散执行（CTDE）：
    - 训练时使用全局信息（可选）
    - 执行时使用局部观测
    - 支持共享Critic（集中式价值函数）
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
        初始化MAPPO训练器

        Args:
            agent: AgentManager实例（多Agent）
            env: 环境实例
            config: 训练配置，与PPOTrainer相同，额外包含：
                - use_centralized_critic: 是否使用集中式Critic（默认False）
                - global_obs_dim: 全局观测维度（如果使用集中式Critic）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
        """
        super().__init__(agent, env, config, logger, tracker)

        # MAPPO特定配置
        self.use_centralized_critic = config.get("use_centralized_critic", False)
        self.global_obs_dim = config.get("global_obs_dim", None)

        if self.use_centralized_critic and self.global_obs_dim is None:
            if self.logger:
                if hasattr(self.logger, "logger"):
                    self.logger.logger.warning(
                        "use_centralized_critic=True but global_obs_dim not provided, disabling centralized critic"
                    )
                elif hasattr(self.logger, "warning"):
                    self.logger.warning(
                        "use_centralized_critic=True but global_obs_dim not provided, disabling centralized critic"
                    )
            self.use_centralized_critic = False

    def _train_step_multi_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        多Agent训练步骤（MAPPO版本）

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

                    # 如果使用集中式Critic，添加全局信息
                    if self.use_centralized_critic:
                        # 这里需要从环境获取全局观测（简化处理：使用所有agent的观测拼接）
                        # 实际实现中，应该从环境获取真正的全局状态
                        global_obs = self._get_global_obs(agent_id, indices, processed_data)
                        batch["global_obs"] = global_obs

                    if agent_id not in batches:
                        batches[agent_id] = []
                    batches[agent_id].append(batch)

        # 批量学习（按agent分组）
        for agent_id, agent_batches in batches.items():
            for batch in agent_batches:
                metrics = self.agent.get_agent(agent_id).learn(batch)
                all_metrics.append(metrics)

        return self._aggregate_metrics(all_metrics)

    def _get_global_obs(self, agent_id: str, indices: np.ndarray, processed_data: Dict[str, Any]) -> np.ndarray:
        """
        获取全局观测（用于集中式Critic）

        Args:
            agent_id: Agent ID
            indices: 批次索引
            processed_data: 所有agent的数据

        Returns:
            全局观测
        """
        # 简化实现：拼接所有agent的观测
        # 实际实现中，应该从环境获取真正的全局状态
        global_obs_list = []
        for other_agent_id, other_data in processed_data.items():
            if len(other_data["obs"]) > 0:
                other_obs = other_data["obs"][indices] if len(indices) <= len(other_data["obs"]) else other_data["obs"]
                global_obs_list.append(other_obs)

        if global_obs_list:
            # 拼接所有agent的观测
            global_obs = np.concatenate(global_obs_list, axis=-1)
        else:
            # 回退到当前agent的观测
            global_obs = processed_data[agent_id]["obs"][indices]

        return global_obs

