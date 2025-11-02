# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : penalty_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : PPO-Penalty训练器实现
实现PPO-Penalty变体的训练循环
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


class PPOPenaltyTrainer(PPOTrainer):
    """
    PPO-Penalty训练器

    在标准PPOTrainer的基础上，支持PPO-Penalty变体。
    主要区别：
    - Agent使用PPOPenaltyAgent（使用KL惩罚而非clipping）
    - 训练过程中跟踪和记录KL散度和beta值
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
        初始化PPO-Penalty训练器

        Args:
            agent: Agent实例（单个Agent）或AgentManager（多Agent），应该是PPOPenaltyAgent
            env: 环境实例
            config: 训练配置，与PPOTrainer相同，但：
                - 不需要clip_coef（PPO-Penalty不使用clipping）
                - 可以包含kl_penalty配置（传递给Agent）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
        """
        super().__init__(agent, env, config, logger, tracker)

        # PPO-Penalty不使用clip_coef
        # 如果配置中有clip_coef，发出警告但不使用
        if hasattr(self, "clip_coef") and self.clip_coef is not None:
            if self.logger:
                if hasattr(self.logger, "logger"):
                    self.logger.logger.warning(
                        "PPO-Penalty does not use clip_coef, ignoring it"
                    )
                elif hasattr(self.logger, "warning"):
                    self.logger.warning("PPO-Penalty does not use clip_coef, ignoring it")
                else:
                    print("Warning: PPO-Penalty does not use clip_coef, ignoring it")

    def _train_step_single_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        单Agent训练步骤（PPO-Penalty版本）

        Args:
            processed_data: 处理后的数据

        Returns:
            训练指标（包含KL散度和beta）
        """
        all_metrics = []

        obs = processed_data["obs"]
        actions = processed_data["actions"]
        old_logprobs = processed_data["logprobs"]
        advantages = processed_data["advantages"]
        returns = processed_data["returns"]

        num_samples = len(obs)

        # PPO多轮更新
        for epoch in range(self.num_epochs):
            # 随机打乱
            indices = np.random.permutation(num_samples)

            # 分批训练
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]

                batch = {
                    "obs": obs[batch_indices],
                    "actions": actions[batch_indices],
                    "logprobs": old_logprobs[batch_indices],
                    "advantages": advantages[batch_indices],
                    "returns": returns[batch_indices],
                    # PPO-Penalty不需要clip_coef
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                }

                metrics = self.agent.learn(batch)
                all_metrics.append(metrics)

        # 聚合指标
        return self._aggregate_metrics(all_metrics)

    def _train_step_multi_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        多Agent训练步骤（PPO-Penalty版本）

        Args:
            processed_data: 处理后的数据（按agent_id组织）

        Returns:
            训练指标（包含KL散度和beta）
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

            # 只取第一批次（简化处理，也可以多批次）
            indices = batch_indices[: min(self.batch_size, num_samples)]

            batches[agent_id] = {
                "obs": obs[indices],
                "actions": actions[indices],
                "logprobs": old_logprobs[indices],
                "advantages": advantages[indices],
                "returns": returns[indices],
                # PPO-Penalty不需要clip_coef
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
            }

        # 批量学习
        metrics_dict = self.agent.learn(batches)

        # 聚合所有agent的指标
        all_metrics = list(metrics_dict.values())
        return self._aggregate_metrics(all_metrics)

