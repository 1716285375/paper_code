# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : penalty_self_play_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : PPO-Penalty自博弈训练器
实现PPO-Penalty算法的自博弈训练
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.ppo.penalty_trainer import PPOPenaltyTrainer
from algorithms.ppo.self_play_trainer import PolicyPool

# Logger是可选的
try:
    from common.utils.logging import LoggerManager

    Logger = LoggerManager
except ImportError:
    Logger = None


class SelfPlayPPOPenaltyTrainer(PPOPenaltyTrainer):
    """
    PPO-Penalty自博弈训练器

    在SelfPlayPPOTrainer的基础上，使用PPO-Penalty变体。
    主要特点：
    - 使用PPO-Penalty算法（KL惩罚而非clipping）
    - 支持策略池和对手策略更新
    - 跟踪KL散度和beta值的变化
    """

    def __init__(
        self,
        agent,
        env,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
        main_team: str = "team_red",
        opponent_team: str = "team_blue",
    ):
        """
        初始化PPO-Penalty自博弈训练器

        Args:
            agent: AgentManager实例，管理所有Agent（应该是PPOPenaltyAgent）
            env: 环境实例
            config: 训练配置，与SelfPlayPPOTrainer相同
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
            main_team: 主团队名称
            opponent_team: 对手团队名称
        """
        # 先调用PPOPenaltyTrainer的初始化
        super().__init__(agent, env, config, logger, tracker=tracker)

        # 初始化自博弈相关的属性
        self.main_team = main_team
        self.opponent_team = opponent_team

        # 自博弈配置
        self.self_play_update_freq = config.get("self_play_update_freq", 10)
        self.self_play_mode = config.get("self_play_mode", "copy")
        self.use_policy_pool = config.get("use_policy_pool", False)

        # 策略池
        if self.use_policy_pool:
            pool_size = config.get("policy_pool_size", 10)
            self.policy_pool = PolicyPool(max_size=pool_size)
        else:
            self.policy_pool = None

        # 验证AgentManager是否有团队分组
        from core.agent import AgentManager

        if not isinstance(agent, AgentManager):
            raise ValueError("SelfPlayPPOPenaltyTrainer requires AgentManager")

        # 检查团队是否存在
        all_groups = agent.get_all_groups()
        if main_team not in all_groups:
            raise ValueError(f"Main team '{main_team}' not found in agent groups: {all_groups}")
        if opponent_team not in all_groups:
            raise ValueError(
                f"Opponent team '{opponent_team}' not found in agent groups: {all_groups}"
            )

        # 替换为多智能体评估器（提供更多自博弈和元学习相关指标）
        from core.trainer.multi_agent_evaluator import MultiAgentEvaluator

        # 获取团队信息
        team_names = {}
        if hasattr(agent, "get_all_groups"):
            all_groups = agent.get_all_groups()
            for group_name in all_groups:
                team_names[group_name] = agent.get_group_members(group_name)

        self.evaluator = MultiAgentEvaluator(
            agent=agent,
            env=env,
            max_steps_per_episode=self.max_steps_per_episode,
            is_multi_agent=self.is_multi_agent,
            team_names=team_names,
        )

        # 初始化时同步策略（对手使用主策略的初始版本）
        self._sync_opponent_policy()

    def train(self, num_updates: int) -> None:
        """
        执行自博弈训练

        Args:
            num_updates: 训练更新次数
        """
        self.agent.to_training_mode()

        for update in range(num_updates):
            # 收集数据
            rollout_data = self.rollout_collector.collect()
            self.episode_count += 1

            # 计算步数
            if isinstance(rollout_data, dict) and "obs" in rollout_data:
                self.step_count += len(rollout_data["obs"])
            else:
                for agent_id, data in rollout_data.items():
                    if isinstance(data, dict) and "obs" in data:
                        self.step_count += len(data["obs"])

            # 计算优势和回报
            processed_data = self._process_rollout(rollout_data)

            # 批量训练（只训练主团队）
            metrics = self._train_step_self_play(processed_data)

            # 更新计数
            self.update_count += 1

            # 记录日志
            if self.update_count % self.log_freq == 0:
                self._log_metrics(metrics, rollout_data)

            # 自博弈更新：更新对手策略
            if self.update_count % self.self_play_update_freq == 0:
                self._update_opponent_policy()
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(
                            f"Updated opponent policy at step {self.update_count}"
                        )
                    elif hasattr(self.logger, "info"):
                        self.logger.info(f"Updated opponent policy at step {self.update_count}")
                    else:
                        print(f"Updated opponent policy at step {self.update_count}")

            # 评估
            if self.update_count % self.eval_freq == 0:
                self._sync_opponent_policy()
                eval_metrics = self.evaluator.evaluate(num_episodes=10)
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(
                            f"Evaluation at update {self.update_count}: {eval_metrics}"
                        )
                    elif hasattr(self.logger, "info"):
                        self.logger.info(
                            f"Evaluation at update {self.update_count}: {eval_metrics}"
                        )
                    else:
                        print(f"Evaluation at update {self.update_count}: {eval_metrics}")

                if self.tracker:
                    for key, value in eval_metrics.items():
                        if isinstance(value, (int, float)):
                            self.tracker.log({f"eval/{key}": float(value)}, step=self.update_count)

            # 保存检查点
            if self.update_count % self.save_freq == 0:
                self.save(f"checkpoints/penalty_selfplay_checkpoint_{self.update_count}.pt")

    def _train_step_self_play(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        自博弈训练步骤（只训练主团队）

        Args:
            processed_data: 处理后的数据

        Returns:
            训练指标（包含KL散度和beta）
        """
        # 只训练主团队的Agent
        main_agent_ids = self.agent.get_group_members(self.main_team)

        batches = {}
        all_metrics = []

        for agent_id in main_agent_ids:
            if agent_id in processed_data:
                data = processed_data[agent_id]
                obs = data["obs"]
                actions = data["actions"]
                old_logprobs = data["logprobs"]
                advantages = data["advantages"]
                returns = data["returns"]

                num_samples = len(obs)

                # PPO多轮更新
                for epoch in range(self.num_epochs):
                    indices = np.random.permutation(num_samples)

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

                        batches[agent_id] = batch

                # 训练主团队
                if batches:
                    metrics_dict = {}
                    for aid, batch in batches.items():
                        agent = self.agent.get_agent(aid)
                        metrics_dict[aid] = agent.learn(batch)
                    all_metrics.extend(metrics_dict.values())

        return self._aggregate_metrics(all_metrics)

    def _sync_opponent_policy(self) -> None:
        """同步对手策略（从主策略复制或从策略池采样）"""
        main_agent = self.agent.get_group_agent(self.main_team)
        opponent_agent = self.agent.get_group_agent(self.opponent_team)

        if main_agent is None or opponent_agent is None:
            return

        # 从主策略复制到对手
        opponent_state = main_agent.state_dict()
        opponent_agent.load_state_dict(opponent_state)

    def _update_opponent_policy(self) -> None:
        """
        更新对手策略

        根据配置的模式：
        - "copy": 直接复制主策略
        - "pool": 从策略池中采样（如果有）
        """
        main_agent = self.agent.get_group_agent(self.main_team)
        opponent_agent = self.agent.get_group_agent(self.opponent_team)

        if main_agent is None or opponent_agent is None:
            return

        if self.self_play_mode == "pool" and self.policy_pool is not None:
            # 从策略池采样
            sampled_state = self.policy_pool.sample()
            if sampled_state is not None:
                opponent_agent.load_state_dict(sampled_state)
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info("Updated opponent from policy pool")
                    elif hasattr(self.logger, "info"):
                        self.logger.info("Updated opponent from policy pool")
        else:
            # 直接复制
            self._sync_opponent_policy()

        # 将当前主策略添加到策略池
        if self.policy_pool is not None:
            main_state = main_agent.state_dict()
            self.policy_pool.add_policy(main_state, metadata={"update": self.update_count})

