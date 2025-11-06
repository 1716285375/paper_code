# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : self_play_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : PPO自博弈训练器
实现PPO算法的自博弈训练，适用于对抗性多Agent环境
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from algorithms.ppo.trainers.base_trainer import PPOTrainer
from core.agent import AgentManager
from core.trainer.multi_agent_evaluator import MultiAgentEvaluator

# Logger是可选的
try:
    from common.utils.logging import LoggerManager

    Logger = LoggerManager
except ImportError:
    Logger = None


# 使用 core 中的 OpponentPool
from core.agent.opponent_pool import OpponentPool


class SelfPlayPPOTrainer(PPOTrainer):
    """
    PPO自博弈训练器

    在对抗性多Agent环境中实现自博弈训练：
    - 两个团队（主团队和对手团队）使用相同的策略进行训练
    - 定期更新对手策略（从当前策略或策略池中采样）
    - 支持多种自博弈策略：固定更新、策略池采样等
    """

    def __init__(
        self,
        agent: AgentManager,
        env,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
        main_team: str = "team_red",
        opponent_team: str = "team_blue",
    ):
        """
        初始化自博弈训练器

        Args:
            agent: AgentManager实例，管理所有Agent
            env: 环境实例
            config: 训练配置，额外包括：
                - self_play_update_freq: 自博弈更新频率（每N个更新更新一次对手策略）
                - self_play_mode: 自博弈模式
                    - "copy": 直接将主策略复制给对手
                    - "pool": 从策略池中采样对手
                - use_policy_pool: 是否使用策略池
                - policy_pool_size: 策略池大小（如果启用）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选，用于wandb/tensorboard）
            main_team: 主团队名称（用于训练）
            opponent_team: 对手团队名称（用于自博弈）
        """
        super().__init__(agent, env, config, logger, tracker=tracker)

        self.main_team = main_team
        self.opponent_team = opponent_team

        # 自博弈配置
        self.self_play_update_freq = config.get("self_play_update_freq", 10)
        self.self_play_mode = config.get("self_play_mode", "copy")
        self.use_policy_pool = config.get("use_policy_pool", False)

        # 策略池
        if self.use_policy_pool:
            pool_size = config.get("policy_pool_size", 10)
            strategy = config.get("opponent_pool_strategy", "pfsp")
            elo_temperature = config.get("elo_temperature", 1.0)
            pfsp_temperature = config.get("pfsp_temperature", 1.0)
            device = getattr(self.agent.get_agent(list(self.agent.agent_ids)[0]), "device", "cpu") if hasattr(self.agent, "agent_ids") else "cpu"
            
            self.policy_pool = OpponentPool(
                max_size=pool_size,
                strategy=strategy,
                elo_temperature=elo_temperature,
                pfsp_temperature=pfsp_temperature,
                device=device,
            )
        else:
            self.policy_pool = None

        # 验证AgentManager是否有团队分组
        if not isinstance(agent, AgentManager):
            raise ValueError("SelfPlayPPOTrainer requires AgentManager")

        # 检查团队是否存在
        all_groups = agent.get_all_groups()
        if main_team not in all_groups:
            raise ValueError(f"Main team '{main_team}' not found in agent groups: {all_groups}")
        if opponent_team not in all_groups:
            raise ValueError(
                f"Opponent team '{opponent_team}' not found in agent groups: {all_groups}"
            )

        # 替换为多智能体评估器（提供更多自博弈和元学习相关指标）
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
        
        # 输出训练开始信息
        if self.logger:
            log_msg = f"开始训练，共 {num_updates} 个更新..."
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(log_msg)
            elif hasattr(self.logger, "info"):
                self.logger.info(log_msg)
            else:
                print(log_msg)
        else:
            print(f"开始训练，共 {num_updates} 个更新...")

        for update in range(num_updates):
            # 输出进度（每10个更新或第一个更新）
            if update == 0 or (update + 1) % 10 == 0:
                progress_msg = f"更新进度: {update + 1}/{num_updates} (Episode {self.episode_count + 1})"
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(progress_msg)
                    elif hasattr(self.logger, "info"):
                        self.logger.info(progress_msg)
                    else:
                        print(progress_msg)
                else:
                    print(progress_msg)
            
            # 收集数据
            rollout_data = self.rollout_collector.collect()
            self.episode_count += 1
            
            # 输出收集完成的提示
            if update == 0:
                print(f"✅ 第一个episode收集完成（{self.episode_count}个episodes）", flush=True)

            # 计算步数（多Agent情况下需要累加所有agent的步数）
            if isinstance(rollout_data, dict) and "obs" in rollout_data:
                # 单Agent格式
                self.step_count += len(rollout_data["obs"])
            else:
                # 多Agent格式：累加所有agent的步数
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

            # 评估（使用扩展的多智能体评估器）
            if self.update_count % self.eval_freq == 0:
                # 评估前同步对手策略（确保评估使用当前策略）
                # 这样评估结果更准确反映当前训练状态
                self._sync_opponent_policy()

                # 使用扩展评估方法，计算多样性和协作指标
                # 增加episode数量以提高评估准确性
                eval_metrics = self.evaluator.evaluate(
                    num_episodes=10,  # 从5增加到10，提高评估准确性
                    compute_diversity=True,  # 计算策略多样性（自博弈）
                    compute_cooperation=True,  # 计算协作指标（多智能体）
                )

                # 添加eval前缀以便区分
                eval_log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}

                # 使用logger输出评估结果
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

                # 记录到tracker（wandb/tensorboard）
                if self.tracker and self.tracker.is_initialized:
                    try:
                        # 过滤掉非标量值（字典、列表等），只保留标量值
                        eval_metrics_for_tracker = {}
                        for k, v in eval_log_dict.items():
                            # 跳过元数据字段
                            if k == "eval/update":
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
                                self.logger.logger.warning(
                                    f"Failed to log eval metrics to tracker: {e}"
                                )
                            elif hasattr(self.logger, "warning"):
                                self.logger.warning(f"Failed to log eval metrics to tracker: {e}")

            # 保存
            if self.update_count % self.save_freq == 0:
                self.save(f"checkpoints/selfplay_checkpoint_{self.update_count}.pt")

    def _train_step_self_play(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        自博弈训练步骤（只训练主团队）

        Args:
            processed_data: 处理后的数据

        Returns:
            训练指标
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
                            "value_coef": self.value_coef,
                            "entropy_coef": self.entropy_coef,
                        }
                        # PPO-Penalty模式不使用clip_coef
                        if not self.use_penalty:
                            batch["clip_coef"] = self.clip_coef

                        batches[agent_id] = batch

                # 训练主团队
                if batches:
                    metrics_dict = {}
                    for aid, batch in batches.items():
                        agent = self.agent.get_agent(aid)
                        metrics_dict[aid] = agent.learn(batch)
                    all_metrics.extend(metrics_dict.values())

        return self._aggregate_metrics(all_metrics)

    def _update_opponent_policy(self) -> None:
        """
        更新对手策略

        根据自博弈模式，将主策略的状态同步到对手策略。
        """
        if self.self_play_mode == "copy":
            # 直接复制主策略
            self._sync_opponent_policy()

            # 如果使用策略池，添加当前策略到池中
            if self.policy_pool:
                main_agent = self.agent.get_agent(self.agent.get_group_members(self.main_team)[0])
                state_dict = main_agent.state_dict()
                self.policy_pool.add_policy(state_dict)  # core 的 OpponentPool 不接受 metadata

        elif self.self_play_mode == "pool":
            # 从策略池中采样
            if self.policy_pool and self.policy_pool.get_size() > 0:
                sampled_policy = self.policy_pool.sample_opponent()  # 使用 core 的 API
                if sampled_policy:
                    self._load_policy_to_opponent(sampled_policy)

            # 将当前策略添加到池中
            if self.policy_pool:
                main_agent = self.agent.get_agent(self.agent.get_group_members(self.main_team)[0])
                state_dict = main_agent.state_dict()
                self.policy_pool.add_policy(state_dict)  # core 的 OpponentPool 不接受 metadata

        else:
            raise ValueError(f"Unknown self-play mode: {self.self_play_mode}")

    def _sync_opponent_policy(self) -> None:
        """同步主策略到对手团队"""
        main_agent_ids = self.agent.get_group_members(self.main_team)
        opponent_agent_ids = self.agent.get_group_members(self.opponent_team)

        if len(main_agent_ids) == 0 or len(opponent_agent_ids) == 0:
            return

        # 获取主团队的策略状态
        main_agent = self.agent.get_agent(main_agent_ids[0])
        main_state_dict = main_agent.state_dict()

        # 加载到所有对手Agent
        for opponent_id in opponent_agent_ids:
            opponent_agent = self.agent.get_agent(opponent_id)
            opponent_agent.load_state_dict(main_state_dict)

    def _load_policy_to_opponent(self, state_dict: Dict[str, Any]) -> None:
        """
        将策略状态加载到对手团队

        Args:
            state_dict: 策略状态字典
        """
        opponent_agent_ids = self.agent.get_group_members(self.opponent_team)

        for opponent_id in opponent_agent_ids:
            opponent_agent = self.agent.get_agent(opponent_id)
            opponent_agent.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        """保存训练器状态（包括策略池）"""
        from pathlib import Path

        # 确保目录存在
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "agent": self.agent.state_dict(),
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "config": self.config,
        }

        # 保存策略池
        if self.policy_pool:
            # core 的 OpponentPool 使用 deque，需要转换为列表
            state["policy_pool"] = {
                "policies": list(self.policy_pool.policies),
                "max_size": self.policy_pool.max_size,
                "strategy": self.policy_pool.strategy,
                "elo": self.policy_pool.elo.ratings,
                "win_rates": self.policy_pool.win_rates,
            }

        torch.save(state, path)
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Saved checkpoint to {path}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Saved checkpoint to {path}")
            else:
                print(f"Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """加载训练器状态（包括策略池）"""
        state = torch.load(path)
        self.agent.load_state_dict(state["agent"])
        self.update_count = state.get("update_count", 0)
        self.episode_count = state.get("episode_count", 0)
        self.step_count = state.get("step_count", 0)

        # 加载策略池
        if self.policy_pool and "policy_pool" in state:
            pool_data = state["policy_pool"]
            # 恢复策略池（core 的 OpponentPool 使用 deque）
            from collections import deque
            policies_list = pool_data.get("policies", [])
            self.policy_pool.policies = deque(policies_list, maxlen=self.policy_pool.max_size)
            # 恢复 Elo 和胜率统计
            if "elo" in pool_data:
                self.policy_pool.elo.ratings = pool_data["elo"]
            if "win_rates" in pool_data:
                self.policy_pool.win_rates = pool_data["win_rates"]

        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Loaded checkpoint from {path}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Loaded checkpoint from {path}")
            else:
                print(f"Loaded checkpoint from {path}")
