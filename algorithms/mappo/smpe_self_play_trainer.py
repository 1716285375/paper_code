# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : smpe_self_play_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE²自博弈训练器
结合VAE状态建模、Filter过滤、SimHash内在奖励与自博弈对手池
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import copy
import numpy as np
import torch

from algorithms.mappo.trainer import MAPPOTrainer
from core.agent.opponent_pool import OpponentPool
from core.agent.smpe_agent import SMPEPolicyAgent
from core.agent import AgentManager

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


class SMPESelfPlayTrainer(MAPPOTrainer):
    """
    SMPE²自博弈训练器

    在MAPPO基础上，集成：
    - VAE状态建模（每N步更新VAE）
    - Filter过滤（软更新目标网络）
    - SimHash内在奖励（组合奖励）
    - 对手池（PFSP/Elo匹配）
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
        初始化SMPE²自博弈训练器

        Args:
            agent: AgentManager实例（应该是SMPEPolicyAgent）
            env: 环境实例
            config: 训练配置，额外包含：
                - vae_update_freq: VAE更新频率（每N个训练步，默认每1024环境步）
                - vae_epochs: VAE训练轮数（默认3）
                - filter_update_freq: Filter更新频率（默认每步更新）
                - intrinsic_reward_beta1: 内在奖励权重（默认0.1-0.3）
                - intrinsic_reward_beta2: 自博弈奖励权重（默认0.05-0.2）
                - intrinsic_warmup_steps: 内在奖励warm-up步数（默认20000）
                - opponent_pool_strategy: 对手池策略（"pfsp", "elo", "uniform"）
                - opponent_pool_size: 对手池大小（默认15）
                - snapshot_freq: 快照频率（每N环境步，默认50000）
                - self_play_update_freq: 对手更新频率（每N个更新，默认10）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
            main_team: 主团队名称
            opponent_team: 对手团队名称
        """
        super().__init__(agent, env, config, logger, tracker)

        self.main_team = main_team
        self.opponent_team = opponent_team

        # SMPE²特定配置
        self.vae_update_freq = config.get("vae_update_freq", 1024)  # 每1024环境步更新VAE
        self.vae_epochs = config.get("vae_epochs", 3)
        self.filter_update_freq = config.get("filter_update_freq", 1)
        self.intrinsic_reward_beta1 = config.get("intrinsic_reward_beta1", 0.1)
        self.intrinsic_reward_beta2 = config.get("intrinsic_reward_beta2", 0.05)
        self.intrinsic_warmup_steps = config.get("intrinsic_warmup_steps", 20000)

        # 对手池配置
        opponent_pool_config = config.get("opponent_pool", {})
        self.opponent_pool_strategy = opponent_pool_config.get("strategy", "pfsp")
        self.opponent_pool_size = opponent_pool_config.get("size", 15)
        self.snapshot_freq = config.get("snapshot_freq", 50000)  # 每50000环境步快照

        # 自博弈配置
        self.self_play_update_freq = config.get("self_play_update_freq", 10)

        # 创建对手池
        self.opponent_pool = OpponentPool(
            max_size=self.opponent_pool_size,
            strategy=self.opponent_pool_strategy,
            device=self.agent.get_agent(list(self.agent.agent_ids)[0]).device if hasattr(self.agent, "agent_ids") else "cpu",
        )

        # 训练状态
        self.total_env_steps = 0
        self.vae_update_count = 0
        self.filter_update_count = 0

        # 初始化对手池（添加初始对手策略）
        self._initialize_opponent_pool()

    def _initialize_opponent_pool(self) -> None:
        """初始化对手池（添加初始对手策略）"""
        # 获取对手团队的策略状态
        opponent_agents = self.agent.get_group_members(self.opponent_team)
        if opponent_agents:
            sample_agent_id = opponent_agents[0]
            opponent_state = self.agent.get_agent(sample_agent_id).state_dict()
            self.opponent_pool.add_policy(opponent_state)

    def _update_opponent_from_pool(self) -> None:
        """从对手池采样并更新对手策略"""
        opponent_state = self.opponent_pool.sample_opponent()
        if opponent_state is not None:
            # 更新对手团队的所有agent
            opponent_agents = self.agent.get_group_members(self.opponent_team)
            for agent_id in opponent_agents:
                agent = self.agent.get_agent(agent_id)
                agent.load_state_dict(opponent_state)

    def _compute_combined_reward(
        self,
        env_rewards: Dict[str, np.ndarray],
        rollout_data: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        计算组合奖励

        r_total = r_env + β1 * r_intrinsic + β2 * r_selfplay

        Args:
            env_rewards: 环境奖励 {agent_id: rewards}
            rollout_data: Rollout数据

        Returns:
            组合奖励 {agent_id: combined_rewards}
        """
        combined_rewards = {}

        # Warm-up系数（线性增长）
        warmup_factor = min(1.0, self.total_env_steps / self.intrinsic_warmup_steps)
        beta1 = self.intrinsic_reward_beta1 * warmup_factor
        beta2 = self.intrinsic_reward_beta2 * warmup_factor

        for agent_id, env_reward in env_rewards.items():
            combined_reward = env_reward.copy()

            # 获取agent
            agent = self.agent.get_agent(agent_id)

            # 1. 内在奖励（SimHash）
            if isinstance(agent, SMPEPolicyAgent) and agent.use_intrinsic:
                # 从rollout数据中提取obs和z
                agent_data = rollout_data.get(agent_id, {})
                obs = agent_data.get("obs", [])
                if len(obs) > 0:
                    # 计算内在奖励
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                    intrinsic_reward = agent.compute_intrinsic_reward(obs_tensor)

                    # 转换为numpy并添加到组合奖励
                    intrinsic_reward_np = intrinsic_reward.detach().cpu().numpy()
                    if intrinsic_reward_np.ndim == 0:
                        # 标量，扩展到所有步
                        intrinsic_reward_np = np.full(len(env_reward), intrinsic_reward_np.item())
                    elif len(intrinsic_reward_np) != len(env_reward):
                        # 长度不匹配，取前N个
                        intrinsic_reward_np = intrinsic_reward_np[: len(env_reward)]

                    combined_reward = combined_reward + beta1 * intrinsic_reward_np

            # 2. 自博弈奖励（简化：使用胜率或Elo差的Sigmoid）
            # 这里简化处理：如果episode结束时主团队获胜，给予奖励
            # 实际实现中应该基于Elo或更复杂的评估
            # 暂时跳过，在evaluation时更新Elo

            combined_rewards[agent_id] = combined_reward

        return combined_rewards

    def _update_vae(self, rollout_data: Dict[str, Any]) -> Dict[str, float]:
        """
        更新VAE参数

        Args:
            rollout_data: Rollout数据

        Returns:
            VAE损失字典
        """
        all_vae_losses = {}

        for agent_id, data in rollout_data.items():
            agent = self.agent.get_agent(agent_id)

            if isinstance(agent, SMPEPolicyAgent) and agent.use_vae and agent.vae is not None:
                # 准备VAE训练批次
                obs = data.get("obs", [])
                actions = data.get("actions", [])
                if len(obs) == 0 or len(actions) == 0:
                    continue

                # 构建批次
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=agent.device)

                # 上一次动作（使用actions的前一个，或默认0）
                last_actions = np.roll(actions, 1)
                last_actions[0] = 0  # 第一步使用默认动作
                last_actions_tensor = torch.as_tensor(last_actions, dtype=torch.long, device=agent.device)

                # Agent ID one-hot
                agent_id_batch = agent.agent_id_onehot.unsqueeze(0).expand(len(obs), -1)

                vae_batch = {
                    "obs": obs_tensor,
                    "last_action": last_actions_tensor,
                    "agent_id": agent_id_batch,
                    "target_obs": obs_tensor,  # 目标观测（自编码）
                    "target_action": actions_tensor,  # 目标动作
                }

                # 更新VAE
                vae_losses = agent.update_vae(vae_batch, epochs=self.vae_epochs)
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
                actions = data.get("actions", [])
                if len(obs) == 0:
                    continue

                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
                if len(actions) > 0:
                    last_actions = np.roll(actions, 1)
                    last_actions[0] = 0
                    last_actions_tensor = torch.as_tensor(last_actions, dtype=torch.long, device=agent.device)
                else:
                    last_actions_tensor = torch.zeros(len(obs), dtype=torch.long, device=agent.device)

                # 获取z
                z = agent.get_vae_z(obs_tensor, last_actions_tensor)

                # 更新Filter
                filter_losses = agent.update_filter(z)
                all_filter_losses[f"{agent_id}_filter"] = filter_losses

        return all_filter_losses

    def train(self, num_updates: int) -> None:
        """
        执行SMPE²自博弈训练

        Args:
            num_updates: 训练更新次数
        """
        self.agent.to_training_mode()

        for update in range(num_updates):
            # 1. 从对手池采样对手
            if update % self.self_play_update_freq == 0:
                self._update_opponent_from_pool()

            # 2. 收集数据
            rollout_data = self.rollout_collector.collect()
            self.episode_count += 1

            # 更新环境步数
            if isinstance(rollout_data, dict) and "obs" in rollout_data:
                self.total_env_steps += len(rollout_data["obs"])
            else:
                for agent_id, data in rollout_data.items():
                    self.total_env_steps += len(data.get("obs", []))

            # 3. 统一批后更新（SMPE²时序）
            # 3.1 更新VAE（每N步）
            if self.total_env_steps % self.vae_update_freq == 0:
                vae_losses = self._update_vae(rollout_data)
                self.vae_update_count += 1
                if self.logger and vae_losses:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(f"VAE updated: {vae_losses}")
                    elif hasattr(self.logger, "info"):
                        self.logger.info(f"VAE updated: {vae_losses}")

            # 3.2 更新Filter（每N步或每步）
            if self.filter_update_count % self.filter_update_freq == 0:
                filter_losses = self._update_filter(rollout_data)
                self.filter_update_count += 1

            # 3.3 计算组合奖励
            env_rewards = {
                agent_id: data.get("rewards", np.zeros(len(data.get("obs", []))))
                for agent_id, data in rollout_data.items()
            }
            combined_rewards = self._compute_combined_reward(env_rewards, rollout_data)

            # 更新rollout_data中的奖励
            for agent_id, combined_reward in combined_rewards.items():
                if agent_id in rollout_data:
                    rollout_data[agent_id]["rewards"] = combined_reward

            # 4. 计算优势和回报
            processed_data = self._process_rollout(rollout_data)

            # 5. 更新RL（Actor/Critic）
            metrics = self._train_step(processed_data)

            # 6. 更新计数
            self.update_count += 1

            # 7. 记录日志
            if self.update_count % self.log_freq == 0:
                # 合并VAE和Filter损失到metrics
                if self.vae_update_count > 0 and "vae_losses" in locals():
                    metrics.update(vae_losses)
                if self.filter_update_count > 0 and "filter_losses" in locals():
                    metrics.update(filter_losses)
                self._log_metrics(metrics, rollout_data)

            # 8. 快照对手策略到池中
            if self.total_env_steps % self.snapshot_freq == 0:
                opponent_agents = self.agent.get_group_members(self.opponent_team)
                if opponent_agents:
                    sample_agent_id = opponent_agents[0]
                    opponent_state = self.agent.get_agent(sample_agent_id).state_dict()
                    pool_index = self.opponent_pool.add_policy(opponent_state)
                    if self.logger:
                        log_msg = f"Added opponent snapshot to pool (index={pool_index}, pool_size={self.opponent_pool.get_size()})"
                        if hasattr(self.logger, "logger"):
                            self.logger.logger.info(log_msg)
                        elif hasattr(self.logger, "info"):
                            self.logger.info(log_msg)

            # 9. 评估
            if self.update_count % self.eval_freq == 0:
                # 使用MultiAgentEvaluator进行评估（提供更多自博弈相关指标）
                from core.trainer.multi_agent_evaluator import MultiAgentEvaluator
                
                # 获取团队信息
                team_names = {}
                if hasattr(self.agent, "get_all_groups"):
                    all_groups = self.agent.get_all_groups()
                    for group_name in all_groups:
                        team_names[group_name] = self.agent.get_group_members(group_name)
                
                eval_evaluator = MultiAgentEvaluator(
                    agent=self.agent,
                    env=self.env,
                    max_steps_per_episode=self.max_steps_per_episode,
                    is_multi_agent=True,
                    team_names=team_names,
                )
                eval_metrics = eval_evaluator.evaluate(num_episodes=10)
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
                        eval_metrics_for_tracker = {
                            k: v for k, v in eval_log_dict.items() if k != "update"
                        }
                        self.tracker.log(eval_metrics_for_tracker, step=self.update_count)
                    except Exception as e:
                        if self.logger:
                            if hasattr(self.logger, "logger"):
                                self.logger.logger.warning(f"Failed to log eval metrics: {e}")
                            elif hasattr(self.logger, "warning"):
                                self.logger.warning(f"Failed to log eval metrics: {e}")

            # 10. 保存
            if self.update_count % self.save_freq == 0:
                self.save(f"checkpoints/smpe_selfplay_checkpoint_{self.update_count}.pt")

