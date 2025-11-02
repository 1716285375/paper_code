# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : multi_agent_evaluator.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 多智能体评估器扩展
为自博弈和元学习场景提供丰富的评估指标
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from core.agent import AgentManager
from core.trainer.evaluator import Evaluator


class MultiAgentEvaluator(Evaluator):
    """
    多智能体评估器扩展

    在基础Evaluator的基础上，为自博弈和元学习场景提供更多评估指标：
    - 团队性能指标
    - 个体性能指标
    - 协作指标
    - 对抗性能指标
    - 策略多样性指标（自博弈）
    """

    def __init__(
        self,
        agent: AgentManager,
        env: Any,
        max_steps_per_episode: int = 1000,
        is_multi_agent: bool = True,
        team_names: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初始化多智能体评估器

        Args:
            agent: AgentManager实例
            env: 环境实例
            max_steps_per_episode: 每个episode的最大步数
            is_multi_agent: 是否为多Agent环境（应为True）
            team_names: 团队名称到agent ID列表的映射，例如：
                {"team_red": ["red_0", "red_1"], "team_blue": ["blue_0", "blue_1"]}
        """
        super().__init__(agent, env, max_steps_per_episode, is_multi_agent)

        if not isinstance(agent, AgentManager):
            raise ValueError("MultiAgentEvaluator requires AgentManager")

        self.agent_manager = agent
        self.team_names = team_names or {}

    def evaluate(
        self,
        num_episodes: int,
        compute_diversity: bool = True,
        compute_cooperation: bool = True,
    ) -> Dict[str, float]:
        """
        评估多智能体性能

        Args:
            num_episodes: 评估的episode数量
            compute_diversity: 是否计算策略多样性指标（自博弈场景）
            compute_cooperation: 是否计算协作指标

        Returns:
            评估指标字典，包含：
            - 基础指标（继承自Evaluator）
            - 团队指标：每个团队的平均奖励、胜率等
            - 个体指标：每个agent的平均奖励
            - 协作指标：团队协作效率、动作一致性等
            - 多样性指标：策略KL散度、动作熵等（自博弈）
            - 对抗指标：团队间的对抗性能差异
        """
        self.agent.to_eval_mode()

        # 基础指标
        episode_rewards = []
        episode_lengths = []

        # 团队指标
        team_rewards = {team: [] for team in self.team_names.keys()}
        team_episode_rewards = {team: [] for team in self.team_names.keys()}

        # 个体指标
        agent_rewards = {}  # {agent_id: [rewards...]}
        agent_episode_rewards = {}  # {agent_id: total_reward_per_episode}

        # 协作指标
        action_consistency = []  # 团队内动作一致性
        coordination_scores = []  # 协作分数

        # 多样性指标（自博弈）
        action_entropies = []  # 动作分布熵
        policy_differences = []  # 策略差异（如果有多个策略）

        # 添加随机性：每次评估使用不同的随机种子（如果环境支持）
        import random

        for episode_idx in range(num_episodes):
            # 为每个episode设置不同的随机种子（增加评估多样性）
            # 这确保即使环境是确定性的，不同episode也会有变化
            episode_seed = random.randint(0, 2**31 - 1)
            np.random.seed(episode_seed)
            random.seed(episode_seed)

            episode_metrics = self._run_one_episode_extended(
                compute_diversity=compute_diversity,
                compute_cooperation=compute_cooperation,
            )

            # 累计总奖励和长度
            total_reward = episode_metrics["total_reward"]
            episode_length = episode_metrics["episode_length"]
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)

            # 团队指标
            for team_name, team_reward in episode_metrics.get("team_rewards", {}).items():
                if team_name in team_rewards:
                    team_rewards[team_name].append(team_reward)
                    team_episode_rewards[team_name].append(team_reward)

            # 个体指标
            for agent_id, agent_reward in episode_metrics.get("agent_rewards", {}).items():
                if agent_id not in agent_rewards:
                    agent_rewards[agent_id] = []
                agent_rewards[agent_id].append(agent_reward)

            # 协作指标
            if "action_consistency" in episode_metrics:
                action_consistency.append(episode_metrics["action_consistency"])
            if "coordination_score" in episode_metrics:
                coordination_scores.append(episode_metrics["coordination_score"])

            # 多样性指标
            if "action_entropy" in episode_metrics:
                action_entropies.append(episode_metrics["action_entropy"])
            if "policy_difference" in episode_metrics:
                policy_differences.append(episode_metrics["policy_difference"])

        self.agent.to_training_mode()

        # 构建返回指标
        metrics = {
            # 基础指标
            "eval_mean_reward": float(np.mean(episode_rewards)),
            "eval_std_reward": float(np.std(episode_rewards)),
            "eval_min_reward": float(np.min(episode_rewards)),
            "eval_max_reward": float(np.max(episode_rewards)),
            "eval_mean_length": float(np.mean(episode_lengths)),
            "eval_std_length": float(np.std(episode_lengths)),
        }

        # 团队指标
        for team_name in self.team_names.keys():
            if team_name in team_episode_rewards and len(team_episode_rewards[team_name]) > 0:
                team_reward_list = team_episode_rewards[team_name]
                metrics[f"eval_team_{team_name}_mean_reward"] = float(np.mean(team_reward_list))
                metrics[f"eval_team_{team_name}_std_reward"] = float(np.std(team_reward_list))
                metrics[f"eval_team_{team_name}_min_reward"] = float(np.min(team_reward_list))
                metrics[f"eval_team_{team_name}_max_reward"] = float(np.max(team_reward_list))

        # 团队对抗指标（如果有两个团队）
        if len(self.team_names) == 2:
            team_list = list(self.team_names.keys())
            team1_name, team2_name = team_list[0], team_list[1]

            if (
                team1_name in team_episode_rewards
                and team2_name in team_episode_rewards
                and len(team_episode_rewards[team1_name]) > 0
                and len(team_episode_rewards[team2_name]) > 0
            ):

                team1_rewards = np.array(team_episode_rewards[team1_name])
                team2_rewards = np.array(team_episode_rewards[team2_name])

                # 计算胜率（奖励更高的团队获胜）
                team1_wins = np.sum(team1_rewards > team2_rewards)
                team2_wins = np.sum(team2_rewards > team1_rewards)
                draws = len(team1_rewards) - team1_wins - team2_wins

                metrics[f"eval_team_{team1_name}_win_rate"] = float(team1_wins / len(team1_rewards))
                metrics[f"eval_team_{team2_name}_win_rate"] = float(team2_wins / len(team2_rewards))
                metrics["eval_draw_rate"] = float(draws / len(team1_rewards))

                # 奖励差异
                reward_diff = team1_rewards - team2_rewards
                metrics["eval_team_reward_diff_mean"] = float(np.mean(reward_diff))
                metrics["eval_team_reward_diff_std"] = float(np.std(reward_diff))

        # 个体指标（统计所有agent的平均）
        if agent_rewards:
            all_agent_rewards = []
            for agent_id, rewards in agent_rewards.items():
                all_agent_rewards.extend(rewards)

            if all_agent_rewards:
                metrics["eval_agent_mean_reward"] = float(np.mean(all_agent_rewards))
                metrics["eval_agent_std_reward"] = float(np.std(all_agent_rewards))
                metrics["eval_agent_min_reward"] = float(np.min(all_agent_rewards))
                metrics["eval_agent_max_reward"] = float(np.max(all_agent_rewards))

            # 每个团队的平均个体奖励
            for team_name, team_agents in self.team_names.items():
                team_agent_rewards = []
                for agent_id in team_agents:
                    if agent_id in agent_rewards:
                        team_agent_rewards.extend(agent_rewards[agent_id])

                if team_agent_rewards:
                    metrics[f"eval_team_{team_name}_agent_mean_reward"] = float(
                        np.mean(team_agent_rewards)
                    )

        # 协作指标
        if action_consistency:
            metrics["eval_action_consistency_mean"] = float(np.mean(action_consistency))
            metrics["eval_action_consistency_std"] = float(np.std(action_consistency))

        if coordination_scores:
            metrics["eval_coordination_score_mean"] = float(np.mean(coordination_scores))
            metrics["eval_coordination_score_std"] = float(np.std(coordination_scores))

        # 多样性指标（自博弈）
        if action_entropies:
            metrics["eval_action_entropy_mean"] = float(np.mean(action_entropies))
            metrics["eval_action_entropy_std"] = float(np.std(action_entropies))

        if policy_differences:
            metrics["eval_policy_difference_mean"] = float(np.mean(policy_differences))
            metrics["eval_policy_difference_std"] = float(np.std(policy_differences))

        return metrics

    def _run_one_episode_extended(
        self,
        compute_diversity: bool = True,
        compute_cooperation: bool = True,
    ) -> Dict[str, Any]:
        """
        运行一个episode并收集扩展指标

        Returns:
            包含详细指标的字典
        """
        obs = self.env.reset()
        done = False
        step = 0
        total_reward = 0.0

        # 团队奖励累计
        team_rewards = {team: 0.0 for team in self.team_names.keys()}

        # 个体奖励累计
        agent_rewards = {}  # {agent_id: total_reward}

        # 动作历史（用于计算协作指标）
        team_actions = {team: [] for team in self.team_names.keys()}

        # 动作分布（用于计算多样性）
        all_actions = []
        action_distributions = {}  # {agent_id: [actions]}

        while not done and step < self.max_steps_per_episode:
            # Agent选择动作
            actions_dict = self.agent.act(obs, deterministic=True)

            # 环境步进
            next_obs, rewards, dones, info = self.env.step(actions_dict)

            # 累计奖励
            if isinstance(rewards, dict):
                for agent_id, reward in rewards.items():
                    total_reward += reward

                    # 个体奖励
                    if agent_id not in agent_rewards:
                        agent_rewards[agent_id] = 0.0
                    agent_rewards[agent_id] += reward

                    # 团队奖励
                    for team_name, team_agents in self.team_names.items():
                        if agent_id in team_agents:
                            team_rewards[team_name] += reward
                            team_actions[team_name].append(actions_dict.get(agent_id))
                            break

                    # 动作分布
                    if agent_id not in action_distributions:
                        action_distributions[agent_id] = []
                    action_distributions[agent_id].append(actions_dict.get(agent_id))
                    all_actions.append(actions_dict.get(agent_id))
            else:
                total_reward += rewards

            # 检查是否结束
            if isinstance(dones, dict):
                done = dones.get("__all__", False)
            else:
                done = dones

            obs = next_obs
            step += 1

        # 构建返回指标
        episode_metrics = {
            "total_reward": total_reward,
            "episode_length": step,
            "team_rewards": team_rewards,
            "agent_rewards": agent_rewards,
        }

        # 计算协作指标
        if compute_cooperation:
            # 动作一致性：团队内agent动作的相似度
            action_consistency = self._compute_action_consistency(team_actions)
            episode_metrics["action_consistency"] = action_consistency

            # 协作分数：基于团队奖励和个体奖励的关系
            coordination_score = self._compute_coordination_score(team_rewards, agent_rewards)
            episode_metrics["coordination_score"] = coordination_score

        # 计算多样性指标
        if compute_diversity:
            # 动作熵：动作分布的熵
            action_entropy = self._compute_action_entropy(all_actions)
            episode_metrics["action_entropy"] = action_entropy

            # 策略差异：如果有多个策略版本，计算它们的差异
            policy_difference = self._compute_policy_difference()
            if policy_difference is not None:
                episode_metrics["policy_difference"] = policy_difference

        return episode_metrics

    def _compute_action_consistency(self, team_actions: Dict[str, List]) -> float:
        """
        计算团队内动作一致性

        使用团队成员在同一时间步动作的相似度来衡量

        Args:
            team_actions: {team_name: [actions...]}

        Returns:
            平均动作一致性分数 [0, 1]
        """
        if not team_actions:
            return 0.0

        consistency_scores = []

        for team_name, actions in team_actions.items():
            if len(actions) < 2:
                continue

            # 将动作转换为数组
            try:
                actions_array = np.array(actions)

                # 计算动作的方差（方差越小，一致性越高）
                if actions_array.ndim > 0:
                    # 对于离散动作，计算动作分布的一致性
                    # 使用动作的标准差，归一化到[0,1]
                    if actions_array.size > 0:
                        std = np.std(actions_array)
                        # 归一化：假设动作范围是[0, action_space_size]
                        # 这里使用简单的归一化，实际应根据动作空间调整
                        max_std = (
                            np.max(actions_array) - np.min(actions_array)
                            if len(actions_array) > 1
                            else 1.0
                        )
                        consistency = 1.0 - (std / max_std) if max_std > 0 else 1.0
                        consistency = max(0.0, min(1.0, consistency))
                        consistency_scores.append(consistency)
            except:
                pass

        return float(np.mean(consistency_scores)) if consistency_scores else 0.0

    def _compute_coordination_score(
        self, team_rewards: Dict[str, float], agent_rewards: Dict[str, float]
    ) -> float:
        """
        计算协作分数

        基于团队奖励与个体奖励之和的关系来衡量协作效果

        Args:
            team_rewards: {team_name: total_reward}
            agent_rewards: {agent_id: total_reward}

        Returns:
            协作分数，值越大表示协作越好
        """
        if not team_rewards or not agent_rewards:
            return 0.0

        # 计算每个团队的实际团队奖励与个体奖励之和的比例
        coordination_scores = []

        for team_name, team_agents in self.team_names.items():
            if team_name not in team_rewards:
                continue

            # 计算该团队所有agent的个体奖励总和
            team_agent_reward_sum = sum(
                agent_rewards.get(agent_id, 0.0) for agent_id in team_agents
            )

            if team_agent_reward_sum > 0:
                # 团队奖励 / 个体奖励总和
                # 如果协作好，团队奖励应该大于简单的个体奖励之和（有协同效应）
                coordination = team_rewards[team_name] / team_agent_reward_sum
                coordination_scores.append(coordination)

        return float(np.mean(coordination_scores)) if coordination_scores else 0.0

    def _compute_action_entropy(self, actions: List) -> float:
        """
        计算动作分布的熵（多样性指标）

        Args:
            actions: 动作列表

        Returns:
            动作熵，值越大表示动作多样性越高
        """
        if not actions:
            return 0.0

        try:
            # 统计动作频次
            unique_actions, counts = np.unique(actions, return_counts=True)

            # 计算概率分布
            probs = counts / np.sum(counts)

            # 计算熵：H = -sum(p * log(p))
            entropy = -np.sum(probs * np.log(probs + 1e-10))

            return float(entropy)
        except:
            return 0.0

    def _compute_policy_difference(self) -> Optional[float]:
        """
        计算策略差异（自博弈场景）

        如果有策略池或历史策略，计算当前策略与它们的差异

        Returns:
            策略差异（KL散度或其他度量），如果无法计算则返回None
        """
        # 这里需要访问策略参数来计算差异
        # 默认返回None，可以由子类或外部实现
        # 如果需要，可以：
        # 1. 访问AgentManager的策略参数
        # 2. 与策略池中的策略比较
        # 3. 计算参数差异或动作分布差异

        # 示例：如果有策略池，可以从self_play_trainer中获取
        # 这里暂时返回None，可以在SelfPlayPPOTrainer中实现具体逻辑

        return None
