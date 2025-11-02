# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : evaluator.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : Agent评估器
通用的Agent性能评估组件，支持单Agent和多Agent环境
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.agent import AgentManager
from core.base.agent import Agent
from core.base.environment import Env


class Evaluator:
    """
    通用的Agent评估器

    用于评估Agent在环境中的性能，使用确定性策略进行评估。
    支持单Agent和多Agent环境。
    """

    def __init__(
        self,
        agent: Agent,
        env: Env,
        max_steps_per_episode: int = 1000,
        is_multi_agent: bool = False,
    ):
        """
        初始化评估器

        Args:
            agent: Agent实例（单个Agent）或AgentManager（多Agent）
            env: 环境实例
            max_steps_per_episode: 每个episode的最大步数
            is_multi_agent: 是否为多Agent环境
        """
        self.agent = agent
        self.env = env
        self.max_steps_per_episode = max_steps_per_episode
        self.is_multi_agent = is_multi_agent

    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """
        评估Agent性能

        Args:
            num_episodes: 评估的episode数量

        Returns:
            评估指标字典，包含：
            - eval_mean_reward: 平均episode奖励
            - eval_std_reward: 奖励标准差
            - eval_min_reward: 最小奖励
            - eval_max_reward: 最大奖励
            - eval_mean_length: 平均episode长度
        """
        self.agent.to_eval_mode()

        episode_rewards = []
        episode_lengths = []

        for _ in range(num_episodes):
            episode_reward, episode_length = self._run_one_episode()
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        self.agent.to_training_mode()

        return {
            "eval_mean_reward": float(np.mean(episode_rewards)),
            "eval_std_reward": float(np.std(episode_rewards)),
            "eval_min_reward": float(np.min(episode_rewards)),
            "eval_max_reward": float(np.max(episode_rewards)),
            "eval_mean_length": float(np.mean(episode_lengths)),
        }

    def _run_one_episode(self) -> tuple[float, int]:
        """
        运行一个episode

        Returns:
            (episode_reward, episode_length) 元组
        """
        if self.is_multi_agent:
            return self._run_multi_agent_episode()
        else:
            return self._run_single_agent_episode()

    def _run_single_agent_episode(self) -> tuple[float, int]:
        """运行单Agent episode"""
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < self.max_steps_per_episode:
            action, _, _ = self.agent.act(obs, deterministic=True)
            next_obs, reward, done, info = self.env.step(action)

            episode_reward += reward
            obs = next_obs
            step += 1

        return episode_reward, step

    def _run_multi_agent_episode(self) -> tuple[float, int]:
        """运行多Agent episode"""
        obs = self.env.reset()
        done = False
        step = 0
        episode_reward = 0.0

        while not done and step < self.max_steps_per_episode:
            actions_dict = self.agent.act(obs, deterministic=True)
            next_obs, rewards, dones, info = self.env.step(actions_dict)

            # 累计所有agent的奖励
            if isinstance(rewards, dict):
                episode_reward += sum(rewards.values())
            else:
                episode_reward += rewards

            # 检查是否结束
            if isinstance(dones, dict):
                done = dones.get("__all__", False)
            else:
                done = dones

            obs = next_obs
            step += 1

        return episode_reward, step
