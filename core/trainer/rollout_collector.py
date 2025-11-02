# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : rollout_collector.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : Rollout数据收集器
通用的rollout数据收集组件，支持单Agent和多Agent环境
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from core.agent import AgentManager
from core.base.agent import Agent
from core.base.environment import Env


class RolloutCollector:
    """
    通用的Rollout数据收集器

    负责Agent与环境交互，收集训练所需的经验数据。
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
        初始化Rollout收集器

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

    def collect(self) -> Dict[str, Any]:
        """
        收集一个rollout的数据

        Returns:
            收集的数据字典，包含：
            - obs: 观测数据
            - actions: 动作数据
            - rewards: 奖励数据
            - next_obs: 下一个观测数据
            - dones: 完成标志
            - logprobs: 对数概率
            - values: 价值估计
        """
        if self.is_multi_agent:
            return self._collect_multi_agent()
        else:
            return self._collect_single_agent()

    def _collect_single_agent(self) -> Dict[str, Any]:
        """
        收集单Agent rollout数据

        Returns:
            单Agent的rollout数据字典
        """
        obs_list = []
        actions_list = []
        rewards_list = []
        next_obs_list = []
        dones_list = []
        logprobs_list = []
        values_list = []

        obs = self.env.reset()
        done = False
        step = 0

        while not done and step < self.max_steps_per_episode:
            # Agent选择动作
            action, logprob, value = self.agent.act(obs, deterministic=False)

            # 环境步进
            next_obs, reward, done, info = self.env.step(action)

            # 存储数据
            obs_list.append(obs)
            actions_list.append(action)
            rewards_list.append(reward)
            next_obs_list.append(next_obs)
            dones_list.append(done)
            logprobs_list.append(logprob)
            values_list.append(value)

            obs = next_obs
            step += 1

        return {
            "obs": np.array(obs_list),
            "actions": np.array(actions_list),
            "rewards": np.array(rewards_list),
            "next_obs": np.array(next_obs_list),
            "dones": np.array(dones_list),
            "logprobs": np.array(logprobs_list),
            "values": np.array(values_list),
        }

    def _collect_multi_agent(self) -> Dict[str, Any]:
        """
        收集多Agent rollout数据

        Returns:
            多Agent的rollout数据字典，格式为 {agent_id: rollout_data}
        """
        # 初始化存储结构
        agent_ids = self.agent.agent_ids if isinstance(self.agent, AgentManager) else []
        all_obs = {aid: [] for aid in agent_ids}
        all_actions = {aid: [] for aid in agent_ids}
        all_rewards = {aid: [] for aid in agent_ids}
        all_next_obs = {aid: [] for aid in agent_ids}
        all_dones = {aid: [] for aid in agent_ids}
        all_logprobs = {aid: [] for aid in agent_ids}
        all_values = {aid: [] for aid in agent_ids}

        obs = self.env.reset()
        done = False
        step = 0

        while not done and step < self.max_steps_per_episode:
            # 批量选择动作
            actions_dict = self.agent.act(obs, deterministic=False)

            # 环境步进
            next_obs, rewards, dones, info = self.env.step(actions_dict)

            # 存储数据
            for agent_id in agent_ids:
                if agent_id in obs:
                    action, logprob, value = actions_dict[agent_id]

                    all_obs[agent_id].append(obs[agent_id])
                    all_actions[agent_id].append(action)
                    all_rewards[agent_id].append(rewards.get(agent_id, 0.0))
                    all_next_obs[agent_id].append(next_obs.get(agent_id, obs[agent_id]))
                    all_dones[agent_id].append(dones.get(agent_id, False))
                    all_logprobs[agent_id].append(logprob)
                    all_values[agent_id].append(value)

            # 检查是否结束
            if isinstance(dones, dict):
                done = dones.get("__all__", False) or all(dones.values())
            else:
                done = dones

            obs = next_obs
            step += 1

        # 转换为numpy数组
        return {
            agent_id: {
                "obs": np.array(all_obs[agent_id]),
                "actions": np.array(all_actions[agent_id]),
                "rewards": np.array(all_rewards[agent_id]),
                "next_obs": np.array(all_next_obs[agent_id]),
                "dones": np.array(all_dones[agent_id]),
                "logprobs": np.array(all_logprobs[agent_id]),
                "values": np.array(all_values[agent_id]),
            }
            for agent_id in agent_ids
        }
