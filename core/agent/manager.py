# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : manager.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:53
@Update Date    :
@Description    : Agent管理器
用于管理多Agent环境中的多个Agent实例，支持策略共享、团队分组等功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Union

from core.agent.factory import build_agent_from_config
from core.base.agent import Agent


class AgentManager:
    """
    Agent管理器

    用于在多Agent环境中管理多个Agent实例，支持：
    - 为每个agent创建独立的Agent实例
    - 按团队/角色共享策略
    - 批量操作（act, learn等）
    - 状态保存和加载
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int,
        action_dim: int,
        config: Dict[str, Any],
        device: str = "cpu",
        shared_agents: Optional[Dict[str, List[str]]] = None,
    ):
        """
        初始化Agent管理器

        Args:
            agent_ids: 所有agent的ID列表
            obs_dim: 观测维度
            action_dim: 动作空间维度
            config: Agent配置字典
            device: 设备
            shared_agents: 共享策略的agent分组，例如 {"team_red": ["red_0", "red_1"], "team_blue": ["blue_0", "blue_1"]}
                        如果为None，则每个agent独立
        """
        self.agent_ids = agent_ids
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.config = config

        # 构建agent映射
        self._agents: Dict[str, Agent] = {}
        self._agent_to_group: Dict[str, str] = {}  # agent_id -> group_name

        # 如果指定了共享策略
        if shared_agents is not None:
            # 为每个组创建一个Agent实例
            for group_name, group_agent_ids in shared_agents.items():
                agent = build_agent_from_config(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    config=config,
                    device=device,
                )
                # 这个组的每个agent都共享同一个Agent实例
                for agent_id in group_agent_ids:
                    self._agents[agent_id] = agent
                    self._agent_to_group[agent_id] = group_name
        else:
            # 每个agent独立
            for agent_id in agent_ids:
                agent = build_agent_from_config(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    config=config,
                    device=device,
                )
                self._agents[agent_id] = agent
                self._agent_to_group[agent_id] = agent_id

    def get_agent(self, agent_id: str) -> Agent:
        """获取指定agent的Agent实例"""
        if agent_id not in self._agents:
            raise KeyError(f"Agent {agent_id} not found. Available: {list(self._agents.keys())}")
        return self._agents[agent_id]

    def act(
        self,
        observations: Dict[str, Any],
        deterministic: bool = False,
        show_progress: bool = False,
    ) -> Dict[str, tuple]:
        """
        批量执行动作选择

        对于共享策略的agent（同一组），会批量处理观测以提高效率。

        Args:
            observations: {agent_id: observation}
            deterministic: 是否使用确定性策略
            show_progress: 是否显示进度（对于大量agent很有用）

        Returns:
            {agent_id: (action, logprob, value)}
        """
        results = {}
        num_agents = len(observations)
        
        # 如果agent数量很多且需要显示进度，输出进度
        if show_progress:
            print(f"  正在为 {num_agents} 个agent选择动作...", flush=True)
        
        # 如果使用共享策略，按组批量处理
        if self._has_shared_agents():
            # 按组组织观测
            group_observations = {}
            agent_to_group_map = {}
            
            for agent_id, obs in observations.items():
                group_name = self._agent_to_group.get(agent_id, agent_id)
                if group_name not in group_observations:
                    group_observations[group_name] = []
                group_observations[group_name].append((agent_id, obs))
                agent_to_group_map[agent_id] = group_name
            
            # 对每个组批量处理
            processed = 0
            for group_name, agent_obs_list in group_observations.items():
                # 获取该组的Agent实例（所有agent共享同一个）
                sample_agent_id = agent_obs_list[0][0]
                agent = self.get_agent(sample_agent_id)
                
                # 提取观测列表
                obs_list = [obs for _, obs in agent_obs_list]
                agent_ids_in_group = [aid for aid, _ in agent_obs_list]
                
                # 尝试批量处理（如果agent支持批量）
                try:
                    # 检查agent是否有批量act方法
                    if hasattr(agent, "act_batch"):
                        batch_results = agent.act_batch(obs_list, deterministic=deterministic)
                        for aid, (action, logprob, value) in zip(agent_ids_in_group, batch_results):
                            results[aid] = (action, logprob, value)
                    else:
                        # 顺序处理（但至少是同一个agent实例，forward pass会更快）
                        for agent_id, obs in agent_obs_list:
                            action, logprob, value = agent.act(obs, deterministic=deterministic)
                            results[agent_id] = (action, logprob, value)
                except Exception:
                    # 如果批量处理失败，回退到顺序处理
                    for agent_id, obs in agent_obs_list:
                        action, logprob, value = agent.act(obs, deterministic=deterministic)
                        results[agent_id] = (action, logprob, value)
                
                processed += len(agent_obs_list)
                if show_progress:
                    print(f"  进度: {processed}/{num_agents} agents (组: {group_name})", end='\r', flush=True)
            
            if show_progress:
                print()  # 换行
        else:
            # 没有共享策略，顺序处理每个agent
            progress_freq = max(10, num_agents // 20)
            for idx, (agent_id, obs) in enumerate(observations.items()):
                agent = self.get_agent(agent_id)
                action, logprob, value = agent.act(obs, deterministic=deterministic)
                results[agent_id] = (action, logprob, value)
                
                if show_progress and (idx + 1) % progress_freq == 0:
                    print(f"  动作选择进度: {idx + 1}/{num_agents} agents", end='\r', flush=True)
            
            if show_progress and num_agents > progress_freq:
                print()  # 换行
        
        return results

    def learn(
        self,
        batches: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        批量学习

        Args:
            batches: {agent_id: batch_dict} 或 {group_name: batch_dict}（如果共享策略）

        Returns:
            {agent_id/group_name: metrics}
        """
        results = {}

        # 如果共享策略，需要合并同一组的批次数据
        if self._has_shared_agents():
            # 按组聚合批次
            group_batches = {}
            for agent_id, batch in batches.items():
                group_name = self._agent_to_group[agent_id]
                if group_name not in group_batches:
                    group_batches[group_name] = self._merge_batches([batch])
                else:
                    group_batches[group_name] = self._merge_batches(
                        [group_batches[group_name], batch]
                    )

            # 对每个组学习
            for group_name, batch in group_batches.items():
                # 获取组内任意一个agent的实例（它们共享同一个实例）
                sample_agent_id = next(
                    aid for aid, gname in self._agent_to_group.items() if gname == group_name
                )
                agent = self.get_agent(sample_agent_id)
                metrics = agent.learn(batch)
                results[group_name] = metrics
        else:
            # 每个agent独立学习
            for agent_id, batch in batches.items():
                agent = self.get_agent(agent_id)
                metrics = agent.learn(batch)
                results[agent_id] = metrics

        return results

    def state_dict(self) -> Dict[str, Any]:
        """
        获取所有Agent的状态字典

        Returns:
            {group_name/agent_id: state_dict}
        """
        states = {}
        seen_groups: Set[str] = set()

        for agent_id, agent in self._agents.items():
            group_name = self._agent_to_group[agent_id]

            # 如果共享策略，只保存一次
            if group_name in seen_groups:
                continue
            seen_groups.add(group_name)

            states[group_name] = agent.state_dict()

        return states

    def load_state_dict(self, states: Dict[str, Any]) -> None:
        """
        加载所有Agent的状态字典

        Args:
            states: {group_name/agent_id: state_dict}
        """
        for agent_id, agent in self._agents.items():
            group_name = self._agent_to_group[agent_id]

            if group_name in states:
                agent.load_state_dict(states[group_name])
            elif agent_id in states:
                agent.load_state_dict(states[agent_id])
            else:
                raise KeyError(f"No state dict found for {agent_id} or {group_name}")

    def to_training_mode(self) -> None:
        """将所有Agent切换到训练模式"""
        seen_agents: Set[Agent] = set()
        for agent in self._agents.values():
            if agent not in seen_agents:
                agent.to_training_mode()
                seen_agents.add(agent)

    def to_eval_mode(self) -> None:
        """将所有Agent切换到评估模式"""
        seen_agents: Set[Agent] = set()
        for agent in self._agents.values():
            if agent not in seen_agents:
                agent.to_eval_mode()
                seen_agents.add(agent)

    def get_group_members(self, group_name: str) -> List[str]:
        """获取指定组的所有agent ID"""
        return [agent_id for agent_id, gname in self._agent_to_group.items() if gname == group_name]

    def get_all_groups(self) -> Set[str]:
        """获取所有组名"""
        return set(self._agent_to_group.values())

    def _has_shared_agents(self) -> bool:
        """检查是否有共享策略的agent"""
        unique_agents = len(set(id(agent) for agent in self._agents.values()))
        return unique_agents < len(self._agents)

    def _merge_batches(self, batches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并多个批次数据

        Args:
            batches: 批次列表

        Returns:
            合并后的批次
        """
        if len(batches) == 0:
            return {}

        if len(batches) == 1:
            return batches[0]

        merged = {}
        for key in batches[0].keys():
            # 假设所有批次的数据都是列表或数组，需要拼接
            import numpy as np
            import torch

            values = [batch[key] for batch in batches]

            # 尝试转换为tensor或numpy数组
            try:
                if isinstance(values[0], torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(values[0], np.ndarray):
                    merged[key] = np.concatenate(values, axis=0)
                elif isinstance(values[0], list):
                    merged[key] = sum(values, [])  # 展平列表
                else:
                    # 尝试使用numpy concatenate
                    merged[key] = np.concatenate([np.array(v) for v in values], axis=0)
            except Exception:
                # 如果无法合并，使用列表
                merged[key] = sum(values, [])

        return merged


def create_agent_manager_from_config(
    agent_ids: List[str],
    obs_dim: int,
    action_dim: int,
    config: Dict[str, Any],
    device: str = "cpu",
) -> AgentManager:
    """
    从配置创建Agent管理器

    配置可以包含：
        - agent_config: Agent的配置
        - shared_groups: 共享策略的分组，例如 {"team_red": ["red_0", "red_1"]}

    Args:
        agent_ids: 所有agent的ID列表
        obs_dim: 观测维度
        action_dim: 动作空间维度
        config: 配置字典
        device: 设备

    Returns:
        AgentManager实例
    """
    agent_config = config.get("agent_config", config)
    shared_groups = config.get("shared_groups", None)

    return AgentManager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device=device,
        shared_agents=shared_groups,
    )
