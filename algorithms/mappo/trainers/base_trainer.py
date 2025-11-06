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
import torch

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer

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


class MAPPOTrainer(BaseAlgorithmTrainer):
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
        # 调用基类初始化
        super().__init__(agent, env, config, logger, tracker)
        
        # MAPPO特定配置
        self.use_centralized_critic = config.get("use_centralized_critic", False)
        self.global_obs_dim = config.get("global_obs_dim", None)
        
        # PPO相关配置（从基类获取或设置默认值）
        self.num_epochs = config.get("num_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)

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
    
    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练（实现基类的抽象方法）
        
        Args:
            processed_data: 处理后的数据
        
        Returns:
            训练指标
        """
        # MAPPO只支持多Agent，所以直接调用多Agent训练步骤
        return self._train_step_multi_agent(processed_data)

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
            values = data.get("values", data.get("old_values", []))  # 获取values用于old_values回退

            # 优势标准化（统一在所有算法中）
            if isinstance(advantages, np.ndarray):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif isinstance(advantages, torch.Tensor):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 更新data中的advantages（标准化后）
            data["advantages"] = advantages

            num_samples = len(obs)
            batch_indices = np.random.permutation(num_samples)

            # PPO多轮更新
            for epoch in range(self.num_epochs):
                # 分批训练
                for i in range(0, num_samples, self.batch_size):
                    indices = batch_indices[i : i + self.batch_size]
                    
                    # 处理advantages（确保是numpy数组）
                    if isinstance(advantages, torch.Tensor):
                        batch_advantages = advantages[indices].cpu().numpy()
                    else:
                        batch_advantages = advantages[indices] if isinstance(advantages, np.ndarray) else np.array(advantages)[indices]

                    batch = {
                        "obs": obs[indices],
                        "actions": actions[indices],
                        "logprobs": old_logprobs[indices],
                        "advantages": batch_advantages,
                        "returns": returns[indices],
                        "old_values": data.get("old_values", values)[indices] if len(values) > 0 else returns[indices],  # 用于价值函数裁剪
                        "clip_coef": self.clip_coef,
                        "value_coef": self.value_coef,
                        "entropy_coef": self.entropy_coef,
                        "vf_clip_param": self.config.get("vf_clip_param"),  # 价值函数裁剪参数
                    }

                    # 如果使用集中式Critic，添加全局信息
                    if self.use_centralized_critic:
                        # 从processed_data获取全局状态（优先使用后处理后的state）
                        if "state" in data and len(data["state"]) > 0:
                            # 使用后处理函数提供的state
                            if len(indices) <= len(data["state"]):
                                batch["state"] = data["state"][indices]
                            else:
                                batch["state"] = data["state"]
                        else:
                            # 回退到原来的方法
                            global_state = self._get_global_obs(agent_id, indices, processed_data)
                            batch["state"] = global_state
                        
                        batch["use_centralized_critic"] = True
                        
                        # 如果配置了对手动作输入，添加对手动作
                        if self.config.get("opp_action_in_cc", False):
                            opponent_actions = self._get_opponent_actions(agent_id, indices, processed_data)
                            if opponent_actions is not None:
                                batch["opponent_actions"] = opponent_actions

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
        获取全局状态（用于集中式Critic）

        Args:
            agent_id: Agent ID
            indices: 批次索引
            processed_data: 所有agent的数据（应该包含state字段）

        Returns:
            全局状态
        """
        # 优先使用环境提供的全局状态
        if "state" in processed_data.get(agent_id, {}):
            states = processed_data[agent_id]["state"]
            if len(states) > 0:
                # 根据indices获取对应的状态
                if len(indices) <= len(states):
                    return states[indices]
                else:
                    # 如果索引超出范围，返回所有状态
                    return states
        
        # 回退：拼接所有agent的观测（如果环境没有提供state）
        global_obs_list = []
        for other_agent_id, other_data in processed_data.items():
            if len(other_data.get("obs", [])) > 0:
                other_obs = other_data["obs"][indices] if len(indices) <= len(other_data["obs"]) else other_data["obs"]
                global_obs_list.append(other_obs)

        if global_obs_list:
            # 拼接所有agent的观测
            global_obs = np.concatenate(global_obs_list, axis=-1)
        else:
            # 最终回退：当前agent的观测
            global_obs = processed_data[agent_id]["obs"][indices]

        return global_obs
    
    def _get_opponent_actions(
        self, agent_id: str, indices: np.ndarray, processed_data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        获取对手动作（用于集中式Critic）
        
        Args:
            agent_id: 当前Agent ID
            indices: 批次索引
            processed_data: 所有agent的数据
        
        Returns:
            对手动作数组，形状为 (B, n_opponents, action_dim) 或 None
        """
        # 获取所有其他agent的动作
        opponent_actions_list = []
        for other_agent_id, other_data in processed_data.items():
            if other_agent_id != agent_id and "actions" in other_data:
                actions = other_data["actions"]
                if len(indices) <= len(actions):
                    opponent_actions_list.append(actions[indices])
                else:
                    opponent_actions_list.append(actions)
        
        if opponent_actions_list:
            # 堆叠成 (B, n_opponents, action_dim)
            # 注意：如果动作是标量，需要先扩展维度
            opponent_actions = np.stack(opponent_actions_list, axis=1)
            return opponent_actions
        return None
    
    def _aggregate_metrics(self, metrics_list: list) -> Dict[str, float]:
        """聚合指标列表"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = float(np.mean(values))
        
        return aggregated

