# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-06 00:00
@Update Date    :
@Description    : HP3O基础训练器
实现HP3O算法的训练循环，包括轨迹重放缓冲区管理
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer
from algorithms.hp3o.core import TrajectoryBuffer, RolloutBufferSamples
from algorithms.hp3o.config import HP3OConfig

# Logger是可选的
try:
    from common.utils.logging import LoggerManager
    Logger = LoggerManager
except ImportError:
    Logger = None


class HP3OTrainer(BaseAlgorithmTrainer):
    """
    HP3O训练器
    
    HP3O (High-Performance PPO with Trajectory Replay) 算法实现。
    核心特点：
    - 使用轨迹重放缓冲区存储完整轨迹
    - 从轨迹缓冲区采样轨迹，然后从轨迹中采样数据
    - 支持基于最佳轨迹的价值函数增强
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
    ):
        """
        初始化HP3O训练器
        
        Args:
            agent: Agent实例（单个Agent）或AgentManager（多Agent）
            env: 环境实例
            config: 训练配置
                - trajectory_buffer_size: 轨迹缓冲区大小
                - trajectory_sample_size: 每次采样轨迹数量
                - data_sample_size: 从轨迹中采样数据量
                - threshold: 轨迹筛选阈值
                - use_best_value: 是否使用最佳轨迹的价值函数
                - num_epochs: PPO更新轮数
                - batch_size: 批次大小
                - clip_coef: PPO裁剪系数
                - value_coef: 价值损失系数
                - entropy_coef: 熵正则化系数
                - gamma: 折扣因子
                - gae_lambda: GAE lambda参数
                - max_grad_norm: 梯度裁剪（可选）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
        """
        # 调用基类初始化
        super().__init__(agent, env, config, logger, tracker)
        
        # HP3O特定配置
        self.trajectory_buffer_size = config.get("trajectory_buffer_size", HP3OConfig.DEFAULT_TRAJECTORY_BUFFER_SIZE)
        self.trajectory_sample_size = config.get("trajectory_sample_size", HP3OConfig.DEFAULT_TRAJECTORY_SAMPLE_SIZE)
        self.data_sample_size = config.get("data_sample_size", HP3OConfig.DEFAULT_DATA_SAMPLE_SIZE)
        self.threshold = config.get("threshold", HP3OConfig.DEFAULT_THRESHOLD)
        self.use_best_value = config.get("use_best_value", HP3OConfig.DEFAULT_USE_BEST_VALUE)
        
        # PPO参数
        self.num_epochs = config.get("num_epochs", HP3OConfig.DEFAULT_NUM_EPOCHS)
        self.batch_size = config.get("batch_size", HP3OConfig.DEFAULT_BATCH_SIZE)
        self.clip_coef = config.get("clip_coef", HP3OConfig.DEFAULT_CLIP_COEF)
        self.value_coef = config.get("value_coef", HP3OConfig.DEFAULT_VALUE_COEF)
        self.entropy_coef = config.get("entropy_coef", HP3OConfig.DEFAULT_ENTROPY_COEF)
        self.max_grad_norm = config.get("max_grad_norm", HP3OConfig.DEFAULT_MAX_GRAD_NORM)
        self.log_trajectory_warnings = config.get("log_trajectory_warnings", False)
        
        # 初始化轨迹缓冲区
        self._init_trajectory_buffer()
    
    def _init_trajectory_buffer(self) -> None:
        """初始化轨迹缓冲区"""
        # 获取观测和动作空间形状
        if self.is_multi_agent:
            # 检查是否有主团队agent列表（自博弈场景）
            main_agent_ids = getattr(self, 'main_agent_ids', None)
            
            if main_agent_ids is not None and len(main_agent_ids) > 0:
                # 自博弈场景：需要聚合所有主团队agent的观测和动作
                # 观测维度 = 单个agent观测维度 × 主团队agent数量
                # 动作维度 = 单个agent动作维度 × 主团队agent数量
                first_agent = self.agent.get_agent(main_agent_ids[0])
                obs_dim = first_agent.obs_dim
                action_dim = first_agent.action_dim
                
                # 计算聚合后的维度
                num_main_agents = len(main_agent_ids)
                if isinstance(obs_dim, int):
                    aggregated_obs_dim = obs_dim * num_main_agents
                    obs_shape = (aggregated_obs_dim,)
                else:
                    # 如果是元组（如CNN的shape），需要特殊处理
                    obs_shape = obs_dim  # 暂时使用原始shape，实际会在处理时拼接
                
                if isinstance(action_dim, int):
                    aggregated_action_dim = action_dim * num_main_agents
                    action_shape = (aggregated_action_dim,)
                else:
                    action_shape = action_dim  # 暂时使用原始shape
            else:
                # 普通多Agent场景：使用第一个agent的观测和动作空间
                agent_ids = self.agent.agent_ids if hasattr(self.agent, "agent_ids") else []
                if len(agent_ids) > 0:
                    first_agent = self.agent.get_agent(agent_ids[0])
                    obs_dim = first_agent.obs_dim
                    action_dim = first_agent.action_dim
                    obs_shape = (obs_dim,) if isinstance(obs_dim, int) else obs_dim
                    action_shape = (action_dim,) if isinstance(action_dim, int) else action_dim
                else:
                    obs_shape = (1,)
                    action_shape = (1,)
        else:
            # 单Agent
            obs_dim = self.agent.obs_dim
            action_dim = self.agent.action_dim
            obs_shape = (obs_dim,) if isinstance(obs_dim, int) else obs_dim
            action_shape = (action_dim,) if isinstance(action_dim, int) else action_dim
        
        device = self.agent.device if hasattr(self.agent, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gamma = self.config.get("gamma", HP3OConfig.DEFAULT_GAMMA)
        gae_lambda = self.config.get("gae_lambda", HP3OConfig.DEFAULT_GAE_LAMBDA)
        
        self.trajectory_buffer = TrajectoryBuffer(
            buffer_size=self.trajectory_buffer_size,
            observation_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    
    def _process_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理rollout数据，转换为轨迹格式并存储到轨迹缓冲区
        
        Args:
            rollout_data: 原始rollout数据
        
        Returns:
            处理后的数据（用于训练）
        """
        if self.is_multi_agent:
            return self._process_multi_agent_rollout(rollout_data)
        else:
            return self._process_single_agent_rollout(rollout_data)
    
    def _process_single_agent_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单Agent rollout数据"""
        # 开始新轨迹
        self.trajectory_buffer.start_trajectory()
        
        # 将rollout数据添加到轨迹缓冲区
        obs = rollout_data["obs"]
        actions = rollout_data["actions"]
        rewards = rollout_data["rewards"]
        dones = rollout_data["dones"]
        values = rollout_data["values"]
        logprobs = rollout_data["logprobs"]
        
        num_steps = len(obs)
        for i in range(num_steps):
            self.trajectory_buffer.add_step(
                obs=obs[i],
                action=actions[i],
                reward=float(rewards[i]),
                done=bool(dones[i]),
                value=float(values[i]),
                log_prob=float(logprobs[i]),
            )
        
        # 返回空字典（实际训练数据从轨迹缓冲区采样）
        return {}
    
    def _process_multi_agent_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理多Agent rollout数据
        
        对于自博弈训练，只处理主团队agent的数据。
        对于普通多Agent训练，处理所有agent的数据。
        """
        # 检查是否有主团队agent列表（自博弈场景）
        main_agent_ids = getattr(self, 'main_agent_ids', None)
        
        if main_agent_ids is not None:
            # 自博弈场景：只处理主团队agent的数据
            agent_ids_to_process = [aid for aid in main_agent_ids if aid in rollout_data]
        else:
            # 普通多Agent场景：处理所有agent的数据
            agent_ids_to_process = list(rollout_data.keys())
        
        if len(agent_ids_to_process) == 0:
            return {}
        
        # 聚合所有主团队agent的数据
        # 策略：将所有agent的观测、动作等拼接在一起，形成一个"虚拟"的轨迹
        # 这样可以充分利用所有主团队agent的经验
        
        # 开始新轨迹
        self.trajectory_buffer.start_trajectory()
        
        # 获取所有agent的数据并确定最大步数
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_logprobs = []
        
        max_steps = 0
        for agent_id in agent_ids_to_process:
            if agent_id not in rollout_data:
                continue
            data = rollout_data[agent_id]
            num_steps = len(data["obs"])
            max_steps = max(max_steps, num_steps)
        
        # 按步聚合所有agent的数据
        for step in range(max_steps):
            step_obs = []
            step_actions = []
            step_rewards = []
            step_dones = []
            step_values = []
            step_logprobs = []
            
            for agent_id in agent_ids_to_process:
                if agent_id not in rollout_data:
                    continue
                data = rollout_data[agent_id]
                if step < len(data["obs"]):
                    step_obs.append(data["obs"][step])
                    step_actions.append(data["actions"][step])
                    step_rewards.append(float(data["rewards"][step]))
                    step_dones.append(bool(data["dones"][step]))
                    step_values.append(float(data["values"][step]))
                    step_logprobs.append(float(data["logprobs"][step]))
            
            if len(step_obs) > 0:
                # 聚合这一步所有agent的数据
                # 观测：展平后拼接所有agent的观测
                step_obs_flat = [np.array(obs).flatten() for obs in step_obs]
                aggregated_obs = np.concatenate(step_obs_flat) if len(step_obs_flat) > 1 else step_obs_flat[0]
                # 动作：展平后拼接所有agent的动作
                step_actions_flat = [np.array(action).flatten() for action in step_actions]
                aggregated_action = np.concatenate(step_actions_flat) if len(step_actions_flat) > 1 else step_actions_flat[0]
                # 奖励：使用平均奖励或总奖励
                aggregated_reward = np.mean(step_rewards) if len(step_rewards) > 0 else 0.0
                # Done：如果任何agent完成，则整个轨迹完成
                aggregated_done = any(step_dones)
                # 价值：使用平均价值
                aggregated_value = np.mean(step_values) if len(step_values) > 0 else 0.0
                # 对数概率：使用平均对数概率
                aggregated_logprob = np.mean(step_logprobs) if len(step_logprobs) > 0 else 0.0
                
                # 添加到轨迹缓冲区
                self.trajectory_buffer.add_step(
                    obs=aggregated_obs,
                    action=aggregated_action,
                    reward=aggregated_reward,
                    done=aggregated_done,
                    value=aggregated_value,
                    log_prob=aggregated_logprob,
                )
        
        return {}
    
    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练（使用轨迹重放）
        
        Args:
            processed_data: 处理后的数据（通常是空的，因为数据从轨迹缓冲区采样）
        
        Returns:
            训练指标
        """
        # 检查轨迹缓冲区是否有足够的轨迹
        if len(self.trajectory_buffer.trajectories) < self.trajectory_sample_size:
            # 如果轨迹不足，返回空指标
            if self.log_trajectory_warnings and self.logger:
                if hasattr(self.logger, "logger"):
                    self.logger.logger.warning(
                        f"Not enough trajectories: {len(self.trajectory_buffer.trajectories)} < {self.trajectory_sample_size}"
                    )
            return {}
        
        all_metrics = []
        values_epoch = []
        returns_epoch = []
        
        # 训练多个epoch
        for epoch in range(self.num_epochs):
            # 采样轨迹
            sampled_trajectory_indices = self.trajectory_buffer.sample_trajectories(
                self.trajectory_sample_size
            )
            
            # 从采样的轨迹中采样数据并训练
            for rollout_data in self.trajectory_buffer.sample(
                sampled_trajectory_indices,
                self.data_sample_size,
                batch_size=self.batch_size,
                use_best_value=self.use_best_value,
                threshold=self.threshold,
            ):
                # 转换为numpy数组（如果需要）
                obs = rollout_data.observations.cpu().numpy() if isinstance(rollout_data.observations, torch.Tensor) else rollout_data.observations
                actions = rollout_data.actions.cpu().numpy() if isinstance(rollout_data.actions, torch.Tensor) else rollout_data.actions
                old_logprobs = rollout_data.old_log_prob.cpu().numpy() if isinstance(rollout_data.old_log_prob, torch.Tensor) else rollout_data.old_log_prob
                advantages = rollout_data.advantages.cpu().numpy() if isinstance(rollout_data.advantages, torch.Tensor) else rollout_data.advantages
                returns = rollout_data.returns.cpu().numpy() if isinstance(rollout_data.returns, torch.Tensor) else rollout_data.returns
                old_values = rollout_data.old_values.cpu().numpy() if isinstance(rollout_data.old_values, torch.Tensor) else rollout_data.old_values
                
                # 确保actions是numpy数组，并且形状正确
                if not isinstance(actions, np.ndarray):
                    actions = np.array(actions)
                # 确保actions至少是1维数组
                if actions.ndim == 0:
                    actions = np.array([actions])
                elif actions.ndim == 1 and len(actions.shape) == 1 and actions.shape[0] == 0:
                    # 空数组，跳过这个批次
                    continue
                
                # 优势标准化
                if isinstance(advantages, np.ndarray):
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # 验证数据形状
                # obs应该是(batch_size, obs_dim)或(obs_dim,)
                # actions应该是(batch_size, action_dim)或(action_dim,)
                # 对于批次数据，确保是2D数组
                if isinstance(obs, np.ndarray):
                    if obs.ndim == 0:
                        # 标量，转换为1D数组
                        obs = np.array([obs])
                    elif obs.ndim == 1:
                        # 单个样本，需要添加batch维度
                        obs = obs.reshape(1, -1)
                    # 如果已经是2D或更高维度，保持不变
                
                if isinstance(actions, np.ndarray):
                    if actions.ndim == 0:
                        # 标量，转换为1D数组
                        actions = np.array([actions], dtype=np.int64)
                    elif actions.ndim == 1:
                        # 单个样本，需要添加batch维度
                        # 对于离散动作，action_dim=1，所以应该是(batch_size, 1)
                        if len(actions) == 1:
                            # 单个动作值，转换为(batch_size=1, action_dim=1)
                            actions = np.array([[int(actions[0])]], dtype=np.int64)
                        else:
                            # 多个动作值，转换为(batch_size=len, action_dim=1)
                            actions = actions.reshape(-1, 1).astype(np.int64)
                    # 如果已经是2D或更高维度，确保是整数类型
                    elif actions.ndim >= 2:
                        actions = actions.astype(np.int64)
                
                # 准备批次数据
                batch = {
                    "obs": obs,
                    "actions": actions,
                    "logprobs": old_logprobs,
                    "advantages": advantages,
                    "returns": returns,
                    "old_values": old_values,
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                    "clip_coef": self.clip_coef,
                    "vf_clip_param": self.config.get("vf_clip_param"),
                }
                
                # 训练agent
                if self.is_multi_agent:
                    # 检查是否有主团队agent列表（自博弈场景）
                    main_agent_ids = getattr(self, 'main_agent_ids', None)
                    
                    if main_agent_ids is not None and len(main_agent_ids) > 0:
                        # 自博弈场景：需要将聚合的数据拆分回各个主团队agent
                        # 获取单个agent的观测和动作维度
                        first_agent = self.agent.get_agent(main_agent_ids[0])
                        single_obs_dim = first_agent.obs_dim if isinstance(first_agent.obs_dim, int) else np.prod(first_agent.obs_dim)
                        single_action_dim = first_agent.action_dim if isinstance(first_agent.action_dim, int) else np.prod(first_agent.action_dim)
                        num_main_agents = len(main_agent_ids)
                        
                        # 拆分聚合的观测和动作
                        batches = {}
                        batch_size = len(obs) if isinstance(obs, np.ndarray) else obs.shape[0] if hasattr(obs, 'shape') else 1
                        
                        for idx, agent_id in enumerate(main_agent_ids):
                            # 计算该agent在聚合数据中的索引范围
                            obs_start = int(idx * single_obs_dim)
                            obs_end = int((idx + 1) * single_obs_dim)
                            action_start = int(idx * single_action_dim)
                            action_end = int((idx + 1) * single_action_dim)
                            
                            # 拆分观测和动作
                            if isinstance(obs, np.ndarray):
                                if len(obs.shape) > 1:
                                    # 批次数据：shape = (batch_size, aggregated_obs_dim)
                                    agent_obs = obs[:, obs_start:obs_end]
                                else:
                                    # 单个样本：shape = (aggregated_obs_dim,)
                                    agent_obs = obs[obs_start:obs_end]
                            else:
                                agent_obs = obs
                            
                            # 拆分动作数据
                            if isinstance(actions, np.ndarray):
                                if len(actions.shape) > 1:
                                    # 批次数据：shape = (batch_size, aggregated_action_dim)
                                    # 对于离散动作，每个agent的动作是标量，所以action_dim=1
                                    if single_action_dim == 1:
                                        # 单个动作：直接取对应列
                                        if action_start < actions.shape[1]:
                                            agent_actions = actions[:, action_start].astype(np.int64)
                                        else:
                                            # 索引超出范围，使用第一个agent的动作
                                            agent_actions = actions[:, 0].astype(np.int64)
                                        # 确保是1维数组（batch_size,），不是2D
                                        # actions[:, action_start] 应该返回 (batch_size,) 的1维数组
                                        if agent_actions.ndim == 0:
                                            agent_actions = np.array([agent_actions], dtype=np.int64)
                                        elif agent_actions.ndim == 2:
                                            # 如果仍然是2D，展平
                                            agent_actions = agent_actions.flatten()
                                        # 确保是1维数组
                                        assert agent_actions.ndim == 1, f"Expected 1D array, got shape {agent_actions.shape}"
                                    else:
                                        # 多个动作维度：取对应范围
                                        if action_end <= actions.shape[1]:
                                            agent_actions = actions[:, action_start:action_end].astype(np.int64)
                                        else:
                                            # 索引超出范围，使用第一个agent的动作范围
                                            agent_actions = actions[:, 0:single_action_dim].astype(np.int64)
                                else:
                                    # 单个样本：shape = (aggregated_action_dim,)
                                    if single_action_dim == 1:
                                        # 单个动作：直接取对应位置
                                        if action_start < len(actions):
                                            agent_actions = np.array([int(actions[action_start])], dtype=np.int64)
                                        else:
                                            # 索引超出范围，使用第一个agent的动作
                                            agent_actions = np.array([int(actions[0])], dtype=np.int64)
                                    else:
                                        # 多个动作维度：取对应范围
                                        if action_end <= len(actions):
                                            agent_actions = actions[action_start:action_end].astype(np.int64)
                                        else:
                                            # 索引超出范围，使用第一个agent的动作范围
                                            agent_actions = actions[0:single_action_dim].astype(np.int64)
                            else:
                                # 如果不是数组，尝试转换
                                if isinstance(actions, (list, tuple)):
                                    actions_array = np.array(actions, dtype=np.int64)
                                    if len(actions_array.shape) > 1:
                                        # 批次数据
                                        if single_action_dim == 1:
                                            if action_start < actions_array.shape[1]:
                                                agent_actions = actions_array[:, action_start]
                                            else:
                                                agent_actions = actions_array[:, 0]
                                        else:
                                            if action_end <= actions_array.shape[1]:
                                                agent_actions = actions_array[:, action_start:action_end]
                                            else:
                                                agent_actions = actions_array[:, 0:single_action_dim]
                                    else:
                                        # 单个样本
                                        if single_action_dim == 1:
                                            if action_start < len(actions_array):
                                                agent_actions = np.array([int(actions_array[action_start])], dtype=np.int64)
                                            else:
                                                agent_actions = np.array([int(actions_array[0])], dtype=np.int64)
                                        else:
                                            if action_end <= len(actions_array):
                                                agent_actions = actions_array[action_start:action_end].astype(np.int64)
                                            else:
                                                agent_actions = actions_array[0:single_action_dim].astype(np.int64)
                                else:
                                    # 其他类型，尝试转换
                                    try:
                                        agent_actions = np.array([int(actions)], dtype=np.int64) if single_action_dim == 1 else np.array(actions, dtype=np.int64)
                                    except (ValueError, TypeError):
                                        # 如果转换失败，使用默认值
                                        agent_actions = np.array([0], dtype=np.int64) if single_action_dim == 1 else np.zeros(single_action_dim, dtype=np.int64)
                            
                            # 验证agent_actions不为空，并确保格式正确
                            if isinstance(agent_actions, np.ndarray):
                                if agent_actions.size == 0:
                                    # 空数组，跳过这个agent
                                    continue
                                # 确保是1维数组（对于批次数据）或标量（对于单个样本）
                                if agent_actions.ndim == 0:
                                    # 标量，转换为1维数组
                                    agent_actions = np.array([agent_actions], dtype=np.int64)
                                elif agent_actions.ndim == 2:
                                    # 2D数组，需要展平为1D
                                    # 对于离散动作，应该是 (batch_size, 1) -> (batch_size,)
                                    if agent_actions.shape[1] == 1:
                                        agent_actions = agent_actions.flatten()
                                    else:
                                        # 如果第二维不是1，说明可能是多个动作维度
                                        # 对于离散动作，这不应该发生，但为了安全，我们取第一列
                                        if single_action_dim == 1:
                                            agent_actions = agent_actions[:, 0].flatten()
                                        else:
                                            # 多个动作维度，保持形状但确保是整数
                                            agent_actions = agent_actions.astype(np.int64)
                                elif agent_actions.ndim > 2:
                                    # 多维数组，展平为1D
                                    agent_actions = agent_actions.flatten()
                            elif isinstance(agent_actions, (int, np.integer)):
                                # 标量整数，转换为1维数组
                                agent_actions = np.array([int(agent_actions)], dtype=np.int64)
                            elif isinstance(agent_actions, (list, tuple)):
                                # 列表或元组，转换为numpy数组
                                agent_actions = np.array(agent_actions, dtype=np.int64)
                                if agent_actions.size == 0:
                                    continue
                                # 确保是1维数组
                                if agent_actions.ndim > 1:
                                    agent_actions = agent_actions.flatten()
                            else:
                                # 其他类型，尝试转换
                                try:
                                    agent_actions = np.array([int(agent_actions)], dtype=np.int64)
                                except (ValueError, TypeError):
                                    # 转换失败，跳过这个agent
                                    continue
                            
                            # 确保动作数组的长度与批次大小一致
                            # 对于离散动作，动作应该是1维数组 (batch_size,)
                            if isinstance(obs, np.ndarray) and len(obs.shape) > 1:
                                # 批次数据
                                batch_size = obs.shape[0]
                                if isinstance(agent_actions, np.ndarray):
                                    # 确保是1维数组
                                    if agent_actions.ndim > 1:
                                        agent_actions = agent_actions.flatten()
                                    
                                    if len(agent_actions.shape) == 1:
                                        # 1D数组，检查长度
                                        if len(agent_actions) != batch_size:
                                            # 长度不匹配，尝试调整
                                            if len(agent_actions) == 1:
                                                # 单个值，广播到批次大小
                                                agent_actions = np.repeat(agent_actions, batch_size)
                                            elif len(agent_actions) > batch_size:
                                                # 截断
                                                agent_actions = agent_actions[:batch_size]
                                            else:
                                                # 填充（使用最后一个值）
                                                last_val = agent_actions[-1] if len(agent_actions) > 0 else 0
                                                agent_actions = np.concatenate([agent_actions, np.full(batch_size - len(agent_actions), last_val, dtype=np.int64)])
                                    else:
                                        # 不是1D数组，强制展平
                                        agent_actions = agent_actions.flatten()
                                        if len(agent_actions) != batch_size:
                                            # 调整长度
                                            if len(agent_actions) == 1:
                                                agent_actions = np.repeat(agent_actions, batch_size)
                                            elif len(agent_actions) > batch_size:
                                                agent_actions = agent_actions[:batch_size]
                                            else:
                                                last_val = agent_actions[-1] if len(agent_actions) > 0 else 0
                                                agent_actions = np.concatenate([agent_actions, np.full(batch_size - len(agent_actions), last_val, dtype=np.int64)])
                            
                            # 为每个agent创建独立的batch
                            # 注意：advantages、returns、old_values等是聚合后的平均值，直接使用
                            batches[agent_id] = {
                                "obs": agent_obs,
                                "actions": agent_actions,
                                "logprobs": old_logprobs,  # 使用平均对数概率
                                "advantages": advantages,
                                "returns": returns,
                                "old_values": old_values,
                                "value_coef": self.value_coef,
                                "entropy_coef": self.entropy_coef,
                                "clip_coef": self.clip_coef,
                                "vf_clip_param": self.config.get("vf_clip_param"),
                            }
                        
                        # 只训练主团队agent
                        metrics = self.agent.learn(batches)
                        all_metrics.extend(list(metrics.values()) if isinstance(metrics, dict) else [metrics])
                    else:
                        # 普通多Agent场景：为每个agent创建批次（使用聚合的数据）
                        batches = {}
                        agent_ids = self.agent.agent_ids if hasattr(self.agent, "agent_ids") else []
                        for agent_id in agent_ids:
                            batches[agent_id] = batch
                        metrics = self.agent.learn(batches)
                        all_metrics.extend(list(metrics.values()) if isinstance(metrics, dict) else [metrics])
                else:
                    # 单Agent
                    metrics = self.agent.learn(batch)
                    all_metrics.append(metrics)
                
                # 收集价值和回报用于计算explained variance
                values_epoch.append(rollout_data.old_values)
                returns_epoch.append(rollout_data.returns)
        
        # 聚合指标
        if not all_metrics:
            return {}
        
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = float(np.mean(values))
        
        # 计算explained variance
        if values_epoch and returns_epoch:
            if isinstance(values_epoch[0], torch.Tensor):
                values_all = torch.cat([v.flatten() for v in values_epoch])
                returns_all = torch.cat([r.flatten() for r in returns_epoch])
            else:
                values_all = np.concatenate([v.flatten() for v in values_epoch])
                returns_all = np.concatenate([r.flatten() for r in returns_epoch])
            
            if isinstance(values_all, torch.Tensor):
                explained_var = self._explained_variance(values_all, returns_all)
            else:
                explained_var = self._explained_variance(
                    torch.tensor(values_all), torch.tensor(returns_all)
                )
            aggregated["explained_variance"] = float(explained_var.item() if isinstance(explained_var, torch.Tensor) else explained_var)
        
        return aggregated
    
    def _explained_variance(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算解释方差
        
        Args:
            y_pred: 预测值
            y_true: 真实值
        
        Returns:
            解释方差
        """
        var_y_true = torch.var(y_true, unbiased=True)
        if var_y_true == 0:
            return torch.tensor(float('nan'))
        variance_explained = 1 - torch.var(y_true - y_pred, unbiased=True) / var_y_true
        return variance_explained

