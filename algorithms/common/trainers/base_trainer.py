# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : 算法训练器基类
所有算法训练器的基础类，包含通用的训练循环、数据处理、评估等功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from core.base.trainer import Trainer
from core.agent import AgentManager
from core.base.agent import Agent
from core.base.environment import Env
from core.modules.advantage_estimators import GAE
from core.trainer import Evaluator, RolloutCollector

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


class BaseAlgorithmTrainer(Trainer):
    """
    算法训练器基类
    
    所有算法训练器的基础类，提供：
    - 通用的训练循环框架
    - Rollout数据收集
    - 优势和回报计算
    - 评估功能
    - 保存/加载功能
    - 日志记录
    
    各个算法训练器继承此类，只需实现：
    - _train_step: 具体的训练步骤（算法特定的更新逻辑）
    """
    
    def __init__(
        self,
        agent: Agent | AgentManager,
        env: Env,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
    ):
        """
        初始化基础训练器
        
        Args:
            agent: Agent实例（单个Agent）或AgentManager（多Agent）
            env: 环境实例
            config: 训练配置
                - max_steps_per_episode: 每个episode最大步数
                - gamma: 折扣因子
                - gae_lambda: GAE lambda参数
                - eval_freq: 评估频率（每N个更新）
                - save_freq: 保存频率（每N个更新）
                - log_freq: 日志记录频率
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选，用于wandb/tensorboard）
        """
        self.agent = agent
        self.env = env
        self.config = config
        
        # 如果没有提供logger，尝试创建一个简单的
        if logger is None:
            try:
                self.logger = Logger(name="base_trainer") if Logger else None
            except:
                self.logger = None
        else:
            self.logger = logger
        
        # 实验跟踪器（用于wandb/tensorboard）
        self.tracker = tracker
        
        # 数据管理器（用于保存指标和视频，可选）
        self.data_manager = None  # 由外部设置
        
        # 配置参数
        self.max_steps_per_episode = config.get("max_steps_per_episode", 1000)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.eval_freq = config.get("eval_freq", 10)
        self.save_freq = config.get("save_freq", 50)
        self.log_freq = config.get("log_freq", 1)
        
        # GAE估计器
        self.gae = GAE(gamma=self.gamma, lam=self.gae_lambda)
        
        # 判断是否为多Agent
        self.is_multi_agent = isinstance(agent, AgentManager)
        
        # 使用通用的RolloutCollector
        self.rollout_collector = RolloutCollector(
            agent=agent,
            env=env,
            max_steps_per_episode=self.max_steps_per_episode,
            is_multi_agent=self.is_multi_agent,
        )
        
        # 使用通用的Evaluator
        self.evaluator = Evaluator(
            agent=agent,
            env=env,
            max_steps_per_episode=self.max_steps_per_episode,
            is_multi_agent=self.is_multi_agent,
        )
        
        # 训练状态
        self.update_count = 0
        self.episode_count = 0
        self.step_count = 0
        
        # Checkpoint目录
        self.checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    
    def train(self, num_updates: int) -> None:
        """
        执行训练循环
        
        Args:
            num_updates: 训练更新次数
        """
        self.agent.to_training_mode()
        
        for update in range(num_updates):
            # 收集数据
            rollout_data = self.rollout_collector.collect()
            self.episode_count += 1
            
            # 更新步数
            if self.is_multi_agent:
                for agent_id, data in rollout_data.items():
                    if isinstance(data, dict) and "obs" in data:
                        self.step_count += len(data["obs"])
            else:
                self.step_count += len(rollout_data.get("obs", []))
            
            # 计算优势和回报
            processed_data = self._process_rollout(rollout_data)
            
            # 执行训练步骤（由子类实现）
            metrics = self._train_step(processed_data)
            
            # 更新计数
            self.update_count += 1
            
            # 记录日志
            if self.update_count % self.log_freq == 0:
                self._log_metrics(metrics, rollout_data)
            
            # 评估
            if self.update_count % self.eval_freq == 0:
                eval_metrics = self.evaluates(num_episodes=5)
                self._log_eval_metrics(eval_metrics)
            
            # 保存
            if self.update_count % self.save_freq == 0:
                self.save(f"{self.checkpoint_dir}/checkpoint_{self.update_count}.pt")
    
    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        训练步骤（由子类实现）
        
        Args:
            processed_data: 处理后的rollout数据（包含advantages和returns）
        
        Returns:
            训练指标字典
        """
        raise NotImplementedError("Subclasses must implement _train_step")
    
    def _process_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理rollout数据，计算优势和回报
        
        Args:
            rollout_data: 原始rollout数据
        
        Returns:
            处理后的数据（包含advantages和returns）
        """
        if self.is_multi_agent:
            processed_data = self._process_multi_agent_rollout(rollout_data)
        else:
            processed_data = self._process_single_agent_rollout(rollout_data)
        
        # 集中式Critic后处理（如果使用）
        if self.config.get("use_centralized_critic", False):
            from algorithms.common.postprocessing.centralized_critic import (
                centralized_critic_postprocessing,
            )
            
            processed_data = centralized_critic_postprocessing(
                processed_data=processed_data,
                agent_manager=self.agent,
                use_centralized_critic=self.config.get("use_centralized_critic", False),
                opp_action_in_cc=self.config.get("opp_action_in_cc", False),
            )
        
        return processed_data
    
    def _process_single_agent_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单Agent rollout数据"""
        rewards = rollout_data["rewards"]
        values = rollout_data["values"]
        dones = rollout_data["dones"]
        
        # 保存旧的价值估计（用于价值函数裁剪）
        rollout_data["old_values"] = values.copy()
        
        # 计算returns和advantages
        returns, advantages = self.gae.compute(rewards, values, dones)
        
        rollout_data["returns"] = returns
        rollout_data["advantages"] = advantages
        
        return rollout_data
    
    def _process_multi_agent_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理多Agent rollout数据"""
        processed_data = {}
        
        for agent_id, data in rollout_data.items():
            rewards = data["rewards"]
            values = data["values"]
            dones = data["dones"]
            
            # 保存旧的价值估计（用于价值函数裁剪）
            processed_data[agent_id] = {
                **data,
                "old_values": values.copy() if isinstance(values, np.ndarray) else values,
            }
            
            # 计算returns和advantages
            returns, advantages = self.gae.compute(rewards, values, dones)
            
            processed_data[agent_id]["returns"] = returns
            processed_data[agent_id]["advantages"] = advantages
        
        return processed_data
    
    def evaluates(self, num_episodes: int) -> Dict[str, Any]:
        """
        评估Agent的性能
        
        Args:
            num_episodes: 评估的episode数量
        
        Returns:
            评估指标字典
        """
        return self.evaluator.evaluate(num_episodes=num_episodes)
    
    def save(self, path: str) -> None:
        """
        保存训练器状态（包括Agent模型）
        
        Args:
            path: 保存路径
        """
        import os
        import torch
        
        # 创建目录
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存状态
        state = {
            "agent_state": self.agent.state_dict() if hasattr(self.agent, "state_dict") else None,
            "update_count": self.update_count,
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "config": self.config,
        }
        
        torch.save(state, path)
        
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Checkpoint saved to {path}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Checkpoint saved to {path}")
    
    def load(self, path: str) -> None:
        """
        从文件加载训练器状态（包括Agent模型）
        
        Args:
            path: 加载路径
        """
        import torch
        
        state = torch.load(path, map_location="cpu")
        
        # 加载agent状态
        if state.get("agent_state") is not None and hasattr(self.agent, "load_state_dict"):
            self.agent.load_state_dict(state["agent_state"])
        
        # 恢复训练状态
        self.update_count = state.get("update_count", 0)
        self.episode_count = state.get("episode_count", 0)
        self.step_count = state.get("step_count", 0)
        
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Checkpoint loaded from {path}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Checkpoint loaded from {path}")
    
    def _log_metrics(self, metrics: Dict[str, float], rollout_data: Dict[str, Any]) -> None:
        """
        记录训练指标
        
        Args:
            metrics: 训练指标字典
            rollout_data: rollout数据（用于计算额外指标）
        """
        # 计算额外指标
        log_dict = {**metrics}
        
        if self.is_multi_agent:
            # 计算平均奖励
            rewards = []
            for agent_id, data in rollout_data.items():
                if isinstance(data, dict) and "rewards" in data:
                    rewards.extend(data["rewards"])
            if rewards:
                log_dict["avg_reward"] = np.mean(rewards)
                log_dict["total_reward"] = np.sum(rewards)
        else:
            if "rewards" in rollout_data:
                log_dict["avg_reward"] = np.mean(rollout_data["rewards"])
                log_dict["total_reward"] = np.sum(rollout_data["rewards"])
        
        log_dict["update"] = self.update_count
        log_dict["episode"] = self.episode_count
        log_dict["steps"] = self.step_count
        
        # 使用logger输出
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Update {self.update_count}: {log_dict}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Update {self.update_count}: {log_dict}")
        
        # 记录到tracker
        if self.tracker and self.tracker.is_initialized:
            try:
                # 过滤掉非标量值（字典、列表等），只保留标量值
                tracker_metrics = {}
                for k, v in log_dict.items():
                    # 跳过元数据字段
                    if k in ["update", "episode", "steps"]:
                        continue
                    # 只保留标量值（int, float, numpy标量）
                    if isinstance(v, (int, float, np.number)):
                        tracker_metrics[k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
                    elif isinstance(v, np.ndarray) and v.size == 1:
                        # 单元素数组，提取标量
                        tracker_metrics[k] = float(v.item())
                    # 跳过字典、列表等非标量类型
                
                if tracker_metrics:
                    self.tracker.log(tracker_metrics, step=self.update_count)
            except Exception as e:
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.warning(f"Failed to log to tracker: {e}")
        
        # 保存到数据管理器
        if self.data_manager is not None:
            self.data_manager.update_metrics(log_dict, step=self.update_count)
    
    def _log_eval_metrics(self, eval_metrics: Dict[str, Any]) -> None:
        """
        记录评估指标
        
        Args:
            eval_metrics: 评估指标字典
        """
        # 添加eval前缀
        eval_log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
        eval_log_dict["update"] = self.update_count
        
        # 使用logger输出
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
                # 过滤掉非标量值（字典、列表等），只保留标量值
                eval_metrics_for_tracker = {}
                for k, v in eval_log_dict.items():
                    # 跳过元数据字段
                    if k == "update":
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
                        self.logger.logger.warning(f"Failed to log eval metrics to tracker: {e}")
        
        # 保存到数据管理器
        if self.data_manager is not None:
            eval_metrics_for_save = {
                k.replace("eval/", ""): v for k, v in eval_log_dict.items() if k != "update"
            }
            self.data_manager.update_metrics(eval_metrics_for_save, step=self.update_count)

