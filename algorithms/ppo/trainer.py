# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : PPO训练器实现
实现完整的PPO训练循环
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from core.agent import AgentManager
from core.base.agent import Agent
from core.base.environment import Env
from core.base.trainer import Trainer
from core.modules.advantage_estimators import GAE
from core.trainer import Evaluator, RolloutCollector

# Logger是可选的
try:
    from common.utils.logging import LoggerManager

    Logger = LoggerManager  # 使用LoggerManager作为Logger
except ImportError:
    Logger = None

# Tracker是可选的
try:
    from common.tracking import ExperimentTracker

    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    ExperimentTracker = None


class PPOTrainer(Trainer):
    """
    PPO训练器

    实现完整的PPO训练循环，包括：
    - 与环境交互收集数据（使用通用RolloutCollector）
    - 计算优势和回报
    - 批量训练Agent
    - 评估（使用通用Evaluator）
    - 保存/加载
    """

    def __init__(
        self,
        agent: Agent,
        env: Env,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
    ):
        """
        初始化PPO训练器

        Args:
            agent: Agent实例（单个Agent）或AgentManager（多Agent）
            env: 环境实例
            config: 训练配置
                - max_steps_per_episode: 每个episode最大步数
                - num_epochs: PPO更新轮数
                - batch_size: 批次大小
                - clip_coef: PPO裁剪系数
                - value_coef: 价值损失系数
                - entropy_coef: 熵正则化系数
                - gamma: 折扣因子
                - gae_lambda: GAE lambda参数
                - eval_freq: 评估频率（每N个更新）
                - save_freq: 保存频率（每N个更新）
                - log_freq: 日志记录频率
                - max_grad_norm: 梯度裁剪（可选）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选，用于wandb/tensorboard）
        """
        self.agent = agent
        self.env = env
        self.config = config

        # 如果没有提供logger，尝试创建一个简单的
        if logger is None:
            try:
                self.logger = Logger(name="ppo_trainer") if Logger else None
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
        self.num_epochs = config.get("num_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.eval_freq = config.get("eval_freq", 10)
        self.save_freq = config.get("save_freq", 50)
        self.log_freq = config.get("log_freq", 1)
        self.max_grad_norm = config.get("max_grad_norm", None)

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

    def train(self, num_updates: int) -> None:
        """
        执行训练

        Args:
            num_updates: 训练更新次数
        """
        self.agent.to_training_mode()

        for update in range(num_updates):
            # 收集数据
            rollout_data = self.rollout_collector.collect()
            self.episode_count += 1
            self.step_count += len(rollout_data.get("obs", []))

            # 计算优势和回报
            processed_data = self._process_rollout(rollout_data)

            # 批量训练
            metrics = self._train_step(processed_data)

            # 更新计数
            self.update_count += 1

            # 记录日志
            if self.update_count % self.log_freq == 0:
                self._log_metrics(metrics, rollout_data)

            # 评估
            if self.update_count % self.eval_freq == 0:
                eval_metrics = self.evaluates(num_episodes=5)

                # 添加eval前缀以便区分
                eval_log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
                eval_log_dict["update"] = self.update_count

                # 使用logger输出评估结果
                if self.logger:
                    # 构建格式化的评估日志消息
                    eval_log_lines = [
                        "=" * 60,
                        f"Evaluation at Update {self.update_count}",
                        "-" * 60,
                    ]
                    for key, value in eval_metrics.items():
                        eval_log_lines.append(f"{key}: {value:.4f}")
                    eval_log_lines.append("=" * 60)

                    # 使用logger.info输出（会自动输出到控制台和文件）
                    # LoggerManager有一个logger属性，它是标准的logging.Logger对象
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info("\n" + "\n".join(eval_log_lines))
                    elif hasattr(self.logger, "info"):
                        self.logger.info("\n" + "\n".join(eval_log_lines))
                    else:
                        # 回退到print
                        print("\n" + "\n".join(eval_log_lines))

                    # 同时尝试使用log方法（如果支持）
                    try:
                        if hasattr(self.logger, "log"):
                            self.logger.log("eval", eval_metrics, step=self.update_count)
                    except:
                        # 如果logger不支持log方法，则只使用info（已经输出）
                        pass
                else:
                    # 如果没有logger，回退到print
                    print(f"\n{'='*60}")
                    print(f"Evaluation at Update {self.update_count}")
                    print(f"{'-'*60}")
                    for key, value in eval_metrics.items():
                        print(f"{key}: {value:.4f}")
                    print(f"{'='*60}\n")

                # 记录到tracker（wandb/tensorboard）
                if self.tracker and self.tracker.is_initialized:
                    try:
                        # 过滤掉update字段（已通过step参数传递）
                        eval_metrics_for_tracker = {
                            k: v for k, v in eval_log_dict.items() if k != "update"
                        }
                        self.tracker.log(eval_metrics_for_tracker, step=self.update_count)
                    except Exception as e:
                        if self.logger:
                            if hasattr(self.logger, "logger"):
                                self.logger.logger.warning(
                                    f"Failed to log eval metrics to tracker: {e}"
                                )
                            elif hasattr(self.logger, "warning"):
                                self.logger.warning(f"Failed to log eval metrics to tracker: {e}")

                # 保存评估指标到数据管理器
                if self.data_manager is not None:
                    # 移除eval前缀，保存原始指标名
                    eval_metrics_for_save = {
                        k.replace("eval/", ""): v for k, v in eval_log_dict.items() if k != "update"
                    }
                    self.data_manager.update_metrics(eval_metrics_for_save, step=self.update_count)

                    # 可选：录制评估episode的视频
                    if self.data_manager.video_recorder is not None:
                        try:
                            video_path = self.data_manager.record_video(
                                env=self.env,
                                agent=self.agent,
                                episode_name=f"eval_update_{self.update_count}",
                                max_steps=self.max_steps_per_episode,
                                deterministic=True,
                                metadata={"update": self.update_count, "type": "evaluation"},
                            )
                            if video_path and self.logger:
                                if hasattr(self.logger, "logger"):
                                    self.logger.logger.info(
                                        f"Recorded evaluation video: {video_path}"
                                    )
                                elif hasattr(self.logger, "info"):
                                    self.logger.info(f"Recorded evaluation video: {video_path}")
                        except Exception as e:
                            if self.logger:
                                if hasattr(self.logger, "logger"):
                                    self.logger.logger.warning(
                                        f"Failed to record evaluation video: {e}"
                                    )
                                elif hasattr(self.logger, "warning"):
                                    self.logger.warning(f"Failed to record evaluation video: {e}")
                            else:
                                print(f"Warning: Failed to record evaluation video: {e}")

            # 保存
            if self.update_count % self.save_freq == 0:
                self.save(f"checkpoints/ppo_checkpoint_{self.update_count}.pt")

    def _process_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理rollout数据，计算优势和回报

        Args:
            rollout_data: 原始rollout数据

        Returns:
            处理后的数据（包含advantages和returns）
        """
        if self.is_multi_agent:
            return self._process_multi_agent_rollout(rollout_data)
        else:
            return self._process_single_agent_rollout(rollout_data)

    def _process_single_agent_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单Agent rollout数据"""
        rewards = rollout_data["rewards"]
        values = rollout_data["values"]
        dones = rollout_data["dones"]

        # 计算returns和advantages
        returns, advantages = self.gae.compute(rewards, values, dones)

        rollout_data["returns"] = returns
        rollout_data["advantages"] = advantages

        return rollout_data

    def _process_multi_agent_rollout(self, rollout_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理多Agent rollout数据"""
        processed = {}

        for agent_id, data in rollout_data.items():
            rewards = data["rewards"]
            values = data["values"]
            dones = data["dones"]

            # 计算每个agent的returns和advantages
            returns, advantages = self.gae.compute(rewards, values, dones)

            processed[agent_id] = {
                **data,
                "returns": returns,
                "advantages": advantages,
            }

        return processed

    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练

        Args:
            processed_data: 处理后的数据

        Returns:
            训练指标
        """
        if self.is_multi_agent:
            return self._train_step_multi_agent(processed_data)
        else:
            return self._train_step_single_agent(processed_data)

    def _train_step_single_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """单Agent训练步骤"""
        all_metrics = []

        obs = processed_data["obs"]
        actions = processed_data["actions"]
        old_logprobs = processed_data["logprobs"]
        advantages = processed_data["advantages"]
        returns = processed_data["returns"]

        num_samples = len(obs)

        # PPO多轮更新
        for epoch in range(self.num_epochs):
            # 随机打乱
            indices = np.random.permutation(num_samples)

            # 分批训练
            for i in range(0, num_samples, self.batch_size):
                batch_indices = indices[i : i + self.batch_size]

                batch = {
                    "obs": obs[batch_indices],
                    "actions": actions[batch_indices],
                    "logprobs": old_logprobs[batch_indices],
                    "advantages": advantages[batch_indices],
                    "returns": returns[batch_indices],
                    "clip_coef": self.clip_coef,
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                }

                metrics = self.agent.learn(batch)
                all_metrics.append(metrics)

        # 聚合指标
        return self._aggregate_metrics(all_metrics)

    def _train_step_multi_agent(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """多Agent训练步骤"""
        all_metrics = []

        # 为每个agent准备批次数据
        batches = {}
        for agent_id, data in processed_data.items():
            obs = data["obs"]
            actions = data["actions"]
            old_logprobs = data["logprobs"]
            advantages = data["advantages"]
            returns = data["returns"]

            num_samples = len(obs)
            batch_indices = np.random.permutation(num_samples)

            # 只取第一批次（简化处理，也可以多批次）
            indices = batch_indices[: min(self.batch_size, num_samples)]

            batches[agent_id] = {
                "obs": obs[indices],
                "actions": actions[indices],
                "logprobs": old_logprobs[indices],
                "advantages": advantages[indices],
                "returns": returns[indices],
                "clip_coef": self.clip_coef,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
            }

        # 批量学习
        metrics_dict = self.agent.learn(batches)

        # 聚合所有agent的指标
        all_metrics = list(metrics_dict.values())
        return self._aggregate_metrics(all_metrics)

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """聚合指标列表"""
        if not metrics_list:
            return {}

        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = float(np.mean(values))

        return aggregated

    def _log_metrics(self, metrics: Dict[str, float], rollout_data: Dict[str, Any]) -> None:
        """记录指标"""
        log_dict = {
            **metrics,
            "update": self.update_count,
            "episode": self.episode_count,
            "step": self.step_count,
        }

        # 添加episode统计
        if not self.is_multi_agent:
            rewards = rollout_data.get("rewards", [])
            if len(rewards) > 0:
                log_dict["episode_reward"] = float(np.sum(rewards))
                log_dict["episode_length"] = len(rewards)
        else:
            total_reward = 0.0
            total_length = 0
            for agent_id, data in rollout_data.items():
                rewards = data.get("rewards", [])
                total_reward += float(np.sum(rewards))
                total_length += len(rewards)
            log_dict["episode_reward"] = total_reward
            log_dict["episode_length"] = total_length

        # 使用logger输出格式化信息
        if self.logger:
            # 构建格式化的日志消息
            log_lines = [
                "=" * 60,
                f"Update: {self.update_count} | Episode: {self.episode_count} | Step: {self.step_count}",
                "-" * 60,
            ]

            if "policy_loss" in log_dict:
                log_lines.append(f"Policy Loss:  {log_dict['policy_loss']:.6f}")
            if "value_loss" in log_dict:
                log_lines.append(f"Value Loss:   {log_dict['value_loss']:.6f}")
            if "entropy" in log_dict:
                log_lines.append(f"Entropy:      {log_dict['entropy']:.6f}")
            if "episode_reward" in log_dict:
                log_lines.append(f"Episode Reward: {log_dict['episode_reward']:.2f}")
            if "episode_length" in log_dict:
                log_lines.append(f"Episode Length: {log_dict['episode_length']}")

            log_lines.append("=" * 60)

            # 使用logger.info输出（会自动输出到控制台和文件）
            # LoggerManager有一个logger属性，它是标准的logging.Logger对象
            if hasattr(self.logger, "logger"):
                self.logger.logger.info("\n" + "\n".join(log_lines))
            elif hasattr(self.logger, "info"):
                self.logger.info("\n" + "\n".join(log_lines))
            else:
                # 回退到print
                print("\n" + "\n".join(log_lines))

            # 同时尝试使用log方法（如果支持）
            try:
                if hasattr(self.logger, "log"):
                    self.logger.log("train", log_dict, step=self.update_count)
            except:
                # 如果logger不支持log方法，则只使用info（已经输出）
                pass
        else:
            # 如果没有logger，回退到print
            print(f"\n{'='*60}")
            print(
                f"Update: {self.update_count} | Episode: {self.episode_count} | Step: {self.step_count}"
            )
            print(f"{'-'*60}")
            if "policy_loss" in log_dict:
                print(f"Policy Loss:  {log_dict['policy_loss']:.6f}")
            if "value_loss" in log_dict:
                print(f"Value Loss:   {log_dict['value_loss']:.6f}")
            if "entropy" in log_dict:
                print(f"Entropy:      {log_dict['entropy']:.6f}")
            if "episode_reward" in log_dict:
                print(f"Episode Reward: {log_dict['episode_reward']:.2f}")
            if "episode_length" in log_dict:
                print(f"Episode Length: {log_dict['episode_length']}")
            print(f"{'='*60}\n")

        # 记录到tracker（wandb/tensorboard）
        # 只记录真正的训练指标，排除元数据字段（update, episode, step）
        if self.tracker and self.tracker.is_initialized:
            try:
                # 过滤掉元数据字段，只保留训练指标
                metrics_for_tracker = {
                    k: v for k, v in log_dict.items() if k not in ["update", "episode", "step"]
                }

                # 确保所有值都是数字类型
                train_metrics = {}
                for k, v in metrics_for_tracker.items():
                    # 添加train前缀以便在TensorBoard和WandB中分组
                    metric_key = f"train/{k}" if not k.startswith("train/") else k
                    # 确保值是数字类型
                    try:
                        train_metrics[metric_key] = float(v) if not isinstance(v, int) else int(v)
                    except (ValueError, TypeError):
                        continue  # 跳过无法转换的值

                # 只在有数据时记录
                if train_metrics:
                    self.tracker.log(train_metrics, step=self.update_count)
            except Exception as e:
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.warning(f"Failed to log to tracker: {e}")
                    elif hasattr(self.logger, "warning"):
                        self.logger.warning(f"Failed to log to tracker: {e}")
                    else:
                        print(f"Warning: Failed to log to tracker: {e}")
                else:
                    print(f"Warning: Failed to log to tracker: {e}")

        # 保存到数据管理器（用于本地绘图）
        if self.data_manager is not None:
            # 过滤掉元数据字段
            metrics_for_save = {
                k: v for k, v in log_dict.items() if k not in ["update", "episode", "step"]
            }
            self.data_manager.update_metrics(metrics_for_save, step=self.update_count)

    def evaluates(self, num_episodes: int) -> Dict[str, float]:
        """
        评估Agent性能

        Args:
            num_episodes: 评估的episode数量

        Returns:
            评估指标
        """
        return self.evaluator.evaluate(num_episodes=num_episodes)

    def save(self, path: str) -> None:
        """保存训练器状态"""
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
        torch.save(state, path)
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Saved checkpoint to {path}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Saved checkpoint to {path}")
            else:
                print(f"Saved checkpoint to {path}")

    def load(self, path: str) -> None:
        """加载训练器状态"""
        state = torch.load(path)
        self.agent.load_state_dict(state["agent"])
        self.update_count = state.get("update_count", 0)
        self.episode_count = state.get("episode_count", 0)
        self.step_count = state.get("step_count", 0)
        if self.logger:
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(f"Loaded checkpoint from {path}")
            elif hasattr(self.logger, "info"):
                self.logger.info(f"Loaded checkpoint from {path}")
            else:
                print(f"Loaded checkpoint from {path}")
