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

from algorithms.common.trainers.base_trainer import BaseAlgorithmTrainer

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


class PPOTrainer(BaseAlgorithmTrainer):
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
        agent: Any,
        env: Any,
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
        # 调用基类初始化
        super().__init__(agent, env, config, logger, tracker)
        
        # PPO特定配置
        self.num_epochs = config.get("num_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.clip_coef = config.get("clip_coef", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", None)
        
        # PPO-Penalty模式（使用KL惩罚而非clipping）
        self.use_penalty = config.get("use_penalty", False)
        if self.use_penalty and self.clip_coef is not None:
            if self.logger:
                if hasattr(self.logger, "logger"):
                    self.logger.logger.warning(
                        "PPO-Penalty mode does not use clip_coef, ignoring it"
                    )
                elif hasattr(self.logger, "warning"):
                    self.logger.warning("PPO-Penalty mode does not use clip_coef, ignoring it")
                else:
                    print("Warning: PPO-Penalty mode does not use clip_coef, ignoring it")

    def _train_step(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步训练（实现基类的抽象方法）

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
                    "old_values": processed_data.get("old_values", processed_data.get("values", []))[batch_indices],
                    "value_coef": self.value_coef,
                    "entropy_coef": self.entropy_coef,
                    "vf_clip_param": self.config.get("vf_clip_param"),
                }
                # PPO-Penalty模式不使用clip_coef
                if not self.use_penalty:
                    batch["clip_coef"] = self.clip_coef

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
            values = data.get("values", data.get("old_values", []))

            # 优势标准化（统一在所有算法中）
            if isinstance(advantages, np.ndarray):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
                "old_values": values[indices] if len(values) > 0 else returns[indices],
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "vf_clip_param": self.config.get("vf_clip_param"),
            }
            # PPO-Penalty模式不使用clip_coef
            if not self.use_penalty:
                batches[agent_id]["clip_coef"] = self.clip_coef

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

