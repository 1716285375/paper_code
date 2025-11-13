# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : self_play_trainer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-06 00:00
@Update Date    :
@Description    : HP3O自博弈训练器
实现HP3O算法的自博弈训练循环
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from algorithms.hp3o.trainers.base_trainer import HP3OTrainer
from algorithms.hp3o.config import HP3OConfig

# Logger是可选的
try:
    from common.utils.logging import LoggerManager
    Logger = LoggerManager
except ImportError:
    Logger = None


class SelfPlayHP3OTrainer(HP3OTrainer):
    """
    HP3O自博弈训练器
    
    在HP3O基础上添加自博弈功能：
    - 只训练主团队
    - 定期更新对手策略
    - 支持策略池管理
    """
    
    def __init__(
        self,
        agent: Any,
        env: Any,
        config: Dict[str, Any],
        logger: Optional[Logger] = None,
        tracker: Optional[Any] = None,
        main_team: str = "team_red",
        opponent_team: str = "team_blue",
    ):
        """
        初始化HP3O自博弈训练器
        
        Args:
            agent: AgentManager（多Agent环境必需）
            env: 环境实例
            config: 训练配置（包含HP3O和自博弈参数）
            logger: 日志记录器（可选）
            tracker: 实验跟踪器（可选）
            main_team: 主团队名称（用于训练）
            opponent_team: 对手团队名称（用于自博弈）
        """
        # 调用父类初始化
        super().__init__(agent, env, config, logger, tracker)
        
        # 自博弈参数
        self.main_team = main_team
        self.opponent_team = opponent_team
        self.self_play_update_freq = config.get(
            "self_play_update_freq", HP3OConfig.DEFAULT_SELF_PLAY_UPDATE_FREQ
        )
        
        # 策略池（如果需要）
        self.use_policy_pool = config.get("use_policy_pool", False)
        if self.use_policy_pool:
            from core.agent.opponent_pool import OpponentPool
            policy_pool_size = config.get("policy_pool_size", HP3OConfig.DEFAULT_POLICY_POOL_SIZE)
            self.opponent_pool = OpponentPool(
                max_size=policy_pool_size,
                strategy=config.get("opponent_pool_strategy", "uniform"),
            )
        else:
            self.opponent_pool = None
        
        # 分离主团队和对手团队的agent
        self._separate_teams()
        
        # 初始化轨迹缓冲区（在团队分离后）
        self._init_trajectory_buffer()
    
    def _separate_teams(self) -> None:
        """分离主团队和对手团队的agent"""
        if not hasattr(self.agent, "agent_ids"):
            raise ValueError("Agent must be AgentManager for self-play training")
        
        agent_ids = self.agent.agent_ids
        self.main_agent_ids = [
            aid for aid in agent_ids
            if aid.startswith(self.main_team.replace("team_", "") + "_")
        ]
        self.opponent_agent_ids = [
            aid for aid in agent_ids
            if aid.startswith(self.opponent_team.replace("team_", "") + "_")
        ]
        
        if len(self.main_agent_ids) == 0 or len(self.opponent_agent_ids) == 0:
            # 如果精确匹配失败，尝试模糊匹配
            main_prefix = self.main_team.replace("team_", "").split("_")[0]
            opponent_prefix = self.opponent_team.replace("team_", "").split("_")[0]
            
            self.main_agent_ids = [aid for aid in agent_ids if main_prefix in aid.lower()]
            self.opponent_agent_ids = [aid for aid in agent_ids if opponent_prefix in aid.lower()]
        
        # 去重并确保不重叠
        self.main_agent_ids = list(set(self.main_agent_ids))
        self.opponent_agent_ids = list(set(self.opponent_agent_ids))
        
        # 确保没有重叠
        self.opponent_agent_ids = [
            aid for aid in self.opponent_agent_ids if aid not in self.main_agent_ids
        ]
        
        if self.logger:
            log_msg = f"Main team agents ({len(self.main_agent_ids)}): {self.main_agent_ids[:5]}..."
            if hasattr(self.logger, "logger"):
                self.logger.logger.info(log_msg)
            elif hasattr(self.logger, "info"):
                self.logger.info(log_msg)
    
    def train(self, num_updates: int) -> None:
        """
        执行训练循环
        
        Args:
            num_updates: 训练更新次数
        """
        # 确保agent处于训练模式
        self.agent.to_training_mode()
        
        for update in range(num_updates):
            # 收集rollout数据
            rollout_data = self.rollout_collector.collect()
            
            # 处理rollout数据（转换为轨迹格式）
            processed_data = self._process_rollout(rollout_data)
            
            # 批量训练（只训练主团队）
            metrics = self._train_step_self_play(processed_data)
            
            # 更新计数
            self.update_count += 1
            self.episode_count += 1
            self.step_count += len(rollout_data.get("obs", [])) if not self.is_multi_agent else sum(
                len(data.get("obs", [])) for data in rollout_data.values()
            )
            
            # 记录日志
            if self.update_count % self.log_freq == 0:
                self._log_metrics(metrics, rollout_data)
            
            # 自博弈更新：更新对手策略
            if self.update_count % self.self_play_update_freq == 0:
                self._update_opponent_policy()
                if self.logger:
                    if hasattr(self.logger, "logger"):
                        self.logger.logger.info(
                            f"Updated opponent policy at step {self.update_count}"
                        )
                    elif hasattr(self.logger, "info"):
                        self.logger.info(f"Updated opponent policy at step {self.update_count}")
            
            # 评估
            if self.update_count % self.eval_freq == 0:
                self._sync_opponent_policy()
                # 使用基础评估器（只支持num_episodes）
                # 注意：MultiAgentEvaluator支持compute_diversity和compute_cooperation，
                # 但BaseAlgorithmTrainer使用的是基础Evaluator
                eval_metrics = self.evaluator.evaluate(num_episodes=10)
                eval_log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
                
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
                            if k == "eval/update":
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
                                self.logger.logger.warning(f"Failed to log eval metrics: {e}")
            
            # 保存
            if self.update_count % self.save_freq == 0:
                checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
                self.save(f"{checkpoint_dir}/hp3o_selfplay_checkpoint_{self.update_count}.pt")
    
    def _train_step_self_play(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        执行一步自博弈训练（只训练主团队）
        
        Args:
            processed_data: 处理后的数据
        
        Returns:
            训练指标
        """
        # 使用父类的训练逻辑（轨迹重放）
        return self._train_step(processed_data)
    
    def _update_opponent_policy(self) -> None:
        """更新对手策略"""
        if self.use_policy_pool and self.opponent_pool is not None:
            # 从策略池采样（返回state_dict）
            opponent_state = self.opponent_pool.sample_opponent()
            if opponent_state is not None:
                # 将采样策略应用到对手团队
                for agent_id in self.opponent_agent_ids:
                    self.agent.get_agent(agent_id).load_state_dict(opponent_state)
        else:
            # 直接复制主团队策略
            self._sync_opponent_policy()
            
            # 如果使用策略池，保存当前策略
            if self.use_policy_pool and self.opponent_pool is not None:
                main_agent = self.agent.get_agent(self.main_agent_ids[0])
                main_state_dict = main_agent.state_dict()
                # 使用add_policy方法添加策略
                self.opponent_pool.add_policy(main_state_dict)
    
    def _sync_opponent_policy(self) -> None:
        """同步对手策略（从主团队复制）"""
        if len(self.main_agent_ids) == 0 or len(self.opponent_agent_ids) == 0:
            return
        
        # 获取主团队的第一个agent的策略
        main_agent = self.agent.get_agent(self.main_agent_ids[0])
        main_state_dict = main_agent.state_dict()
        
        # 应用到所有对手agent
        for agent_id in self.opponent_agent_ids:
            opponent_agent = self.agent.get_agent(agent_id)
            opponent_agent.load_state_dict(main_state_dict)

