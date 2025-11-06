# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : mixins.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : 训练器混入类
提供可复用的训练器功能（如自博弈混入）
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

from core.agent.opponent_pool import OpponentPool


class SelfPlayMixin:
    """
    自博弈训练器混入类
    
    提供自博弈训练的通用功能，可以被任何训练器类继承
    """

    def __init_self_play__(self, config: Dict[str, Any], main_team: str, opponent_team: str):
        """
        初始化自博弈组件（由子类在__init__中调用）

        Args:
            config: 训练配置
            main_team: 主团队名称
            opponent_team: 对手团队名称
        """
        self.main_team = main_team
        self.opponent_team = opponent_team

        # 自博弈配置
        self_play_config = config.get("self_play", {})
        self.self_play_update_freq = self_play_config.get("update_freq", 10)
        self.use_policy_pool = self_play_config.get("use_policy_pool", True)
        self.policy_pool_size = self_play_config.get("policy_pool_size", 15)
        self.snapshot_freq = self_play_config.get("snapshot_freq", 50000)

        # 创建策略池（使用 core 中的 OpponentPool）
        if self.use_policy_pool:
            strategy = self_play_config.get("strategy", "pfsp")
            elo_temperature = self_play_config.get("elo_temperature", 1.0)
            pfsp_temperature = self_play_config.get("pfsp_temperature", 1.0)
            device = getattr(self, "device", "cpu")
            
            self.policy_pool = OpponentPool(
                max_size=self.policy_pool_size,
                strategy=strategy,
                elo_temperature=elo_temperature,
                pfsp_temperature=pfsp_temperature,
                device=device,
            )
        else:
            self.policy_pool = None

    def _initialize_opponent_pool(self) -> None:
        """初始化对手池（添加初始对手策略）"""
        if not self.use_policy_pool or self.policy_pool is None:
            return

        # 获取对手团队的策略状态
        opponent_agents = self.agent.get_group_members(self.opponent_team)
        if opponent_agents:
            sample_agent_id = opponent_agents[0]
            opponent_state = self.agent.get_agent(sample_agent_id).state_dict()
            # core 的 OpponentPool.add_policy 不接受 metadata 参数
            self.policy_pool.add_policy(opponent_state)

    def _update_opponent_from_pool(self) -> None:
        """从策略池采样并更新对手策略"""
        if not self.use_policy_pool or self.policy_pool is None:
            return

        opponent_state = self.policy_pool.sample_opponent()
        if opponent_state is not None:
            # 更新对手团队的所有agent
            opponent_agents = self.agent.get_group_members(self.opponent_team)
            for agent_id in opponent_agents:
                agent = self.agent.get_agent(agent_id)
                agent.load_state_dict(opponent_state)

    def _snapshot_opponent_policy(self, step: int) -> Optional[int]:
        """快照对手策略到池中"""
        if not self.use_policy_pool or self.policy_pool is None:
            return None

        opponent_agents = self.agent.get_group_members(self.opponent_team)
        if opponent_agents:
            sample_agent_id = opponent_agents[0]
            opponent_state = self.agent.get_agent(sample_agent_id).state_dict()
            
            pool_index = self.policy_pool.add_policy(opponent_state)
            return pool_index
        return None

