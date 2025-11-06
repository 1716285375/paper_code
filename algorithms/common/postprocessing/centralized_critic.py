# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : centralized_critic.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-05 00:00
@Update Date    :
@Description    : 集中式Critic后处理函数
用于在rollout数据处理后，使用集中式Critic计算价值函数预测
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


def centralized_critic_postprocessing(
    processed_data: Dict[str, Any],
    agent_manager: Any,
    use_centralized_critic: bool = True,
    opp_action_in_cc: bool = False,
) -> Dict[str, Any]:
    """
    集中式Critic后处理函数
    
    在rollout数据处理后，使用集中式Critic重新计算价值函数预测。
    这允许使用全局状态和对手动作来改进价值估计。
    
    Args:
        processed_data: 处理后的rollout数据（包含state、advantages等）
        agent_manager: Agent管理器，用于访问集中式Critic
        use_centralized_critic: 是否使用集中式Critic
        opp_action_in_cc: 是否在集中式Critic中使用对手动作
    
    Returns:
        更新后的processed_data，包含新的value预测
    """
    if not use_centralized_critic:
        return processed_data
    
    # 处理多Agent情况
    if isinstance(processed_data, dict) and any(
        isinstance(v, dict) and "obs" in v for v in processed_data.values()
    ):
        # 多Agent格式
        for agent_id, data in processed_data.items():
            agent = agent_manager.get_agent(agent_id)
            
            # 检查是否有集中式Critic
            if not hasattr(agent, "central_critic") or agent.central_critic is None:
                continue
            
            # 获取全局状态
            state = data.get("state")
            if state is None:
                # 如果没有state，跳过（使用局部Critic）
                continue
            
            # 转换为tensor
            if isinstance(state, np.ndarray):
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device)
            else:
                state_tensor = state.to(agent.device) if hasattr(state, "to") else state
            
            # 获取对手动作（如果使用）
            opponent_actions = None
            if opp_action_in_cc:
                opponent_actions = _get_opponent_actions(agent_id, processed_data)
                if opponent_actions is not None:
                    if isinstance(opponent_actions, np.ndarray):
                        opponent_actions = torch.as_tensor(
                            opponent_actions, dtype=torch.long, device=agent.device
                        )
            
            # 使用集中式Critic计算价值
            with torch.no_grad():
                try:
                    central_values = agent.central_value_function(state_tensor, opponent_actions)
                    
                    # 转换为numpy（如果原来是numpy）
                    if isinstance(data.get("values"), np.ndarray):
                        central_values = central_values.cpu().numpy()
                    
                    # 更新values（可选：可以混合使用局部和集中式价值）
                    # 这里直接用集中式价值替换
                    data["values"] = central_values
                    data["use_centralized_critic"] = True
                except Exception as e:
                    # 如果计算失败，保持原值
                    if hasattr(agent_manager, "logger"):
                        agent_manager.logger.warning(
                            f"Failed to compute central value for {agent_id}: {e}"
                        )
                    continue
    
    else:
        # 单Agent格式
        agent = agent_manager if hasattr(agent_manager, "central_critic") else None
        
        if agent is not None and hasattr(agent, "central_critic") and agent.central_critic is not None:
            state = processed_data.get("state")
            if state is not None:
                if isinstance(state, np.ndarray):
                    state_tensor = torch.as_tensor(
                        state, dtype=torch.float32, device=agent.device
                    )
                else:
                    state_tensor = state.to(agent.device) if hasattr(state, "to") else state
                
                with torch.no_grad():
                    try:
                        central_values = agent.central_value_function(state_tensor, None)
                        if isinstance(processed_data.get("values"), np.ndarray):
                            central_values = central_values.cpu().numpy()
                        processed_data["values"] = central_values
                        processed_data["use_centralized_critic"] = True
                    except Exception:
                        pass
    
    return processed_data


def _get_opponent_actions(
    agent_id: str, processed_data: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    获取对手动作
    
    Args:
        agent_id: 当前Agent ID
        processed_data: 所有agent的数据
    
    Returns:
        对手动作数组，形状为 (T, n_opponents) 或 None
    """
    opponent_actions_list = []
    
    for other_agent_id, other_data in processed_data.items():
        if other_agent_id != agent_id and "actions" in other_data:
            actions = other_data["actions"]
            opponent_actions_list.append(actions)
    
    if opponent_actions_list:
        # 堆叠成 (T, n_opponents)
        opponent_actions = np.stack(opponent_actions_list, axis=1)
        return opponent_actions
    
    return None

