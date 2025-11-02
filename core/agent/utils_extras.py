# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : utils_extras.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-01 23:53
@Update Date    :
@Description    : Agent工具函数集合
提供批量操作、统计信息等辅助功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.base.agent import Agent


def batch_act(
    agents: Dict[str, Agent],
    observations: Dict[str, Any],
    deterministic: bool = False,
) -> Dict[str, Tuple[int, float, float]]:
    """
    批量执行动作选择

    Args:
        agents: {agent_id: Agent实例}
        observations: {agent_id: observation}
        deterministic: 是否使用确定性策略

    Returns:
        {agent_id: (action, logprob, value)}
    """
    results = {}
    for agent_id, obs in observations.items():
        if agent_id not in agents:
            raise KeyError(f"Agent {agent_id} not found")
        agent = agents[agent_id]
        result = agent.act(obs, deterministic=deterministic)
        results[agent_id] = result
    return results


def batch_learn(
    agents: Dict[str, Agent],
    batches: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    批量学习

    Args:
        agents: {agent_id: Agent实例}
        batches: {agent_id: batch_dict}

    Returns:
        {agent_id: metrics}
    """
    results = {}
    for agent_id, batch in batches.items():
        if agent_id not in agents:
            raise KeyError(f"Agent {agent_id} not found")
        agent = agents[agent_id]
        metrics = agent.learn(batch)
        results[agent_id] = metrics
    return results


def get_agent_statistics(agent: Agent) -> Dict[str, Any]:
    """
    获取Agent的统计信息

    Args:
        agent: Agent实例

    Returns:
        统计信息字典
    """
    stats = {
        "num_parameters": 0,
        "trainable_parameters": 0,
    }

    try:
        # 计算参数量
        for param in agent.parameters():
            stats["num_parameters"] += param.numel()
            if param.requires_grad:
                stats["trainable_parameters"] += param.numel()

        # 尝试获取模型大小
        total_size = sum(p.numel() * p.element_size() for p in agent.parameters())
        stats["memory_size_mb"] = total_size / (1024 * 1024)

    except Exception as e:
        stats["error"] = str(e)

    return stats


def clone_agent_state(source_agent: Agent, target_agent: Agent) -> None:
    """
    克隆Agent的状态（参数）

    Args:
        source_agent: 源Agent
        target_agent: 目标Agent
    """
    target_agent.load_state_dict(source_agent.state_dict())


def aggregate_metrics(
    metrics_list: List[Dict[str, float]],
    method: str = "mean",
) -> Dict[str, float]:
    """
    聚合多个metrics字典

    Args:
        metrics_list: metrics字典列表
        method: 聚合方法（'mean', 'sum', 'max', 'min'）

    Returns:
        聚合后的metrics
    """
    if len(metrics_list) == 0:
        return {}

    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m.get(key, 0.0) for m in metrics_list]

        if method == "mean":
            aggregated[key] = float(np.mean(values))
        elif method == "sum":
            aggregated[key] = float(np.sum(values))
        elif method == "max":
            aggregated[key] = float(np.max(values))
        elif method == "min":
            aggregated[key] = float(np.min(values))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    return aggregated


def prepare_batch_for_agent(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    old_logprobs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Dict[str, Any]:
    """
    准备PPO训练批次数据

    Args:
        obs: 观测数组
        actions: 动作数组
        rewards: 奖励数组
        next_obs: 下一状态观测数组
        dones: 结束标志数组
        old_logprobs: 旧的对数概率
        advantages: 优势值
        returns: 回报值
        clip_coef: PPO裁剪系数
        value_coef: 价值损失系数
        entropy_coef: 熵正则化系数

    Returns:
        批次字典
    """
    return {
        "obs": obs,
        "actions": actions,
        "logprobs": old_logprobs,
        "advantages": advantages,
        "returns": returns,
        "clip_coef": clip_coef,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
    }
