# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : trajectory_filter.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-01-XX
@Update Date    :
@Description    : 轨迹优先级过滤模块
实现Trajectory-Filtering PPO for Multi-Agent Policy Transfer
支持轨迹过滤、分类、分段、重加权等功能
"""
# ------------------------------------------------------------

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FilterStrategy(Enum):
    """轨迹过滤策略"""
    TOP_K = "top_k"  # 选择Top-K轨迹
    PERCENTILE = "percentile"  # 按百分位数过滤
    ADVANTAGE_BASED = "advantage_based"  # 基于优势值过滤
    REWARD_BASED = "reward_based"  # 基于奖励过滤
    DIVERSITY_BASED = "diversity_based"  # 基于多样性过滤
    MIXED = "mixed"  # 混合策略


class TrajectoryFilter:
    """
    轨迹优先级过滤器
    
    用于在PPO训练中对轨迹进行过滤、分类、分段和重加权，
    实现Trajectory-Filtering PPO for Multi-Agent Policy Transfer。
    
    主要功能：
    1. 轨迹优先级排序（基于奖励、优势、多样性等）
    2. 轨迹过滤（Top-K、百分位数等）
    3. 轨迹分段（按时间步或重要性分段）
    4. 轨迹重加权（为不同优先级的轨迹分配不同权重）
    """
    
    def __init__(
        self,
        strategy: str = "top_k",
        filter_ratio: float = 0.5,
        segment_length: Optional[int] = None,
        reweight_enabled: bool = False,
        reweight_scheme: str = "linear",
        diversity_threshold: float = 0.1,
        **kwargs
    ):
        """
        初始化轨迹过滤器
        
        Args:
            strategy: 过滤策略
                - "top_k": 选择Top-K轨迹（默认）
                - "percentile": 按百分位数过滤
                - "advantage_based": 基于优势值过滤
                - "reward_based": 基于奖励过滤
                - "diversity_based": 基于多样性过滤
                - "mixed": 混合策略
            filter_ratio: 过滤比例（保留的比例，0-1之间）
            segment_length: 轨迹分段长度（None表示不分段）
            reweight_enabled: 是否启用重加权
            reweight_scheme: 重加权方案
                - "linear": 线性权重
                - "exponential": 指数权重
                - "inverse": 反比例权重
            diversity_threshold: 多样性阈值（用于多样性过滤）
            **kwargs: 其他参数
        """
        self.strategy = FilterStrategy(strategy.lower())
        self.filter_ratio = filter_ratio
        self.segment_length = segment_length
        self.reweight_enabled = reweight_enabled
        self.reweight_scheme = reweight_scheme
        self.diversity_threshold = diversity_threshold
        
        # 验证参数
        if not 0 < filter_ratio <= 1.0:
            raise ValueError(f"filter_ratio必须在(0, 1]之间，得到: {filter_ratio}")
    
    def filter_trajectories(
        self,
        processed_data: Dict[str, Any],
        is_multi_agent: bool = False
    ) -> Dict[str, Any]:
        """
        过滤轨迹数据
        
        Args:
            processed_data: 处理后的rollout数据（包含advantages、returns等）
            is_multi_agent: 是否为多Agent环境
        
        Returns:
            过滤后的数据（可能包含weights字段用于重加权）
        """
        if is_multi_agent:
            return self._filter_multi_agent_trajectories(processed_data)
        else:
            return self._filter_single_agent_trajectories(processed_data)
    
    def _filter_single_agent_trajectories(
        self,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """过滤单Agent轨迹"""
        # 检查数据是否为空
        advantages = processed_data.get("advantages", [])
        if isinstance(advantages, np.ndarray):
            if len(advantages) == 0:
                # 如果数据为空，直接返回原始数据（不进行过滤）
                return processed_data
        elif isinstance(advantages, list) and len(advantages) == 0:
            return processed_data
        
        # 计算优先级分数
        priority_scores = self._compute_priority_scores(processed_data)
        
        # 如果优先级分数为空，直接返回原始数据
        if len(priority_scores) == 0:
            return processed_data
        
        # 选择轨迹索引
        selected_indices = self._select_indices(priority_scores)
        
        # 如果选中的索引为空，直接返回原始数据
        if len(selected_indices) == 0:
            return processed_data
        
        # 过滤数据
        filtered_data = self._apply_filter(processed_data, selected_indices)
        
        # 重加权（如果启用）
        if self.reweight_enabled:
            weights = self._compute_weights(priority_scores, selected_indices)
            filtered_data["weights"] = weights
        
        # 分段（如果启用）
        if self.segment_length is not None:
            filtered_data = self._segment_trajectories(filtered_data)
        
        return filtered_data
    
    def _filter_multi_agent_trajectories(
        self,
        processed_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """过滤多Agent轨迹"""
        filtered_data = {}
        
        for agent_id, data in processed_data.items():
            # 为每个agent单独过滤（直接传递data，而不是{agent_id: data}）
            agent_filtered = self._filter_single_agent_trajectories(data)
            filtered_data[agent_id] = agent_filtered
        
        return filtered_data
    
    def _compute_priority_scores(
        self,
        processed_data: Dict[str, Any]
    ) -> np.ndarray:
        """
        计算轨迹优先级分数
        
        Args:
            processed_data: 处理后的数据
        
        Returns:
            优先级分数数组（越高越好）
        """
        if self.strategy == FilterStrategy.TOP_K:
            # 使用总回报作为优先级
            returns = processed_data.get("returns", [])
            if isinstance(returns, list):
                returns = np.array(returns)
            return returns
        
        elif self.strategy == FilterStrategy.ADVANTAGE_BASED:
            # 使用优势值作为优先级
            advantages = processed_data.get("advantages", [])
            if isinstance(advantages, list):
                advantages = np.array(advantages)
            return advantages
        
        elif self.strategy == FilterStrategy.REWARD_BASED:
            # 使用奖励总和作为优先级
            rewards = processed_data.get("rewards", [])
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            return rewards
        
        elif self.strategy == FilterStrategy.PERCENTILE:
            # 使用总回报，但会在_select_indices中按百分位数过滤
            returns = processed_data.get("returns", [])
            if isinstance(returns, list):
                returns = np.array(returns)
            return returns
        
        elif self.strategy == FilterStrategy.DIVERSITY_BASED:
            # 基于观测多样性计算优先级
            obs = processed_data.get("obs", [])
            if len(obs) == 0:
                return np.array([])
            
            # 计算观测之间的多样性（使用L2距离）
            obs_array = np.array(obs)
            if obs_array.ndim > 2:
                obs_array = obs_array.reshape(len(obs_array), -1)
            
            # 计算每个轨迹与所有其他轨迹的平均距离
            diversity_scores = []
            for i in range(len(obs_array)):
                distances = np.linalg.norm(obs_array - obs_array[i], axis=1)
                diversity_scores.append(np.mean(distances))
            
            return np.array(diversity_scores)
        
        elif self.strategy == FilterStrategy.MIXED:
            # 混合策略：结合优势、奖励和多样性
            advantages = processed_data.get("advantages", [])
            rewards = processed_data.get("rewards", [])
            obs = processed_data.get("obs", [])
            
            if isinstance(advantages, list):
                advantages = np.array(advantages)
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            
            # 检查数据是否为空
            if len(advantages) == 0:
                return np.array([])
            
            # 归一化各项分数（添加空数组检查）
            if len(advantages) > 0 and advantages.std() > 1e-8:
                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                adv_norm = np.zeros_like(advantages)
            
            if len(rewards) > 0 and rewards.std() > 1e-8:
                reward_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                reward_norm = np.zeros_like(advantages)
            
            # 计算多样性分数
            diversity_scores = np.ones(len(advantages))
            if len(obs) > 0:
                obs_array = np.array(obs)
                if obs_array.ndim > 2:
                    obs_array = obs_array.reshape(len(obs_array), -1)
                if len(obs_array) > 0:
                    for i in range(len(obs_array)):
                        distances = np.linalg.norm(obs_array - obs_array[i], axis=1)
                        diversity_scores[i] = np.mean(distances)
                    if diversity_scores.std() > 1e-8:
                        diversity_scores = (diversity_scores - diversity_scores.mean()) / (diversity_scores.std() + 1e-8)
                    else:
                        diversity_scores = np.zeros_like(advantages)
            
            # 加权组合（优势0.5，奖励0.3，多样性0.2）
            mixed_scores = 0.5 * adv_norm + 0.3 * reward_norm + 0.2 * diversity_scores
            return mixed_scores
        
        else:
            # 默认使用总回报
            returns = processed_data.get("returns", [])
            if isinstance(returns, list):
                returns = np.array(returns)
            return returns
    
    def _select_indices(
        self,
        priority_scores: np.ndarray
    ) -> np.ndarray:
        """
        根据优先级分数选择轨迹索引
        
        Args:
            priority_scores: 优先级分数数组
        
        Returns:
            选中的轨迹索引
        """
        if len(priority_scores) == 0:
            return np.array([], dtype=int)
        
        if self.strategy == FilterStrategy.PERCENTILE:
            # 按百分位数过滤
            threshold = np.percentile(priority_scores, (1 - self.filter_ratio) * 100)
            selected = np.where(priority_scores >= threshold)[0]
            # 确保至少选择一个（如果数据不为空）
            if len(selected) == 0 and len(priority_scores) > 0:
                selected = np.array([np.argmax(priority_scores)])
        else:
            # Top-K或其他策略
            num_select = max(1, int(len(priority_scores) * self.filter_ratio))
            # 确保不超过实际数据量
            num_select = min(num_select, len(priority_scores))
            selected = np.argsort(priority_scores)[-num_select:]
        
        return selected
    
    def _apply_filter(
        self,
        processed_data: Dict[str, Any],
        selected_indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        应用过滤，只保留选中的轨迹
        
        Args:
            processed_data: 原始数据
            selected_indices: 选中的索引
        
        Returns:
            过滤后的数据
        """
        filtered_data = {}
        
        for key, value in processed_data.items():
            if key == "weights":  # 跳过已有的weights，会重新计算
                continue
            
            if isinstance(value, (list, np.ndarray)):
                if len(value) > 0:
                    filtered_data[key] = np.array(value)[selected_indices]
                else:
                    filtered_data[key] = value
            else:
                filtered_data[key] = value
        
        return filtered_data
    
    def _compute_weights(
        self,
        priority_scores: np.ndarray,
        selected_indices: np.ndarray
    ) -> np.ndarray:
        """
        计算轨迹权重（用于重加权）
        
        Args:
            priority_scores: 优先级分数
            selected_indices: 选中的索引
        
        Returns:
            权重数组
        """
        # 检查选中的索引是否为空
        if len(selected_indices) == 0:
            return np.array([])
        
        selected_scores = priority_scores[selected_indices]
        
        # 检查选中的分数是否为空
        if len(selected_scores) == 0:
            return np.array([])
        
        if self.reweight_scheme == "linear":
            # 线性权重：归一化到[0.5, 1.5]
            if len(selected_scores) > 0:
                min_score = selected_scores.min()
                max_score = selected_scores.max()
                if max_score > min_score:
                    weights = 0.5 + (selected_scores - min_score) / (max_score - min_score)
                else:
                    weights = np.ones(len(selected_scores))
            else:
                weights = np.ones(len(selected_indices))
        
        elif self.reweight_scheme == "exponential":
            # 指数权重：exp(归一化分数)
            if len(selected_scores) > 0:
                min_score = selected_scores.min()
                max_score = selected_scores.max()
                if max_score > min_score:
                    normalized = (selected_scores - min_score) / (max_score - min_score)
                    weights = np.exp(normalized)
                    weights = weights / weights.mean()  # 归一化到均值1
                else:
                    weights = np.ones(len(selected_scores))
            else:
                weights = np.ones(len(selected_indices))
        
        elif self.reweight_scheme == "inverse":
            # 反比例权重：1 / (1 + 归一化分数)
            if len(selected_scores) > 0:
                min_score = selected_scores.min()
                max_score = selected_scores.max()
                if max_score > min_score:
                    normalized = (selected_scores - min_score) / (max_score - min_score)
                    weights = 1.0 / (1.0 + normalized)
                    weights = weights / weights.mean()  # 归一化到均值1
                else:
                    weights = np.ones(len(selected_scores))
            else:
                weights = np.ones(len(selected_indices))
        
        else:
            # 默认均匀权重
            weights = np.ones(len(selected_indices))
        
        return weights
    
    def _segment_trajectories(
        self,
        filtered_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        将轨迹分段
        
        Args:
            filtered_data: 过滤后的数据
        
        Returns:
            分段后的数据
        """
        if self.segment_length is None or self.segment_length <= 0:
            return filtered_data
        
        # 获取轨迹长度
        obs = filtered_data.get("obs", [])
        if len(obs) == 0:
            return filtered_data
        
        # 分段处理（简化实现：按固定长度分段）
        # 注意：这里假设所有轨迹长度相同，实际可能需要更复杂的处理
        segmented_data = {}
        
        for key, value in filtered_data.items():
            if key == "weights":  # weights不需要分段
                segmented_data[key] = value
                continue
            
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                value_array = np.array(value)
                
                # 如果数据是1D的，按segment_length分段
                if value_array.ndim == 1:
                    num_segments = len(value_array) // self.segment_length
                    if num_segments > 0:
                        # 只保留完整的段
                        truncated_length = num_segments * self.segment_length
                        segmented_data[key] = value_array[:truncated_length]
                    else:
                        segmented_data[key] = value_array
                else:
                    # 多维数据保持原样
                    segmented_data[key] = value_array
            else:
                segmented_data[key] = value
        
        return segmented_data


def trajectory_filter_postprocessing(
    processed_data: Dict[str, Any],
    agent_manager: Any,
    use_trajectory_filter: bool = False,
    filter_strategy: str = "top_k",
    filter_ratio: float = 0.5,
    segment_length: Optional[int] = None,
    reweight_enabled: bool = False,
    reweight_scheme: str = "linear",
    **kwargs
) -> Dict[str, Any]:
    """
    轨迹过滤后处理函数
    
    在计算advantages和returns之后，对轨迹进行过滤、分类、分段和重加权。
    用于实现Trajectory-Filtering PPO for Multi-Agent Policy Transfer。
    
    Args:
        processed_data: 处理后的rollout数据（包含advantages、returns等）
        agent_manager: Agent管理器（用于判断是否为多Agent）
        use_trajectory_filter: 是否启用轨迹过滤
        filter_strategy: 过滤策略
        filter_ratio: 过滤比例（保留的比例）
        segment_length: 轨迹分段长度
        reweight_enabled: 是否启用重加权
        reweight_scheme: 重加权方案
        **kwargs: 其他参数
    
    Returns:
        过滤后的数据
    """
    if not use_trajectory_filter:
        return processed_data
    
    # 判断是否为多Agent
    is_multi_agent = hasattr(agent_manager, "agent_ids") or isinstance(processed_data, dict) and any(
        isinstance(v, dict) and "obs" in v for v in processed_data.values()
    )
    
    # 创建过滤器
    filter_obj = TrajectoryFilter(
        strategy=filter_strategy,
        filter_ratio=filter_ratio,
        segment_length=segment_length,
        reweight_enabled=reweight_enabled,
        reweight_scheme=reweight_scheme,
        **kwargs
    )
    
    # 应用过滤
    filtered_data = filter_obj.filter_trajectories(processed_data, is_multi_agent=is_multi_agent)
    
    return filtered_data

