#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型评估脚本 - Magent2环境

用于评估训练好的HP3O和MAPPO模型在Magent2环境中的性能。

使用方法:
    # 评估单个模型
    python examples/eval_models_magent2.py --checkpoint checkpoints/hp3o_magent2_selfplay_12v12_8gb/hp3o_final.pt --algorithm hp3o
    
    # 评估两个模型对战
    python examples/eval_models_magent2.py \
        --checkpoint1 checkpoints/hp3o_magent2_selfplay_12v12_8gb/hp3o_final.pt --algorithm1 hp3o \
        --checkpoint2 checkpoints/mappo_magent2_selfplay_12v12_8gb/mappo_final.pt --algorithm2 mappo \
        --mode vs
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms import SelfPlayHP3OTrainer, SelfPlayMAPPOTrainer
from common.config import load_config
from core.agent import AgentManager
from core.trainer.multi_agent_evaluator import MultiAgentEvaluator
try:
    from environments import make_env
except ImportError:
    from environments.factory import make_env


def setup_seed(seed: int) -> None:
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_environment_info(env, config: dict) -> tuple:
    """
    获取环境信息
    
    Returns:
        (agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents)
    """
    env.reset()
    agent_ids = getattr(env, "agents", [])
    
    if not agent_ids:
        raise ValueError("无法获取agent ID列表，请检查环境配置")
    
    # 分离红队和蓝队
    red_agents = [aid for aid in agent_ids if aid.lower().startswith("red_")]
    blue_agents = [aid for aid in agent_ids if aid.lower().startswith("blue_")]
    
    if not red_agents and not blue_agents:
        red_agents = [aid for aid in agent_ids if "red" in aid.lower() and "blue" not in aid.lower()]
        blue_agents = [aid for aid in agent_ids if "blue" in aid.lower() and "red" not in aid.lower()]
    
    if not red_agents or not blue_agents:
        mid = len(agent_ids) // 2
        red_agents = sorted(agent_ids[:mid])
        blue_agents = sorted(agent_ids[mid:])
    
    red_agents = sorted(list(set(red_agents)))
    blue_agents = sorted(list(set(blue_agents)))
    
    # 获取观测和动作空间
    obs_dim = config.get("agent", {}).get("obs_dim", None)
    action_dim = config.get("agent", {}).get("action_dim", None)
    
    if red_agents:
        try:
            obs_space = env.observation_space(red_agents[0])
            action_space = env.action_space(red_agents[0])
            
            if obs_dim is None:
                if hasattr(obs_space, "shape"):
                    obs_dim = int(np.prod(obs_space.shape))
                elif hasattr(obs_space, "spaces"):
                    first_space = list(obs_space.spaces.values())[0]
                    if hasattr(first_space, "shape"):
                        obs_dim = int(np.prod(first_space.shape))
                    else:
                        obs_dim = 845
                else:
                    obs_dim = 845
            
            if action_dim is None:
                if hasattr(action_space, "n"):
                    action_dim = action_space.n
                else:
                    action_dim = 21
        except Exception as e:
            print(f"无法自动获取空间信息: {e}，使用配置中的默认值")
            obs_dim = obs_dim or 845
            action_dim = action_dim or 21
    else:
        obs_dim = obs_dim or 845
        action_dim = action_dim or 21
    
    state_dim = None
    if hasattr(env, "get_state_size"):
        try:
            state_dim = env.get_state_size()
        except Exception:
            pass
    
    if state_dim is None:
        state_dim = config.get("agent", {}).get("centralized_critic", {}).get("state_dim", None)
    
    return agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents


def create_agent_manager(
    agent_ids: list,
    obs_dim: int,
    action_dim: int,
    config: dict,
    device: str,
    red_agents: list,
    blue_agents: list,
) -> AgentManager:
    """创建Agent管理器"""
    agent_config = config.get("agent", {})
    
    agent_manager = AgentManager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device=device,
        shared_agents={
            "team_red": red_agents,
            "team_blue": blue_agents,
        },
    )
    
    return agent_manager


def load_checkpoint_and_create_trainer(
    checkpoint_path: str,
    algorithm: str,
    config: dict,
    agent_manager: AgentManager,
    env: Any,
    device: str,
) -> Any:
    """
    加载checkpoint并创建训练器
    
    Args:
        checkpoint_path: checkpoint文件路径
        algorithm: 算法名称 (hp3o, mappo)
        config: 配置字典
        agent_manager: Agent管理器
        env: 环境实例
        device: 设备
    
    Returns:
        训练器实例
    """
    # 获取训练器类
    trainer_map = {
        "hp3o": SelfPlayHP3OTrainer,
        "mappo": SelfPlayMAPPOTrainer,
    }
    
    TrainerClass = trainer_map.get(algorithm.lower())
    if TrainerClass is None:
        raise ValueError(f"未知的算法: {algorithm}。支持的算法: {list(trainer_map.keys())}")
    
    # 创建训练器
    training_config = config.get("training", {})
    training_config.update(config)
    
    trainer = TrainerClass(
        agent=agent_manager,
        env=env,
        config=training_config,
        logger=None,
        tracker=None,
        main_team="team_red",
        opponent_team="team_blue",
    )
    
    # 加载checkpoint
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.exists():
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"正在加载checkpoint: {checkpoint_path}")
    trainer.load(str(checkpoint_path))
    print(f"✓ Checkpoint加载成功")
    
    return trainer


def evaluate_single_model(
    checkpoint_path: str,
    algorithm: str,
    config_path: Optional[str] = None,
    num_episodes: int = 20,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, float]:
    """
    评估单个模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        algorithm: 算法名称
        config_path: 配置文件路径（可选，如果不提供则使用默认配置）
        num_episodes: 评估episode数量
        device: 设备
        seed: 随机种子
    
    Returns:
        评估指标字典
    """
    setup_seed(seed)
    
    # 加载配置
    if config_path is None:
        # 使用默认配置文件
        default_configs = {
            "hp3o": "configs/hp3o/hp3o_magent2_selfplay_12v12_8gb.yaml",
            "mappo": "configs/mappo/mappo_magent2_selfplay_12v12_8gb.yaml",
        }
        config_path = default_configs.get(algorithm.lower())
        if config_path is None:
            raise ValueError(f"未知的算法: {algorithm}，请提供配置文件路径")
    
    config = load_config(config_path, as_dict=True, project_root=project_root)
    
    # 创建环境
    env_config = config.get("env", {})
    env_name = env_config.pop("name", "magent2:battle_v4")
    env = make_env(env_name, **env_config)
    print(f"✓ 环境创建成功: {env_name}")
    
    # 获取环境信息
    agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents = get_environment_info(env, config)
    
    print(f"\n环境信息:")
    print(f"  总Agent数: {len(agent_ids)}")
    print(f"  红队Agent: {len(red_agents)}个")
    print(f"  蓝队Agent: {len(blue_agents)}个")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    
    # 创建Agent管理器
    agent_manager = create_agent_manager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        device=device,
        red_agents=red_agents,
        blue_agents=blue_agents,
    )
    
    # 加载checkpoint并创建训练器
    trainer = load_checkpoint_and_create_trainer(
        checkpoint_path=checkpoint_path,
        algorithm=algorithm,
        config=config,
        agent_manager=agent_manager,
        env=env,
        device=device,
    )
    
    # 创建评估器
    team_names = {
        "team_red": red_agents,
        "team_blue": blue_agents,
    }
    
    evaluator = MultiAgentEvaluator(
        agent=agent_manager,
        env=env,
        max_steps_per_episode=config.get("training", {}).get("max_steps_per_episode", 100),
        is_multi_agent=True,
        team_names=team_names,
    )
    
    # 评估
    print(f"\n开始评估，共 {num_episodes} 个episodes...")
    metrics = evaluator.evaluate(
        num_episodes=num_episodes,
        compute_diversity=True,
        compute_cooperation=True,
    )
    
    return metrics


def evaluate_two_models_vs(
    checkpoint1_path: str,
    algorithm1: str,
    checkpoint2_path: str,
    algorithm2: str,
    config1_path: Optional[str] = None,
    config2_path: Optional[str] = None,
    num_episodes: int = 20,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, float]:
    """
    评估两个模型对战
    
    Args:
        checkpoint1_path: 第一个模型的checkpoint路径（红队）
        algorithm1: 第一个算法名称
        checkpoint2_path: 第二个模型的checkpoint路径（蓝队）
        algorithm2: 第二个算法名称
        config1_path: 第一个模型的配置文件路径（可选）
        config2_path: 第二个模型的配置文件路径（可选）
        num_episodes: 评估episode数量
        device: 设备
        seed: 随机种子
    
    Returns:
        评估指标字典
    """
    setup_seed(seed)
    
    # 加载配置
    default_configs = {
        "hp3o": "configs/hp3o/hp3o_magent2_selfplay_12v12_8gb.yaml",
        "mappo": "configs/mappo/mappo_magent2_selfplay_12v12_8gb.yaml",
    }
    
    if config1_path is None:
        config1_path = default_configs.get(algorithm1.lower())
    if config2_path is None:
        config2_path = default_configs.get(algorithm2.lower())
    
    config1 = load_config(config1_path, as_dict=True, project_root=project_root)
    config2 = load_config(config2_path, as_dict=True, project_root=project_root)
    
    # 使用第一个配置创建环境（两个模型应该使用相同的环境配置）
    env_config = config1.get("env", {})
    env_name = env_config.pop("name", "magent2:battle_v4")
    env = make_env(env_name, **env_config)
    print(f"✓ 环境创建成功: {env_name}")
    
    # 获取环境信息
    agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents = get_environment_info(env, config1)
    
    print(f"\n环境信息:")
    print(f"  总Agent数: {len(agent_ids)}")
    print(f"  红队Agent: {len(red_agents)}个 ({algorithm1})")
    print(f"  蓝队Agent: {len(blue_agents)}个 ({algorithm2})")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    
    # 创建Agent管理器（使用第一个配置）
    agent_manager = create_agent_manager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config1,
        device=device,
        red_agents=red_agents,
        blue_agents=blue_agents,
    )
    
    # 加载第一个模型（红队）
    print(f"\n加载红队模型 ({algorithm1})...")
    trainer1 = load_checkpoint_and_create_trainer(
        checkpoint_path=checkpoint1_path,
        algorithm=algorithm1,
        config=config1,
        agent_manager=agent_manager,
        env=env,
        device=device,
    )
    
    # 加载第二个模型（蓝队）- 需要创建新的Agent管理器或替换蓝队的agent
    print(f"\n加载蓝队模型 ({algorithm2})...")
    # 创建第二个训练器来加载蓝队的模型
    agent_manager2 = create_agent_manager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config2,
        device=device,
        red_agents=red_agents,
        blue_agents=blue_agents,
    )
    
    trainer2 = load_checkpoint_and_create_trainer(
        checkpoint_path=checkpoint2_path,
        algorithm=algorithm2,
        config=config2,
        agent_manager=agent_manager2,
        env=env,
        device=device,
    )
    
    # 将蓝队的agent替换为第二个模型的agent
    for blue_agent_id in blue_agents:
        blue_agent_from_model2 = agent_manager2.get_agent(blue_agent_id)
        agent_manager._agents[blue_agent_id] = blue_agent_from_model2
    
    # 创建评估器
    team_names = {
        "team_red": red_agents,
        "team_blue": blue_agents,
    }
    
    evaluator = MultiAgentEvaluator(
        agent=agent_manager,
        env=env,
        max_steps_per_episode=config1.get("training", {}).get("max_steps_per_episode", 100),
        is_multi_agent=True,
        team_names=team_names,
    )
    
    # 评估
    print(f"\n开始对战评估，共 {num_episodes} 个episodes...")
    print(f"  红队 ({algorithm1}) vs 蓝队 ({algorithm2})")
    metrics = evaluator.evaluate(
        num_episodes=num_episodes,
        compute_diversity=True,
        compute_cooperation=True,
    )
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "评估结果"):
    """打印评估指标"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    
    # 基础指标
    print("\n基础指标:")
    print(f"  平均奖励: {metrics.get('eval_mean_reward', 0):.4f} ± {metrics.get('eval_std_reward', 0):.4f}")
    print(f"  奖励范围: [{metrics.get('eval_min_reward', 0):.4f}, {metrics.get('eval_max_reward', 0):.4f}]")
    print(f"  平均episode长度: {metrics.get('eval_mean_length', 0):.2f} ± {metrics.get('eval_std_length', 0):.2f}")
    
    # 团队指标
    team_metrics = {}
    for key, value in metrics.items():
        if key.startswith("eval_team_"):
            parts = key.split("_")
            if len(parts) >= 3:
                team_name = parts[2]
                metric_name = "_".join(parts[3:])
                if team_name not in team_metrics:
                    team_metrics[team_name] = {}
                team_metrics[team_name][metric_name] = value
    
    if team_metrics:
        print("\n团队指标:")
        for team_name, team_data in team_metrics.items():
            print(f"  {team_name}:")
            if "mean_reward" in team_data:
                print(f"    平均奖励: {team_data['mean_reward']:.4f} ± {team_data.get('std_reward', 0):.4f}")
            if "win_rate" in team_data:
                print(f"    胜率: {team_data['win_rate']:.2%}")
    
    # 对抗指标
    if "eval_team_reward_diff_mean" in metrics:
        print("\n对抗指标:")
        print(f"  团队奖励差异: {metrics['eval_team_reward_diff_mean']:.4f} ± {metrics.get('eval_team_reward_diff_std', 0):.4f}")
        if "eval_draw_rate" in metrics:
            print(f"  平局率: {metrics['eval_draw_rate']:.2%}")
    
    # 协作指标
    if "eval_coordination_score_mean" in metrics:
        print("\n协作指标:")
        print(f"  协作分数: {metrics['eval_coordination_score_mean']:.4f} ± {metrics.get('eval_coordination_score_std', 0):.4f}")
        if "eval_action_consistency_mean" in metrics:
            print(f"  动作一致性: {metrics['eval_action_consistency_mean']:.4f} ± {metrics.get('eval_action_consistency_std', 0):.4f}")
    
    # 多样性指标
    if "eval_action_entropy_mean" in metrics:
        print("\n多样性指标:")
        print(f"  动作熵: {metrics['eval_action_entropy_mean']:.4f} ± {metrics.get('eval_action_entropy_std', 0):.4f}")
    
    print(f"{'=' * 60}\n")


def main():
    