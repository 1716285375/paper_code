#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Magent2环境评估脚本

用于评估训练好的HP3O模型在Magent2 battle_v4环境中的性能。

使用方法:
    python test/eval_magent2.py --checkpoint checkpoints/hp3o_final.pt --num_episodes 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.config import load_config
from core.agent import AgentManager
from core.trainer.multi_agent_evaluator import MultiAgentEvaluator
from environments import make_env


def get_environment_info(env, config: dict) -> tuple:
    """
    获取环境信息
    
    Returns:
        (agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents)
    """
    # 重置环境以获取agent列表
    env.reset()
    agent_ids = getattr(env, "agents", [])
    
    if not agent_ids:
        raise ValueError("无法获取agent ID列表，请检查环境配置")
    
    # 分离红队和蓝队
    red_agents = [aid for aid in agent_ids if aid.lower().startswith("red_")]
    blue_agents = [aid for aid in agent_ids if aid.lower().startswith("blue_")]
    
    # 如果精确匹配失败，尝试包含"red"或"blue"的agent
    if not red_agents and not blue_agents:
        red_agents = [aid for aid in agent_ids if "red" in aid.lower() and "blue" not in aid.lower()]
        blue_agents = [aid for aid in agent_ids if "blue" in aid.lower() and "red" not in aid.lower()]
    
    # 如果仍然无法分离，则按顺序平分
    if not red_agents or not blue_agents or len(red_agents) + len(blue_agents) != len(agent_ids):
        mid = len(agent_ids) // 2
        red_agents = sorted(agent_ids[:mid])
        blue_agents = sorted(agent_ids[mid:])
    
    # 最终验证：确保没有重复且总数正确
    red_agents = sorted(list(set(red_agents)))
    blue_agents = sorted(list(set(blue_agents)))
    
    # 移除重复
    common = set(red_agents) & set(blue_agents)
    if common:
        blue_agents = [aid for aid in blue_agents if aid not in common]
    
    # 最终验证总数
    if len(red_agents) + len(blue_agents) != len(agent_ids):
        mid = len(agent_ids) // 2
        red_agents = sorted(agent_ids[:mid])
        blue_agents = sorted(agent_ids[mid:])
    
    # 获取观测和动作空间
    obs_dim = config.get("agent", {}).get("obs_dim", None)
    action_dim = config.get("agent", {}).get("action_dim", None)
    
    if red_agents:
        try:
            obs_space = env.observation_space(red_agents[0])
            action_space = env.action_space(red_agents[0])
            
            # 计算观测维度
            if obs_dim is None:
                if hasattr(obs_space, "shape"):
                    obs_dim = int(np.prod(obs_space.shape))
                else:
                    obs_dim = 845  # Magent2默认
            
            # 计算动作维度
            if action_dim is None:
                if hasattr(action_space, "n"):
                    action_dim = action_space.n
                else:
                    action_dim = 21  # Magent2默认
        except Exception as e:
            print(f"无法自动获取空间信息: {e}，使用配置中的默认值")
            obs_dim = obs_dim or 845
            action_dim = action_dim or 21
    else:
        obs_dim = obs_dim or 845
        action_dim = action_dim or 21
    
    # 如果环境提供了state，获取state维度
    state_dim = None
    if hasattr(env, "get_state_size"):
        try:
            state_dim = env.get_state_size()
        except Exception:
            pass
    
    if state_dim is None:
        state_dim = config.get("agent", {}).get("centralized_critic", {}).get("state_dim", None)
    
    return agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents


def main():
    parser = argparse.ArgumentParser(description="Magent2环境评估脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/hp3o_final.pt",
        help="检查点文件路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（如果不指定，将尝试从检查点推断）",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="评估的episode数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu），如果指定则覆盖配置文件",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Magent2环境评估")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"✓ 随机种子: {args.seed}")
    
    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"✓ 加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 尝试从检查点获取配置信息
    if "config" in checkpoint:
        config = checkpoint["config"]
        print("✓ 从检查点加载配置")
    elif args.config:
        config = load_config(args.config, as_dict=True, project_root=project_root)
        print(f"✓ 从配置文件加载: {args.config}")
    else:
        # 使用默认配置
        print("⚠️  未找到配置，使用默认配置")
        config = {
            "env": {
                "name": "magent2:battle_v4",
                "map_size": 20,
                "max_cycles": 100,
            },
            "agent": {
                "type": "ppo",
                "obs_dim": 845,
                "action_dim": 21,
                "encoder": {
                    "type": "networks/mlp",
                    "params": {
                        "in_dim": 845,
                        "hidden_dims": [128, 64],
                        "use_layer_norm": True,
                        "dropout": 0.0,
                    },
                },
                "policy_head": {
                    "type": "policy_heads/discrete",
                    "params": {"hidden_dims": [32]},
                },
                "value_head": {
                    "type": "value_heads/mlp",
                    "params": {"hidden_dims": [32]},
                },
                "optimizer": {
                    "type": "optimizers/adam",
                    "params": {
                        "lr": 3e-4,
                        "weight_decay": 0.0,
                        "max_grad_norm": 0.5,
                    },
                },
            },
        }
    
    # 设备配置
    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = "cpu"
    print(f"✓ 使用设备: {device}")
    
    # 创建环境
    env_config = config.get("env", {})
    env_name = env_config.pop("name", "magent2:battle_v4")
    env = make_env(env_name, **env_config)
    print(f"✓ 环境创建成功: {env_name}")
    
    # 获取环境信息
    agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents = get_environment_info(env, config)
    
    print(f"\n{'=' * 60}")
    print("环境信息")
    print(f"{'=' * 60}")
    print(f"总Agent数: {len(agent_ids)}")
    print(f"红队Agent: {red_agents[:5]}{'...' if len(red_agents) > 5 else ''} ({len(red_agents)}个)")
    print(f"蓝队Agent: {blue_agents[:5]}{'...' if len(blue_agents) > 5 else ''} ({len(blue_agents)}个)")
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    if state_dim:
        print(f"状态维度: {state_dim}")
    
    # 创建Agent管理器
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
    print(f"✓ Agent管理器创建成功")
    
    # 加载Agent状态
    if "agent_state" in checkpoint:
        agent_manager.load_state_dict(checkpoint["agent_state"])
        print("✓ Agent状态加载成功")
    elif "agent" in checkpoint:
        agent_manager.load_state_dict(checkpoint["agent"])
        print("✓ Agent状态加载成功（使用'agent'键）")
    else:
        raise KeyError("检查点中未找到agent状态（'agent_state'或'agent'键）")
    
    # 切换到评估模式
    agent_manager.to_eval_mode()
    print("✓ Agent切换到评估模式")
    
    # 创建评估器
    max_steps_per_episode = config.get("training", {}).get("max_steps_per_episode", 1000)
    if "max_cycles" in env_config:
        max_steps_per_episode = env_config["max_cycles"]
    
    team_names = {
        "team_red": red_agents,
        "team_blue": blue_agents,
    }
    
    evaluator = MultiAgentEvaluator(
        agent=agent_manager,
        env=env,
        max_steps_per_episode=max_steps_per_episode,
        is_multi_agent=True,
        team_names=team_names,
    )
    print(f"✓ 评估器创建成功")
    
    # 运行评估
    print(f"\n{'=' * 60}")
    print(f"开始评估 ({args.num_episodes} episodes)")
    print(f"{'=' * 60}\n")
    
    eval_metrics = evaluator.evaluate(
        num_episodes=args.num_episodes,
        compute_diversity=True,
        compute_cooperation=True,
    )
    
    # 打印评估结果
    print(f"\n{'=' * 60}")
    print("评估结果")
    print(f"{'=' * 60}")
    for key, value in eval_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"{'=' * 60}\n")
    
    # 关闭环境
    env.close()
    print("✓ 评估完成")


if __name__ == "__main__":
    main()