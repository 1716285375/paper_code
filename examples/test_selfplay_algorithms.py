#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自博弈算法测试脚本

在Magent2环境中测试MAPPO、MATRPO、HAPPO、HATRPO的自博弈训练算法。

使用方法:
    # 测试MAPPO自博弈
    python examples/test_selfplay_algorithms.py --algorithm mappo --num_updates 100
    
    # 测试MATRPO自博弈
    python examples/test_selfplay_algorithms.py --algorithm matrpo --num_updates 100
    
    # 测试HAPPO自博弈
    python examples/test_selfplay_algorithms.py --algorithm happo --num_updates 100
    
    # 测试HATRPO自博弈
    python examples/test_selfplay_algorithms.py --algorithm hatrpo --num_updates 100
    
    # 测试所有算法
    python examples/test_selfplay_algorithms.py --algorithm all --num_updates 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms import (
    SelfPlayMAPPOTrainer,
    SelfPlayMATRPOTrainer,
    SelfPlayHAPPOTrainer,
    SelfPlayHATRPOTrainer,
)
from common.config import load_config
from common.tracking import TensorBoardTracker
from common.utils.data_manager import TrainingDataManager
from common.utils.logging import LoggerManager
from core.agent import AgentManager
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


def create_agent_manager(
    agent_ids: list,
    obs_dim: int,
    action_dim: int,
    config: dict,
    device: str,
    red_agents: list,
    blue_agents: list,
) -> AgentManager:
    """
    创建Agent管理器

    Args:
        agent_ids: 所有Agent ID列表
        obs_dim: 观测维度
        action_dim: 动作空间维度
        config: Agent配置
        device: 设备
        red_agents: 红队Agent列表
        blue_agents: 蓝队Agent列表

    Returns:
        AgentManager实例
    """
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


def test_algorithm(
    algorithm_name: str,
    config: dict,
    num_updates: int = 100,
    verbose: bool = True,
):
    """
    测试指定的自博弈算法

    Args:
        algorithm_name: 算法名称（mappo, matrpo, happo, hatrpo）
        config: 配置字典
        num_updates: 训练更新次数
        verbose: 是否输出详细信息

    Returns:
        是否测试成功
    """
    print(f"\n{'='*60}")
    print(f"测试 {algorithm_name.upper()} 自博弈算法")
    print(f"{'='*60}\n")

    # 设置随机种子
    seed = config.get("seed", 42)
    setup_seed(seed)

    # 设备配置
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = "cpu"
    else:
        print(f"✓ 使用设备: {device}")

    # 环境配置
    env_config = config.get("env", {})
    env_id = env_config.get("id", "magent2:battle_v4")
    env_kwargs = env_config.get("kwargs", {})

    # 创建环境
    try:
        env = make_env(env_id, **env_kwargs)
        env.reset()
        print(f"✓ 环境创建成功: {env_id}")
    except Exception as e:
        print(f"✗ 环境创建失败: {e}")
        return False

    # 获取Agent信息
    agent_ids = getattr(env, "agents", [])
    if not agent_ids:
        print("✗ 无法获取agent ID列表")
        return False

    # 分离团队
    red_agents = [aid for aid in agent_ids if "red" in aid.lower()]
    blue_agents = [aid for aid in agent_ids if "blue" in aid.lower()]

    if not red_agents or not blue_agents:
        # 如果无法自动分离，则平分
        mid = len(agent_ids) // 2
        red_agents = agent_ids[:mid]
        blue_agents = agent_ids[mid:]

    print(f"✓ 总Agent数: {len(agent_ids)}")
    print(f"  - 红队: {len(red_agents)} 个")
    print(f"  - 蓝队: {len(blue_agents)} 个")

    # 获取观测和动作维度
    sample_agent_id = agent_ids[0]
    try:
        obs_space = env.observation_space(sample_agent_id)
        action_space = env.action_space(sample_agent_id)

        if hasattr(obs_space, "shape"):
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = config.get("agent", {}).get("obs_dim", 845)

        if hasattr(action_space, "n"):
            action_dim = action_space.n
        else:
            action_dim = config.get("agent", {}).get("action_dim", 21)
    except Exception:
        obs_dim = config.get("agent", {}).get("obs_dim", 845)
        action_dim = config.get("agent", {}).get("action_dim", 21)

    print(f"✓ 观测维度: {obs_dim}, 动作维度: {action_dim}")

    # 创建Agent管理器
    try:
        agent_manager = create_agent_manager(
            agent_ids=agent_ids,
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config,
            device=device,
            red_agents=red_agents,
            blue_agents=blue_agents,
        )
        print(f"✓ Agent管理器创建成功")
    except Exception as e:
        print(f"✗ Agent管理器创建失败: {e}")
        return False

    # 创建训练器
    training_config = config.get("training", {})
    
    # 根据配置文件名生成checkpoint目录
    checkpoint_dir = config.get("checkpoint_dir", None)
    if checkpoint_dir is None:
        # 自动生成：checkpoints/{algorithm_name}_test
        checkpoint_dir = f"checkpoints/{algorithm_name}_selfplay_test"
    training_config["checkpoint_dir"] = checkpoint_dir
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint目录: {checkpoint_dir}")

    # 数据保存配置
    data_saving_config = config.get("data_saving", {})
    data_output_dir = data_saving_config.get("output_dir", None)
    if data_output_dir is None:
        data_output_dir = f"training_data/{algorithm_name}_selfplay_test"
    print(f"✓ 训练数据目录: {data_output_dir}")

    # 日志记录器
    logger = LoggerManager(
        name=f"{algorithm_name}_selfplay_test",
        log_dir=str(project_root / "logs"),
        enable_file=True,
        enable_console=True,
    )

    # 实验跟踪器
    tracker = None
    tracking_config = config.get("tracking", {})
    if tracking_config.get("enabled", False):
        if tracking_config.get("tensorboard", {}).get("enabled", False):
            tb_config = tracking_config["tensorboard"]
            tb_tracker = TensorBoardTracker()
            project = tb_config.get("project", f"{algorithm_name}-selfplay-test")
            name = tb_config.get("name", None) or f"{algorithm_name}_selfplay_test"
            tb_tracker.init(
                log_dir=str(project_root / "runs"),
                project=project,
                name=name,
                config=config,
            )
            tracker = tb_tracker

    # 数据管理器
    data_manager = None
    if data_saving_config.get("enabled", False):
        data_manager = TrainingDataManager(
            output_dir=data_output_dir,
            save_format=data_saving_config.get("format", "json"),
        )

    # 选择训练器
    trainer_class_map = {
        "mappo": SelfPlayMAPPOTrainer,
        "matrpo": SelfPlayMATRPOTrainer,
        "happo": SelfPlayHAPPOTrainer,
        "hatrpo": SelfPlayHATRPOTrainer,
    }

    if algorithm_name.lower() not in trainer_class_map:
        print(f"✗ 未知算法: {algorithm_name}")
        return False

    TrainerClass = trainer_class_map[algorithm_name.lower()]

    try:
        trainer = TrainerClass(
            agent=agent_manager,
            env=env,
            config=training_config,
            logger=logger,
            tracker=tracker,
            main_team="team_red",
            opponent_team="team_blue",
        )
        print(f"✓ {algorithm_name.upper()}训练器创建成功")
    except Exception as e:
        print(f"✗ {algorithm_name.upper()}训练器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 设置数据管理器
    if data_manager:
        trainer.data_manager = data_manager

    # 开始训练
    print(f"\n开始训练，共 {num_updates} 个更新...")
    print("-" * 60)

    try:
        trainer.train(num_updates=num_updates)
        print(f"\n✓ {algorithm_name.upper()} 自博弈训练完成！")
        return True
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        checkpoint_path = Path(checkpoint_dir) / f"interrupted_{algorithm_name}_checkpoint.pt"
        trainer.save(str(checkpoint_path))
        print(f"✓ 已保存检查点: {checkpoint_path}")
        return True
    except Exception as e:
        print(f"\n✗ {algorithm_name.upper()} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试自博弈算法")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["mappo", "matrpo", "happo", "hatrpo", "all"],
        default="mappo",
        help="要测试的算法（mappo, matrpo, happo, hatrpo, all）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选，如果不指定则使用默认配置）",
    )
    parser.add_argument(
        "--num_updates",
        type=int,
        default=100,
        help="训练更新次数（默认100）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu），如果指定则覆盖配置",
    )

    args = parser.parse_args()

    # 默认配置
    default_config = {
        "seed": args.seed,
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "env": {
            "id": "magent2:battle_v4",
            "kwargs": {
                "map_size": 20,
                "max_cycles": 50,
            },
        },
        "agent": {
            "encoder": {
                "type": "networks/mlp",
                "params": {
                    "in_dim": 845,
                    "hidden_dims": [128, 64],
                },
            },
            "policy_head": {
                "type": "policy_heads/discrete",
                "params": {
                    "hidden_dims": [32],
                },
            },
            "value_head": {
                "type": "value_heads/mlp",
                "params": {
                    "hidden_dims": [32],
                },
            },
            "optimizer": {
                "type": "optimizers/adam",
                "params": {
                    "lr": 3e-4,
                },
            },
        },
        "training": {
            "max_steps_per_episode": 50,
            "num_epochs": 2,
            "batch_size": 64,
            "clip_coef": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "eval_freq": 20,
            "save_freq": 50,
            "log_freq": 10,
            "self_play_update_freq": 10,
            "self_play_mode": "copy",
            "use_policy_pool": False,
        },
        "tracking": {
            "enabled": True,
            "tensorboard": {
                "enabled": True,
                "project": "selfplay-algorithms-test",
                "name": None,
            },
        },
        "data_saving": {
            "enabled": True,
            "format": "json",
        },
    }

    # 加载配置文件（如果指定）
    if args.config:
        from common.config import load_config
        try:
            file_config = load_config(args.config, as_dict=True, project_root=project_root)
            # 合并配置
            default_config.update(file_config)
            print(f"✓ 已加载配置文件: {args.config}")
        except Exception as e:
            print(f"⚠️  配置文件加载失败: {e}，使用默认配置")

    # MATRPO/HATRPO特定配置
    trpo_config = {
        "kl_threshold": 0.01,
        "max_line_search_steps": 10,
        "accept_ratio": 0.1,
        "back_ratio": 0.8,
        "cg_damping": 0.1,
        "cg_max_iters": 10,
        "critic_lr": 5e-3,
    }

    # HAPPO/HATRPO特定配置
    happo_config = {
        "update_order": "random",
        "use_marginal_advantage": True,
    }

    # 测试算法
    algorithms_to_test = []
    if args.algorithm.lower() == "all":
        algorithms_to_test = ["mappo", "matrpo", "happo", "hatrpo"]
    else:
        algorithms_to_test = [args.algorithm.lower()]

    results = {}
    for algo in algorithms_to_test:
        # 准备算法特定配置
        algo_config = default_config.copy()
        if algo in ["matrpo", "hatrpo"]:
            algo_config["training"].update(trpo_config)
        if algo in ["happo", "hatrpo"]:
            algo_config["training"].update(happo_config)

        success = test_algorithm(
            algorithm_name=algo,
            config=algo_config,
            num_updates=args.num_updates,
            verbose=True,
        )
        results[algo] = success

    # 输出测试结果摘要
    print(f"\n{'='*60}")
    print("测试结果摘要")
    print(f"{'='*60}")
    for algo, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {algo.upper()}: {status}")
    print(f"{'='*60}\n")

    # 返回退出码
    if all(results.values()):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

