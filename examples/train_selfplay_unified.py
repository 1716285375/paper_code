#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的自博弈训练脚本

支持所有算法的自博弈训练：
- PPO
- MAPPO
- MATRPO
- HAPPO
- HATRPO
- SMPE

使用方法:
    # PPO自博弈
    python examples/train_selfplay_unified.py --algorithm ppo --config configs/ppo/ppo_magent2_selfplay.yaml
    
    # MAPPO自博弈
    python examples/train_selfplay_unified.py --algorithm mappo --config configs/mappo/mappo_magent2_selfplay.yaml
    
    # MATRPO自博弈
    python examples/train_selfplay_unified.py --algorithm matrpo --config configs/matrpo/matrpo_magent2_selfplay.yaml
    
    # HAPPO自博弈
    python examples/train_selfplay_unified.py --algorithm happo --config configs/happo/happo_magent2_selfplay.yaml
    
    # HATRPO自博弈
    python examples/train_selfplay_unified.py --algorithm hatrpo --config configs/hatrpo/hatrpo_magent2_selfplay.yaml
    
    # SMPE自博弈
    python examples/train_selfplay_unified.py --algorithm smpe --config configs/smpe/smpe_magent2_selfplay.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.ppo.trainers.self_play_trainer import SelfPlayPPOTrainer
try:
    from algorithms import (
        SelfPlayMAPPOTrainer,
        SelfPlayMATRPOTrainer,
        SelfPlayHAPPOTrainer,
        SelfPlayHATRPOTrainer,
    )
except ImportError:
    # 如果统一导入失败，尝试分别导入
    from algorithms.mappo.trainers.self_play_trainer import SelfPlayMAPPOTrainer
    from algorithms.matrpo.trainers.self_play_trainer import SelfPlayMATRPOTrainer
    from algorithms.happo.trainers.self_play_trainer import SelfPlayHAPPOTrainer
    from algorithms.hatrpo.trainers.self_play_trainer import SelfPlayHATRPOTrainer

try:
    from algorithms.smpe.trainers.self_play_trainer import SMPESelfPlayTrainer
except ImportError:
    SMPESelfPlayTrainer = None
from common.config import load_config
from common.tracking import ExperimentTracker, TensorBoardTracker, WandBTracker
from common.utils.data_manager import TrainingDataManager
from common.utils.logging import LoggerManager
from common.video import VideoRecorder
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


# 训练器类映射
TRAINER_CLASS_MAP = {
    "ppo": SelfPlayPPOTrainer,
    "mappo": SelfPlayMAPPOTrainer,
    "matrpo": SelfPlayMATRPOTrainer,
    "happo": SelfPlayHAPPOTrainer,
    "hatrpo": SelfPlayHATRPOTrainer,
}

# 如果SMPE可用，添加到映射中
if SMPESelfPlayTrainer is not None:
    TRAINER_CLASS_MAP["smpe"] = SMPESelfPlayTrainer


def create_agent_manager(
    agent_ids: list,
    obs_dim: int,
    action_dim: int,
    config: dict,
    device: str,
    red_agents: list,
    blue_agents: list,
    algorithm: str = "ppo",
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
        algorithm: 算法名称（用于特殊处理，如SMPE）
    
    Returns:
        AgentManager实例
    """
    agent_config = config.get("agent", {})
    
    # SMPE需要特殊处理
    if algorithm.lower() == "smpe":
        # SMPE需要agent_id维度等特殊配置
        agent_config.setdefault("agent_id_dim", 32)
        agent_config.setdefault("n_agents", len(agent_ids))
    
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


def get_environment_info(env, config: dict) -> tuple:
    """
    获取环境信息
    
    Returns:
        (agent_ids, obs_dim, action_dim, red_agents, blue_agents)
    """
    # 重置环境以获取agent列表
    env.reset()
    agent_ids = getattr(env, "agents", [])
    
    if not agent_ids:
        raise ValueError("无法获取agent ID列表，请检查环境配置")
    
    # 分离红队和蓝队（精确匹配）
    # 方法1: 优先使用明确的团队前缀（red_ 或 blue_）
    red_agents = [aid for aid in agent_ids if aid.lower().startswith("red_")]
    blue_agents = [aid for aid in agent_ids if aid.lower().startswith("blue_")]
    
    # 方法2: 如果方法1没有找到，尝试包含"red"或"blue"的agent
    if not red_agents and not blue_agents:
        red_agents = [aid for aid in agent_ids if "red" in aid.lower() and not "blue" in aid.lower()]
        blue_agents = [aid for aid in agent_ids if "blue" in aid.lower() and not "red" in aid.lower()]
    
    # 方法3: 如果仍然无法分离，则按顺序平分
    if not red_agents or not blue_agents or len(red_agents) + len(blue_agents) != len(agent_ids):
        # 重新分配：先尝试精确匹配
        red_agents = [aid for aid in agent_ids if aid.lower().startswith("red_")]
        blue_agents = [aid for aid in agent_ids if aid.lower().startswith("blue_")]
        
        # 如果精确匹配后总数不匹配，说明有未分类的agent，需要重新分配
        if len(red_agents) + len(blue_agents) != len(agent_ids):
            # 找到所有已分类的agent
            classified = set(red_agents) | set(blue_agents)
            unclassified = [aid for aid in agent_ids if aid not in classified]
            
            # 将未分类的agent平分
            if unclassified:
                mid = len(unclassified) // 2
                red_agents.extend(unclassified[:mid])
                blue_agents.extend(unclassified[mid:])
        
        # 如果还是有问题，完全平分
        if len(red_agents) + len(blue_agents) != len(agent_ids) or not red_agents or not blue_agents:
            mid = len(agent_ids) // 2
            red_agents = sorted(agent_ids[:mid])
            blue_agents = sorted(agent_ids[mid:])
    
    # 最终验证：确保没有重复且总数正确
    red_agents = sorted(list(set(red_agents)))  # 去重并排序
    blue_agents = sorted(list(set(blue_agents)))  # 去重并排序
    
    # 移除重复（如果存在）
    common = set(red_agents) & set(blue_agents)
    if common:
        # 保留在红队，从蓝队移除
        blue_agents = [aid for aid in blue_agents if aid not in common]
    
    # 最终验证总数
    if len(red_agents) + len(blue_agents) != len(agent_ids):
        # 如果总数仍然不匹配，强制平分
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
                elif hasattr(obs_space, "spaces"):  # 字典空间
                    first_space = list(obs_space.spaces.values())[0]
                    if hasattr(first_space, "shape"):
                        obs_dim = int(np.prod(first_space.shape))
                    else:
                        obs_dim = 845  # Magent2默认
                else:
                    obs_dim = 845
            
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
    parser = argparse.ArgumentParser(description="统一的自博弈训练脚本")
    
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=list(TRAINER_CLASS_MAP.keys()),
        default="ppo",
        help=f"算法名称: {', '.join(TRAINER_CLASS_MAP.keys())}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（如果不指定，将使用默认配置）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu），如果指定则覆盖配置文件",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（如果指定则覆盖配置文件）",
    )
    
    args = parser.parse_args()
    
    # 默认配置文件路径
    if args.config is None:
        default_configs = {
            "ppo": "configs/ppo/ppo_magent2_selfplay.yaml",
            "mappo": "configs/mappo/mappo_magent2_selfplay.yaml",
            "matrpo": "configs/matrpo/matrpo_magent2_selfplay.yaml",
            "happo": "configs/happo/happo_magent2_selfplay.yaml",
            "hatrpo": "configs/hatrpo/hatrpo_magent2_selfplay.yaml",
            "smpe": "configs/smpe/smpe_magent2_selfplay.yaml",
        }
        args.config = default_configs.get(args.algorithm, "configs/ppo/ppo_magent2_selfplay.yaml")
    
    print("=" * 60)
    print(f"开始 {args.algorithm.upper()} 自博弈训练")
    print("=" * 60)
    
    # 加载配置
    config = load_config(args.config, as_dict=True, project_root=project_root)
    
    # 设置随机种子
    seed = args.seed or config.get("seed", 42)
    setup_seed(seed)
    print(f"✓ 随机种子: {seed}")
    
    # 设备配置
    device = args.device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = "cpu"
    print(f"✓ 使用设备: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
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
    agent_manager = create_agent_manager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        device=device,
        red_agents=red_agents,
        blue_agents=blue_agents,
        algorithm=args.algorithm,
    )
    print(f"✓ Agent管理器创建成功")
    
    # 创建日志和跟踪器
    experiment_name = f"{args.algorithm}_selfplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = LoggerManager(
        name=experiment_name,
        log_dir="logs",
        enable_file=True,
        enable_console=True,
    )
    
    # 创建跟踪器
    tracker = None
    tracking_config = config.get("tracking", {})
    if tracking_config.get("enabled", False):
        tracker_type = tracking_config.get("type", "tensorboard")
        try:
            if tracker_type == "tensorboard":
                # TensorBoardTracker初始化只需要log_dir（可选）
                tb_tracker = TensorBoardTracker(log_dir="runs")
                # 然后调用init方法设置project和name
                project = tracking_config.get("project", "marl-selfplay")
                tb_tracker.init(
                    project=project,
                    name=experiment_name,
                    config=config,
                )
                tracker = tb_tracker
            elif tracker_type == "wandb":
                # WandBTracker需要先初始化，然后调用init方法
                wandb_tracker = WandBTracker()
                project = tracking_config.get("project", "marl-selfplay")
                wandb_tracker.init(
                    project=project,
                    name=experiment_name,
                    config=config,
                )
                tracker = wandb_tracker
        except ImportError as e:
            print(f"⚠️  跟踪器初始化失败: {e}")
            print("  继续训练，但不记录指标到跟踪器")
            tracker = None
        except Exception as e:
            print(f"⚠️  跟踪器初始化失败: {e}")
            print("  继续训练，但不记录指标到跟踪器")
            tracker = None
    
    # 创建数据管理器
    data_manager = None
    data_saving = config.get("data_saving", {})
    if data_saving.get("enabled", False):
        # 根据配置文件名创建数据目录
        config_name = Path(args.config).stem
        output_dir = data_saving.get("output_dir", f"training_data/{config_name}")
        
        data_manager = TrainingDataManager(
            output_dir=output_dir,
            save_format=data_saving.get("format", "json"),  # 使用save_format而不是format
        )
        # save_freq在训练器中处理，不需要在这里设置
        print(f"✓ 训练数据保存目录: {output_dir}")
    
    # 创建检查点目录
    config_name = Path(args.config).stem
    checkpoint_dir = Path("checkpoints") / config_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Checkpoint目录: {checkpoint_dir}")
    
    # 获取训练器类
    TrainerClass = TRAINER_CLASS_MAP.get(args.algorithm.lower())
    if TrainerClass is None:
        raise ValueError(f"未知的算法: {args.algorithm}")
    
    # 创建训练器
    training_config = config.get("training", {})
    training_config.update(config)  # 合并配置
    
    # 特殊处理：SMPE需要额外的配置
    if args.algorithm.lower() == "smpe":
        training_config["agent"] = agent_manager
    
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
        print(f"✓ {args.algorithm.upper()}训练器创建成功")
    except Exception as e:
        print(f"✗ {args.algorithm.upper()}训练器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置数据管理器
    if data_manager:
        trainer.data_manager = data_manager
    
    # 恢复训练（如果指定）
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.load(str(checkpoint_path))
            print(f"✓ 已加载检查点: {checkpoint_path}")
        else:
            print(f"⚠️  检查点不存在: {checkpoint_path}")
    
    # 获取训练参数
    num_updates = training_config.get("num_updates", 1000)
    save_freq = training_config.get("save_freq", 100)
    eval_freq = training_config.get("eval_freq", 50)
    
    print(f"\n{'=' * 60}")
    print(f"开始{args.algorithm.upper()}自博弈训练")
    print(f"{'=' * 60}")
    print(f"更新次数: {num_updates}")
    print(f"保存频率: {save_freq}")
    print(f"评估频率: {eval_freq}")
    print(f"设备: {device}")
    print(f"{'=' * 60}\n")
    
    # 开始训练
    try:
        trainer.train(num_updates=num_updates)
        print(f"\n{'=' * 60}")
        print(f"✓ {args.algorithm.upper()} 自博弈训练完成！")
        print(f"{'=' * 60}")
        
        # 保存最终检查点
        final_checkpoint = checkpoint_dir / f"{args.algorithm}_final.pt"
        trainer.save(str(final_checkpoint))
        print(f"✓ 最终检查点已保存: {final_checkpoint}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断")
        checkpoint_path = checkpoint_dir / f"interrupted_{args.algorithm}_checkpoint.pt"
        trainer.save(str(checkpoint_path))
        print(f"✓ 已保存检查点: {checkpoint_path}")
    except Exception as e:
        print(f"\n✗ {args.algorithm.upper()} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 保存错误检查点
        error_checkpoint = checkpoint_dir / f"error_{args.algorithm}_checkpoint.pt"
        try:
            trainer.save(str(error_checkpoint))
            print(f"✓ 已保存错误检查点: {error_checkpoint}")
        except Exception:
            pass


if __name__ == "__main__":
    main()

