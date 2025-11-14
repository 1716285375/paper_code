#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A5000 GPU训练脚本
针对NVIDIA RTX A5000 (24GB显存)优化的训练脚本

支持：
- 所有算法（PPO, MAPPO, MATRPO, HAPPO, HATRPO, SMPE, HP3O）
- 轨迹过滤PPO（Trajectory-Filtering PPO）
- 自博弈训练
- 大batch size和网络规模（充分利用24GB显存）

使用方法:
    # PPO训练（带轨迹过滤）
    python examples/train_a5000.py --algorithm ppo --config configs/ppo/ppo_magent2_selfplay_a5000.yaml
    
    # MAPPO训练
    python examples/train_a5000.py --algorithm mappo --config configs/mappo/mappo_magent2_selfplay_a5000.yaml
    
    # 指定GPU
    python examples/train_a5000.py --algorithm ppo --gpu 0
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

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
        SelfPlayHP3OTrainer,
    )
except ImportError:
    from algorithms.mappo.trainers.self_play_trainer import SelfPlayMAPPOTrainer
    from algorithms.matrpo.trainers.self_play_trainer import SelfPlayMATRPOTrainer
    from algorithms.happo.trainers.self_play_trainer import SelfPlayHAPPOTrainer
    from algorithms.hatrpo.trainers.self_play_trainer import SelfPlayHATRPOTrainer
    try:
        from algorithms.hp3o.trainers.self_play_trainer import SelfPlayHP3OTrainer
    except ImportError:
        SelfPlayHP3OTrainer = None

try:
    from algorithms.smpe.trainers.self_play_trainer import SMPESelfPlayTrainer
except ImportError:
    SMPESelfPlayTrainer = None

from common.config import load_config
from common.tracking import ExperimentTracker, TensorBoardTracker, WandBTracker
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


def setup_gpu(gpu_id: int = 0) -> str:
    """
    设置GPU并检查显存
    
    Args:
        gpu_id: GPU ID
    
    Returns:
        设备字符串（"cuda:X"或"cpu"）
    """
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        return "cpu"
    
    if gpu_id >= torch.cuda.device_count():
        print(f"警告: GPU {gpu_id}不存在，使用GPU 0")
        gpu_id = 0
    
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # 检查显存
    total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
    print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    print(f"总显存: {total_memory:.2f} GB")
    
    # 清空显存缓存
    torch.cuda.empty_cache()
    
    return device


def check_memory(device: str) -> None:
    """检查当前显存使用"""
    if device.startswith("cuda"):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"当前显存使用: {allocated:.2f} GB / {reserved:.2f} GB")


# 训练器类映射
TRAINER_CLASS_MAP = {
    "ppo": SelfPlayPPOTrainer,
    "mappo": SelfPlayMAPPOTrainer,
    "matrpo": SelfPlayMATRPOTrainer,
    "happo": SelfPlayHAPPOTrainer,
    "hatrpo": SelfPlayHATRPOTrainer,
}

if SMPESelfPlayTrainer is not None:
    TRAINER_CLASS_MAP["smpe"] = SMPESelfPlayTrainer

if SelfPlayHP3OTrainer is not None:
    TRAINER_CLASS_MAP["hp3o"] = SelfPlayHP3OTrainer


def create_agent_manager(
    agent_ids: list,
    obs_dim: int,
    action_dim: int,
    config,
    device: str,
    red_agents: list,
    blue_agents: list,
    algorithm: str = "ppo",
) -> AgentManager:
    """创建Agent管理器"""
    agent_config = getattr(config, "agent", EasyDict())
    
    # SMPE需要特殊处理
    if algorithm.lower() == "smpe":
        if not hasattr(agent_config, "agent_id_dim"):
            agent_config.agent_id_dim = 32
        if not hasattr(agent_config, "n_agents"):
            agent_config.n_agents = len(agent_ids)
    
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


def main():
    parser = argparse.ArgumentParser(description="A5000 GPU训练脚本")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=list(TRAINER_CLASS_MAP.keys()),
        help="算法名称",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="配置文件路径",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID（默认: 0）",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的checkpoint路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（覆盖配置文件）",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="禁用实验跟踪（WandB/TensorBoard）",
    )
    
    args = parser.parse_args()
    
    # 加载配置（返回字典格式，因为配置文件使用嵌套结构）
    config = load_config(args.config, as_dict=True)
    
    # 覆盖设备配置
    device = setup_gpu(args.gpu)
    config.device = device
    
    # 覆盖随机种子
    if args.seed is not None:
        config.seed = args.seed
    
    seed = getattr(config, "seed", 42)
    setup_seed(seed)
    
    print("=" * 80)
    print("A5000 GPU训练脚本")
    print("=" * 80)
    print(f"算法: {args.algorithm.upper()}")
    print(f"配置文件: {args.config}")
    print(f"设备: {device}")
    print(f"随机种子: {seed}")
    print("=" * 80)
    
    # 创建环境
    env_config = getattr(config, "env", EasyDict())
    env_name = getattr(env_config, "name", None) or getattr(env_config, "id", "magent2:battle_v4")
    env_kwargs = getattr(env_config, "kwargs", EasyDict())
    if hasattr(env_config, "name"):
        env_kwargs.update({k: v for k, v in env_config.items() if k != "name"})
    
    print(f"\n创建环境: {env_name}")
    env = make_env(env_name, **env_kwargs)
    
    # 获取Agent信息
    obs = env.reset()
    if isinstance(obs, dict):
        agent_ids = list(obs.keys())
    else:
        agent_ids = ["agent_0"]
    
    # 分离红队和蓝队
    red_agents = [aid for aid in agent_ids if "red" in aid.lower()]
    blue_agents = [aid for aid in agent_ids if "blue" in aid.lower()]
    
    if not red_agents or not blue_agents:
        # 如果没有明确的红蓝队，按顺序分组
        mid = len(agent_ids) // 2
        red_agents = agent_ids[:mid]
        blue_agents = agent_ids[mid:]
    
    print(f"总Agent数: {len(agent_ids)}")
    print(f"红队: {len(red_agents)}个Agent")
    print(f"蓝队: {len(blue_agents)}个Agent")
    
    # 获取观测和动作维度
    sample_agent_id = agent_ids[0]
    obs_space = env.observation_space(sample_agent_id)
    action_space = env.action_space(sample_agent_id)
    
    if hasattr(obs_space, "shape"):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        obs_dim = obs_space.n if hasattr(obs_space, "n") else 845
    
    action_dim = action_space.n if hasattr(action_space, "n") else 21
    
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    
    # 获取state维度（如果环境提供）
    try:
        state_dim = env.get_state_size()
        if isinstance(state_dim, tuple):
            state_dim = int(np.prod(state_dim))
        print(f"状态维度: {state_dim}")
    except:
        state_dim = obs_dim * len(agent_ids)
        print(f"状态维度（估算）: {state_dim}")
    
    # 更新Agent配置
    agent_config = getattr(config, "agent", EasyDict())
    if not hasattr(agent_config, "obs_dim"):
        agent_config.obs_dim = obs_dim
    if not hasattr(agent_config, "action_dim"):
        agent_config.action_dim = action_dim
    
    # 如果使用集中式Critic，更新state_dim
    training_config = getattr(config, "training", EasyDict())
    if getattr(training_config, "use_centralized_critic", False):
        if hasattr(agent_config, "centralized_critic"):
            agent_config.centralized_critic.state_dim = state_dim
    
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
    
    print("\n检查显存使用（初始化后）...")
    check_memory(device)
    
    # 创建训练器
    TrainerClass = TRAINER_CLASS_MAP[args.algorithm]
    training_config = getattr(config, "training", EasyDict())
    
    # 创建日志记录器
    logger = LoggerManager(
        name=f"{args.algorithm}_a5000",
        log_dir="logs",
        enable_console=True,
        enable_file=True,
    )
    
    # 创建实验跟踪器
    tracker = None
    if not args.no_track:
        tracking_config = getattr(config, "tracking", EasyDict())
        if getattr(tracking_config, "enabled", True):
            experiment_name = getattr(tracking_config, "name", None) or f"{args.algorithm}_a5000_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            trackers_list = []
            
            # WandB
            wandb_config = getattr(tracking_config, "wandb", EasyDict())
            if getattr(wandb_config, "enabled", False):
                try:
                    wandb_tracker = WandBTracker()
                    wandb_tracker.init(
                        project=getattr(wandb_config, "project", "marl-a5000"),
                        name=getattr(wandb_config, "name", None) or experiment_name,
                        config=config,
                    )
                    trackers_list.append(wandb_tracker)
                    print(f"✓ WandB跟踪已启用: {experiment_name}")
                except Exception as e:
                    print(f"⚠ WandB初始化失败: {e}")
            
            # TensorBoard
            tb_config = getattr(tracking_config, "tensorboard", EasyDict())
            if getattr(tb_config, "enabled", True):
                try:
                    tb_tracker = TensorBoardTracker(log_dir="runs")
                    tb_tracker.init(
                        project=getattr(tb_config, "experiment_name", experiment_name),
                        name=experiment_name,
                        config=config,
                    )
                    trackers_list.append(tb_tracker)
                    print(f"✓ TensorBoard跟踪已启用")
                except Exception as e:
                    print(f"⚠ TensorBoard初始化失败: {e}")
            
            # 如果有多个tracker，使用ExperimentTracker组合
            if len(trackers_list) == 1:
                tracker = trackers_list[0]
            elif len(trackers_list) > 1:
                tracker = ExperimentTracker(trackers_list)
    
    # 创建数据管理器
    data_manager = None
    data_saving_config = getattr(config, "data_saving", EasyDict())
    if getattr(data_saving_config, "enabled", False):
        output_dir = getattr(data_saving_config, "output_dir", "training_data")
        # 创建带时间戳的子目录
        experiment_name = f"{args.algorithm}_a5000_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        full_output_dir = Path(output_dir) / experiment_name
        data_manager = TrainingDataManager(
            output_dir=str(full_output_dir),
            save_format=getattr(data_saving_config, "format", "both"),
        )
        print(f"✓ 数据保存已启用: {full_output_dir}")
    
    # 创建训练器
    trainer = TrainerClass(
        agent=agent_manager,
        env=env,
        config=training_config,
        logger=logger,
        tracker=tracker,
    )
    
    # 设置数据管理器
    if data_manager is not None:
        trainer.data_manager = data_manager
    
    # 恢复训练（如果指定）
    if args.resume:
        print(f"\n恢复训练: {args.resume}")
        trainer.load(args.resume)
    
    print("\n检查显存使用（训练器创建后）...")
    check_memory(device)
    
    # 开始训练
    num_updates = getattr(training_config, "num_updates", 1000)
    print(f"\n开始训练: {num_updates}个更新")
    print("=" * 80)
    
    try:
        trainer.train(num_updates=num_updates)
        print("\n训练完成！")
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存checkpoint
        checkpoint_dir = getattr(training_config, "checkpoint_dir", "checkpoints")
        checkpoint_path = f"{checkpoint_dir}/{args.algorithm}_a5000_interrupted.pt"
        trainer.save(checkpoint_path)
        print(f"已保存中断checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        # 保存checkpoint
        checkpoint_dir = getattr(training_config, "checkpoint_dir", "checkpoints")
        checkpoint_path = f"{checkpoint_dir}/{args.algorithm}_a5000_error.pt"
        try:
            trainer.save(checkpoint_path)
            print(f"已保存错误checkpoint: {checkpoint_path}")
        except:
            pass
        raise
    
    # 最终显存检查
    print("\n最终显存使用...")
    check_memory(device)
    
    # 关闭环境
    env.close()
    print("\n训练脚本执行完成")


if __name__ == "__main__":
    main()

