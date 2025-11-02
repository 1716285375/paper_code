#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PPO-Penalty自博弈训练脚本

在MAgent2 battle_v4环境中使用PPO-Penalty算法进行自博弈训练。
两个团队（红队和蓝队）进行对抗，通过不断更新对手策略来提升自身策略。

PPO-Penalty与标准PPO的区别：
- 标准PPO使用clipping限制策略更新：min(ratio * adv, clip(ratio) * adv)
- PPO-Penalty使用自适应KL惩罚：ratio * adv - beta * KL(old, new)
- beta系数根据KL散度自适应调整，保持KL散度在目标值附近

使用方法:
    python examples/train_selfplay_penalty.py --config configs/ppo_penalty_magent2_selfplay.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

# 添加项目根目录到路径（从examples目录向上找项目根目录）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.ppo.penalty_self_play_trainer import SelfPlayPPOPenaltyTrainer
from common.config import load_config
from common.tracking import ExperimentTracker, TensorBoardTracker, WandBTracker
from common.utils.data_manager import TrainingDataManager
from common.utils.logging import LoggerManager
from common.video import VideoRecorder
from core.agent import AgentManager
from environments.factory import make_env


def setup_seed(seed: int) -> None:
    """设置随机种子"""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="PPO-Penalty自博弈训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_penalty_magent2_selfplay.yaml",
        help="配置文件路径（相对于项目根目录）",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="设备（cuda/cpu），如果指定则覆盖配置文件"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的检查点路径（相对于项目根目录或绝对路径）",
    )

    args = parser.parse_args()

    # 加载配置（使用common/config模块的加载函数，返回字典格式）
    config = load_config(args.config, as_dict=True, project_root=project_root)

    # 设置随机种子
    seed = config.get("seed", 42)
    setup_seed(seed)

    # 设备配置
    device = args.device or config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = "cpu"

    # 创建环境
    env_config = config.get("env", {})
    env_name = env_config.pop("name", "magent2:battle_v4")
    env = make_env(env_name, **env_config)

    # 获取Agent IDs和团队信息
    env.reset()
    if hasattr(env, "agents"):
        agent_ids = env.agents
    else:
        # 如果没有agents属性，尝试从观测空间获取
        obs_spaces = env.observation_space()
        if isinstance(obs_spaces, dict):
            agent_ids = list(obs_spaces.keys())
        else:
            raise ValueError("无法自动获取agent IDs")

    # 分离红队和蓝队
    red_agents = [aid for aid in agent_ids if aid.startswith("red_")]
    blue_agents = [aid for aid in agent_ids if aid.startswith("blue_")]

    print(f"检测到 {len(agent_ids)} 个Agent")
    print(f"红队Agent: {red_agents}")
    print(f"蓝队Agent: {blue_agents}")

    # 获取观测和动作维度
    if hasattr(env, "observation_space") and hasattr(env, "action_space"):
        try:
            # 尝试从第一个agent获取
            first_agent_id = agent_ids[0]
            obs_space = env.observation_space(first_agent_id)
            action_space = env.action_space(first_agent_id)

            # 处理Dict观察空间（多Agent环境常见）
            if hasattr(obs_space, "shape"):
                obs_dim = int(obs_space.shape[0]) if len(obs_space.shape) > 0 else int(obs_space.shape)
            elif isinstance(obs_space, dict):
                # 如果是Dict，尝试获取第一个键的shape
                first_key = list(obs_space.spaces.keys())[0]
                obs_dim = int(obs_space.spaces[first_key].shape[0])
            else:
                obs_dim = config.get("agent", {}).get("obs_dim", 845)

            if hasattr(action_space, "n"):
                action_dim = int(action_space.n)
            elif hasattr(action_space, "shape"):
                action_dim = int(action_space.shape[0]) if len(action_space.shape) > 0 else 1
            else:
                action_dim = config.get("agent", {}).get("action_dim", 21)
        except Exception as e:
            print(f"无法自动获取空间信息: {e}，使用配置中的默认值")
            obs_dim = config.get("agent", {}).get("obs_dim", 845)
            action_dim = config.get("agent", {}).get("action_dim", 21)
    else:
        obs_dim = config.get("agent", {}).get("obs_dim", 845)
        action_dim = config.get("agent", {}).get("action_dim", 21)

    print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")

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

    # 创建日志记录器
    logger = LoggerManager(
        name="selfplay_penalty_training",
        log_dir=str(project_root / "logs"),
        enable_file=True,
        enable_console=True,
    )

    # 创建实验跟踪器（wandb/tensorboard）
    tracker = None
    tracking_config = config.get("tracking", {})
    if tracking_config.get("enabled", False):
        trackers = []

        # WandB跟踪器
        if tracking_config.get("wandb", {}).get("enabled", False):
            wandb_cfg = tracking_config["wandb"]
            try:
                # 自动生成运行名称：环境名_时间戳
                if wandb_cfg.get("name", None) is None:
                    env_name_clean = env_name.split(":")[-1] if ":" in env_name else env_name
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_name = f"{env_name_clean}_penalty_{timestamp}"
                else:
                    run_name = wandb_cfg.get("name")

                wandb_tracker = WandBTracker()
                wandb_tracker.init(
                    project=wandb_cfg.get("project", "magent2-selfplay-penalty"),
                    name=run_name,
                    config=config,
                    dir=str(project_root / "wandb"),
                )
                trackers.append(wandb_tracker)
                print(
                    f"✅ WandB tracking enabled: project={wandb_cfg.get('project')}, run={run_name}"
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize WandB: {e}")

        # TensorBoard跟踪器
        if tracking_config.get("tensorboard", {}).get("enabled", False):
            tb_cfg = tracking_config["tensorboard"]
            try:
                tb_tracker = TensorBoardTracker(log_dir=str(project_root / "runs"))
                tb_tracker.init(
                    project=tb_cfg.get("experiment_name", "magent2-selfplay-penalty"),
                    name=None,
                    config=config,
                )
                trackers.append(tb_tracker)
                print(
                    f"✅ TensorBoard tracking enabled: log_dir=runs/{tb_cfg.get('experiment_name')}"
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize TensorBoard: {e}")

        if trackers:
            tracker = ExperimentTracker(trackers)
            print(f"✅ Experiment tracking initialized with {len(trackers)} tracker(s)")
        else:
            print("⚠️  No tracking enabled or all trackers failed to initialize")

    # 创建数据管理器（用于保存指标和录制视频）
    data_manager = None
    data_config = config.get("data_saving", {})
    if data_config.get("enabled", True):
        output_dir = data_config.get("output_dir", str(project_root / "training_data"))
        save_format = data_config.get("format", "both")

        video_recorder = None
        if data_config.get("record_video", {}).get("enabled", False):
            video_recorder = VideoRecorder(
                output_dir=output_dir,
                fps=data_config.get("record_video", {}).get("fps", 30),
            )

        data_manager = TrainingDataManager(
            output_dir=output_dir,
            save_format=save_format,
            video_recorder=video_recorder,
        )

    # 创建PPO-Penalty自博弈训练器
    training_config = config.get("training", {})
    trainer = SelfPlayPPOPenaltyTrainer(
        agent=agent_manager,
        env=env,
        config=training_config,
        logger=logger,
        tracker=tracker,
        main_team=training_config.get("main_team", "team_red"),
        opponent_team=training_config.get("opponent_team", "team_blue"),
    )

    # 如果指定了data_manager，添加到trainer
    if data_manager is not None:
        trainer.data_manager = data_manager

    # 恢复训练（如果指定）
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / args.resume
        if resume_path.exists():
            trainer.load(str(resume_path))
            logger.logger.info(f"恢复训练从: {resume_path}")
        else:
            logger.logger.warning(f"检查点不存在: {resume_path}，从头开始训练")

    # 开始训练
    num_updates = training_config.get("num_updates", 10000)
    logger.logger.info(f"开始PPO-Penalty自博弈训练，共 {num_updates} 个更新...")

    try:
        trainer.train(num_updates=num_updates)
    except KeyboardInterrupt:
        logger.logger.info("训练被用户中断")
    except Exception as e:
        logger.logger.error(f"训练出错: {e}", exc_info=True)
        raise
    finally:
        # 保存最终检查点
        final_checkpoint = project_root / "checkpoints" / "penalty_selfplay_final.pt"
        final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(final_checkpoint))
        logger.logger.info(f"保存最终检查点到: {final_checkpoint}")

        # 保存所有数据
        if data_manager is not None:
            data_manager.save_all()
            logger.logger.info("训练数据已保存")

    logger.logger.info("训练完成！")


if __name__ == "__main__":
    main()

