#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自博弈训练脚本

在MAgent2 battle_v4环境中使用PPO算法进行自博弈训练。
两个团队（红队和蓝队）进行对抗，通过不断更新对手策略来提升自身策略。

使用方法:
    python examples/train_selfplay.py --config configs/ppo_magent2_selfplay.yaml
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

# 添加项目根目录到路径（从examples目录向上找项目根目录）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.ppo.self_play_trainer import SelfPlayPPOTrainer
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
    parser = argparse.ArgumentParser(description="PPO自博弈训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_magent2_selfplay.yaml",
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

    # 获取环境信息
    # 先重置环境以获取agent列表
    env.reset()
    agent_ids = getattr(env, "agents", [])

    if not agent_ids:
        raise ValueError("无法获取agent ID列表，请检查环境配置")

    print(f"检测到 {len(agent_ids)} 个Agent")

    # 分离红队和蓝队
    red_agents = [aid for aid in agent_ids if "red" in aid.lower() or "0" in aid]
    blue_agents = [aid for aid in agent_ids if "blue" in aid.lower() or "1" in aid]

    # 如果无法自动分离，则平分
    if not red_agents or not blue_agents:
        mid = len(agent_ids) // 2
        red_agents = agent_ids[:mid]
        blue_agents = agent_ids[mid:]

    print(f"红队Agent: {red_agents}")
    print(f"蓝队Agent: {blue_agents}")

    # 获取观测和动作空间
    if red_agents:
        try:
            obs_space = env.observation_space(red_agents[0])
            action_space = env.action_space(red_agents[0])

            # 计算观测维度
            if hasattr(obs_space, "shape"):
                obs_dim = int(np.prod(obs_space.shape))
            elif hasattr(obs_space, "spaces"):  # 字典空间
                # 获取第一个空间的维度
                first_space = list(obs_space.spaces.values())[0]
                if hasattr(first_space, "shape"):
                    obs_dim = int(np.prod(first_space.shape))
                else:
                    obs_dim = config.get("agent", {}).get("obs_dim", 845)
            else:
                obs_dim = config.get("agent", {}).get("obs_dim", 845)

            # 计算动作维度
            if hasattr(action_space, "n"):
                action_dim = action_space.n
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
        name="selfplay_training",
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
                    # 从环境名称中提取（去掉前缀，如"magent2:"）
                    env_name_clean = env_name.split(":")[-1] if ":" in env_name else env_name
                    # 生成时间戳（格式：YYYYMMDD_HHMMSS）
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    # 组合运行名称：环境名_时间戳
                    run_name = f"{env_name_clean}_{timestamp}"
                else:
                    run_name = wandb_cfg.get("name")

                wandb_tracker = WandBTracker()
                wandb_tracker.init(
                    project=wandb_cfg.get("project", "magent2-selfplay"),
                    name=run_name,
                    config=config,
                    dir=str(project_root / "wandb"),  # wandb的dir参数用于指定保存目录
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
                tb_tracker = TensorBoardTracker(
                    log_dir=str(project_root / "runs"),
                )
                tb_tracker.init(
                    project=tb_cfg.get("experiment_name", "magent2-selfplay"),
                    name=None,  # 可以添加运行名称
                    config=config,
                )
                trackers.append(tb_tracker)
                print(
                    f"✅ TensorBoard tracking enabled: log_dir=runs/{tb_cfg.get('experiment_name')}"
                )
            except Exception as e:
                print(f"⚠️  Failed to initialize TensorBoard: {e}")

        # 如果至少有一个tracker成功创建，则创建ExperimentTracker
        if trackers:
            tracker = ExperimentTracker(trackers)
            print(f"✅ Experiment tracking initialized with {len(trackers)} tracker(s)")
        else:
            print("⚠️  No tracking enabled or all trackers failed to initialize")

    # 创建数据管理器（用于保存指标和录制视频）
    data_manager = None
    data_config = config.get("data_saving", {})
    if data_config.get("enabled", True):  # 默认启用
        output_dir = data_config.get("output_dir", str(project_root / "training_data"))
        save_format = data_config.get("format", "both")  # "json", "csv", "both"

        # 视频录制器（可选）
        video_recorder = None
        if data_config.get("record_video", {}).get("enabled", False):
            video_config = data_config["record_video"]
            try:
                video_recorder = VideoRecorder(
                    output_dir=str(project_root / "videos"),
                    fps=video_config.get("fps", 30),
                )
                print(f"✅ Video recording enabled: output_dir=videos")
            except Exception as e:
                print(f"⚠️  Failed to initialize video recorder: {e}")

        data_manager = TrainingDataManager(
            output_dir=output_dir,
            save_format=save_format,
            video_recorder=video_recorder,
        )
        print(f"✅ Data manager enabled: output_dir={output_dir}, format={save_format}")

    # 创建训练器
    training_config = config.get("training", {})
    trainer = SelfPlayPPOTrainer(
        agent=agent_manager,
        env=env,
        config=training_config,
        logger=logger,
        tracker=tracker,
        main_team="team_red",
        opponent_team="team_blue",
    )

    # 设置数据管理器
    trainer.data_manager = data_manager

    # 恢复训练
    if args.resume:
        print(f"从检查点恢复训练: {args.resume}")
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = project_root / args.resume
        trainer.load(str(resume_path))

    # 开始训练
    num_updates = training_config.get("num_updates", 10000)
    print(f"开始自博弈训练，共 {num_updates} 个更新...")

    try:
        trainer.train(num_updates=num_updates)
        print("训练完成！")
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        checkpoint_path = project_root / "checkpoints" / "interrupted_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(checkpoint_path))
        print("检查点已保存")
    finally:
        # 保存所有训练数据
        if data_manager is not None:
            try:
                saved_paths = data_manager.save_all(prefix="selfplay_training")
                print("\n✅ Training data saved:")
                for file_type, path in saved_paths.items():
                    print(f"   - {file_type}: {path}")
            except Exception as e:
                print(f"⚠️  Failed to save training data: {e}")

        # 关闭tracker
        if tracker and tracker.is_initialized:
            try:
                tracker.close()
                print("✅ Tracker closed")
            except:
                pass
        env.close()


if __name__ == "__main__":
    import numpy as np

    main()
