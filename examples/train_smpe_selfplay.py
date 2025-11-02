# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : train_smpe_selfplay.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-02 00:00
@Update Date    :
@Description    : SMPE²自博弈训练脚本
结合VAE状态建模、Filter过滤、SimHash内在奖励与自博弈对手池的训练
"""
# ------------------------------------------------------------

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithms.mappo import SMPESelfPlayTrainer
from common.config import load_config
from common.tracking import ExperimentTracker, TensorBoardTracker, WandBTracker
from common.utils.data_manager import TrainingDataManager
from common.utils.logging import LoggerManager
from core.agent import AgentManager
from environments import make_env


def create_smpe_agent_manager(config: dict, agent_ids: list, obs_dim: int, action_dim: int, device: str, team_name: str) -> AgentManager:
    """
    创建SMPE² Agent管理器

    Args:
        config: Agent配置
        agent_ids: Agent ID列表
        obs_dim: 观测维度
        action_dim: 动作空间维度
        device: 设备
        team_name: 团队名称

    Returns:
        AgentManager实例
    """
    from core.agent.smpe_agent import SMPEPolicyAgent

    # 创建AgentManager（共享策略）
    agent_manager = AgentManager(
        agent_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        device=device,
        shared_agents={team_name: agent_ids},  # 共享策略
    )

    # 替换为SMPEPolicyAgent实例
    # 对于共享策略，所有agent使用同一个实例
    base_agent = SMPEPolicyAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        device=device,
        agent_id=0,  # 使用0作为agent_id（团队共享）
        agent_id_dim=config.get("agent_id_dim", 32),
    )

    # 替换所有agent为SMPEPolicyAgent（共享策略）
    for agent_id in agent_ids:
        agent_manager._agents[agent_id] = base_agent

    return agent_manager


def main():
    parser = argparse.ArgumentParser(description="SMPE²自博弈训练")
    parser.add_argument("--config", type=str, default="configs/smpe_magent2_selfplay_12v12.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    args = parser.parse_args()

    # 加载配置（使用common/config模块的加载函数，返回字典格式）
    config = load_config(args.config, as_dict=True, project_root=project_root)

    # 环境配置
    env_config = config.get("env", {})
    env_id = env_config.get("id", "magent2:battle_v4")
    env_kwargs = env_config.get("kwargs", {})

    # 创建环境
    env = make_env(env_id, **env_kwargs)
    env.reset()

    # 获取Agent信息
    agent_ids = getattr(env, "agents", [])
    if not agent_ids:
        agent_ids = list(env.observation_space.keys()) if hasattr(env.observation_space, "keys") else []

    # 分离团队
    red_agents = [aid for aid in agent_ids if "red" in aid.lower()]
    blue_agents = [aid for aid in agent_ids if "blue" in aid.lower()]

    # 获取观测和动作维度
    sample_agent_id = agent_ids[0] if agent_ids else None
    if sample_agent_id:
        obs_space = env.observation_space(sample_agent_id)
        action_space = env.action_space(sample_agent_id)

        if hasattr(obs_space, "shape"):
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = obs_space.n if hasattr(obs_space, "n") else 845  # 默认battle_v4

        action_dim = action_space.n if hasattr(action_space, "n") else 21  # 默认battle_v4
    else:
        obs_dim = env_config.get("obs_dim", 845)
        action_dim = env_config.get("action_dim", 21)

    print(f"检测到 {len(agent_ids)} 个Agent")
    print(f"红队Agent: {red_agents[:10]}... ({len(red_agents)}个)" if len(red_agents) > 10 else f"红队Agent: {red_agents}")
    print(f"蓝队Agent: {blue_agents[:10]}... ({len(blue_agents)}个)" if len(blue_agents) > 10 else f"蓝队Agent: {blue_agents}")
    print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")

    # Agent配置
    agent_config = config.get("agent", {})

    # 创建Agent管理器（红队和蓝队）
    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        device = "cpu"

    # 创建合并的AgentManager（包含两个团队）
    all_agent_ids = red_agents + blue_agents
    combined_agent_manager = AgentManager(
        agent_ids=all_agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device=device,
        shared_agents={
            "team_red": red_agents,
            "team_blue": blue_agents,
        },
    )

    # 为每个团队创建SMPEPolicyAgent并替换到combined_agent_manager
    from core.agent.smpe_agent import SMPEPolicyAgent

    # 红队Agent（共享策略）
    red_agent = SMPEPolicyAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device=device,
        agent_id=0,
        agent_id_dim=agent_config.get("agent_id_dim", 32),
    )
    for agent_id in red_agents:
        combined_agent_manager._agents[agent_id] = red_agent

    # 蓝队Agent（共享策略）
    blue_agent = SMPEPolicyAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device=device,
        agent_id=1,
        agent_id_dim=agent_config.get("agent_id_dim", 32),
    )
    for agent_id in blue_agents:
        combined_agent_manager._agents[agent_id] = blue_agent

    # 训练配置
    training_config = config.get("training", {})

    # 日志记录器
    logger = LoggerManager(
        name="smpe_selfplay_training",
        log_dir=str(project_root / "logs"),
        enable_file=True,
        enable_console=True,
    )

    # 实验跟踪器
    tracker = None
    tracking_config = config.get("tracking", {})
    if tracking_config.get("enabled", True):
        trackers = []

        # WandB
        if tracking_config.get("wandb", {}).get("enabled", False):
            wandb_config = tracking_config["wandb"]
            wandb_tracker = WandBTracker()
            project = wandb_config.get("project", "magent2-smpe-selfplay")
            name = wandb_config.get("name", None)
            if name is None:
                # 自动生成名称：环境名_时间戳
                from datetime import datetime

                env_name = env_id.split(":")[-1] if ":" in env_id else env_id
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"{env_name}_{timestamp}"
            wandb_tracker.init(
                project=project,
                name=name,
                config=config,
                dir=str(project_root / "wandb"),
            )
            trackers.append(wandb_tracker)

        # TensorBoard
        if tracking_config.get("tensorboard", {}).get("enabled", False):
            tb_config = tracking_config["tensorboard"]
            tb_tracker = TensorBoardTracker()
            project = tb_config.get("project", "magent2-smpe-selfplay")
            name = tb_config.get("name", None) or "smpe_selfplay"
            tb_tracker.init(
                log_dir=str(project_root / "runs"),
                project=project,
                name=name,
                config=config,
            )
            trackers.append(tb_tracker)

        if trackers:
            tracker = ExperimentTracker(trackers)
            # 注意：各个tracker已经单独初始化，不需要再调用tracker.init()

    # 数据管理器
    data_manager = None
    data_saving_config = config.get("data_saving", {})
    if data_saving_config.get("enabled", False):
        data_manager = TrainingDataManager(
            output_dir=data_saving_config.get("output_dir", "training_data"),
            save_format=data_saving_config.get("format", "both"),  # "json", "csv", "both"
        )

    # 创建训练器
    trainer = SMPESelfPlayTrainer(
        agent=combined_agent_manager,
        env=env,
        config=training_config,
        logger=logger,
        tracker=tracker,
        main_team="team_red",
        opponent_team="team_blue",
    )

    # 设置数据管理器
    trainer.data_manager = data_manager

    # 恢复训练（如果指定）
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"恢复训练: {resume_path}")
            trainer.load(str(resume_path))

    # 开始训练
    num_updates = training_config.get("num_updates", 10000)
    print(f"开始SMPE²自博弈训练，共 {num_updates} 个更新...")

    try:
        trainer.train(num_updates=num_updates)
        print("训练完成！")
    except KeyboardInterrupt:
        print("\n训练被中断，保存检查点...")
        checkpoint_path = project_root / "checkpoints" / "interrupted_smpe_checkpoint.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save(str(checkpoint_path))
        print("检查点已保存")
    finally:
        # 保存所有训练数据
        if data_manager is not None:
            try:
                saved_paths = data_manager.save_all(prefix="smpe_selfplay_training")
                print("\n✅ Training data saved:")
                for file_type, path in saved_paths.items():
                    print(f"   - {file_type}: {path}")
            except Exception as e:
                print(f"⚠️  Failed to save training data: {e}")

        # 关闭tracker
        if tracker and tracker.is_initialized:
            try:
                tracker.finish()
            except Exception as e:
                print(f"⚠️  Failed to finish tracker: {e}")


if __name__ == "__main__":
    import numpy as np
    import torch

    main()

