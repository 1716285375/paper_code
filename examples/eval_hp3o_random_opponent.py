# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : eval_hp3o_random_opponent.py
@Author         : GPT-5 Codex (generated)
@Description    : Evaluate an HP3O checkpoint against a random opponent.

运行示例:
    python examples/eval_hp3o_random_opponent.py \
        --config configs/hp3o/hp3o_magent2_selfplay_2v2_1M_4090.yaml \
        --checkpoint checkpoints/hp3o_magent2_selfplay_2v2_1M_4090/hp3o_selfplay_checkpoint_15000.pt \
        --episodes 5

脚本功能:
    - 加载项目配置与环境
    - 使用训练好的HP3O红队策略对抗蓝队随机策略
    - 记录每一步双方执行的动作、平均奖励、剩余单位（存活数量/生命值）
    - 可选将详细日志保存为 JSONLines / CSV 文件，便于进一步分析
"""
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from common.config import load_config
from core.agent.manager import AgentManager
from environments import make_env

# 复用训练脚本中的工具函数，避免重复代码
from examples.train_selfplay_unified import (  # type: ignore
    get_environment_info,
)


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    red_total_reward: float
    blue_total_reward: float
    red_alive_final: int
    blue_alive_final: int
    winner: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate HP3O self-play checkpoint against random opponent"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hp3o/hp3o_magent2_selfplay_2v2_1M_4090.yaml",
        help="路径: 训练使用的配置文件",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=r"C:\A\RL\paper-code\checkpoints\hp3o_magent2_selfplay_2v2_1M_4090\hp3o_selfplay_checkpoint_15000.pt",
        help="路径: HP3O自博弈训练生成的checkpoint (pt文件)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="评估Episode数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="评估使用的设备（cuda/cpu），默认自动检测",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（影响随机对手与评估一致性）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="每个episode的最大步数（优先级高于配置的max_cycles）",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="评估时渲染环境（需环境支持）",
    )
    parser.add_argument(
        "--log-jsonl",
        type=str,
        default=None,
        help="可选: 保存逐步日志到JSONLines文件",
    )
    parser.add_argument(
        "--log-csv",
        type=str,
        default=None,
        help="可选: 保存逐步日志到CSV文件",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="控制台仅输出总结信息，逐步日志写入文件",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(agent_manager, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent_state = checkpoint.get("agent_state")
    if not agent_state:
        raise KeyError("Checkpoint missing 'agent_state'.")
    agent_manager.load_state_dict(agent_state)
    agent_manager.to_eval_mode()


def sample_random_actions(
    env,
    agent_ids: List[str],
    observations: Dict[str, np.ndarray],
) -> Dict[str, int]:
    actions = {}
    for agent_id in agent_ids:
        if agent_id not in observations:
            # 该智能体可能已经退出（死亡）
            continue
        avail_actions = env.get_avail_agent_actions(agent_id)
        if not avail_actions:
            # 默认使用全部动作空间
            action_space = env.action_space(agent_id)
            if hasattr(action_space, "sample"):
                actions[agent_id] = action_space.sample()
            else:
                actions[agent_id] = 0
        else:
            actions[agent_id] = random.choice(avail_actions)
    return actions


def compute_alive_count(
    agent_ids: List[str],
    dones: Dict[str, bool],
) -> int:
    """
    统计仍存活的智能体数量（done == False）。
    """
    return sum(0 if dones.get(aid, False) else 1 for aid in agent_ids)


def log_step(
    buffer: List[Dict[str, object]],
    episode_idx: int,
    step_idx: int,
    red_actions: Dict[str, int],
    blue_actions: Dict[str, int],
    red_rewards: Dict[str, float],
    blue_rewards: Dict[str, float],
    red_alive: int,
    blue_alive: int,
) -> None:
    buffer.append(
        {
            "episode": episode_idx,
            "step": step_idx,
            "red_actions": red_actions,
            "blue_actions": blue_actions,
            "red_reward": float(sum(red_rewards.values())) if red_rewards else 0.0,
            "blue_reward": float(sum(blue_rewards.values())) if blue_rewards else 0.0,
            "red_alive": red_alive,
            "blue_alive": blue_alive,
        }
    )


def maybe_write_jsonl(path: Optional[Path], records: List[Dict[str, object]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def maybe_write_csv(path: Optional[Path], records: List[Dict[str, object]]) -> None:
    if path is None or not records:
        return
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    episodes: int,
    device: torch.device,
    max_steps_override: Optional[int],
    render: bool,
    quiet: bool,
    jsonl_path: Optional[Path],
    csv_path: Optional[Path],
) -> None:
    # 加载配置
    project_root = Path(__file__).parent.parent
    config = load_config(str(config_path), as_dict=True, project_root=project_root)

    # 初始化环境
    env_config = config.get("env", {}).copy()
    env_name = env_config.pop("name", "magent2:battle_v4")
    env = make_env(env_name, **env_config)

    # 获取环境信息
    agent_ids, obs_dim, action_dim, _, _, = (None, None, None, None, None)
    agent_ids, obs_dim, action_dim, state_dim, red_agents, blue_agents = get_environment_info(
        env, config
    )

    if not red_agents or not blue_agents:
        raise ValueError("Failed to split agents into red/blue teams.")

    # 构建只包含红队的AgentManager
    agent_config = config.get("agent", {})
    red_agent_manager = AgentManager(
        agent_ids=red_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=agent_config,
        device=str(device),
        shared_agents={"team_red": red_agents},
    )

    # 加载checkpoint中的模型参数
    load_checkpoint(red_agent_manager, checkpoint_path, device=torch.device(device))

    # 环境Episode限制
    episode_limit = max_steps_override or env.episode_limit or config.get("training", {}).get("max_steps_per_episode", 500)

    if not quiet:
        print("=" * 60)
        print("HP3O Checkpoint Evaluation (Random Opponent)")
        print("=" * 60)
        print(f"Config:      {config_path}")
        print(f"Checkpoint:  {checkpoint_path}")
        print(f"Device:      {device}")
        print(f"Episodes:    {episodes}")
        print(f"Episode Max Steps: {episode_limit}")
        print(f"Red agents:  {red_agents}")
        print(f"Blue agents: {blue_agents}")
        print("=" * 60)

    # 评估循环
    step_records: List[Dict[str, object]] = []
    episode_summaries: List[EpisodeSummary] = []

    for ep in range(1, episodes + 1):
        observations = env.reset()
        red_agent_manager.to_eval_mode()

        red_episode_reward = 0.0
        blue_episode_reward = 0.0
        step_idx = 0
        done_all = False
        red_alive = len(red_agents)
        blue_alive = len(blue_agents)

        while not done_all and step_idx < episode_limit:
            step_idx += 1

            # 当前仍在场的智能体列表（基于observations，这是step之前的存活状态）
            active_red = [aid for aid in red_agents if aid in observations]
            active_blue = [aid for aid in blue_agents if aid in observations]

            # 在step之前统计存活数量（基于observations中存在的agent）
            red_alive = len(active_red)
            blue_alive = len(active_blue)

            # 选择动作
            red_obs = {aid: observations[aid] for aid in active_red}
            red_actions_batch = red_agent_manager.act(
                red_obs,
                deterministic=True,
                show_progress=False,
            )
            red_actions = {aid: int(action) for aid, (action, _, _) in red_actions_batch.items()}

            blue_actions = sample_random_actions(env, active_blue, observations)

            joint_actions = {}
            joint_actions.update(red_actions)
            joint_actions.update(blue_actions)

            # 环境前进一步
            next_obs, rewards, dones, infos = env.step(joint_actions)

            if render:
                env.render()

            # 汇总奖励
            red_step_reward = {aid: float(rewards.get(aid, 0.0)) for aid in red_agents}
            blue_step_reward = {aid: float(rewards.get(aid, 0.0)) for aid in blue_agents}
            red_episode_reward += sum(red_step_reward.values())
            blue_episode_reward += sum(blue_step_reward.values())

            # 注意：存活数量已经在step之前统计了（基于observations）
            # 这样即使环境因为达到max_steps而将所有agent标记为done，也不会影响统计
            # 因为如果agent真的死了，它就不会在observations中出现

            log_step(
                step_records,
                ep,
                step_idx,
                red_actions,
                blue_actions,
                red_step_reward,
                blue_step_reward,
                red_alive,
                blue_alive,
            )

            if not quiet:
                print(
                    f"[Episode {ep:02d} | Step {step_idx:03d}] "
                    f"RedAlive={red_alive} "
                    f"BlueAlive={blue_alive} "
                    f"RedActions={red_actions} "
                    f"BlueActions={blue_actions} "
                    f"RReward={sum(red_step_reward.values()):+.3f} "
                    f"BReward={sum(blue_step_reward.values()):+.3f}"
                )

            observations = next_obs

            done_all = (
                dones.get("__all__", False)
                or red_alive == 0
                or blue_alive == 0
                or all(dones.get(aid, False) for aid in set(red_agents + blue_agents))
            )

        # Episode总结
        winner = "draw"
        if red_episode_reward > blue_episode_reward:
            winner = "red"
        elif blue_episode_reward > red_episode_reward:
            winner = "blue"

        episode_summary = EpisodeSummary(
            episode=ep,
            steps=step_idx,
            red_total_reward=red_episode_reward,
            blue_total_reward=blue_episode_reward,
            red_alive_final=red_alive,
            blue_alive_final=blue_alive,
            winner=winner,
        )
        episode_summaries.append(episode_summary)

        if not quiet:
            print(
                "-" * 60
                + f"\nEpisode {ep} finished in {step_idx} steps. "
                f"RedReward={red_episode_reward:.3f}, BlueReward={blue_episode_reward:.3f}, Winner={winner.upper()}"
                + "\n"
                + "-" * 60
            )

    env.close()

    # 输出总结
    print("\nEvaluation Summary")
    print("=" * 60)
    for summary in episode_summaries:
        print(
            f"Episode {summary.episode:02d} | Steps={summary.steps:03d} | "
            f"RedReward={summary.red_total_reward:+.3f} | BlueReward={summary.blue_total_reward:+.3f} | "
            f"Winner={summary.winner.upper()} | "
            f"RedAlive={summary.red_alive_final} | BlueAlive={summary.blue_alive_final}"
        )

    red_wins = sum(1 for s in episode_summaries if s.winner == "red")
    blue_wins = sum(1 for s in episode_summaries if s.winner == "blue")
    draws = episodes - red_wins - blue_wins
    print("-" * 60)
    print(
        f"Overall -> RedWins: {red_wins}, BlueWins: {blue_wins}, Draws: {draws}, "
        f"Average Red Reward: {np.mean([s.red_total_reward for s in episode_summaries]):.3f}, "
        f"Average Blue Reward: {np.mean([s.blue_total_reward for s in episode_summaries]):.3f}"
    )
    print("=" * 60)

    maybe_write_jsonl(jsonl_path, step_records)
    maybe_write_csv(csv_path, step_records)

    if jsonl_path or csv_path:
        print("Step-level logs saved to:")
        if jsonl_path:
            print(f"  JSONL: {jsonl_path}")
        if csv_path:
            print(f"  CSV:   {csv_path}")


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    jsonl_path = Path(args.log_jsonl) if args.log_jsonl else None
    csv_path = Path(args.log_csv) if args.log_csv else None

    evaluate(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        device=device,
        max_steps_override=args.max_steps,
        render=args.render,
        quiet=args.quiet,
        jsonl_path=jsonl_path,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()

