#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的Magent2评估脚本 - 单文件，无依赖

只需要：
- torch
- magent2
- numpy

使用方法:
    python test/eval_simple.py --checkpoint checkpoints/hp3o_final.pt --num_episodes 10
"""

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from magent2.environments import battle_v4


class SimpleMLP(nn.Module):
    """简单的MLP网络用于推理"""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class SimpleAgent:
    """简单的Agent用于推理"""
    
    def __init__(self, encoder, policy_head, device="cpu"):
        self.encoder = encoder
        self.policy_head = policy_head
        self.device = device
        self.encoder.eval()
        self.policy_head.eval()
    
    def act(self, obs, deterministic=True, action_dim=21):
        """根据观测选择动作"""
        # 处理观测
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        
        # 展平观测
        if obs.dim() > 1:
            obs = obs.flatten()
        
        # 添加batch维度
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        obs = obs.to(self.device)
        
        with torch.no_grad():
            # 编码
            features = self.encoder(obs)
            
            # 策略头
            logits = self.policy_head(features)
            
            # 选择动作
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1)
            
            # 确保动作在有效范围内
            action = torch.clamp(action, 0, action_dim - 1)
            
            return int(action.item())


def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    """从检查点加载模型"""
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 尝试获取agent状态
    agent_state = None
    if "agent_state" in checkpoint:
        agent_state = checkpoint["agent_state"]
    elif "agent" in checkpoint:
        agent_state = checkpoint["agent"]
    else:
        raise KeyError("检查点中未找到agent状态")
    
    # 尝试从配置获取网络结构
    config = checkpoint.get("config", {})
    agent_config = config.get("agent", {})
    
    # 默认配置
    obs_dim = agent_config.get("obs_dim", 845)
    action_dim = agent_config.get("action_dim", 21)
    
    encoder_config = agent_config.get("encoder", {})
    encoder_params = encoder_config.get("params", {})
    hidden_dims = encoder_params.get("hidden_dims", [128, 64])
    
    policy_config = agent_config.get("policy_head", {})
    policy_params = policy_config.get("params", {})
    policy_hidden = policy_params.get("hidden_dims", [32])
    
    # 构建编码器
    encoder_layers = []
    prev_dim = obs_dim
    for hidden_dim in hidden_dims:
        encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())
        if encoder_params.get("use_layer_norm", False):
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        prev_dim = hidden_dim
    
    encoder = nn.Sequential(*encoder_layers).to(device)
    
    # 构建策略头
    policy_layers = []
    prev_dim = hidden_dims[-1] if hidden_dims else obs_dim
    for hidden_dim in policy_hidden:
        policy_layers.append(nn.Linear(prev_dim, hidden_dim))
        policy_layers.append(nn.ReLU())
        prev_dim = hidden_dim
    policy_layers.append(nn.Linear(prev_dim, action_dim))
    policy_head = nn.Sequential(*policy_layers).to(device)
    
    # 加载权重
    if "encoder" in agent_state:
        encoder.load_state_dict(agent_state["encoder"])
    if "policy_head" in agent_state:
        policy_head.load_state_dict(agent_state["policy_head"])
    
    return SimpleAgent(encoder, policy_head, device)


def evaluate(agent, env, num_episodes=10, deterministic=True, action_dim=21):
    """评估agent性能"""
    episode_rewards = []
    episode_lengths = []
    team_rewards = defaultdict(list)
    
    for episode in range(num_episodes):
        try:
            obs = env.reset()
        except Exception as e:
            print(f"Error resetting environment: {e}")
            raise
        
        done = False
        step = 0
        episode_reward = 0.0
        team_episode_rewards = defaultdict(float)
        
        # 获取活跃agent列表
        active_agents = list(env.agents) if hasattr(env, 'agents') else list(obs.keys())
        if not active_agents:
            print(f"Warning: No active agents after reset in episode {episode + 1}")
            continue
        
        # 分离红队和蓝队
        red_agents = [aid for aid in active_agents if aid.lower().startswith("red_")]
        blue_agents = [aid for aid in active_agents if aid.lower().startswith("blue_")]
        
        while not done and step < 1000:
            # 获取当前所有活跃的agent（每次step后可能变化）
            current_agents = list(env.agents) if hasattr(env, 'agents') else list(obs.keys())
            if not current_agents:
                print(f"Warning: No active agents at step {step}")
                break
            
            # 为所有活跃agent选择动作
            actions = {}
            for agent_id in current_agents:
                if agent_id in obs:
                    try:
                        agent_obs = obs[agent_id]
                        # 确保观测是numpy数组
                        if not isinstance(agent_obs, np.ndarray):
                            agent_obs = np.array(agent_obs)
                        action = agent.act(agent_obs, deterministic=deterministic, action_dim=action_dim)
                        # 确保动作在有效范围内且是整数
                        action = int(max(0, min(action, action_dim - 1)))
                        actions[agent_id] = action
                    except Exception as e:
                        # 如果出错，使用默认动作0
                        print(f"Warning: Error getting action for {agent_id}: {e}, using action 0")
                        actions[agent_id] = 0
                else:
                    # 如果agent不在obs中，使用默认动作0
                    actions[agent_id] = 0
            
            # 确保所有活跃agent都有动作（双重检查）
            for agent_id in current_agents:
                if agent_id not in actions:
                    actions[agent_id] = 0
            
            # 验证动作字典
            if len(actions) != len(current_agents):
                print(f"Warning: Actions count ({len(actions)}) != Active agents count ({len(current_agents)})")
                print(f"Actions keys: {list(actions.keys())}")
                print(f"Active agents: {current_agents}")
            
            # 确保所有动作都是整数
            actions = {k: int(v) for k, v in actions.items()}
            
            try:
                # 执行动作
                next_obs, rewards, terminations, truncations, infos = env.step(actions)
            except Exception as e:
                print(f"Error in env.step at step {step}: {e}")
                print(f"Actions: {actions}")
                print(f"Active agents: {current_agents}")
                print(f"Actions count: {len(actions)}, Agents count: {len(current_agents)}")
                # 尝试打印动作值的范围
                if actions:
                    action_values = list(actions.values())
                    print(f"Action value range: [{min(action_values)}, {max(action_values)}]")
                    print(f"Action dim: {action_dim}")
                raise
            
            # 合并terminations和truncations
            dones = {}
            for aid in current_agents:
                term = terminations.get(aid, False) if isinstance(terminations, dict) else False
                trunc = truncations.get(aid, False) if isinstance(truncations, dict) else False
                dones[aid] = bool(term or trunc)
            
            done = dones.get("__all__", False) if isinstance(dones, dict) else False
            if not done:
                # 检查是否所有agent都结束了
                if all(dones.values()):
                    done = True
            
            # 累计奖励
            if isinstance(rewards, dict):
                for agent_id, reward in rewards.items():
                    episode_reward += reward
                    # 按团队分类
                    if agent_id in red_agents:
                        team_episode_rewards["red"] += reward
                    elif agent_id in blue_agents:
                        team_episode_rewards["blue"] += reward
            else:
                episode_reward += rewards
            
            obs = next_obs
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        for team, reward in team_episode_rewards.items():
            team_rewards[team].append(reward)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, Steps={step}")
    
    # 计算统计信息
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
    }
    
    for team, rewards in team_rewards.items():
        metrics[f"team_{team}_mean_reward"] = np.mean(rewards)
        metrics[f"team_{team}_std_reward"] = np.std(rewards)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="简单的Magent2评估脚本")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="C:\A\RL\paper-code\checkpoints\mappo_magent2_selfplay_12v12_8gb\mappo_final.pt",
        help="检查点文件路径",
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
        default="cpu",
        help="设备（cuda/cpu）",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=20,
        help="地图大小",
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=100,
        help="最大步数",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="使用确定性策略（默认True）",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="使用随机策略（覆盖deterministic）",
    )
    
    args = parser.parse_args()
    
    # 确定是否使用确定性策略
    deterministic = not args.stochastic if args.stochastic else args.deterministic
    
    print("=" * 60)
    print("简单的Magent2评估")
    print("=" * 60)
    print(f"检查点: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")
    print(f"设备: {args.device}")
    print(f"策略: {'确定性' if deterministic else '随机'}")
    print("=" * 60)
    
    # 加载模型
    agent = load_model_from_checkpoint(args.checkpoint, device=args.device)
    print("✓ 模型加载成功")
    
    # 创建环境
    env = battle_v4.parallel_env(
        map_size=args.map_size,
        max_cycles=args.max_cycles,
    )
    print(f"✓ 环境创建成功: map_size={args.map_size}, max_cycles={args.max_cycles}")
    
    # 获取动作维度
    action_dim = 21  # 默认值
    try:
        if env.agents:
            action_space = env.action_space(env.agents[0])
            if hasattr(action_space, "n"):
                action_dim = action_space.n
    except:
        pass
    
    # 运行评估
    print("\n开始评估...")
    metrics = evaluate(agent, env, num_episodes=args.num_episodes, deterministic=deterministic, action_dim=action_dim)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)
    
    # 关闭环境
    env.close()
    print("\n✓ 评估完成")


if __name__ == "__main__":
    main()

