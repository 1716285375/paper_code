# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : train_selfplay_magent2.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-10 16:25
@Update Date    : 
@Description    : 
"""
# ------------------------------------------------------------

import torch
import copy
from typing import Dict, List
from collections import deque
from ding.config import compile_config
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import DQNPolicy
from ding.model import DQN
from ding.utils import set_pkg_seed

# simple_magent2_wrapper.py
from easydict import EasyDict


def create_magent2_env(cfg: dict = None):
    """创建 MAgent2 环境"""
    if cfg is None:
        cfg = {}

    from magent2.environments import battle_v4

    env = battle_v4.parallel_env(
        map_size=cfg.get('map_size', 45),
        max_cycles=cfg.get('max_cycles', 300),
    )

    wrapper_cfg = EasyDict({'env_id': cfg.get('env_id', 'battle_v4'),
                            'manager': cfg.get('manager', {})})
    # 使用 DI-engine 的通用包装器
    return DingEnvWrapper(env, wrapper_cfg)


# magent2_selfplay_config.py
from easydict import EasyDict

magent2_selfplay_config = dict(
    exp_name='magent2_battle_selfplay',
    env=dict(
        env_id='battle_v4',
        map_size=45,
        max_cycles=300,
        collector_env_num=4,
        evaluator_env_num=2,
        n_evaluator_episode=5,
        stop_value=100,
        manager=dict(shared_memory=False),
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        model=dict(
            agent_obs_shape=13 * 13 * 5,  # MAgent2 观测空间
            agent_num=81,  # 每个团队的智能体数量
            action_shape=21,  # MAgent2 动作空间
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=128,
            learning_rate=0.0005,
            target_update_freq=100,
            ignore_done=False,
        ),
        collect=dict(
            n_sample=1000,
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=100)),
        other=dict(
            eps=dict(
                type='exp',
                start=1.0,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=100000,
                enable_track_used_data=True,
            ),
        ),
    ),
    # 自博弈配置
    selfplay=dict(
        # 对手池配置
        opponent_pool_size=5,  # 保留最近5个历史策略
        update_opponent_freq=1000,  # 每1000步更新对手池

        # 联赛训练配置
        league=dict(
            enabled=True,
            main_exploiter_ratio=0.35,  # 35% 时间对战主要利用者
            league_exploiter_ratio=0.25,  # 25% 时间对战联盟利用者
            main_agents_ratio=0.40,  # 40% 时间对战历史版本
        ),

        # Elo 评分
        use_elo=True,
        initial_elo=1500,
    ),
)



class OpponentPool:
    """对手池管理器"""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.pool = deque(maxlen=max_size)
        self.elo_ratings = {}
        self.match_history = []

    def add_policy(self, policy_state: dict, policy_id: str, elo: float = 1500):
        """添加策略到对手池"""
        self.pool.append({
            'id': policy_id,
            'state': copy.deepcopy(policy_state),
            'elo': elo,
            'games_played': 0
        })
        self.elo_ratings[policy_id] = elo
        print(f"Added policy {policy_id} to opponent pool (Elo: {elo:.0f})")

    def sample_opponent(self, strategy: str = 'latest'):
        """采样对手"""
        if len(self.pool) == 0:
            return None

        if strategy == 'latest':
            return self.pool[-1]
        elif strategy == 'random':
            import random
            return random.choice(self.pool)
        elif strategy == 'elo_weighted':
            # 根据 Elo 评分加权采样
            import random
            elos = [p['elo'] for p in self.pool]
            weights = [elo / sum(elos) for elo in elos]
            return random.choices(list(self.pool), weights=weights)[0]
        else:
            return self.pool[-1]

    def update_elo(self, policy1_id: str, policy2_id: str,
                   score1: float, k: float = 32):
        """更新 Elo 评分"""
        elo1 = self.elo_ratings.get(policy1_id, 1500)
        elo2 = self.elo_ratings.get(policy2_id, 1500)

        expected1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        expected2 = 1 / (1 + 10 ** ((elo1 - elo2) / 400))

        new_elo1 = elo1 + k * (score1 - expected1)
        new_elo2 = elo2 + k * ((1 - score1) - expected2)

        self.elo_ratings[policy1_id] = new_elo1
        self.elo_ratings[policy2_id] = new_elo2

        self.match_history.append({
            'policy1': policy1_id,
            'policy2': policy2_id,
            'score1': score1,
            'elo1_before': elo1,
            'elo1_after': new_elo1,
            'elo2_before': elo2,
            'elo2_after': new_elo2,
        })

        return new_elo1, new_elo2


class SelfPlayTrainer:
    """自博弈训练器"""

    def __init__(self, cfg):
        self.cfg = cfg
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        # 创建环境
        self.collector_env = self._create_env_manager(cfg.env.collector_env_num)
        self.evaluator_env = self._create_env_manager(cfg.env.evaluator_env_num)

        # 创建主策略
        self.policy = self._create_policy()

        # 创建对手池
        self.opponent_pool = OpponentPool(
            max_size=cfg.selfplay.opponent_pool_size
        )

        # 训练计数器
        self.train_iter = 0
        self.collect_count = 0

    def _create_env_manager(self, env_num: int):
        """创建环境管理器"""
        return BaseEnvManager(
            env_fn=[lambda: create_magent2_env(self.cfg.env) for _ in range(env_num)],
            cfg=self.cfg.env.manager
        )

    def _create_policy(self):
        """创建策略"""
        model = DQN(**self.cfg.policy.model)
        policy = DQNPolicy(self.cfg.policy, model=model)
        return policy

    def train(self, max_iterations: int = 10000):
        """主训练循环"""

        for iteration in range(max_iterations):
            # 1. 收集数据
            self._collect_data()

            # 2. 训练策略
            if self.policy.get_attribute('replay_buffer').count() > self.cfg.policy.learn.batch_size:
                for _ in range(self.cfg.policy.learn.update_per_collect):
                    train_data = self.policy.get_attribute('replay_buffer').sample(
                        self.cfg.policy.learn.batch_size,
                        train_iter=self.train_iter
                    )
                    self.policy.forward(train_data, stage='learn')
                    self.train_iter += 1

            # 3. 更新对手池
            if iteration % self.cfg.selfplay.update_opponent_freq == 0:
                self._update_opponent_pool()

            # 4. 评估
            if iteration % self.cfg.policy.eval.evaluator.eval_freq == 0:
                self._evaluate()

            # 5. 保存检查点
            if iteration % 1000 == 0:
                self._save_checkpoint(iteration)

        print("Training completed!")

    def _collect_data(self):
        """收集训练数据（自博弈）"""
        # 采样对手
        opponent = self.opponent_pool.sample_opponent(strategy='elo_weighted')

        if opponent is not None:
            # 加载对手策略
            opponent_policy = self._create_policy()
            opponent_policy.load_state_dict(opponent['state'])
            opponent_policy.eval()
        else:
            # 如果没有对手，使用自己
            opponent_policy = self.policy

        # 这里需要实现双方对战的数据收集逻辑
        # 简化示例：收集主策略的数据
        self.collector_env.launch()
        self.policy.reset()

        # 收集 n_sample 个样本
        # 实际实现需要处理多智能体的对战逻辑
        pass

    def _update_opponent_pool(self):
        """更新对手池"""
        policy_id = f"policy_iter_{self.train_iter}"
        policy_state = self.policy.state_dict()

        # 添加当前策略到对手池
        current_elo = self.opponent_pool.elo_ratings.get('current', 1500)
        self.opponent_pool.add_policy(policy_state, policy_id, current_elo)

    def _evaluate(self):
        """评估当前策略"""
        print(f"\n=== Evaluation at iteration {self.train_iter} ===")

        # 对战对手池中的所有策略
        for opponent in self.opponent_pool.pool:
            win_rate = self._evaluate_vs_opponent(opponent)
            print(f"vs {opponent['id']}: win rate = {win_rate:.2%}")

            # 更新 Elo 评分
            self.opponent_pool.update_elo(
                'current', opponent['id'], win_rate
            )

        print(f"Current Elo: {self.opponent_pool.elo_ratings.get('current', 1500):.0f}")
        print("=" * 50)

    def _evaluate_vs_opponent(self, opponent: dict) -> float:
        """评估对战特定对手的胜率"""
        # 这里需要实现评估逻辑
        # 返回胜率 (0-1)
        return 0.5  # 占位符

    def _save_checkpoint(self, iteration: int):
        """保存检查点"""
        save_path = f'./checkpoints/magent2_selfplay_iter_{iteration}.pth'
        torch.save({
            'iteration': iteration,
            'policy_state_dict': self.policy.state_dict(),
            'opponent_pool': self.opponent_pool.pool,
            'elo_ratings': self.opponent_pool.elo_ratings,
        }, save_path)
        print(f"Checkpoint saved to {save_path}")


magent2_selfplay_config = EasyDict(magent2_selfplay_config)


# train_selfplay_magent2.py (修正后的 main 函数)

def main():
    """主函数"""

    # 修正: 使用 dict() 构造函数将 EasyDict 转换为 dict
    # 这样做可以避免 'to_dict' 错误，并确保 compile_config 接收到纯字典。
    cfg_dict = dict(magent2_selfplay_config)

    # 重新尝试 compile_config
    try:
        cfg = compile_config(
            cfg_dict,
            seed=0,
            auto=True
        )
    except AssertionError:
        # 如果 compile_config 仍然失败 (最可能的原因)，直接使用 EasyDict 对象
        print("Warning: DI-engine compile_config failed, falling back to raw EasyDict config.")
        cfg = magent2_selfplay_config
        # 确保 seed 存在
        if 'seed' not in cfg:
            cfg.seed = 0

    trainer = SelfPlayTrainer(cfg)
    trainer.train(max_iterations=10000)


if __name__ == '__main__':
    main()