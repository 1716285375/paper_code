#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HAPPO自博弈训练测试脚本

在Magent2环境中测试HAPPO自博弈算法。

使用方法:
    python examples/test_happo_selfplay.py --num_updates 100
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from examples.test_selfplay_algorithms import test_algorithm

if __name__ == "__main__":
    import torch
    
    # HAPPO配置（包含顺序更新参数）
    config = {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
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
            # HAPPO特定参数
            "update_order": "random",
            "use_marginal_advantage": True,
        },
        "tracking": {
            "enabled": True,
            "tensorboard": {
                "enabled": True,
                "project": "happo-selfplay-test",
            },
        },
        "data_saving": {
            "enabled": True,
            "format": "json",
        },
    }
    
    import argparse
    parser = argparse.ArgumentParser(description="HAPPO自博弈训练测试")
    parser.add_argument("--num_updates", type=int, default=100, help="训练更新次数")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    
    if args.config:
        from common.config import load_config
        file_config = load_config(args.config, as_dict=True, project_root=project_root)
        config.update(file_config)
    
    success = test_algorithm("happo", config, args.num_updates)
    sys.exit(0 if success else 1)

