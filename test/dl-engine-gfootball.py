import gfootball.env as football_env
from ding.config import compile_config
from ding.policy import PPOPolicy
from ding.worker import SampleSerialCollector, InteractionSerialEvaluator, BaseLearner
from ding.envs import BaseEnvManager, DingEnvWrapper
from easydict import EasyDict

# 创建环境函数
def create_env():
    env = football_env.create_environment(
        env_name='academy_3_vs_1_with_keeper',  # 简单场景
        representation='simple115v2',
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=1,  # 启用自博弈
        render=False,
    )
    return DingEnvWrapper(env)

# 配置
config = EasyDict(dict(
    env=dict(
        collector_env_num=4,
        evaluator_env_num=2,
        n_evaluator_episode=5,
        stop_value=100,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=115,
            action_shape=19,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
        ),
        collect=dict(n_sample=256),
    ),
))

# 编译配置
config = compile_config(config, auto=True)

# 创建环境管理器
collector_env = BaseEnvManager([create_env for _ in range(config.env.collector_env_num)])
evaluator_env = BaseEnvManager([create_env for _ in range(config.env.evaluator_env_num)])

# 训练循环
# ... (添加 policy, collector, learner 的初始化和训练循环)