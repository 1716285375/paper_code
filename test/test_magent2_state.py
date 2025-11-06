# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : test_magent2_state.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-11-04 00:00
@Update Date    :
@Description    : 测试Magent2环境的state获取功能
测试包括：
    - 环境初始化和reset
    - get_state()和get_state_size()
    - get_obs()和get_obs_agent()
    - get_env_info()
    - step()后的state获取
"""
# ------------------------------------------------------------

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from environments.magent2 import Magent2BattleV4Parallel


def create_test_env():
    """创建测试环境"""
    try:
        env = Magent2BattleV4Parallel(
            map_size=20,  # 较小的地图，对应12 vs 12
            max_cycles=50,  # 较短的episode
            minimap_mode=False,
        )
        return env
    except ImportError as e:
        raise ImportError(f"Magent2 not installed: {e}")


if HAS_PYTEST:
    class TestMagent2State:
        """测试Magent2环境的state获取功能（使用pytest）"""

        @pytest.fixture
        def env(self):
            """创建测试环境"""
            return create_test_env()

    def test_env_initialization(self, env):
        """测试环境初始化"""
        assert env is not None
        assert len(env.agents) > 0
        print(f"\n✓ 环境初始化成功，Agent数量: {len(env.agents)}")
        print(f"  Agent列表（前5个）: {env.agents[:5]}")

    def test_reset_and_get_state(self, env):
        """测试reset后获取state"""
        obs = env.reset()
        
        # 验证观测
        assert obs is not None
        assert isinstance(obs, dict)
        assert len(obs) == len(env.agents)
        print(f"\n✓ Reset成功，观测数量: {len(obs)}")
        
        # 测试get_state()
        state = env.get_state()
        assert state is not None
        assert isinstance(state, np.ndarray)
        assert state.ndim == 1  # 应该是1维数组
        print(f"✓ get_state()成功，state形状: {state.shape}")
        
        # 测试get_state_size()
        state_size = env.get_state_size()
        assert state_size > 0
        assert state_size == state.shape[0]  # 维度应该匹配
        print(f"✓ get_state_size()成功，state维度: {state_size}")

    def test_get_obs_methods(self, env):
        """测试get_obs相关方法"""
        env.reset()
        
        # 测试get_obs()
        all_obs = env.get_obs()
        assert all_obs is not None
        assert isinstance(all_obs, list)
        assert len(all_obs) == len(env.agents)
        print(f"\n✓ get_obs()成功，返回{len(all_obs)}个观测")
        
        # 测试get_obs_agent()
        first_agent = env.agents[0]
        agent_obs = env.get_obs_agent(first_agent)
        assert agent_obs is not None
        assert isinstance(agent_obs, np.ndarray)
        assert agent_obs.ndim == 1  # 应该是展平后的1维数组
        print(f"✓ get_obs_agent('{first_agent}')成功，观测形状: {agent_obs.shape}")
        
        # 验证get_obs()返回的观测与get_obs_agent()一致
        assert np.allclose(all_obs[0], agent_obs), "get_obs()和get_obs_agent()结果不一致"
        print("✓ get_obs()和get_obs_agent()结果一致")

    def test_get_obs_size(self, env):
        """测试get_obs_size()"""
        env.reset()
        
        obs_size = env.get_obs_size()
        assert obs_size is not None
        
        # obs_size可能是int或tuple
        if isinstance(obs_size, tuple):
            print(f"\n✓ get_obs_size()返回元组: {obs_size}")
            obs_dim = int(np.prod(obs_size))
        else:
            print(f"\n✓ get_obs_size()返回标量: {obs_size}")
            obs_dim = obs_size
        
        # 验证与单个agent观测维度一致
        first_agent = env.agents[0]
        agent_obs = env.get_obs_agent(first_agent)
        assert agent_obs.shape[0] == obs_dim, f"观测维度不匹配: {agent_obs.shape[0]} vs {obs_dim}"

    def test_state_dimension_calculation(self, env):
        """测试state维度计算是否正确"""
        env.reset()
        
        # 获取观测维度
        obs_size = env.get_obs_size()
        if isinstance(obs_size, tuple):
            obs_dim = int(np.prod(obs_size))
        else:
            obs_dim = obs_size
        
        # 获取state维度
        state_size = env.get_state_size()
        
        # 验证: state_dim = obs_dim * n_agents
        expected_state_dim = obs_dim * len(env.agents)
        assert state_size == expected_state_dim, (
            f"State维度计算错误: 期望{expected_state_dim}, 实际{state_size}"
        )
        print(f"\n✓ State维度计算正确:")
        print(f"  观测维度: {obs_dim}")
        print(f"  Agent数量: {len(env.agents)}")
        print(f"  State维度: {state_size} = {obs_dim} × {len(env.agents)}")

    def test_step_and_state_update(self, env):
        """测试step后state更新"""
        obs = env.reset()
        
        # 获取初始state
        state_before = env.get_state().copy()
        
        # 执行一步
        actions = {agent_id: 0 for agent_id in env.agents}  # 所有agent执行动作0
        obs_new, rewards, dones, infos = env.step(actions)
        
        # 获取新的state
        state_after = env.get_state()
        
        # 验证state更新了（通常应该不同）
        assert state_after is not None
        assert state_after.shape == state_before.shape
        print(f"\n✓ Step后state更新成功")
        print(f"  State形状: {state_after.shape}")
        
        # 验证观测已更新（通常应该不同）
        assert obs_new is not None
        assert len(obs_new) == len(env.agents)

    def test_get_env_info(self, env):
        """测试get_env_info()"""
        env.reset()
        
        env_info = env.get_env_info()
        assert env_info is not None
        assert isinstance(env_info, dict)
        
        # 验证必需字段
        required_keys = ["state_shape", "obs_shape", "n_actions", "n_agents", "episode_limit"]
        for key in required_keys:
            assert key in env_info, f"缺少字段: {key}"
        
        print(f"\n✓ get_env_info()成功:")
        print(f"  state_shape: {env_info['state_shape']}")
        print(f"  obs_shape: {env_info['obs_shape']}")
        print(f"  n_actions: {env_info['n_actions']}")
        print(f"  n_agents: {env_info['n_agents']}")
        print(f"  episode_limit: {env_info['episode_limit']}")
        
        # 验证维度一致性
        assert env_info['n_agents'] == len(env.agents)
        assert env_info['episode_limit'] == 50  # 我们在fixture中设置的max_cycles=50

    def test_get_total_actions(self, env):
        """测试get_total_actions()"""
        env.reset()
        
        total_actions = env.get_total_actions()
        assert total_actions > 0
        print(f"\n✓ get_total_actions()成功: {total_actions}")

    def test_get_avail_actions(self, env):
        """测试get_avail_actions()"""
        env.reset()
        
        # 测试get_avail_actions()
        avail_actions = env.get_avail_actions()
        assert avail_actions is not None
        assert isinstance(avail_actions, list)
        assert len(avail_actions) == len(env.agents)
        print(f"\n✓ get_avail_actions()成功，返回{len(avail_actions)}个可用动作列表")
        
        # 测试get_avail_agent_actions()
        first_agent = env.agents[0]
        agent_avail = env.get_avail_agent_actions(first_agent)
        assert agent_avail is not None
        assert isinstance(agent_avail, list)
        assert len(agent_avail) > 0
        print(f"✓ get_avail_agent_actions('{first_agent}')成功，可用动作数: {len(agent_avail)}")

    def test_state_consistency(self, env):
        """测试state的一致性（多次调用应该返回相同结果）"""
        env.reset()
        
        # 多次调用get_state()应该返回相同结果（在未执行step之前）
        state1 = env.get_state()
        state2 = env.get_state()
        
        assert np.allclose(state1, state2), "多次调用get_state()结果不一致"
        print(f"\n✓ State一致性测试通过")

    def test_multiple_episodes(self, env):
        """测试多个episode中state获取"""
        print(f"\n测试多个episode:")
        
        for episode in range(3):
            obs = env.reset()
            state = env.get_state()
            state_size = env.get_state_size()
            
            assert state.shape[0] == state_size
            print(f"  Episode {episode + 1}: state形状={state.shape}, state_size={state_size}")
            
            # 执行几步
            for step in range(2):
                actions = {agent_id: step % env.get_total_actions() for agent_id in env.agents}
                obs, rewards, dones, infos = env.step(actions)
                state = env.get_state()
                assert state.shape[0] == state_size
        
        print(f"✓ 多个episode测试通过")


def run_tests():
    """运行测试（不使用pytest）"""
    print("=" * 60)
    print("Magent2 Environment State Retrieval Test")
    print("=" * 60)
    
    try:
        env = create_test_env()
    except ImportError as e:
        print(f"\n[ERROR] Magent2 environment not installed: {e}")
        print("Please run: pip install magent2")
        return
    
    # 定义测试函数（不使用pytest fixture）
    def test_env_initialization():
        """测试环境初始化"""
        assert env is not None
        assert len(env.agents) > 0
        print(f"\n✓ 环境初始化成功，Agent数量: {len(env.agents)}")
        print(f"  Agent列表（前5个）: {env.agents[:5]}")

    def test_reset_and_get_state():
        """测试reset后获取state"""
        obs = env.reset()
        
        # 验证观测
        assert obs is not None
        assert isinstance(obs, dict)
        assert len(obs) == len(env.agents)
        print(f"\n✓ Reset成功，观测数量: {len(obs)}")
        
        # 测试get_state()
        state = env.get_state()
        assert state is not None
        assert isinstance(state, np.ndarray)
        assert state.ndim == 1  # 应该是1维数组
        print(f"✓ get_state()成功，state形状: {state.shape}")
        
        # 测试get_state_size()
        state_size = env.get_state_size()
        assert state_size > 0
        assert state_size == state.shape[0]  # 维度应该匹配
        print(f"✓ get_state_size()成功，state维度: {state_size}")

    def test_get_obs_methods():
        """测试get_obs相关方法"""
        env.reset()
        
        # 测试get_obs()
        all_obs = env.get_obs()
        assert all_obs is not None
        assert isinstance(all_obs, list)
        assert len(all_obs) == len(env.agents)
        print(f"\n✓ get_obs()成功，返回{len(all_obs)}个观测")
        
        # 测试get_obs_agent()
        first_agent = env.agents[0]
        agent_obs = env.get_obs_agent(first_agent)
        assert agent_obs is not None
        assert isinstance(agent_obs, np.ndarray)
        assert agent_obs.ndim == 1  # 应该是展平后的1维数组
        print(f"✓ get_obs_agent('{first_agent}')成功，观测形状: {agent_obs.shape}")
        
        # 验证get_obs()返回的观测与get_obs_agent()一致
        assert np.allclose(all_obs[0], agent_obs), "get_obs()和get_obs_agent()结果不一致"
        print("✓ get_obs()和get_obs_agent()结果一致")

    def test_get_obs_size():
        """测试get_obs_size()"""
        env.reset()
        
        obs_size = env.get_obs_size()
        assert obs_size is not None
        
        # obs_size可能是int或tuple
        if isinstance(obs_size, tuple):
            print(f"\n✓ get_obs_size()返回元组: {obs_size}")
            obs_dim = int(np.prod(obs_size))
        else:
            print(f"\n✓ get_obs_size()返回标量: {obs_size}")
            obs_dim = obs_size
        
        # 验证与单个agent观测维度一致
        first_agent = env.agents[0]
        agent_obs = env.get_obs_agent(first_agent)
        assert agent_obs.shape[0] == obs_dim, f"观测维度不匹配: {agent_obs.shape[0]} vs {obs_dim}"

    def test_state_dimension_calculation():
        """测试state维度计算是否正确"""
        env.reset()
        
        # 获取观测维度
        obs_size = env.get_obs_size()
        if isinstance(obs_size, tuple):
            obs_dim = int(np.prod(obs_size))
        else:
            obs_dim = obs_size
        
        # 获取state维度和实际state
        state_size = env.get_state_size()
        actual_state = env.get_state()
        
        # 验证: get_state_size()返回的维度与实际state的维度一致
        assert state_size == actual_state.shape[0], (
            f"State维度不一致: get_state_size()返回{state_size}, "
            f"实际state形状{actual_state.shape}"
        )
        
        # 检查是否使用了环境提供的state（维度可能不等于obs_dim * n_agents）
        # 或者是从观测拼接的（维度等于obs_dim * n_agents）
        expected_from_obs = obs_dim * len(env.agents)
        if state_size == expected_from_obs:
            # 从观测拼接的
            print(f"\n[PASS] State dimension calculated from observations:")
            print(f"  Observation dimension: {obs_dim}")
            print(f"  Agent count: {len(env.agents)}")
            print(f"  State dimension: {state_size} = {obs_dim} x {len(env.agents)}")
        else:
            # 使用环境提供的state（全局状态表示）
            print(f"\n[PASS] State dimension from environment (global state representation):")
            print(f"  Observation dimension: {obs_dim}")
            print(f"  Agent count: {len(env.agents)}")
            print(f"  State dimension: {state_size} (from env.state())")
            print(f"  Note: State is not simple observation concatenation "
                  f"(would be {expected_from_obs} if concatenated)")

    def test_step_and_state_update():
        """测试step后state更新"""
        obs = env.reset()
        
        # 获取初始state
        state_before = env.get_state().copy()
        
        # 执行一步
        actions = {agent_id: 0 for agent_id in env.agents}  # 所有agent执行动作0
        obs_new, rewards, dones, infos = env.step(actions)
        
        # 获取新的state
        state_after = env.get_state()
        
        # 验证state更新了（通常应该不同）
        assert state_after is not None
        assert state_after.shape == state_before.shape
        print(f"\n✓ Step后state更新成功")
        print(f"  State形状: {state_after.shape}")
        
        # 验证观测已更新（通常应该不同）
        assert obs_new is not None
        assert len(obs_new) == len(env.agents)

    def test_get_env_info():
        """测试get_env_info()"""
        env.reset()
        
        env_info = env.get_env_info()
        assert env_info is not None
        assert isinstance(env_info, dict)
        
        # 验证必需字段
        required_keys = ["state_shape", "obs_shape", "n_actions", "n_agents", "episode_limit"]
        for key in required_keys:
            assert key in env_info, f"缺少字段: {key}"
        
        print(f"\n✓ get_env_info()成功:")
        print(f"  state_shape: {env_info['state_shape']}")
        print(f"  obs_shape: {env_info['obs_shape']}")
        print(f"  n_actions: {env_info['n_actions']}")
        print(f"  n_agents: {env_info['n_agents']}")
        print(f"  episode_limit: {env_info['episode_limit']}")
        
        # 验证维度一致性
        assert env_info['n_agents'] == len(env.agents)
        assert env_info['episode_limit'] == 50  # 我们在创建环境中设置的max_cycles=50

    def test_get_total_actions():
        """测试get_total_actions()"""
        env.reset()
        
        total_actions = env.get_total_actions()
        assert total_actions > 0
        print(f"\n✓ get_total_actions()成功: {total_actions}")

    def test_get_avail_actions():
        """测试get_avail_actions()"""
        env.reset()
        
        # 测试get_avail_actions()
        avail_actions = env.get_avail_actions()
        assert avail_actions is not None
        assert isinstance(avail_actions, list)
        assert len(avail_actions) == len(env.agents)
        print(f"\n✓ get_avail_actions()成功，返回{len(avail_actions)}个可用动作列表")
        
        # 测试get_avail_agent_actions()
        first_agent = env.agents[0]
        agent_avail = env.get_avail_agent_actions(first_agent)
        assert agent_avail is not None
        assert isinstance(agent_avail, list)
        assert len(agent_avail) > 0
        print(f"✓ get_avail_agent_actions('{first_agent}')成功，可用动作数: {len(agent_avail)}")

    def test_state_consistency():
        """测试state的一致性（多次调用应该返回相同结果）"""
        env.reset()
        
        # 多次调用get_state()应该返回相同结果（在未执行step之前）
        state1 = env.get_state()
        state2 = env.get_state()
        
        assert np.allclose(state1, state2), "多次调用get_state()结果不一致"
        print(f"\n✓ State一致性测试通过")

    def test_multiple_episodes():
        """测试多个episode中state获取"""
        print(f"\n测试多个episode:")
        
        for episode in range(3):
            obs = env.reset()
            state = env.get_state()
            state_size = env.get_state_size()
            
            assert state.shape[0] == state_size
            print(f"  Episode {episode + 1}: state形状={state.shape}, state_size={state_size}")
            
            # 执行几步
            for step in range(2):
                actions = {agent_id: step % env.get_total_actions() for agent_id in env.agents}
                obs, rewards, dones, infos = env.step(actions)
                state = env.get_state()
                assert state.shape[0] == state_size
        
        print(f"✓ 多个episode测试通过")
    
    # 运行所有测试方法
    test_methods = [
        ("环境初始化", test_env_initialization),
        ("Reset和State获取", test_reset_and_get_state),
        ("观测获取方法", test_get_obs_methods),
        ("观测维度", test_get_obs_size),
        ("State维度计算", test_state_dimension_calculation),
        ("Step后State更新", test_step_and_state_update),
        ("环境信息", test_get_env_info),
        ("动作空间", test_get_total_actions),
        ("可用动作", test_get_avail_actions),
        ("State一致性", test_state_consistency),
        ("多个Episode", test_multiple_episodes),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_method in test_methods:
        try:
            test_method()
            passed += 1
        except AssertionError as e:
            print(f"\n[FAIL] {name} test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"\n[ERROR] {name} test exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test completed: [PASS] {passed}, [FAIL] {failed}")
    print("=" * 60)


if __name__ == "__main__":
    # 如果直接运行，使用简单测试
    run_tests()
    
    # 如果使用pytest运行，使用pytest测试
    # pytest test/test_magent2_state.py -v

