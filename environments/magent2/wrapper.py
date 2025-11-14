# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : wrapper.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 21:15
@Update Date    :
@Description    : MAgent2环境包装器
提供MAgent2各种环境的统一包装接口，支持多Agent并行交互
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from environments.base import AgentParrelEnv

# 尝试导入所有MAgent2环境
try:
    from magent2.environments import battle_v4
except (ModuleNotFoundError, ImportError):
    battle_v4 = None

try:
    from magent2.environments import adversarial_pursuit_v4
except (ModuleNotFoundError, ImportError):
    adversarial_pursuit_v4 = None

try:
    from magent2.environments import battlefield_v4
except (ModuleNotFoundError, ImportError):
    battlefield_v4 = None

try:
    from magent2.environments import combined_arms_v6
except (ModuleNotFoundError, ImportError):
    combined_arms_v6 = None

try:
    from magent2.environments import gather_v4
except (ModuleNotFoundError, ImportError):
    gather_v4 = None

try:
    from magent2.environments import tiger_deer_v3
except (ModuleNotFoundError, ImportError):
    tiger_deer_v3 = None


class Magent2ParallelBase(AgentParrelEnv):
    """
    MAgent2并行环境包装器基类

    提供通用的环境接口实现，所有MAgent2环境包装器都应该继承此类。
    
    公共接口：
        - agents: 获取当前活跃的Agent ID列表（property）
        - n_agents: 获取Agent数量（property，继承自基类，等于 len(agents)）
        - get_env_info(): 获取环境信息字典，包含 n_agents 字段
    """

    def __init__(self, env_module: Any, env_name: str, **kwargs) -> None:
        """
        初始化MAgent2环境包装器

        Args:
            env_module: MAgent2环境模块（如battle_v4, adversarial_pursuit_v4等）
            env_name: 环境名称（用于错误提示）
            **kwargs: 传递给MAgent2环境的配置参数
        """
        if env_module is None:
            raise ImportError(
                f"magent2环境模块 {env_name} 未安装或导入失败，请运行: pip install magent2"
            )

        # 创建MAgent2并行环境
        self._env = env_module.parallel_env(**kwargs)
        self._env_name = env_name
        
        # 存储agents列表（使用私有属性，因为基类中agents是property）
        self._agents = list(getattr(self._env, "agents", []))
        
        # 缓存当前观测（用于get_state和get_obs）
        self._current_observations: Optional[Dict[str, np.ndarray]] = None
        
        # 缓存episode_limit（从max_cycles获取）
        self._episode_limit = kwargs.get("max_cycles", None)

    def reset(self) -> Dict[str, Any]:
        """
        重置环境，开始新的episode

        Returns:
            所有Agent的初始观测值字典，格式为 {agent_id: observation}
        """
        result = self._env.reset()

        # 处理不同版本的返回值格式
        if isinstance(result, tuple) and len(result) == 2:
            observation, _infos = result
        else:
            observation = result

        # 更新活跃Agent列表
        self._agents = list(getattr(self._env, "agents", self._agents))
        
        # 缓存当前观测
        self._current_observations = observation

        return observation

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[Any, Any], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        """
        执行一步环境交互（所有Agent同时执行动作）

        Args:
            actions: Agent动作字典，格式为 {agent_id: action}

        Returns:
            (observations, rewards, dones, infos) 元组，均为字典格式：
                - observations: 下一步的观测值 {agent_id: observation}
                - rewards: 当前步骤的奖励 {agent_id: reward}
                - dones: 结束标志 {agent_id: done}
                - infos: 额外信息 {agent_id: info}
        """
        # 验证和清理动作字典
        # 只包含当前活跃的agent
        active_agents = list(getattr(self._env, "agents", self._agents))
        filtered_actions = {}
        
        for agent_id, action in actions.items():
            # 只包含活跃的agent
            if agent_id not in active_agents:
                continue
            
            # 确保动作是整数类型
            if not isinstance(action, int):
                try:
                    action = int(action)
                except (ValueError, TypeError):
                    raise ValueError(f"Agent {agent_id} 的动作值无法转换为整数: {action}")
            
            # 验证动作值在有效范围内（0-20 for MAgent2 battle_v4）
            # 获取动作空间大小
            action_space = getattr(self._env, "action_space", None)
            if action_space is not None:
                if isinstance(action_space, dict) and agent_id in action_space:
                    max_action = action_space[agent_id].n - 1 if hasattr(action_space[agent_id], 'n') else 20
                elif hasattr(action_space, 'n'):
                    max_action = action_space.n - 1
                else:
                    max_action = 20  # 默认值
            else:
                max_action = 20  # 默认MAgent2 battle_v4的动作空间是21（0-20）
            
            # 裁剪动作到有效范围
            action = max(0, min(action, max_action))
            filtered_actions[agent_id] = action
        
        # 如果没有有效动作，使用默认动作（do_nothing = 0）
        if not filtered_actions:
            # 为所有活跃agent提供默认动作
            filtered_actions = {aid: 0 for aid in active_agents}
        
        try:
            observations, rewards, terminations, truncations, infos = self._env.step(filtered_actions)
        except Exception as e:
            # 添加详细的错误信息
            raise RuntimeError(
                f"MAgent2环境步进失败。\n"
                f"活跃agents: {active_agents}\n"
                f"输入actions keys: {list(actions.keys())}\n"
                f"过滤后actions keys: {list(filtered_actions.keys())}\n"
                f"过滤后actions values: {list(filtered_actions.values())}\n"
                f"原始错误: {str(e)}"
            ) from e

        # 合并terminations和truncations为dones
        dones = {
            aid: bool(terminations.get(aid, False) or truncations.get(aid, False))
            for aid in self.agents
        }
        
        # 缓存当前观测
        self._current_observations = observations

        return observations, rewards, dones, infos

    def close(self) -> None:
        """关闭环境，释放资源"""
        if hasattr(self._env, "close"):
            self._env.close()

    def render(self, mode: str = "human") -> Any:
        """
        渲染环境（可视化）

        Args:
            mode: 渲染模式
                - "human": 显示窗口（默认）
                - "rgb_array": 返回RGB数组，形状为(H, W, 3)，数据类型为uint8

        Returns:
            如果mode="rgb_array"，返回RGB数组（numpy数组）；否则返回None
        """
        if hasattr(self._env, "render"):
            # MAgent2环境支持rgb_array模式
            if mode == "rgb_array":
                try:
                    # 尝试使用rgb_array模式
                    return self._env.render(mode=mode)
                except (TypeError, ValueError):
                    # 如果不支持mode参数，尝试调用无参数的render
                    # 注意：这可能需要根据具体的MAgent2版本调整
                    return None
            else:
                self._env.render()
        return None

    def observation_space(self, agent_id: str) -> Any:
        """
        获取指定Agent的观测空间

        Args:
            agent_id: Agent的标识符

        Returns:
            该Agent的观测空间描述（gym.Space对象）
        """
        return self._env.observation_space(agent_id)

    def action_space(self, agent_id: str) -> Any:
        """
        获取指定Agent的动作空间

        Args:
            agent_id: Agent的标识符

        Returns:
            该Agent的动作空间描述（gym.Space对象）
        """
        return self._env.action_space(agent_id)

    @property
    def agents(self) -> List[str]:
        """
        获取当前活跃的Agent列表

        Returns:
            Agent ID列表
        
        Note:
            可以通过 `n_agents` property 获取Agent数量（继承自基类）：
            >>> env = Magent2BattleV4Parallel(map_size=20)
            >>> agent_list = env.agents  # 获取Agent ID列表
            >>> agent_count = env.n_agents  # 获取Agent数量（等于 len(agent_list)）
        """
        return self._agents

    @property
    def episode_limit(self) -> Optional[int]:
        """
        获取Episode最大步数限制

        Returns:
            Episode最大步数，如果无限制则返回None
        """
        return self._episode_limit

    def get_obs(self) -> List[np.ndarray]:
        """
        获取所有Agent的观测值列表

        Returns:
            所有Agent的观测值列表，按agents顺序排列
        """
        if self._current_observations is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        obs_list = []
        for agent_id in self.agents:
            if agent_id in self._current_observations:
                obs = np.asarray(self._current_observations[agent_id])
                # 展平多维观测
                if obs.ndim > 1:
                    obs = obs.flatten()
                obs_list.append(obs)
        
        return obs_list

    def get_obs_agent(self, agent_id: str) -> np.ndarray:
        """
        获取指定Agent的观测值

        Args:
            agent_id: Agent标识符

        Returns:
            该Agent的观测值，形状为 (obs_dim,)
        """
        if self._current_observations is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        if agent_id not in self._current_observations:
            raise ValueError(f"Agent {agent_id} not found in current observations")
        
        obs = np.asarray(self._current_observations[agent_id])
        # 展平多维观测
        if obs.ndim > 1:
            obs = obs.flatten()
        
        return obs

    def get_obs_size(self) -> Union[int, Tuple[int, ...]]:
        """
        获取观测维度大小

        Returns:
            观测空间的维度大小，可能是标量或元组（对于多Agent环境，通常所有Agent的观测维度相同）
        """
        if not self.agents:
            raise ValueError("No agents available")
        obs_space = self.observation_space(self.agents[0])
        if hasattr(obs_space, "shape"):
            shape = obs_space.shape
            if len(shape) > 1:
                return tuple(shape)
            else:
                return int(shape[0]) if len(shape) == 1 else int(np.prod(shape))
        elif hasattr(obs_space, "n"):
            return obs_space.n
        else:
            raise NotImplementedError(f"Cannot determine obs size from {obs_space}")

    def get_state(self) -> np.ndarray:
        """
        获取全局状态（所有Agent的联合状态）

        Magent2环境提供了state()方法，返回全局地图状态的压缩表示。
        
        State Space结构（根据map_size）：
        - 不使用extra_features: map_size × map_size × 5 channels
          - channels: [obstacle_map, team_0_presence, team_0_hp, team_1_presence, team_1_hp]
        - 使用extra_features: map_size × map_size × 37 channels
          - 额外channels: [binary_agent_id(10), one_hot_action(21), last_reward(1)]
        
        示例：
        - map_size=20: state_dim = 20 × 20 × 5 = 2000
        - map_size=45: state_dim = 45 × 45 × 5 = 10125
        
        如果环境没有state()方法，则通过拼接所有Agent的观测来构建全局状态。
        如果环境有state()方法，优先使用环境提供的state。

        Returns:
            全局状态，形状为 (state_dim,)
        """
        # 首先尝试从环境获取state（如果支持）
        if hasattr(self._env, "state") and callable(getattr(self._env, "state")):
            try:
                state = self._env.state()
                if state is not None:
                    state_array = np.asarray(state)
                    if state_array.ndim > 1:
                        state_array = state_array.flatten()
                    return state_array
            except (AttributeError, NotImplementedError, TypeError):
                pass
        
        # 如果没有state方法，从观测拼接构建
        if self._current_observations is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # 获取所有Agent的观测并展平
        obs_list = []
        for agent_id in sorted(self.agents):
            if agent_id in self._current_observations:
                obs = np.asarray(self._current_observations[agent_id])
                # 展平多维观测
                if obs.ndim > 1:
                    obs = obs.flatten()
                obs_list.append(obs)
        
        if not obs_list:
            raise ValueError("No observations available")
        
        # 拼接所有观测
        state = np.concatenate(obs_list, axis=0)
        
        return state

    def get_state_size(self) -> int:
        """
        获取全局状态维度大小

        优先使用环境提供的state维度，如果环境没有提供，则从观测拼接计算。

        Returns:
            全局状态的维度大小（标量）
        """
        # 首先尝试从环境获取state（如果支持），使用实际state的维度
        if hasattr(self._env, "state") and callable(getattr(self._env, "state")):
            try:
                state = self._env.state()
                if state is not None:
                    state_array = np.asarray(state)
                    if state_array.ndim > 1:
                        return int(np.prod(state_array.shape))
                    else:
                        return int(state_array.shape[0])
            except (AttributeError, NotImplementedError, TypeError):
                pass
        
        # 尝试从环境获取state space
        try:
            if hasattr(self._env, "state_space"):
                state_space = self._env.state_space
                if hasattr(state_space, "shape"):
                    return int(np.prod(state_space.shape))
                elif hasattr(state_space, "n"):
                    return state_space.n
        except (AttributeError, NotImplementedError):
            pass
        
        # 如果没有state相关方法，通过拼接观测计算
        # 获取单个agent的观测维度
        if not self.agents:
            raise ValueError("No agents available")
        
        obs_space = self.observation_space(self.agents[0])
        obs_dim = 1
        if hasattr(obs_space, "shape"):
            obs_dim = int(np.prod(obs_space.shape))
        elif hasattr(obs_space, "n"):
            obs_dim = obs_space.n
        
        # 全局状态 = 所有agent的观测拼接
        state_dim = obs_dim * len(self.agents)
        
        return state_dim

    def get_total_actions(self) -> int:
        """
        获取动作空间大小（所有Agent的动作空间通常相同）

        Returns:
            动作空间的总动作数（适用于离散动作空间）
        """
        if not self.agents:
            raise ValueError("No agents available")
        action_space = self.action_space(self.agents[0])
        if hasattr(action_space, "n"):
            return action_space.n
        elif hasattr(action_space, "shape"):
            # 连续动作空间
            return int(np.prod(action_space.shape))
        else:
            raise NotImplementedError(f"Cannot determine action size from {action_space}")

    def get_avail_actions(self) -> List[List[int]]:
        """
        获取所有Agent的可用动作列表

        Returns:
            所有Agent的可用动作列表，按agents顺序排列
            每个元素是一个可用动作的列表
        """
        avail_actions = []
        for agent_id in self.agents:
            avail_actions.append(self.get_avail_agent_actions(agent_id))
        return avail_actions

    def get_avail_agent_actions(self, agent_id: str) -> List[int]:
        """
        获取指定Agent的可用动作列表

        Magent2环境通常所有动作都可用，但如果环境提供可用动作掩码，则使用掩码。

        Args:
            agent_id: Agent标识符

        Returns:
            可用动作的列表，默认返回所有动作
        """
        # 尝试从环境获取可用动作掩码
        if hasattr(self._env, "get_avail_actions") and callable(getattr(self._env, "get_avail_actions")):
            try:
                avail_actions = self._env.get_avail_actions(agent_id)
                if avail_actions is not None:
                    # 转换为列表并返回可用动作的索引
                    avail_actions = np.asarray(avail_actions)
                    if avail_actions.dtype == bool:
                        # 布尔掩码，返回True的位置索引
                        return np.where(avail_actions)[0].tolist()
                    else:
                        # 已经是索引列表
                        return avail_actions.tolist() if hasattr(avail_actions, "tolist") else list(avail_actions)
            except (AttributeError, NotImplementedError, TypeError, KeyError):
                pass
        
        # 如果没有可用动作掩码，默认所有动作都可用
        total_actions = self.get_total_actions()
        return list(range(total_actions))

    def get_env_info(self) -> Dict[str, Any]:
        """
        获取环境信息字典

        Returns:
            包含环境信息的字典：
                - state_shape: 状态维度
                - obs_shape: 观测维度
                - n_actions: 动作空间大小
                - n_agents: Agent数量
                - episode_limit: Episode最大步数（如果支持）
        """
        obs_size = self.get_obs_size()
        if isinstance(obs_size, tuple):
            obs_shape = obs_size
        else:
            obs_shape = (obs_size,)

        state_size = self.get_state_size()
        state_shape = (state_size,)

        return {
            "state_shape": state_shape,
            "obs_shape": obs_shape,
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }


# ==================== 具体环境包装器 ====================


class Magent2BattleV4Parallel(Magent2ParallelBase):
    """
    MAgent2 battle_v4 并行环境包装器

    将MAgent2的battle_v4环境包装为统一的环境接口，支持多Agent并行交互。
    battle_v4是一个大规模多Agent战斗环境，两个团队相互对抗。

    State Space说明：
        全局state是一个 map_size × map_size 的地图，包含多个channels：
        - 不使用extra_features: 5个channels
          [obstacle_map, team_0_presence, team_0_hp, team_1_presence, team_1_hp]
        - 使用extra_features: 37个channels（额外32个channels）
        
        State维度计算：
        - map_size=20: 20 × 20 × 5 = 2000
        - map_size=45: 45 × 45 × 5 = 10125
        
        注意：state是环境的全局状态表示，不是所有agent观测的拼接。
        全局state通常比观测拼接更紧凑（2000 vs 20280 for map_size=20）。

    配置参数（**kwargs）：
        - map_size: 地图大小（如20表示20x20的地图，45表示45x45的地图）
        - minimap_mode: 小地图模式，如果为True则使用小地图观测
        - step_reward: 每一步的基础奖励
        - dead_penalty: 死亡惩罚（负数）
        - attack_penalty: 攻击惩罚（负数）
        - attack_opponent_reward: 攻击敌人的奖励（正数）
        - max_cycles: 每个episode的最大步数
        - extra_features: 额外特征（可选），如果为True，state会增加32个channels
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(battle_v4, "battle_v4", **kwargs)


class Magent2AdversarialPursuitV4Parallel(Magent2ParallelBase):
    """
    MAgent2 adversarial_pursuit_v4 并行环境包装器

    对抗追击环境：多个Agent（predators，红色）需要合作追捕逃跑的Agent（prey，蓝色）。
    这是一个典型的合作-竞争混合多Agent环境。

    Agents:
        - predators (red): predator_[0-24]，共25个
        - prey (blue): prey_[0-49]，共50个
        - 总计：75个agents

    Action Space:
        - Predators: Discrete(9)
            - [do_nothing, move_4, tag_8]
            - move_4: 移动到4个最近的网格位置之一
            - tag_8: 标记8个相邻位置之一的prey
        - Prey: Discrete(13)
            - [do_nothing, move_8]
            - move_8: 移动到8个相邻位置之一

    Observation Space:
        - Predators: (10, 10, 5) 或 (10, 10, 9) if extra_features=True
            - Channels: [obstacle/off_map, my_team_presence, my_team_hp, other_team_presence, other_team_hp]
            - Extra (if enabled): [binary_agent_id(10), one_hot_action(9), last_reward(1)]
        - Prey: (9, 9, 5) 或 (9, 9, 9) if extra_features=True
            - Channels: [obstacle/off_map, my_team_presence, my_team_hp, other_team_presence, other_team_hp]
            - Extra (if enabled): [binary_agent_id(10), one_hot_action(13), last_reward(1)]
        - State Space: (45, 45, 5) 或 (45, 45, 9) if extra_features=True
            - Channels: [obstacle_map, prey_presence, prey_hp, predator_presence, predator_hp]
            - Extra (if enabled): [binary_agent_id(10), one_hot_action(13), last_reward(1)]

    Reward:
        - Predators:
            - +1.0: 成功标记一个prey
            - -0.2: 标记任何位置（tag_penalty，可配置）
        - Prey:
            - -1.0: 被predator标记

    配置参数（**kwargs）：
        - map_size (int): 地图大小（正方形），最小值7。增加大小会增加agent数量
        - minimap_mode (bool): 是否启用全局小地图观测。包含你和对手的单位密度binned到2D网格，以及agent绝对位置（缩放到0-1）
        - tag_penalty (float): 当红色agents标记任何位置时的奖励惩罚，默认-0.2
        - max_cycles (int): episode的最大帧数（每个agent的步数），默认500
        - extra_features (bool): 是否在观测中添加额外特征，默认False
            - 额外特征包括：binary_agent_id, one_hot_action, last_reward

    示例:
        >>> env = Magent2AdversarialPursuitV4Parallel(
        ...     map_size=45,
        ...     minimap_mode=False,
        ...     tag_penalty=-0.2,
        ...     max_cycles=500,
        ...     extra_features=False
        ... )
        >>> obs = env.reset()
        >>> # obs是字典: {'predator_0': array(...), 'prey_0': array(...), ...}
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(adversarial_pursuit_v4, "adversarial_pursuit_v4", **kwargs)


class Magent2BattlefieldV4Parallel(Magent2ParallelBase):
    """
    MAgent2 battlefield_v4 并行环境包装器

    战场环境：两个团队在战场上对抗，类似battle_v4但可能有不同的配置和规则。

    配置参数（**kwargs）：
        - map_size: 地图大小
        - minimap_mode: 小地图模式
        - step_reward: 每步奖励
        - max_cycles: 最大步数
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(battlefield_v4, "battlefield_v4", **kwargs)


class Magent2CombinedArmsV6Parallel(Magent2ParallelBase):
    """
    MAgent2 combined_arms_v6 并行环境包装器

    联合兵种环境：包含多种类型的单位（如步兵、坦克、飞机等），需要协调不同兵种进行作战。
    这是一个复杂的多兵种协同环境。

    配置参数（**kwargs）：
        - map_size: 地图大小
        - minimap_mode: 小地图模式
        - step_reward: 每步奖励
        - max_cycles: 最大步数
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(combined_arms_v6, "combined_arms_v6", **kwargs)


class Magent2GatherV4Parallel(Magent2ParallelBase):
    """
    MAgent2 gather_v4 并行环境包装器

    收集环境：Agent需要在地图上收集资源或物品，可能涉及竞争或合作。
    这是一个资源收集和竞争的多Agent环境。

    配置参数（**kwargs）：
        - map_size: 地图大小
        - minimap_mode: 小地图模式
        - step_reward: 每步奖励
        - max_cycles: 最大步数
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(gather_v4, "gather_v4", **kwargs)


class Magent2TigerDeerV3Parallel(Magent2ParallelBase):
    """
    MAgent2 tiger_deer_v3 并行环境包装器

    虎鹿环境：模拟捕食者-被捕食者关系，tiger（捕食者）需要追捕deer（被捕食者）。
    这是一个经典的多Agent生态系统模拟环境。

    配置参数（**kwargs）：
        - map_size: 地图大小
        - minimap_mode: 小地图模式
        - step_reward: 每步奖励
        - max_cycles: 最大步数
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(tiger_deer_v3, "tiger_deer_v3", **kwargs)
