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

from typing import Any, Dict, Optional, Tuple

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
        self.agents = list(getattr(self._env, "agents", []))
        self._env_name = env_name

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
        self.agents = list(getattr(self._env, "agents", self.agents))

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
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        # 合并terminations和truncations为dones
        dones = {
            aid: bool(terminations.get(aid, False) or truncations.get(aid, False))
            for aid in self.agents
        }

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


# ==================== 具体环境包装器 ====================


class Magent2BattleV4Parallel(Magent2ParallelBase):
    """
    MAgent2 battle_v4 并行环境包装器

    将MAgent2的battle_v4环境包装为统一的环境接口，支持多Agent并行交互。
    battle_v4是一个大规模多Agent战斗环境，两个团队相互对抗。

    配置参数（**kwargs）：
        - map_size: 地图大小（如45表示45x45的地图）
        - minimap_mode: 小地图模式，如果为True则使用小地图观测
        - step_reward: 每一步的基础奖励
        - dead_penalty: 死亡惩罚（负数）
        - attack_penalty: 攻击惩罚（负数）
        - attack_opponent_reward: 攻击敌人的奖励（正数）
        - max_cycles: 每个episode的最大步数
        - extra_features: 额外特征（可选）
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
