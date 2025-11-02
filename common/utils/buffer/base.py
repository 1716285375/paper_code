# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : base.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 23:20
@Update Date    :
@Description    : 经验缓冲区抽象基类
定义了经验回放缓冲区的标准接口
"""
# ------------------------------------------------------------


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# 可选导入torch用于设备管理
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class Buffer(ABC):
    """
    经验缓冲区抽象基类

    定义了经验回放缓冲区的标准接口，用于存储和采样经验数据。
    所有具体的缓冲区实现（如FIFOBuffer、DictBuffer等）都需要继承此类。

    支持可选的设备管理（如果安装了PyTorch），用于将采样数据自动转换到指定设备。
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        device: Optional[Union[str, Any]] = None,
    ):
        """
        初始化缓冲区

        Args:
            capacity: 缓冲区最大容量（可选，由子类决定是否使用）
            device: 计算设备（可选，如果提供了torch设备，采样时会自动转换数据）
                   可以是 "cpu", "cuda", torch.device对象等
        """
        self.capacity = capacity
        self._size = 0

        # 设备设置（可选）
        if TORCH_AVAILABLE and device is not None:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        else:
            self.device = device  # 可能是None或其他类型

    @abstractmethod
    def add(self, **transition: Any) -> None:
        """
        添加一条经验数据

        Args:
            **transition: 经验数据的键值对，通常包括obs, action, reward, next_obs, done等
        """
        pass

    def add_batch(self, transitions: Sequence[Dict[str, Any]]) -> None:
        """
        批量添加经验数据

        Args:
            transitions: 经验数据字典列表
        """
        for t in transitions:
            self.add(**t)

    @abstractmethod
    def sample(self, batch_size: int, replace: bool = False) -> Dict[str, np.ndarray]:
        """
        采样一批经验数据

        Args:
            batch_size: 批次大小
            replace: 是否允许重复采样（有放回采样）

        Returns:
            采样得到的数据字典，所有值都是numpy数组
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        清空缓冲区
        """
        pass

    @abstractmethod
    def as_dict(self) -> Dict[str, List[Any]]:
        """
        将缓冲区内容转换为字典形式

        Returns:
            包含所有数据的字典，每个键对应一个列表
        """
        pass

    @abstractmethod
    def get_recent(self, n: int) -> Dict[str, List[Any]]:
        """
        获取最近n条经验数据

        Args:
            n: 要获取的数据条数

        Returns:
            最近n条数据的字典
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        获取缓冲区中的数据条数

        Returns:
            数据条数
        """
        pass

    @property
    def size(self) -> int:
        """
        当前缓冲区中存储的经验数量

        Returns:
            int: 缓冲区当前大小
        """
        return len(self)

    @property
    def is_full(self) -> bool:
        """
        检查缓冲区是否已满

        Returns:
            bool: 如果缓冲区已满返回True，否则返回False
        """
        if self.capacity is None:
            return False
        return len(self) >= self.capacity

    @property
    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空

        Returns:
            bool: 如果缓冲区为空返回True，否则返回False
        """
        return len(self) == 0

    def can_sample(self, batch_size: int, replace: bool = False) -> bool:
        """
        检查是否可以采样指定数量的经验

        Args:
            batch_size: 需要的批次大小
            replace: 是否允许重复采样（有放回采样）

        Returns:
            bool: 如果可以采样返回True
        """
        if self.is_empty:
            return False
        # 有放回采样：只要缓冲区不为空就可以采样任意大小
        if replace:
            return True
        # 无放回采样：需要缓冲区大小至少等于batch_size
        return len(self) >= batch_size

    def __repr__(self) -> str:
        """返回缓冲区的字符串表示"""
        return f"{self.__class__.__name__}(capacity={self.capacity}, size={len(self)})"
