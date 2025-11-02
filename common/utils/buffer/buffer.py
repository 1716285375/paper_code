# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : buffer.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 09:09
@Update Date    : 
@Description    : FIFO缓冲区和字典缓冲区的具体实现
实现了基于deque的高效FIFO缓冲区，以及支持多键的字典缓冲区
"""
# ------------------------------------------------------------


from __future__ import annotations

from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import deque
import numpy as np

from common.utils.buffer.base import Buffer


class FIFOBuffer(Buffer):
    """
    FIFO（先进先出）经验缓冲区
    
    使用collections.deque实现高效的FIFO缓冲区，支持自动容量管理。
    使用字典格式存储经验数据，每个字段对应一个deque。
    
    特性：
    - 高效的FIFO操作（O(1)添加和删除）
    - 自动管理容量（超出容量时自动移除最旧数据）
    - 灵活的数据模式（首次添加时自动确定字段结构）
    - 支持批量操作和随机采样
    """

    def __init__(self, capacity: int, device: Optional[Union[str, Any]] = None) -> None:
        """
        初始化FIFO缓冲区
        
        Args:
            capacity: 缓冲区最大容量
            device: 计算设备（可选，用于PyTorch张量转换）
        """
        super().__init__(capacity=capacity, device=device)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buffers: Dict[str, Deque[Any]] = {}
        self._keys: Optional[List[str]] = None
        self._size: int = 0

    def __ensure_schema(self, keys: Iterable[str]) -> None:
        """
        确保数据模式一致性
        
        首次添加数据时确定字段结构，后续添加的数据必须具有相同的字段。
        这保证了数据的一致性和类型安全。
        
        Args:
            keys: 经验数据的字段名集合
        """
        klist = sorted(keys)
        if self._keys is None:
            # 首次添加：初始化所有字段的deque
            self._keys = klist
            for k in self._keys:
                self._buffers[k] = deque(maxlen=self.capacity)
        elif self._keys != klist:
            raise ValueError(f"Inconsistent transitions keys. Expected {self._keys}, got {klist}")

    def add(self, **transition: Any) -> None:
        """
        添加一条经验数据
        
        使用关键字参数传入经验数据的各个字段，例如：
        buffer.add(obs=obs, action=action, reward=reward, done=done)
        
        Args:
            **transition: 经验数据的键值对，例如obs, action, reward, next_obs, done等
        """
        if not transition:
            return
        self.__ensure_schema(transition.keys())
        assert self._keys is not None
        for k in self._keys:
            self._buffers[k].append(transition[k])
        self._size = min(self._size + 1, self.capacity)

    def clear(self) -> None:
        """
        清空缓冲区中的所有数据
        """
        for q in self._buffers.values():
            q.clear()
        self._size = 0

    def as_dict(self) -> Dict[str, List[Any]]:
        """
        将缓冲区内容转换为字典形式
        
        Returns:
            包含所有数据的字典，每个键对应一个列表
        """
        return { k: list(v) for k, v in self._buffers.items() }

    def get_recent(self, n: int) -> Dict[str, List[Any]]:
        """
        获取最近n条经验数据
        
        Args:
            n: 要获取的数据条数
            
        Returns:
            最近n条数据的字典
        """
        n = max(0, int(n))
        if self._keys is None or self._size == 0 or n == 0:
            return { k: [] for k in (self._keys or []) }
        return { k: list(self._buffers[k])[-n:] for k in self._keys}

    def sample(self, batch_size: int, replace: bool = False) -> Dict[str, np.ndarray]:
        """
        随机采样一批经验数据
        
        Args:
            batch_size: 批次大小
            replace: 是否允许重复采样（有放回采样）
            
        Returns:
            采样得到的数据字典，所有值都是numpy数组
        """
        if self._keys is None or self._size == 0:
            return {}
        bs = int(batch_size)
        if bs < 0:
            return { k: np.empty((0, ), dtype=object) for k in self._keys }
        
        # 随机选择索引
        idx = np.random.choice(self._size, size=bs, replace=replace)
        
        # 将deque转换为列表以便索引访问
        mats = { k: list(self._buffers[k]) for k in self._keys }
        
        # 收集采样数据
        out: Dict[str, List[Any]] = { k: [] for k in self._keys }
        for i in idx:
            for k in self._keys:
                out[k].append(mats[k][int(i)])
        
        # 转换为numpy数组
        stacked: Dict[str, np.ndarray] = {}
        for k, vals in out.items():
            try:
                stacked[k] = np.asarray(vals)
            except Exception:
                # 如果转换失败（例如混合类型），使用object类型
                stacked[k] = np.array(vals, dtype=object)

        return stacked

    def __len__(self) -> int:
        """
        返回缓冲区中数据的数量
        
        Returns:
            数据条数
        """
        return self._size


class DictBuffer(Buffer):
    """
    字典经验缓冲区
    
    为每个键（例如agent ID）维护一个独立的FIFOBuffer。
    适用于多Agent场景，每个Agent有独立的经验缓冲区，但可以统一采样。
    
    特性：
    - 支持多个独立的FIFO缓冲区（通过键索引）
    - 统一采样：可以从所有缓冲区中混合采样
    - 按键采样：可以从特定键的缓冲区中采样
    - 自动管理：为每个新键自动创建缓冲区
    """

    def __init__(self, capacity: int, device: Optional[Union[str, Any]] = None) -> None:
        """
        初始化字典缓冲区
        
        Args:
            capacity: 每个子缓冲区的最大容量
            device: 计算设备（可选，用于PyTorch张量转换）
        """
        super().__init__(capacity=capacity, device=device)
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buffers: Dict[Any, FIFOBuffer] = {}
        self._size = 0
        self._schema: Optional[List[str]] = None

    def _ensure_schema(self, keys: Iterable[str]) -> None:
        """
        确保数据模式一致性
        
        所有键的缓冲区必须使用相同的数据模式（字段名）。
        
        Args:
            keys: 经验数据的字段名集合
        """
        schema = sorted(keys)
        if self._schema is None:
            self._schema = schema
        elif self._schema != schema:
            raise ValueError(f"Inconsistent transitions keys. Expected {self._schema}, got {schema}")

    def add(self, key: Any, **transition: Any) -> None:
        """
        向指定键的缓冲区添加一条经验数据
        
        如果该键的缓冲区不存在，会自动创建。
        
        Args:
            key: 缓冲区的键（例如agent ID）
            **transition: 经验数据的键值对
        """
        if not transition:
            return
        self._ensure_schema(transition.keys())
        if key not in self._buffers:
            self._buffers[key] = FIFOBuffer(capacity=self.capacity, device=self.device)
        before = len(self._buffers[key])
        self._buffers[key].add(**transition)
        # 如果这个键的缓冲区新增了数据，更新总大小
        self._size += int(len(self._buffers[key]) > before)


    def add_batch(self, key: Any, transitions: Sequence[Dict[str, Any]]) -> None:
        """
        批量向指定键的缓冲区添加经验数据
        
        Args:
            key: 缓冲区的键
            transitions: 经验数据字典列表
        """
        for t in transitions:
            self.add(key, **t)

    def sample(self, batch_size: int, replace: bool = False) -> Dict[str, np.ndarray]:
        """
        从所有缓冲区中混合采样一批经验数据
        
        从所有键的缓冲区中统一采样，混合在一起。
        
        Args:
            batch_size: 批次大小
            replace: 是否允许重复采样（有放回采样）
            
        Returns:
            采样得到的数据字典，所有值都是numpy数组
        """
        if self._size == 0 or not self._buffers:
            return {}

        # 收集所有缓冲区的索引（键，索引）对
        indices: List[Tuple[Any, int]] = []
        mats: Dict[Any, Dict[str, List[Any]]] = {}
        keys: List[Any] = list(self._buffers.keys())

        for k in keys:
            d = self._buffers[k].as_dict()
            mats[k] = { f: list(v) for f, v in d.items() }
            
            # 获取该缓冲区的数据条数
            n = len(next(iter(mats[k].values()))) if d else 0
            indices.extend((k, i) for i in range(n))

        if not indices:
            return {}
        
        # 随机选择索引
        idx = np.random.choice(len(indices), size=int(batch_size), replace=replace)
        
        # 收集采样数据
        out: Dict[str, List[Any]] = { f: [] for f in (self._schema or []) }
        for j in idx:
            k, i = indices[int(j)]
            for f in out:
                out[f].append(mats[k][f][i])

        # 转换为numpy数组
        stacked: Dict[str, np.ndarray] = {}
        for f, vals in out.items():
            try:
                stacked[f] = np.asarray(vals)
            except Exception:
                stacked[f] = np.array(vals, dtype=object)

        return stacked

    def sample_per_key(self, key: Any, batch_size: int, replace: bool = False) -> Dict[str, np.ndarray]:
        """
        从指定键的缓冲区中采样经验数据
        
        Args:
            key: 要采样的缓冲区键
            batch_size: 批次大小
            replace: 是否允许重复采样（有放回采样）
            
        Returns:
            采样得到的数据字典，如果键不存在则返回空字典
        """
        if key not in self._buffers:
            return {}
        return self._buffers[key].sample(batch_size, replace=replace)

    def clear(self) -> None:
        """
        清空所有缓冲区中的数据
        """
        for b in self._buffers.values():
            b.clear()
        self._size = 0

    def as_dict(self) -> Dict[str, List[Any]]:
        """
        将所有缓冲区的内容合并转换为字典形式
        
        Returns:
            包含所有数据的字典，每个键对应一个列表（所有缓冲区的数据合并）
        """
        if not self._buffers:
            return { f: [] for f in (self._schema or []) }

        merged: Dict[str, List[Any]] = { f: [] for f in (self._schema or []) }
        for b in self._buffers.values():
            d = b.as_dict()
            for f, vals in d.items():
                merged[f].extend(vals)

        return merged

    def get_recent(self, n: int) -> Dict[str, List[Any]]:
        """
        获取所有缓冲区中最近n条经验数据的合并结果
        
        Args:
            n: 每个缓冲区要获取的数据条数
            
        Returns:
            最近n条数据的字典（所有缓冲区合并）
        """
        if not self._buffers:
            return { f: [] for f in (self._schema or []) }
        merged: Dict[str, List[Any]] = { f: [] for f in (self._schema or []) }
        for b in self._buffers.values():
            d = b.get_recent(n)
            for f, vals in d.items():
                merged[f].extend(vals)
        return merged

    def __len__(self) -> int:
        """
        返回所有缓冲区中数据的总数量
        
        Returns:
            数据总条数
        """
        return self._size

