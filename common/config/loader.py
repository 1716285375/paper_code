# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : loader.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:33
@Update Date    :
@Description    : 配置文件加载器
从YAML文件加载配置并转换为配置对象
"""
# ------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from easydict import EasyDict

from .schema import PPOConfig


def _dict_to_easydict(data: Any) -> Any:
    """
    递归地将字典转换为EasyDict
    
    Args:
        data: 要转换的数据（字典、列表或其他类型）
    
    Returns:
        转换后的数据（字典变为EasyDict，列表中的字典也会转换）
    """
    if isinstance(data, dict):
        return EasyDict({k: _dict_to_easydict(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [_dict_to_easydict(item) for item in data]
    else:
        return data


def load_config(
    path: str, 
    as_dict: bool = False,
    project_root: Union[str, Path, None] = None
) -> Union[PPOConfig, EasyDict]:
    """
    从YAML文件加载配置

    Args:
        path: YAML配置文件路径（相对路径或绝对路径）
        as_dict: 如果True，返回字典；如果False，返回PPOConfig对象
        project_root: 项目根目录路径（用于解析相对路径），如果None则从当前文件推断

    Returns:
        如果as_dict=True，返回EasyDict配置对象（支持点号访问）；否则返回PPOConfig配置对象

    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果YAML文件格式错误
        TypeError: 如果配置数据无法转换为PPOConfig（且as_dict=False）
    """
    # 解析路径
    config_file = Path(path)
    if not config_file.is_absolute():
        # 相对路径：如果提供了project_root，使用它；否则尝试从当前文件推断
        if project_root is not None:
            root = Path(project_root)
        else:
            # 从当前文件位置推断项目根目录（common/config/loader.py -> 项目根目录）
            root = Path(__file__).parent.parent.parent
        config_file = root / path
    
    # 确保文件存在
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")
    
    # 加载YAML
    with open(config_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # 根据as_dict参数返回
    if as_dict:
        # 返回EasyDict，支持点号访问
        return _dict_to_easydict(data)
    else:
        return PPOConfig(**data)
