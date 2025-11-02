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

from typing import Any

import yaml

from .schema import PPOConfig


def load_config(path: str) -> PPOConfig:
    """
    从YAML文件加载配置

    Args:
        path: YAML配置文件路径

    Returns:
        PPOConfig配置对象

    Raises:
        FileNotFoundError: 如果配置文件不存在
        yaml.YAMLError: 如果YAML文件格式错误
        TypeError: 如果配置数据无法转换为PPOConfig
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PPOConfig(**data)
