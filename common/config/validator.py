# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : validator.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:33
@Update Date    :
@Description    : 配置验证器
验证配置文件的有效性和完整性
"""
# ------------------------------------------------------------

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ConfigValidator:
    """
    配置验证器

    用于验证配置文件的有效性，检查必需字段、类型、取值范围等。
    """

    def __init__(self):
        """初始化验证器"""
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证配置

        Args:
            config: 配置字典
            schema: 配置模式（可选，如果不提供则使用默认验证）

        Returns:
            验证是否通过
        """
        self.errors.clear()
        self.warnings.clear()

        if schema:
            self._validate_with_schema(config, schema)
        else:
            self._validate_basic(config)

        return len(self.errors) == 0

    def _validate_basic(self, config: Dict[str, Any]) -> None:
        """基本验证"""
        # 检查必需字段
        required_fields = [
            "seed",
            "env_id",
            "num_updates",
        ]

        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")

        # 验证数值范围
        if "clip_coef" in config:
            clip_coef = config["clip_coef"]
            if not isinstance(clip_coef, (int, float)) or not (0 < clip_coef <= 1):
                self.errors.append(f"clip_coef must be in (0, 1], got {clip_coef}")

        if "lr" in config:
            lr = config["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append(f"lr must be positive, got {lr}")

        if "batch_size" in config:
            batch_size = config["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append(f"batch_size must be a positive integer, got {batch_size}")

        if "gamma" in config:
            gamma = config["gamma"]
            if not isinstance(gamma, (int, float)) or not (0 <= gamma <= 1):
                self.errors.append(f"gamma must be in [0, 1], got {gamma}")

    def _validate_with_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """使用模式验证"""
        # 检查必需字段
        required = schema.get("required", [])
        for field in required:
            if field not in config:
                self.errors.append(f"Missing required field: {field}")

        # 检查字段类型
        types = schema.get("types", {})
        for field, expected_type in types.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    self.errors.append(
                        f"Field '{field}' must be of type {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        # 检查取值范围
        ranges = schema.get("ranges", {})
        for field, (min_val, max_val) in ranges.items():
            if field in config:
                value = config[field]
                if not (min_val <= value <= max_val):
                    self.errors.append(
                        f"Field '{field}' must be in [{min_val}, {max_val}], got {value}"
                    )

    def get_errors(self) -> List[str]:
        """获取验证错误列表"""
        return self.errors.copy()

    def get_warnings(self) -> List[str]:
        """获取验证警告列表"""
        return self.warnings.copy()

    def print_report(self) -> None:
        """打印验证报告"""
        if self.errors:
            print("Validation Errors:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("Validation Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("Configuration is valid!")


def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    便捷函数：验证配置

    Args:
        config: 配置字典
        schema: 配置模式（可选）

    Returns:
        验证是否通过
    """
    validator = ConfigValidator()
    is_valid = validator.validate(config, schema)

    if not is_valid:
        validator.print_report()

    return is_valid
