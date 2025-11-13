# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : logging.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-30 19:32
@Update Date    :
@Description    : 日志管理器
提供带颜色输出的控制台日志和文件日志功能
"""
# ------------------------------------------------------------


from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

LogLevel = int


try:
    from colorama import Back, Fore, Style, init

    COLORAMA_AVAILABLE = True
    init(autoreset=True)
except ImportError:
    COLORAMA_AVAILABLE = False

    # ANSI color codes as fallback
    class Fore:
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        RESET = "\033[0m"

    class Style:
        BRIGHT = "\033[1m"
        DIM = "\033[2m"
        RESET_ALL = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """
    自定义颜色格式器，支持控制台彩色输出

    根据日志等级自动为日志消息添加颜色，提高可读性。
    """

    # 映射日志等级到颜色
    COLORS = {
        "DEBUG": Fore.CYAN + Style.DIM if COLORAMA_AVAILABLE else "\033[36m\033[2m",
        "INFO": Fore.GREEN if COLORAMA_AVAILABLE else "\033[32m",
        "WARNING": Fore.YELLOW + Style.BRIGHT if COLORAMA_AVAILABLE else "\033[33m\033[1m",
        "ERROR": Fore.RED if COLORAMA_AVAILABLE else "\033[31m",
        "CRITICAL": (
            Fore.RED + Style.BRIGHT + Back.YELLOW
            if COLORAMA_AVAILABLE
            else "\033[31m\033[1m\033[43m"
        ),
    }
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录，添加颜色代码

        Args:
            record: 日志记录对象

        Returns:
            格式化后的日志字符串（带颜色代码）
        """
        original_level_name = record.levelname
        original_name = record.name

        log_color = self.COLORS.get(record.levelname, "")
        reset = self.RESET
        name_color = Fore.BLUE if COLORAMA_AVAILABLE else "\033[34m"

        # 暂时修改record适配控制台彩色输出
        record.levelname = f"{log_color}{record.levelname}{reset}"
        record.name = f"{name_color}{record.name}{reset}"

        formatted = super().format(record)

        # 颜色格式
        record.levelname = original_level_name
        record.name = original_name

        return formatted


class DebugFormatter(logging.Formatter):
    """
    调试格式器

    提供详细的日志格式，包含文件名、函数名、行号等信息，适用于文件日志。
    """

    DETAILED_FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
    )

    def __init__(self, use_colors: bool = False):
        super().__init__(fmt=self.DETAILED_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录（调试模式）

        Args:
            record: 日志记录对象

        Returns:
            格式化后的日志字符串，包含函数名和行号
        """
        if not hasattr(record, "funcName") or record.funcName == "":
            record.funcName = "<module>"

        return super().format(record)


class LoggerManager:
    """
    日志管理器

    统一管理日志的配置和输出，支持同时输出到控制台和文件。
    控制台输出带颜色，文件输出包含详细信息。
    """

    def __init__(
        self,
        name: str = "ppo",
        log_dir: str | Path = "logs",
        log_level: LogLevel = logging.DEBUG,
        console_level: Optional[LogLevel] = None,
        file_level: Optional[LogLevel] = None,
        enable_file: bool = True,
        enable_console: bool = True,
    ) -> None:
        """
        初始化日志管理器

        Args:
            name: 日志器名称，用于标识日志来源
            log_dir: 日志文件保存目录
            log_level: 默认日志级别（同时应用于控制台和文件）
            console_level: 控制台日志级别（如果为None则使用log_level）
            file_level: 文件日志级别（如果为None则使用log_level）
            enable_file: 是否启用文件日志
            enable_console: 是否启用控制台日志
        """

        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.console_level = console_level or log_level
        self.file_level = file_level or log_level
        self.enable_file = enable_file
        self.enable_console = enable_console

        if self.enable_file:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """
        设置并配置日志器

        Returns:
            配置好的logging.Logger实例
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(min(self.console_level, self.file_level))

        # 移除现存的 handlers, 避免拷贝
        logger.handlers.clear()

        # 文件处理器，使用详细格式（首先添加以避免颜色代码）
        if self.enable_file:
            # 创建带时间戳的日志文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{self.name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(self.file_level)

            file_formatter = DebugFormatter()
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # 控制台输出
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_formatter = ColoredFormatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # 维护一个当前日志文件: latest.log 来调试
            latest_log = self.log_dir / "latest.log"
            try:
                if latest_log.exists():
                    latest_log.unlink()
                elif latest_log.is_symlink():
                    try:
                        latest_log.unlink()
                    except Exception:
                        import os

                        try:
                            os.remove(str(latest_log))
                        except Exception:
                            pass
                import os

                if os.name == "nt":
                    try:
                        latest_log.symlink_to(log_file)
                    except (OSError, NotImplementedError):
                        try:
                            import shutil

                            os.link(log_file, latest_log)
                        except (OSError, AttributeError):
                            shutil.copy(log_file, latest_log)
                else:
                    latest_log.symlink_to(log_file)
            except Exception as e:
                logger.warning(f"Could not create latest log: {e}")

            logger.info(f"Logging to file: {log_file}")

        return logger

    def get_logger(self) -> logging.Logger:
        """
        获取已配置的日志器实例

        Returns:
            配置好的logging.Logger实例
        """
        return self.logger


# 维护一个全局日志管理器实例
_global_logger_manager: Optional[LoggerManager] = None


def setup_logger(
    name: str = "ppo",
    log_dir: str | Path = "logs",
    log_level: LogLevel = logging.INFO,
    console_level: Optional[LogLevel] = None,
    file_level: Optional[LogLevel] = None,
    enable_file: bool = True,
    enable_console: bool = True,
) -> logging.Logger:
    """
    设置全局日志器（便捷函数）

    Args:
        name: 日志器名称
        log_dir: 日志文件保存目录
        log_level: 默认日志级别
        console_level: 控制台日志级别
        file_level: 文件日志级别
        enable_file: 是否启用文件日志
        enable_console: 是否启用控制台日志

    Returns:
        已配置的日志器实例

    示例:
        >>> logger = setup_logger(log_level=logging.DEBUG)
        >>> logger.info("这条消息将被记录")
    """
    global _global_logger_manager
    _global_logger_manager = LoggerManager(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        console_level=console_level,
        file_level=file_level,
        enable_file=enable_file,
        enable_console=enable_console,
    )
    return _global_logger_manager.get_logger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    获取日志器实例

    Args:
        name: 日志器名称（可选），例如 "ppo.trainer"
            如果提供，将返回子日志器

    Returns:
        日志器实例

    Note:
        如果全局日志管理器未初始化，将返回默认的root日志器
    """
    if _global_logger_manager is None:
        if name:
            return logging.getLogger(f"{_global_logger_manager.name}.{name}")
        return _global_logger_manager
    return logging.getLogger(name or "root")


# 便捷函数，用于快速调试（无需获取logger实例）
def debug(msg: str, *args, **kwargs) -> None:
    """记录调试级别日志消息"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """记录信息级别日志消息"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """记录警告级别日志消息"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """记录错误级别日志消息"""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """记录严重错误级别日志消息"""
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """记录异常日志消息（包含完整的堆栈跟踪）"""
    get_logger().exception(msg, *args, **kwargs)
