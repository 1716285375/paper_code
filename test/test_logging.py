# ------------------------------------------------------------
# -*- coding: utf-8 -*-
"""
@File           : test_logging.py
@Author         : Jie
@CopyRight      : Copyright © 2025 Jie. All Rights Reserved
@Create Date    : 2025-10-31 13:19
@Update Date    :
@Description    :
"""
# ------------------------------------------------------------


from __future__ import annotations

import sys
from pathlib import Path

from common.utils.logging import (
    critical,
    debug,
    error,
    exception,
    get_logger,
    info,
    setup_logger,
    warning,
)


def main() -> None:
    # Setup logger with custom configuration
    import logging

    logger = setup_logger(
        name="ppo",
        log_dir="logs",
        log_level=logging.DEBUG,  # Use logging constants instead of numbers
        console_level=logging.INFO,  # Less verbose on console
        file_level=logging.DEBUG,  # More verbose in file
        enable_file=True,
        enable_console=True,
    )

    # Test different log levels
    logger.debug("This is a debug message (only in file)")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test convenience functions
    debug("Debug via convenience function")
    info("Info via convenience function")
    warning("Warning via convenience function")
    error("Error via convenience function")
    critical("Critical via convenience function")

    # Test child logger
    trainer_logger = get_logger("trainer")
    trainer_logger.info("Message from trainer logger")

    # Test exception logging
    try:
        # raise ValueError("Test exception")
        pass
    except Exception:
        exception("Caught exception with traceback")

    # Test with context
    logger.info("Logging with extra context: episode=1, reward=100.5")
    logger.info("Multiple values: %s, %d, %.2f", "string", 42, 3.14)


if __name__ == "__main__":
    main()
