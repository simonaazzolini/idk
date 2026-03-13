"""
Structured logging configuration with rich integration.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: str = "polymarket_bot.log") -> None:
    """Configure root logger with rich console output and file rotation."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any existing handlers
    root.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        level=numeric_level,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    # Rotating file handler (10MB, keep 5 files)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        root.addHandler(file_handler)
    except PermissionError:
        pass  # Can't write log file, console-only is fine

    # Quiet noisy third-party loggers
    for lib in ("urllib3", "aiohttp", "websockets", "asyncio", "httpx"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
