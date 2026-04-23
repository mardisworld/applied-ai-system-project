"""Central logging configuration for the Music Recommender application.

Usage in any module::

    from src.logger import get_logger
    logger = get_logger(__name__)

Call ``configure_logging()`` once at application startup (done by ``main.py``).
Library/test code that never calls ``configure_logging()`` will see no output
because the root logger has no handlers by default.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_CONFIGURED = False

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Safe to call before ``configure_logging()``."""
    return logging.getLogger(name)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    level:
        Logging level name, e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``.
    log_file:
        Optional path to a file where logs are also written. Logs always go
        to stderr as well.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [stderr_handler]

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    for handler in handlers:
        root.addHandler(handler)

    # Silence overly chatty third-party loggers at WARNING unless DEBUG requested.
    if numeric_level > logging.DEBUG:
        for noisy in ("urllib3", "requests", "urllib"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
