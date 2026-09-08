"""Central logging setup. Replaces jarvis's scattered print() calls.

configure_logging() is called once from main.py; every module gets its
logger via get_logger(__name__-ish short name), all under the "luna."
namespace so log level/handlers apply uniformly.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_ROOT_NAME = "luna"
_configured = False


def configure_logging(level: str = "INFO", log_dir: str | Path | None = None) -> None:
    global _configured
    if _configured:
        return

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level.upper())
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s", "%H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "luna.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8",
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
