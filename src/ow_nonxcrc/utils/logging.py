"""Simple logging for experiments."""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with stream handler."""
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(level)
        log.addHandler(h)
    return log
