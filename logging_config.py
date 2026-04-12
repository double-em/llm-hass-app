"""Centralized logging configuration for LLM AI Dashboard.

All modules use get_logger(__name__) instead of logging.getLogger(__name__)
to automatically include version in every log record.
"""
import json
import logging
import os
import sys

# Version is written at Docker build time. Default to 'dev' if not set.
try:
    from version import __version__
except ImportError:
    __version__ = "dev"


class VersionFormatter(logging.Formatter):
    """Plain text formatter: ISO timestamp, level, version, message."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(version)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "version"):
            record.version = __version__
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "version"):
            record.version = __version__
        return json.dumps({
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "version": record.version,
            "message": record.getMessage(),
        })


def configure_logging() -> None:
    """Configure root logger. Called once from app.py on startup."""
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = os.environ.get("LOG_FORMAT", "text")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(VersionFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a LoggerAdapter that injects version into every log record.

    Usage: logger = get_logger(__name__)
    All log calls work normally — version is pre-injected.
    """
    logger = logging.getLogger(name)
    return logging.LoggerAdapter(logger, {"version": __version__})
