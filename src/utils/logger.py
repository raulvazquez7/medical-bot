"""
Structured logging utilities for production observability.
Provides context-rich logging with correlation IDs and structured fields.
"""

import logging
import json
from typing import Any
from contextvars import ContextVar

# Context variable for correlation ID (thread-safe)
correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured logs with context.
    Formats logs as JSON for easy parsing by log aggregation tools.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with structured context.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if available
        correlation_id = correlation_id_ctx.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id

        # Add any extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class StructuredLogger:
    """
    Wrapper around standard logger with structured logging capabilities.
    Makes it easy to add context fields to logs.
    """

    def __init__(self, name: str):
        """
        Initialize structured logger.

        Args:
            name: Logger name (typically __name__)
        """
        self.logger = logging.getLogger(name)

    def _log(
        self, level: int, message: str, **extra_fields: Any
    ) -> None:
        """
        Internal log method with structured fields.

        Args:
            level: Log level
            message: Log message
            **extra_fields: Additional context fields
        """
        extra = {"extra_fields": extra_fields}
        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **extra_fields: Any) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **extra_fields)

    def info(self, message: str, **extra_fields: Any) -> None:
        """Log info message with context."""
        self._log(logging.INFO, message, **extra_fields)

    def warning(self, message: str, **extra_fields: Any) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, message, **extra_fields)

    def error(
        self, message: str, exc_info: bool = False, **extra_fields: Any
    ) -> None:
        """
        Log error message with context.

        Args:
            message: Error message
            exc_info: If True, include exception traceback
            **extra_fields: Additional context fields
        """
        extra = {"extra_fields": extra_fields}
        self.logger.error(message, extra=extra, exc_info=exc_info)


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def set_correlation_id(correlation_id: str | None) -> None:
    """
    Set correlation ID for current context.

    Args:
        correlation_id: Unique ID to track requests/conversations
    """
    correlation_id_ctx.set(correlation_id)


def get_correlation_id() -> str | None:
    """
    Get current correlation ID.

    Returns:
        Current correlation ID or None
    """
    return correlation_id_ctx.get()


def configure_logging(
    level: str = "INFO", use_structured: bool = True
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_structured: If True, use structured JSON logging
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Set formatter
    if use_structured:
        formatter = StructuredFormatter(
            fmt="%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

