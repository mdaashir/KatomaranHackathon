"""Logging setup and the mandatory ``events.log`` formatter.

Runtime code uses :func:`get_logger` instead of ``print``. The ``ift`` logger
tree writes to the console and ``logs/app.log``; the ``ift.events`` child
writes the audit trail to ``logs/events.log``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import time
from typing import Any

_APP_LOGGER = "ift"
_EVENTS_LOGGER = "ift.events"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_DEFAULT_FORMAT = (
    "%(asctime)s.%(msecs)03dZ | %(levelname)-8s | %(name)s | %(message)s"
)
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class UTCFormatter(logging.Formatter):
    """Formatter that renders timestamps in UTC."""

    converter = time.gmtime


class EventFormatter(UTCFormatter):
    """Render one machine-parseable audit line.

    Format: ``<ISO-8601 UTC> | <EVENT_CODE> | key=value ...``
    """

    def format(self, record: logging.LogRecord) -> str:
        code = getattr(record, "event_code", "EVENT")
        timestamp = self.formatTime(record, self.datefmt)
        message = record.getMessage() or "-"
        return f"{timestamp}.{int(record.msecs):03d}Z | {code} | {message}"


@dataclass(frozen=True)
class LoggingSetup:
    """Handles returned by :func:`configure_logging`."""

    app_logger: logging.Logger
    events_logger: logging.Logger
    app_log_path: Path
    events_log_path: Path
    debug_log_path: Path | None


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``ift`` namespace."""

    if name == _APP_LOGGER or name.startswith(f"{_APP_LOGGER}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_APP_LOGGER}.{name}")


def configure_logging(
    log_dir: str | Path,
    level: str = "INFO",
    max_bytes: int = 10_485_760,
    backup_count: int = 5,
) -> LoggingSetup:
    """Configure console, app, debug and audit logging.

    Calling this function again replaces the handlers it owns, so it is safe
    in tests and in a restarted process.
    """

    level_name = level.upper()
    if level_name not in _LOG_LEVELS:
        raise ValueError(
            f"invalid log level '{level}', expected one of {sorted(_LOG_LEVELS)}"
        )

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    app_logger = logging.getLogger(_APP_LOGGER)
    events_logger = logging.getLogger(_EVENTS_LOGGER)
    _reset_logger(app_logger)
    _reset_logger(events_logger)

    app_logger.setLevel(logging.DEBUG)
    formatter = UTCFormatter(_DEFAULT_FORMAT, _DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level_name)
    console.setFormatter(formatter)
    app_logger.addHandler(console)

    app_log_path = directory / "app.log"
    app_file = RotatingFileHandler(
        app_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    app_file.setLevel(level_name)
    app_file.setFormatter(formatter)
    app_logger.addHandler(app_file)

    debug_log_path: Path | None = None
    if level_name == "DEBUG":
        debug_log_path = directory / "debug.log"
        debug_file = RotatingFileHandler(
            debug_log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        debug_file.setLevel(logging.DEBUG)
        debug_file.setFormatter(formatter)
        app_logger.addHandler(debug_file)

    events_log_path = directory / "events.log"
    events_file = RotatingFileHandler(
        events_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    events_file.setLevel(logging.INFO)
    events_file.setFormatter(EventFormatter(datefmt=_DATE_FORMAT))
    events_logger.setLevel(logging.INFO)
    events_logger.addHandler(events_file)

    return LoggingSetup(
        app_logger=app_logger,
        events_logger=events_logger,
        app_log_path=app_log_path,
        events_log_path=events_log_path,
        debug_log_path=debug_log_path,
    )


def log_event(logger: logging.Logger, code: str, **fields: Any) -> None:
    """Write one audit event with ``key=value`` fields."""

    logger.info(_format_fields(fields), extra={"event_code": code})


def shutdown_logging() -> None:
    """Remove and close the handlers owned by this module."""

    _reset_logger(logging.getLogger(_EVENTS_LOGGER))
    _reset_logger(logging.getLogger(_APP_LOGGER))


def _reset_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _format_fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={_format_value(value)}" for key, value in fields.items())


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"

    text = (
        str(value)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("|", "/")
    )
    if not text or any(character.isspace() for character in text) or '"' in text:
        text = '"' + text.replace('"', '\\"') + '"'
    return text
