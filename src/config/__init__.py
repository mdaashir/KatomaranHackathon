"""Typed configuration objects."""

from src.config.loader import ConfigError, load_config
from src.config.models import (
    AppConfig,
    DetectorConfig,
    LoggingConfig,
    LogLevel,
    RecognitionConfig,
    SourceConfig,
    SourceType,
    StorageConfig,
    TrackingConfig,
)

__all__ = [
    "AppConfig",
    "ConfigError",
    "DetectorConfig",
    "LogLevel",
    "LoggingConfig",
    "RecognitionConfig",
    "SourceConfig",
    "SourceType",
    "StorageConfig",
    "TrackingConfig",
    "load_config",
]
