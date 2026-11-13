"""Load ``config.json`` and validate it into typed configuration objects.

The loader is the only place that reads the JSON file. Every other component
receives an :class:`AppConfig` (or one of its sections) through its
constructor.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, cast

from src.config.models import (
    AppConfig,
    DetectorConfig,
    LogLevel,
    LoggingConfig,
    RecognitionConfig,
    SourceConfig,
    SourceType,
    StorageConfig,
    TrackingConfig,
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_MISSING = object()

_TOP_LEVEL_KEYS = {
    "source",
    "frame_skip",
    "detector",
    "recognition",
    "tracking",
    "storage",
    "logging",
}
_SOURCE_KEYS = {
    "type",
    "path",
    "glob",
    "rtsp_url",
    "device_index",
    "realtime",
    "rtsp_transport",
    "reconnect_max_backoff_sec",
}
_DETECTOR_KEYS = {"model_path", "conf_threshold", "imgsz", "min_face_size", "device"}
_RECOGNITION_KEYS = {
    "model_name",
    "model_dir",
    "device",
    "similarity_threshold",
    "min_face_size",
    "min_confidence",
    "blur_threshold",
    "register_quality_threshold",
    "max_embeddings_per_face",
    "retry_interval_sec",
}
_TRACKING_KEYS = {
    "tracker",
    "track_activation_threshold",
    "minimum_matching_threshold",
    "confirm_hits",
    "exit_timeout_frames",
    "lost_track_buffer",
}
_STORAGE_KEYS = {"db_path", "log_dir", "image_size", "jpeg_quality"}
_LOGGING_KEYS = {"level", "max_bytes", "backup_count"}

_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_SOURCE_TYPES = {"file", "folder", "rtsp", "webcam"}
_TRACKERS = {"bytetrack", "deepsort"}


class ConfigError(ValueError):
    """Raised when the configuration is missing, malformed or invalid."""


def load_config(path: str | Path) -> AppConfig:
    """Read *path*, validate it, and return an immutable :class:`AppConfig`.

    Raises:
        ConfigError: if the file cannot be read, is not valid JSON, or any
            value fails validation. The message names the exact key.
    """

    raw = _read_json(Path(path))
    data = _expand_env(raw, "config")
    if not isinstance(data, dict):
        raise ConfigError("config: top-level value must be an object")

    _reject_unknown(data, _TOP_LEVEL_KEYS, "config")

    config = AppConfig(
        source=_parse_source(data),
        frame_skip=_get_int(data, "frame_skip", "config"),
        detector=_parse_detector(data),
        recognition=_parse_recognition(data),
        tracking=_parse_tracking(data),
        storage=_parse_storage(data),
        logging=_parse_logging(data),
    )
    _validate(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Validate an in-memory config with the same rules as file loading."""

    _validate(config)


def _parse_source(data: Mapping[str, Any]) -> SourceConfig:
    section = _section(data, "source", _SOURCE_KEYS)
    return SourceConfig(
        type=cast(SourceType, _get_str(section, "type", "source")),
        path=_get_str(section, "path", "source", default="", allow_empty=True),
        glob=_get_str(section, "glob", "source", default="*.mp4"),
        rtsp_url=_get_str(section, "rtsp_url", "source", default="", allow_empty=True),
        device_index=_get_int(section, "device_index", "source", default=0),
        realtime=_get_bool(section, "realtime", "source", default=False),
        rtsp_transport=_get_str(
            section, "rtsp_transport", "source", default="tcp"
        ).lower(),
        reconnect_max_backoff_sec=_get_int(
            section, "reconnect_max_backoff_sec", "source", default=30
        ),
    )


def _parse_detector(data: Mapping[str, Any]) -> DetectorConfig:
    section = _section(data, "detector", _DETECTOR_KEYS)
    return DetectorConfig(
        model_path=_get_str(
            section,
            "model_path",
            "detector",
            default="models/yolov8n-face.pt",
        ),
        conf_threshold=_get_float(section, "conf_threshold", "detector", default=0.5),
        imgsz=_get_int(section, "imgsz", "detector", default=1600),
        min_face_size=_get_int(section, "min_face_size", "detector", default=40),
        device=_get_str(section, "device", "detector", default="auto"),
    )


def _parse_recognition(data: Mapping[str, Any]) -> RecognitionConfig:
    section = _section(data, "recognition", _RECOGNITION_KEYS)
    return RecognitionConfig(
        model_name=_get_str(section, "model_name", "recognition", default="buffalo_l"),
        model_dir=_get_str(
            section,
            "model_dir",
            "recognition",
            default="models/insightface/models/buffalo_l",
        ),
        device=_get_str(section, "device", "recognition", default="auto"),
        similarity_threshold=_get_float(
            section,
            "similarity_threshold",
            "recognition",
            default=0.45,
        ),
        min_face_size=_get_int(section, "min_face_size", "recognition", default=60),
        min_confidence=_get_float(
            section, "min_confidence", "recognition", default=0.6
        ),
        blur_threshold=_get_float(
            section, "blur_threshold", "recognition", default=60.0
        ),
        register_quality_threshold=_get_float(
            section,
            "register_quality_threshold",
            "recognition",
            default=0.65,
        ),
        max_embeddings_per_face=_get_int(
            section,
            "max_embeddings_per_face",
            "recognition",
            default=3,
        ),
        retry_interval_sec=_get_float(
            section, "retry_interval_sec", "recognition", default=1.0
        ),
    )


def _parse_tracking(data: Mapping[str, Any]) -> TrackingConfig:
    section = _section(data, "tracking", _TRACKING_KEYS)
    return TrackingConfig(
        tracker=_get_str(section, "tracker", "tracking", default="bytetrack").lower(),
        track_activation_threshold=_get_float(
            section,
            "track_activation_threshold",
            "tracking",
            default=0.25,
        ),
        minimum_matching_threshold=_get_float(
            section,
            "minimum_matching_threshold",
            "tracking",
            default=0.8,
        ),
        confirm_hits=_get_int(section, "confirm_hits", "tracking", default=3),
        exit_timeout_frames=_get_int(
            section, "exit_timeout_frames", "tracking", default=75
        ),
        lost_track_buffer=_get_int(
            section, "lost_track_buffer", "tracking", default=30
        ),
    )


def _parse_storage(data: Mapping[str, Any]) -> StorageConfig:
    section = _section(data, "storage", _STORAGE_KEYS)
    return StorageConfig(
        db_path=_get_str(section, "db_path", "storage", default="data/faces.db"),
        log_dir=_get_str(section, "log_dir", "storage", default="logs"),
        image_size=_get_int(section, "image_size", "storage", default=256),
        jpeg_quality=_get_int(section, "jpeg_quality", "storage", default=90),
    )


def _parse_logging(data: Mapping[str, Any]) -> LoggingConfig:
    section = _section(data, "logging", _LOGGING_KEYS)
    level = _get_str(section, "level", "logging", default="INFO").upper()
    return LoggingConfig(
        level=cast(LogLevel, level),
        max_bytes=_get_int(section, "max_bytes", "logging", default=10_485_760),
        backup_count=_get_int(section, "backup_count", "logging", default=5),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise ConfigError(f"{path}: configuration file not found") from exc
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read configuration: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level value must be an object")
    return data


def _expand_env(value: Any, path: str) -> Any:
    """Replace ``${VAR}`` references with environment values."""

    if isinstance(value, dict):
        return {key: _expand_env(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [
            _expand_env(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ConfigError(f"{path}: environment variable '{name}' is not set")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    return value


def _section(
    data: Mapping[str, Any], key: str, allowed_keys: set[str]
) -> dict[str, Any]:
    if key not in data:
        raise ConfigError(f"{key}: missing required section")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{key}: must be an object")
    section = dict(value)
    _reject_unknown(section, allowed_keys, key)
    return section


def _reject_unknown(section: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ConfigError(
            f"{path}: unknown key(s): {', '.join(repr(key) for key in unknown)}"
        )


def _get_value(
    section: Mapping[str, Any], key: str, path: str, default: Any = _MISSING
) -> Any:
    if key in section:
        return section[key]
    if default is _MISSING:
        raise ConfigError(f"{path}.{key}: missing required value")
    return default


def _get_str(
    section: Mapping[str, Any],
    key: str,
    path: str,
    default: Any = _MISSING,
    *,
    allow_empty: bool = False,
) -> str:
    value = _get_value(section, key, path, default)
    if not isinstance(value, str):
        raise ConfigError(
            f"{path}.{key}: expected a string, got {type(value).__name__}"
        )
    if not allow_empty and value == "":
        raise ConfigError(f"{path}.{key}: must not be empty")
    return value


def _get_int(
    section: Mapping[str, Any], key: str, path: str, default: Any = _MISSING
) -> int:
    value = _get_value(section, key, path, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{path}.{key}: expected an integer, got {type(value).__name__}"
        )
    return value


def _get_float(
    section: Mapping[str, Any], key: str, path: str, default: Any = _MISSING
) -> float:
    value = _get_value(section, key, path, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{path}.{key}: expected a number, got {type(value).__name__}"
        )
    return float(value)


def _get_bool(
    section: Mapping[str, Any], key: str, path: str, default: Any = _MISSING
) -> bool:
    value = _get_value(section, key, path, default)
    if not isinstance(value, bool):
        raise ConfigError(
            f"{path}.{key}: expected a boolean, got {type(value).__name__}"
        )
    return value


def _validate(config: AppConfig) -> None:
    if not 1 <= config.frame_skip <= 1000:
        raise ConfigError("config.frame_skip: must be between 1 and 1000")

    _validate_source(config.source)
    _validate_detector(config.detector)
    _validate_recognition(config.recognition, config.detector)
    _validate_tracking(config.tracking)
    _validate_storage(config.storage)
    _validate_logging(config.logging)


def _validate_source(source: SourceConfig) -> None:
    if source.type not in _SOURCE_TYPES:
        raise ConfigError(
            f"source.type: expected one of {sorted(_SOURCE_TYPES)}, got '{source.type}'"
        )
    if source.type in {"file", "folder"} and not source.path:
        raise ConfigError(
            "source.path: required when source.type is 'file' or 'folder'"
        )
    if source.type == "folder" and not source.glob:
        raise ConfigError("source.glob: required when source.type is 'folder'")
    if source.type == "rtsp" and not source.rtsp_url:
        raise ConfigError("source.rtsp_url: required when source.type is 'rtsp'")
    if source.device_index < 0:
        raise ConfigError("source.device_index: must not be negative")
    if source.rtsp_transport not in {"tcp", "udp"}:
        raise ConfigError("source.rtsp_transport: expected 'tcp' or 'udp'")
    if source.reconnect_max_backoff_sec < 1:
        raise ConfigError("source.reconnect_max_backoff_sec: must be at least 1")


def _validate_detector(detector: DetectorConfig) -> None:
    if detector.imgsz < 320:
        raise ConfigError("detector.imgsz: must be at least 320")
    if detector.imgsz % 32 != 0:
        raise ConfigError("detector.imgsz: must be a multiple of 32")
    if not 0.0 <= detector.conf_threshold <= 1.0:
        raise ConfigError("detector.conf_threshold: must be between 0 and 1")
    if detector.min_face_size < 1:
        raise ConfigError("detector.min_face_size: must be at least 1")
    _validate_device(detector.device, "detector.device")


def _validate_device(device: str, path: str) -> None:
    if device not in {"auto", "cpu"} and not re.fullmatch(r"cuda(:\d+)?", device):
        raise ConfigError(f"{path}: expected 'auto', 'cpu', 'cuda' or 'cuda:<index>'")


def _validate_recognition(
    recognition: RecognitionConfig, detector: DetectorConfig
) -> None:
    if not recognition.model_name:
        raise ConfigError("recognition.model_name: must not be empty")
    if not recognition.model_dir:
        raise ConfigError("recognition.model_dir: must not be empty")
    _validate_device(recognition.device, "recognition.device")
    if not 0.0 <= recognition.similarity_threshold <= 1.0:
        raise ConfigError("recognition.similarity_threshold: must be between 0 and 1")
    if recognition.min_face_size < detector.min_face_size:
        raise ConfigError(
            "recognition.min_face_size: must be greater than or equal to "
            "detector.min_face_size"
        )
    if not 0.0 <= recognition.min_confidence <= 1.0:
        raise ConfigError("recognition.min_confidence: must be between 0 and 1")
    if recognition.blur_threshold < 0:
        raise ConfigError("recognition.blur_threshold: must not be negative")
    if not 0.0 <= recognition.register_quality_threshold <= 1.0:
        raise ConfigError(
            "recognition.register_quality_threshold: must be between 0 and 1"
        )
    if recognition.max_embeddings_per_face < 1:
        raise ConfigError("recognition.max_embeddings_per_face: must be at least 1")
    if recognition.retry_interval_sec <= 0:
        raise ConfigError("recognition.retry_interval_sec: must be greater than 0")


def _validate_tracking(tracking: TrackingConfig) -> None:
    if tracking.tracker not in _TRACKERS:
        raise ConfigError(
            f"tracking.tracker: expected one of {sorted(_TRACKERS)}, "
            f"got '{tracking.tracker}'"
        )
    if not 0.0 <= tracking.track_activation_threshold <= 1.0:
        raise ConfigError(
            "tracking.track_activation_threshold: must be between 0 and 1"
        )
    if not 0.0 <= tracking.minimum_matching_threshold <= 1.0:
        raise ConfigError(
            "tracking.minimum_matching_threshold: must be between 0 and 1"
        )
    if tracking.confirm_hits < 1:
        raise ConfigError("tracking.confirm_hits: must be at least 1")
    if tracking.exit_timeout_frames < 1:
        raise ConfigError("tracking.exit_timeout_frames: must be at least 1")
    if tracking.lost_track_buffer < 1:
        raise ConfigError("tracking.lost_track_buffer: must be at least 1")


def _validate_storage(storage: StorageConfig) -> None:
    if not storage.db_path:
        raise ConfigError("storage.db_path: must not be empty")
    if not storage.log_dir:
        raise ConfigError("storage.log_dir: must not be empty")
    if not 32 <= storage.image_size <= 2048:
        raise ConfigError("storage.image_size: must be between 32 and 2048")
    if not 1 <= storage.jpeg_quality <= 100:
        raise ConfigError("storage.jpeg_quality: must be between 1 and 100")


def _validate_logging(logging_config: LoggingConfig) -> None:
    if logging_config.level not in _LOG_LEVELS:
        raise ConfigError(
            f"logging.level: expected one of {sorted(_LOG_LEVELS)}, "
            f"got '{logging_config.level}'"
        )
    if logging_config.max_bytes < 1024:
        raise ConfigError("logging.max_bytes: must be at least 1024")
    if logging_config.backup_count < 0:
        raise ConfigError("logging.backup_count: must not be negative")
