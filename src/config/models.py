"""Immutable typed configuration objects for the face tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceType = Literal["file", "folder", "rtsp", "webcam"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass(frozen=True)
class SourceConfig:
    """Where frames come from."""

    type: SourceType
    path: str = ""
    glob: str = "*.mp4"
    rtsp_url: str = ""
    device_index: int = 0
    realtime: bool = False
    rtsp_transport: str = "tcp"
    reconnect_max_backoff_sec: int = 30


@dataclass(frozen=True)
class DetectorConfig:
    """YOLO face detector settings."""

    model_path: str = "models/yolov8n-face.pt"
    conf_threshold: float = 0.5
    imgsz: int = 1600
    min_face_size: int = 40
    device: str = "auto"


@dataclass(frozen=True)
class RecognitionConfig:
    """Embedding, quality and re-identification settings."""

    model_name: str = "buffalo_l"
    model_dir: str = "models/insightface/models/buffalo_l"
    device: str = "auto"
    similarity_threshold: float = 0.45
    min_face_size: int = 60
    min_confidence: float = 0.6
    blur_threshold: float = 60.0
    register_quality_threshold: float = 0.65
    max_embeddings_per_face: int = 3
    retry_interval_sec: float = 1.0


@dataclass(frozen=True)
class TrackingConfig:
    """Tracker and track-lifecycle settings."""

    tracker: str = "bytetrack"
    track_activation_threshold: float = 0.25
    minimum_matching_threshold: float = 0.8
    confirm_hits: int = 3
    exit_timeout_frames: int = 75
    lost_track_buffer: int = 30


@dataclass(frozen=True)
class StorageConfig:
    """SQLite and image-store settings."""

    db_path: str = "data/faces.db"
    log_dir: str = "logs"
    image_size: int = 256
    jpeg_quality: int = 90


@dataclass(frozen=True)
class LoggingConfig:
    """Rotating log settings."""

    level: LogLevel = "INFO"
    max_bytes: int = 10_485_760
    backup_count: int = 5


@dataclass(frozen=True)
class AppConfig:
    """Complete validated application configuration."""

    source: SourceConfig
    frame_skip: int
    detector: DetectorConfig
    recognition: RecognitionConfig
    tracking: TrackingConfig
    storage: StorageConfig
    logging: LoggingConfig
