"""Runtime config overrides layered over ``config.json``."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
import re
import threading

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import AppConfig, ConfigError, validate_config

LOGGER = logging.getLogger(__name__)


class DetectorOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conf_threshold: float | None = Field(None, ge=0.0, le=1.0)
    min_face_size: int | None = Field(None, ge=1)


class RecognitionOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    min_face_size: int | None = Field(None, ge=1)


class TrackingOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_timeout_frames: int | None = Field(None, ge=1)


class ConfigOverrides(BaseModel):
    """Only the hot-reloadable tunables; frame_skip stays file-owned."""

    model_config = ConfigDict(extra="forbid")

    detector: DetectorOverrides | None = None
    recognition: RecognitionOverrides | None = None
    tracking: TrackingOverrides | None = None


class ConfigStore:
    """Base config + persisted runtime overrides + atomic save-default."""

    def __init__(
        self,
        base: AppConfig,
        *,
        base_path: str | Path | None = None,
        overrides_path: str | Path | None = None,
    ) -> None:
        self._base = base
        self._base_path = Path(base_path) if base_path is not None else None
        self._overrides_path = (
            Path(overrides_path)
            if overrides_path is not None
            else Path("data/runtime_overrides.json")
        )
        self._lock = threading.RLock()
        self._overrides = self._load_overrides()
        try:
            self._effective = self._merge(self._base, self._overrides)
        except ConfigError as exc:
            LOGGER.warning("ignoring incompatible runtime overrides: %s", exc)
            self._overrides = {}
            self._effective = self._base

    @property
    def effective(self) -> AppConfig:
        with self._lock:
            return self._effective

    @property
    def base(self) -> AppConfig:
        with self._lock:
            return self._base

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            base = _redact(asdict(self._base))
            effective = _redact(asdict(self._effective))
            return {
                "base": base,
                "effective": effective,
                "overrides": dict(self._overrides),
                "hot_fields": [
                    "detector.conf_threshold",
                    "detector.min_face_size",
                    "recognition.similarity_threshold",
                    "recognition.min_face_size",
                    "tracking.exit_timeout_frames",
                ],
                "file_owned": ["frame_skip", "source", "storage", "logging"],
            }

    def apply(self, overrides: ConfigOverrides) -> AppConfig:
        payload = overrides.model_dump(exclude_none=True)
        effective = self._merge(self._base, payload)
        with self._lock:
            self._overrides = payload
            self._effective = effective
        self._write_overrides()
        return effective

    def reset(self) -> AppConfig:
        with self._lock:
            self._overrides = {}
            self._effective = self._base
        self._overrides_path.unlink(missing_ok=True)
        return self._base

    def save_default(self) -> AppConfig:
        with self._lock:
            if self._base_path is None:
                raise ConfigError("base config path is not configured")
            effective = self._effective
            validate_config(effective)
            self._atomic_write(
                self._base_path,
                json.dumps(asdict(effective), indent=2) + "\n",
            )
            self._base = effective
            self._overrides = {}
            self._effective = effective
        self._overrides_path.unlink(missing_ok=True)
        return effective

    def _merge(
        self,
        base: AppConfig,
        overrides: dict[str, object],
    ) -> AppConfig:
        detector_data = _section(overrides, "detector")
        recognition_data = _section(overrides, "recognition")
        tracking_data = _section(overrides, "tracking")
        detector = replace(base.detector, **detector_data)
        recognition = replace(base.recognition, **recognition_data)
        tracking = replace(base.tracking, **tracking_data)
        candidate = replace(
            base,
            detector=detector,
            recognition=recognition,
            tracking=tracking,
        )
        validate_config(candidate)
        return candidate

    def _load_overrides(self) -> dict[str, object]:
        if not self._overrides_path.is_file():
            return {}
        try:
            data = json.loads(self._overrides_path.read_text(encoding="utf-8"))
            parsed = ConfigOverrides.model_validate(data)
        except (OSError, ValueError, ValidationError) as exc:
            LOGGER.warning("ignoring invalid runtime overrides: %s", exc)
            return {}
        return parsed.model_dump(exclude_none=True)

    def _write_overrides(self) -> None:
        self._overrides_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(
            self._overrides_path,
            json.dumps(self._overrides, indent=2) + "\n",
        )

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)


def _section(overrides: dict[str, object], name: str) -> dict[str, object]:
    section = overrides.get(name)
    return dict(section) if isinstance(section, dict) else {}


def _redact(payload: dict[str, object]) -> dict[str, object]:
    source = payload.get("source")
    if isinstance(source, dict):
        url = source.get("rtsp_url")
        if isinstance(url, str) and url:
            source["rtsp_url"] = re.sub(r"(rtsps?://)[^/@]+@", r"\1***@", url)
    return payload
