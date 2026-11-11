"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

import pytest


@pytest.fixture
def config_dict(tmp_path: Path) -> dict[str, Any]:
    """A valid configuration dictionary with temporary paths."""

    return {
        "source": {
            "type": "file",
            "path": str(tmp_path / "input.mp4"),
        },
        "frame_skip": 3,
        "detector": {
            "model_path": "models/yolov8n-face.pt",
            "conf_threshold": 0.5,
            "imgsz": 1600,
            "min_face_size": 40,
            "device": "auto",
        },
        "recognition": {
            "model_name": "buffalo_l",
            "model_dir": str(tmp_path / "insightface"),
            "device": "auto",
            "similarity_threshold": 0.45,
            "min_face_size": 60,
            "min_confidence": 0.6,
            "blur_threshold": 60.0,
            "register_quality_threshold": 0.65,
            "max_embeddings_per_face": 3,
            "retry_interval_sec": 1.0,
        },
        "tracking": {
            "tracker": "bytetrack",
            "track_activation_threshold": 0.25,
            "minimum_matching_threshold": 0.8,
            "confirm_hits": 3,
            "exit_timeout_frames": 75,
            "lost_track_buffer": 30,
        },
        "storage": {
            "db_path": str(tmp_path / "faces.db"),
            "log_dir": str(tmp_path / "logs"),
            "image_size": 256,
            "jpeg_quality": 90,
        },
        "logging": {
            "level": "INFO",
            "max_bytes": 1_048_576,
            "backup_count": 2,
        },
    }


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    """Return a helper that writes a configuration dictionary to disk."""

    def _write(data: dict[str, Any]) -> Path:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Generate a small deterministic MP4 with ffmpeg, or skip if absent."""

    ffmpeg = shutil.which("ffmpeg")

    def _make(
        name: str,
        *,
        frames: int = 10,
        fps: int = 5,
        size: str = "160x120",
    ) -> Path:
        if ffmpeg is None:
            pytest.skip("ffmpeg is not available")
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={size}:rate={fps}",
                "-frames:v",
                str(frames),
                "-c:v",
                "mpeg4",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        return path

    return _make
