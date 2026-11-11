"""Small PyAV helpers shared by file and RTSP sources."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.video.types import VideoSourceError


def video_stream(container: Any, source_id: str) -> Any:
    """Return the first video stream or raise a source error."""

    try:
        return container.streams.video[0]
    except (IndexError, AttributeError) as exc:
        raise VideoSourceError(f"{source_id}: no video stream found") from exc


def stream_fps(stream: Any, default: float = 25.0) -> float:
    """Return a positive frame rate, falling back to *default*."""

    rate = getattr(stream, "average_rate", None)
    if rate is None:
        rate = getattr(stream, "rate", None)
    try:
        fps = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return default
    return fps if fps > 0 else default


def frame_to_bgr(raw: Any) -> np.ndarray:
    """Convert a PyAV frame to a NumPy BGR array."""

    return raw.to_ndarray(format="bgr24")


def mtime_utc(path: Path) -> datetime:
    """Return the file modification time in UTC."""

    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
