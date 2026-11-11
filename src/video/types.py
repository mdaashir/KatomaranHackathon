"""Frame and source contracts shared by every video input."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Protocol

import numpy as np


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened, decoded, or iterated."""


@dataclass(frozen=True)
class Frame:
    """One decoded video frame.

    Attributes:
        index: Frame index within the current window (clip or stream).
        global_index: Frame index across the whole source; used for cadence.
        timestamp_s: Seconds from the start of the current window.
        absolute_time: UTC capture time derived from the clip name or clock.
        image: BGR image at native resolution.
        source_id: Stable identifier for the clip or camera.
        window_id: Logical window that tracks must not cross.
        is_window_end: True on the last frame of a file window.
        fps: Source frame rate, used to configure ByteTrack.
    """

    index: int
    global_index: int
    timestamp_s: float
    absolute_time: datetime
    image: np.ndarray
    source_id: str
    window_id: str
    is_window_end: bool = False
    fps: float = 25.0


class VideoSource(Protocol):
    """Read frames from a file, a folder of clips, or an RTSP stream."""

    @property
    def source_id(self) -> str:
        """Stable identifier used in logs and the database."""
        ...

    def frames(self) -> Iterator[Frame]:
        """Yield frames in order. A source is single-use."""
        ...

    def close(self) -> None:
        """Release the decoder and stop any background reader."""
        ...
