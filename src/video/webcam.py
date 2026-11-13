"""Local webcam source for real-time host demos."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator
import sys
import time

import cv2

from src.video.types import Frame, VideoSourceError

CaptureFactory = Callable[[int, int], Any]


class WebcamSource:
    """Read frames from a local camera with OpenCV.

    The API process must run on the machine that owns the camera. On Linux in
    Docker, pass the device through explicitly. The camera read is blocking at
    its native rate; the pipeline's frame sink handles viewer delivery.
    """

    def __init__(
        self,
        device_index: int = 0,
        *,
        capture_factory: CaptureFactory | None = None,
        source_id: str | None = None,
    ) -> None:
        if device_index < 0:
            raise ValueError("device_index must not be negative")

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        factory = capture_factory or cv2.VideoCapture
        try:
            self._capture = factory(device_index, backend)
        except Exception as exc:
            raise VideoSourceError(
                f"webcam {device_index}: cannot open camera: {exc}"
            ) from exc
        if not self._capture.isOpened():
            self._capture.release()
            raise VideoSourceError(
                f"webcam {device_index}: camera not found or already in use"
            )

        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if 1.0 <= fps <= 240.0 else 25.0
        self._device_index = device_index
        self._source_id = source_id or f"webcam:{device_index}"
        self._closed = False
        self._started = False
        self._index = 0
        self._first_monotonic: float | None = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def fps(self) -> float:
        return self._fps

    def frames(self) -> Iterator[Frame]:
        if self._started:
            raise VideoSourceError(f"{self._source_id}: source is single-use")
        if self._closed:
            raise VideoSourceError(f"{self._source_id}: source is closed")
        self._started = True

        while not self._closed:
            ok, image = self._capture.read()
            if not ok or image is None:
                if self._closed:
                    return
                raise VideoSourceError(f"{self._source_id}: camera frame unavailable")

            now = datetime.now(timezone.utc)
            monotonic = time.monotonic()
            if self._first_monotonic is None:
                self._first_monotonic = monotonic
            index = self._index
            self._index += 1
            yield Frame(
                index=index,
                global_index=index,
                timestamp_s=monotonic - self._first_monotonic,
                absolute_time=now,
                image=image,
                source_id=self._source_id,
                window_id=self._source_id,
                is_window_end=False,
                fps=self._fps,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._capture.release()

    def __enter__(self) -> WebcamSource:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
