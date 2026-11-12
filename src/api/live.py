"""Latest-frame buffer for the MJPEG live view.

The pipeline callback only copies a frame when a viewer is attached, so the
headless pipeline pays nothing when nobody is watching.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Iterator, Sequence

import cv2
import numpy as np

from src.tracking import TrackedBox
from src.video import Frame


@dataclass(frozen=True)
class LiveFrame:
    image: np.ndarray
    boxes: tuple[TrackedBox, ...]
    face_ids: dict[int, int | None]
    captured_at: datetime


class LiveFrameBuffer:
    """Keep only the newest detection frame for MJPEG consumers."""

    def __init__(self, *, jpeg_quality: int = 80) -> None:
        self._condition = threading.Condition()
        self._viewers = 0
        self._latest: LiveFrame | None = None
        self._jpeg_quality = jpeg_quality

    @property
    def has_viewers(self) -> bool:
        with self._condition:
            return self._viewers > 0

    def acquire(self) -> None:
        with self._condition:
            self._viewers += 1

    def release(self) -> None:
        with self._condition:
            self._viewers = max(0, self._viewers - 1)
            if self._viewers == 0:
                self._latest = None
            self._condition.notify_all()

    def publish(
        self,
        frame: Frame,
        boxes: Sequence[TrackedBox],
        face_ids: dict[int, int | None],
    ) -> None:
        with self._condition:
            if self._viewers <= 0:
                return
            self._latest = LiveFrame(
                image=frame.image.copy(),
                boxes=tuple(boxes),
                face_ids=dict(face_ids),
                captured_at=datetime.now(timezone.utc),
            )
            self._condition.notify_all()

    def wait(self, timeout: float = 1.0) -> LiveFrame | None:
        with self._condition:
            self._condition.wait(timeout)
            return self._latest

    def encode(self, live: LiveFrame) -> bytes:
        image = live.image.copy()
        for box in live.boxes:
            x1, y1, x2, y2 = (int(value) for value in box.bbox)
            face_id = live.face_ids.get(box.track_id)
            color = (0, 200, 0) if face_id is not None else (0, 165, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"T{box.track_id}"
            if face_id is not None:
                label += f" F{face_id}"
            cv2.putText(
                image,
                label,
                (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        return encoded.tobytes() if ok else b""

    def mjpeg(self) -> Iterator[bytes]:
        self.acquire()
        try:
            while True:
                live = self.wait(1.0)
                if live is None:
                    continue
                payload = self.encode(live)
                if not payload:
                    continue
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode()
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
        finally:
            self.release()
