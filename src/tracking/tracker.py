"""ByteTrack wrapper configured for detection cadence, not video frame rate."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import supervision as sv

from src.config.models import TrackingConfig
from src.detection.types import FaceDetection
from src.tracking.types import TrackedBox


def _empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        confidence=np.empty((0,), dtype=np.float32),
        class_id=np.empty((0,), dtype=int),
    )


class ByteTrackAdapter:
    """Feed face detections to ByteTrack and return tracked boxes.

    ByteTrack is configured with ``frame_rate = fps / frame_skip`` because it
    is updated once per detection cycle, not once per decoded frame.
    """

    def __init__(
        self,
        config: TrackingConfig,
        *,
        fps: float,
        frame_skip: int,
    ) -> None:
        if config.tracker != "bytetrack":
            raise ValueError(f"unsupported tracker '{config.tracker}'")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        self._config = config
        self._frame_rate = max(1, int(round(fps / frame_skip)))
        self._tracker = self._new_tracker()

    @property
    def frame_rate(self) -> int:
        """Effective detection cycles per second passed to ByteTrack."""

        return self._frame_rate

    def _new_tracker(self) -> sv.ByteTrack:
        return sv.ByteTrack(
            track_activation_threshold=(
                self._config.track_activation_threshold
            ),
            lost_track_buffer=self._config.lost_track_buffer,
            minimum_matching_threshold=(
                self._config.minimum_matching_threshold
            ),
            frame_rate=self._frame_rate,
            minimum_consecutive_frames=1,
        )

    def update(self, detections: Sequence[FaceDetection]) -> list[TrackedBox]:
        """Update the tracker and return boxes that have a tracker ID."""

        if not detections:
            self._tracker.update_with_detections(_empty_detections())
            return []

        xyxy = np.asarray(
            [detection.bbox for detection in detections],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [detection.confidence for detection in detections],
            dtype=np.float32,
        )
        tracked = self._tracker.update_with_detections(
            sv.Detections(xyxy=xyxy, confidence=confidence)
        )
        if tracked.tracker_id is None:
            return []

        boxes: list[TrackedBox] = []
        for index, track_id in enumerate(tracked.tracker_id):
            if track_id is None:
                continue
            confidence_value = (
                float(tracked.confidence[index])
                if tracked.confidence is not None
                else 0.0
            )
            boxes.append(
                TrackedBox(
                    track_id=int(track_id),
                    bbox=tuple(float(value) for value in tracked.xyxy[index]),
                    confidence=confidence_value,
                )
            )
        return boxes

    def reset(self) -> None:
        """Reset tracker IDs, for example at a new video window."""

        self._tracker = self._new_tracker()
