"""Tests for the local webcam source using a fake OpenCV capture."""

from __future__ import annotations

import numpy as np
import pytest

from src.video import VideoSourceError, WebcamSource


class FakeCapture:
    def __init__(self, *, opened: bool = True, frames: int = 3) -> None:
        self._opened = opened
        self._remaining = frames
        self.released = False
        self.settings: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop: int, value: float) -> bool:
        self.settings.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        return 30.0

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_webcam_source_reads_frames_and_releases():
    capture = FakeCapture(frames=3)
    source = WebcamSource(2, capture_factory=lambda index, backend: capture)
    iterator = source.frames()

    frames = [next(iterator) for _ in range(3)]
    source.close()

    assert [frame.index for frame in frames] == [0, 1, 2]
    assert frames[0].source_id == "webcam:2"
    assert frames[0].image.shape == (48, 64, 3)
    assert frames[0].fps == 30.0
    assert capture.released is True
    assert capture.settings


def test_webcam_source_rejects_missing_camera():
    capture = FakeCapture(opened=False)

    with pytest.raises(VideoSourceError, match="not found"):
        WebcamSource(0, capture_factory=lambda index, backend: capture)
    assert capture.released is True


def test_webcam_source_is_single_use():
    capture = FakeCapture(frames=2)
    source = WebcamSource(0, capture_factory=lambda index, backend: capture)
    next(source.frames())

    with pytest.raises(VideoSourceError, match="single-use"):
        list(source.frames())
