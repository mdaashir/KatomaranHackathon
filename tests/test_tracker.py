"""Tests for the ByteTrack adapter."""

from __future__ import annotations

import pytest

from src.config.models import TrackingConfig
from src.detection.types import FaceDetection
from src.tracking import ByteTrackAdapter


def _config() -> TrackingConfig:
    return TrackingConfig(
        tracker="bytetrack",
        track_activation_threshold=0.25,
        minimum_matching_threshold=0.8,
        confirm_hits=3,
        exit_timeout_frames=75,
        lost_track_buffer=30,
    )


def _detection(x: float, y: float = 100.0, confidence: float = 0.9):
    return FaceDetection(
        bbox=(x, y, x + 60, y + 70),
        confidence=confidence,
    )


def test_frame_rate_uses_detection_cadence():
    adapter = ByteTrackAdapter(_config(), fps=25.0, frame_skip=3)
    assert adapter.frame_rate == 8


def test_keeps_id_for_a_moving_detection():
    adapter = ByteTrackAdapter(_config(), fps=25.0, frame_skip=1)
    ids = []
    for index in range(6):
        boxes = adapter.update([_detection(100.0 + index * 5)])
        if boxes:
            ids.append(boxes[0].track_id)

    assert ids
    assert len(set(ids)) == 1


def test_empty_detection_returns_empty():
    adapter = ByteTrackAdapter(_config(), fps=25.0, frame_skip=1)
    assert adapter.update([]) == []


def test_reacquires_after_a_short_gap():
    adapter = ByteTrackAdapter(_config(), fps=25.0, frame_skip=1)
    first = adapter.update([_detection(100.0)])
    adapter.update([])
    second = adapter.update([_detection(102.0)])

    assert first and second
    assert first[0].track_id == second[0].track_id


def test_reset_restarts_the_tracker():
    adapter = ByteTrackAdapter(_config(), fps=25.0, frame_skip=1)
    first = adapter.update([_detection(100.0)])
    adapter.reset()
    second = adapter.update([_detection(100.0)])

    assert first and second
    assert second[0].track_id == 1


def test_confidence_is_preserved():
    adapter = ByteTrackAdapter(_config(), fps=25.0, frame_skip=1)
    boxes = adapter.update([_detection(100.0, confidence=0.77)])

    assert boxes[0].confidence == pytest.approx(0.77)


def test_invalid_tracker_rejected():
    with pytest.raises(ValueError, match="unsupported tracker"):
        ByteTrackAdapter(
            TrackingConfig(tracker="deepsort"),
            fps=25.0,
            frame_skip=1,
        )


def test_invalid_cadence_rejected():
    with pytest.raises(ValueError, match="frame_skip"):
        ByteTrackAdapter(_config(), fps=25.0, frame_skip=0)

    with pytest.raises(ValueError, match="fps"):
        ByteTrackAdapter(_config(), fps=0.0, frame_skip=1)
