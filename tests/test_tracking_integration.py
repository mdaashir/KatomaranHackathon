"""End-to-end seam test: ByteTrack output into the visit lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config.models import TrackingConfig
from src.detection.types import FaceDetection
from src.tracking import ByteTrackAdapter, TrackLifecycle

BASE_TIME = datetime(2025, 6, 20, 18, 39, 3, tzinfo=timezone.utc)


def test_bytetrack_and_lifecycle_emit_one_entry_and_one_exit():
    config = TrackingConfig(
        tracker="bytetrack",
        confirm_hits=1,
        exit_timeout_frames=12,
        lost_track_buffer=30,
    )
    adapter = ByteTrackAdapter(config, fps=25.0, frame_skip=1)
    lifecycle = TrackLifecycle(config)
    events = []

    def step(frame_index: int, detection: FaceDetection | None) -> None:
        boxes = adapter.update([detection] if detection is not None else [])
        events.extend(
            lifecycle.observe(
                boxes,
                frame_index=frame_index,
                timestamp=BASE_TIME + timedelta(seconds=frame_index / 25.0),
                video_time_s=frame_index / 25.0,
                source_id="clip.mp4",
                window_id="clip.mp4",
            )
        )

    for frame_index in range(6):
        step(
            frame_index,
            FaceDetection(
                bbox=(
                    100.0 + frame_index * 3,
                    100.0,
                    160.0 + frame_index * 3,
                    170.0,
                ),
                confidence=0.9,
            ),
        )

    for frame_index in range(6, 20):
        step(frame_index, None)

    events.extend(lifecycle.flush())

    entries = [event for event in events if event.event_type == "entry"]
    exits = [event for event in events if event.event_type == "exit"]
    assert len(entries) == 1
    assert len(exits) == 1
    assert entries[0].visit_id == exits[0].visit_id
    assert lifecycle.summary.open_visits == 0
