"""Tests for the per-track state machine and visit lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.config.models import TrackingConfig
from src.tracking import TrackLifecycle, TrackState, TrackedBox

UTC = timezone.utc
BASE_TIME = datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)


def _config(**overrides) -> TrackingConfig:
    values = {
        "tracker": "bytetrack",
        "confirm_hits": 3,
        "exit_timeout_frames": 12,
        "lost_track_buffer": 30,
    }
    values.update(overrides)
    return TrackingConfig(**values)


def _box(track_id: int = 1, x: float = 100.0) -> TrackedBox:
    return TrackedBox(
        track_id=track_id,
        bbox=(x, 100.0, x + 60.0, 170.0),
        confidence=0.9,
    )


def _observe(
    lifecycle: TrackLifecycle,
    boxes,
    frame: int,
    *,
    face_ids=None,
    window_id: str = "clip",
):
    return lifecycle.observe(
        boxes,
        frame_index=frame,
        timestamp=BASE_TIME + timedelta(seconds=frame / 25.0),
        video_time_s=frame / 25.0,
        source_id="clip.mp4",
        window_id=window_id,
        face_ids=face_ids,
    )


def test_entry_after_confirm_hits():
    lifecycle = TrackLifecycle(_config(confirm_hits=3))

    assert _observe(lifecycle, [_box()], 0) == []
    assert _observe(lifecycle, [_box()], 3) == []
    events = _observe(lifecycle, [_box()], 6)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "entry"
    assert event.track_id == 1
    assert event.visit_id == 1
    assert event.frame_index == 0
    assert event.occurred_at == BASE_TIME
    assert event.is_anonymous is True
    assert lifecycle.tracks[1].state == TrackState.ACTIVE
    assert lifecycle.summary.open_visits == 1
    assert lifecycle.summary.entry_events == 1


def test_one_frame_false_positive_produces_no_events():
    lifecycle = TrackLifecycle(_config())

    _observe(lifecycle, [_box()], 0)
    _observe(lifecycle, [], 3)
    events = lifecycle.flush()

    assert events == []
    assert lifecycle.tracks[1].state == TrackState.DROPPED
    assert lifecycle.visits == {}


def test_gap_before_confirmation_keeps_counting_hits():
    lifecycle = TrackLifecycle(_config(confirm_hits=3, exit_timeout_frames=12))

    assert _observe(lifecycle, [_box()], 0) == []
    assert _observe(lifecycle, [_box()], 3) == []
    assert _observe(lifecycle, [], 6) == []
    assert lifecycle.tracks[1].state == TrackState.NEW
    assert lifecycle.tracks[1].hits == 2

    events = _observe(lifecycle, [_box()], 9)

    assert len(events) == 1
    assert events[0].event_type == "entry"
    assert events[0].frame_index == 0
    assert lifecycle.tracks[1].state == TrackState.ACTIVE


def test_exit_after_timeout_uses_last_seen_timestamp():
    lifecycle = TrackLifecycle(_config(exit_timeout_frames=12))
    _observe(lifecycle, [_box()], 0)
    _observe(lifecycle, [_box()], 3)
    _observe(lifecycle, [_box()], 6)

    assert _observe(lifecycle, [], 9) == []
    assert _observe(lifecycle, [], 12) == []
    assert _observe(lifecycle, [], 15) == []
    events = _observe(lifecycle, [], 18)

    assert len(events) == 1
    exit_event = events[0]
    assert exit_event.event_type == "exit"
    assert exit_event.frame_index == 6
    assert exit_event.occurred_at == BASE_TIME + timedelta(seconds=6 / 25.0)
    assert lifecycle.tracks[1].state == TrackState.EXITED
    assert lifecycle.summary.open_visits == 0
    assert lifecycle.summary.closed_visits == 1


def test_reacquisition_before_timeout_keeps_the_visit():
    lifecycle = TrackLifecycle(_config(exit_timeout_frames=12))
    for frame in (0, 3, 6):
        _observe(lifecycle, [_box()], frame)

    _observe(lifecycle, [], 9)
    assert lifecycle.tracks[1].state == TrackState.LOST
    assert _observe(lifecycle, [_box()], 12) == []
    assert lifecycle.tracks[1].state == TrackState.ACTIVE
    assert lifecycle.summary.entry_events == 1

    events = _observe(lifecycle, [], 24)
    assert len(events) == 1
    assert events[0].event_type == "exit"
    assert lifecycle.summary.exit_events == 1


def test_reappearance_after_exit_creates_a_new_visit_with_same_face():
    lifecycle = TrackLifecycle(_config(confirm_hits=1, exit_timeout_frames=6))
    _observe(lifecycle, [_box(1)], 0, face_ids={1: 42})
    _observe(lifecycle, [], 6)
    _observe(lifecycle, [], 12)

    events = _observe(lifecycle, [_box(2)], 30, face_ids={2: 42})

    assert len(events) == 1
    assert events[0].event_type == "entry"
    assert events[0].face_id == 42
    assert events[0].visit_id == 2
    assert lifecycle.summary.closed_visits == 1
    assert lifecycle.summary.open_visits == 1


def test_identity_anchoring_attaches_to_an_open_visit():
    lifecycle = TrackLifecycle(_config(confirm_hits=1, exit_timeout_frames=6))
    _observe(lifecycle, [_box(1)], 0, face_ids={1: 42})
    assert lifecycle.summary.entry_events == 1

    events = _observe(
        lifecycle,
        [_box(1), _box(2, x=200.0)],
        3,
        face_ids={1: 42, 2: 42},
    )
    assert events == []
    assert len(lifecycle.visits[1].track_ids) == 2

    assert _observe(lifecycle, [_box(2, x=200.0)], 6, face_ids={2: 42}) == []
    assert lifecycle.summary.open_visits == 1

    events = _observe(lifecycle, [], 12)
    assert len(events) == 1
    assert events[0].event_type == "exit"
    assert events[0].visit_id == 1
    assert lifecycle.summary.entry_events == 1
    assert lifecycle.summary.exit_events == 1


def test_window_change_flushes_and_starts_a_new_visit():
    lifecycle = TrackLifecycle(_config(confirm_hits=1, exit_timeout_frames=6))
    _observe(lifecycle, [_box()], 0, window_id="clip-a")

    events = _observe(lifecycle, [_box()], 3, window_id="clip-b")

    assert [event.event_type for event in events] == ["exit", "entry"]
    assert events[0].window_id == "clip-a"
    assert events[1].window_id == "clip-b"
    assert events[1].visit_id == 2
    assert lifecycle.summary.open_visits == 1


def test_end_of_stream_flush_closes_every_open_visit():
    lifecycle = TrackLifecycle(_config(confirm_hits=1))
    _observe(lifecycle, [_box()], 0)

    events = lifecycle.flush()

    assert len(events) == 1
    assert events[0].event_type == "exit"
    assert lifecycle.summary.open_visits == 0
    assert lifecycle.flush() == []


def test_assign_face_to_an_anonymous_visit():
    lifecycle = TrackLifecycle(_config(confirm_hits=1))
    entry_events = _observe(lifecycle, [_box()], 0)
    assert entry_events[0].is_anonymous is True

    assert lifecycle.assign_face(1, 42) is True

    visit = lifecycle.visits[1]
    assert visit.face_id == 42
    assert visit.is_anonymous is False

    exit_events = lifecycle.flush()
    assert exit_events[0].face_id == 42
    assert exit_events[0].is_anonymous is False


def test_assign_face_rejects_unknown_or_conflicting_track():
    lifecycle = TrackLifecycle(_config(confirm_hits=1))
    _observe(lifecycle, [_box()], 0, face_ids={1: 42})

    assert lifecycle.assign_face(999, 7) is False
    assert lifecycle.assign_face(1, 7) is False
    assert lifecycle.visits[1].face_id == 42


def test_multiple_tracks_have_independent_visits():
    lifecycle = TrackLifecycle(_config(confirm_hits=1, exit_timeout_frames=6))
    events = _observe(
        lifecycle,
        [_box(1), _box(2, x=200.0)],
        0,
    )

    assert [event.event_type for event in events] == ["entry", "entry"]
    assert lifecycle.summary.open_visits == 2

    events = _observe(lifecycle, [], 6)
    assert [event.event_type for event in events] == ["exit", "exit"]
    assert lifecycle.summary.open_visits == 0


def test_event_sink_receives_the_same_events():
    received = []
    lifecycle = TrackLifecycle(
        _config(confirm_hits=1),
        event_sink=received.append,
    )

    events = _observe(lifecycle, [_box()], 0)

    assert received == events


def test_unknown_face_lookup_returns_none():
    lifecycle = TrackLifecycle(_config(confirm_hits=1))
    _observe(lifecycle, [_box()], 0, face_ids={1: 42})

    assert lifecycle.open_visit_for_face(42) is not None
    assert lifecycle.open_visit_for_face(999) is None


def test_invalid_observe_arguments():
    lifecycle = TrackLifecycle(_config(confirm_hits=1))

    with pytest.raises(ValueError, match="source_id"):
        lifecycle.observe(
            [],
            frame_index=0,
            timestamp=BASE_TIME,
            video_time_s=0.0,
            source_id="",
            window_id="clip",
        )
    with pytest.raises(ValueError, match="window_id"):
        lifecycle.observe(
            [],
            frame_index=0,
            timestamp=BASE_TIME,
            video_time_s=0.0,
            source_id="clip.mp4",
            window_id="",
        )
