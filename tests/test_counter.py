"""Tests for DB-derived unique counting and run summaries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.counting import UniqueCounter
from src.storage import (
    Database,
    EventRepository,
    FaceRepository,
    VisitRepository,
)

UTC = timezone.utc
BASE = datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)


def _vector() -> bytes:
    return np.zeros(512, dtype=np.float32).tobytes()


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "faces.db")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def faces(db):
    return FaceRepository(db)


@pytest.fixture
def visits(db):
    return VisitRepository(db)


@pytest.fixture
def events(db):
    return EventRepository(db)


def _face(faces: FaceRepository) -> int:
    return faces.create(
        vector=_vector(),
        dim=512,
        quality=0.8,
        model_name="buffalo_l",
        now=BASE,
    )


def _entry(
    events: EventRepository,
    *,
    face_id: int | None,
    track_id: int,
    seconds: float,
    is_anonymous: bool,
) -> tuple[int, int]:
    moment = BASE + timedelta(seconds=seconds)
    return events.create_visit_with_entry(
        face_id=face_id,
        track_id=track_id,
        source_id="clip.mp4",
        entry_at=moment,
        is_anonymous=is_anonymous,
        entry_image_path="entry.jpg",
        occurred_at=moment,
        video_time_s=seconds,
        image_path="entry.jpg",
        confidence=0.9,
    )


def test_empty_summary(db):
    summary = UniqueCounter(db).get_summary()

    assert summary.unique_faces == 0
    assert summary.visits == 0
    assert summary.open_visits == 0
    assert summary.anonymous_visits == 0
    assert summary.entries == 0
    assert summary.exits == 0
    assert summary.first_event_at is None
    assert summary.last_event_at is None


def test_unique_count_is_derived_from_faces(db, faces, events):
    first = _face(faces)
    second = _face(faces)
    _entry(events, face_id=first, track_id=1, seconds=0, is_anonymous=False)
    _entry(events, face_id=second, track_id=2, seconds=1, is_anonymous=False)
    _entry(events, face_id=None, track_id=3, seconds=2, is_anonymous=True)

    counter = UniqueCounter(db)
    summary = counter.get_summary()

    assert counter.get_unique_count() == 2
    assert summary.unique_faces == 2
    assert summary.visits == 3
    assert summary.open_visits == 3
    assert summary.anonymous_visits == 1
    assert summary.entries == 3
    assert summary.exits == 0
    assert summary.first_event_at is not None
    assert summary.last_event_at is not None


def test_clean_run_has_matching_entries_and_exits(db, faces, events):
    face_id = _face(faces)
    visit_id, _ = _entry(
        events,
        face_id=face_id,
        track_id=1,
        seconds=0,
        is_anonymous=False,
    )
    events.close_visit_with_exit(
        visit_id=visit_id,
        face_id=face_id,
        occurred_at=BASE + timedelta(seconds=3),
        video_time_s=3.0,
        source_id="clip.mp4",
        image_path="exit.jpg",
        confidence=0.9,
    )

    summary = UniqueCounter(db).get_summary()

    assert summary.entries == 1
    assert summary.exits == 1
    assert summary.open_visits == 0
    assert summary.visits == 1


def test_deactivated_face_is_not_counted(db, faces):
    face_id = _face(faces)
    assert UniqueCounter(db).get_unique_count() == 1

    faces.deactivate(face_id)

    assert UniqueCounter(db).get_unique_count() == 0


def test_count_survives_database_reopen(tmp_path):
    path = tmp_path / "faces.db"
    first = Database(path)
    first.initialize()
    _face(FaceRepository(first))
    first.close()

    second = Database(path)
    second.initialize()
    try:
        assert UniqueCounter(second).get_unique_count() == 1
    finally:
        second.close()


def test_summary_serializes_to_dict(db):
    data = UniqueCounter(db).get_summary().to_dict()

    assert data["unique_faces"] == 0
    assert set(data) == {
        "unique_faces",
        "visits",
        "open_visits",
        "anonymous_visits",
        "entries",
        "exits",
        "first_event_at",
        "last_event_at",
    }
