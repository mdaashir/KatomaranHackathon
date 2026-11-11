"""Tests for the read-only dashboard builder."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.dashboard import DashboardBuilder
from src.storage import Database, EventRepository, FaceRepository

UTC = timezone.utc
BASE = datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)


def _vector() -> bytes:
    return np.zeros(512, dtype=np.float32).tobytes()


def _image(path: Path) -> Path:
    path.write_bytes(b"\xff\xd8\xff\xd9")
    return path


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "faces.db")
    database.initialize()
    yield database
    database.close()


@pytest.fixture
def seeded(db, tmp_path):
    faces = FaceRepository(db)
    events = EventRepository(db)
    entry_one = _image(tmp_path / "entry-one.jpg")
    exit_one = _image(tmp_path / "exit-one.jpg")
    entry_two = _image(tmp_path / "entry-two.jpg")
    exit_two = _image(tmp_path / "exit-two.jpg")

    first = faces.create(
        vector=_vector(),
        dim=512,
        quality=0.9,
        model_name="buffalo_l",
        now=BASE,
    )
    second = faces.create(
        vector=_vector(),
        dim=512,
        quality=0.8,
        model_name="buffalo_l",
        now=BASE + timedelta(seconds=10),
    )

    first_visit, _ = events.create_visit_with_entry(
        face_id=first,
        track_id=1,
        source_id="clip<script>.mp4",
        entry_at=BASE,
        is_anonymous=False,
        entry_image_path=str(entry_one),
        occurred_at=BASE,
        video_time_s=0.0,
        image_path=str(entry_one),
        confidence=0.9,
    )
    events.close_visit_with_exit(
        visit_id=first_visit,
        face_id=first,
        occurred_at=BASE + timedelta(seconds=3),
        video_time_s=3.0,
        source_id="clip<script>.mp4",
        image_path=str(exit_one),
        confidence=0.8,
    )

    second_visit, _ = events.create_visit_with_entry(
        face_id=second,
        track_id=2,
        source_id="clip.mp4",
        entry_at=BASE + timedelta(seconds=10),
        is_anonymous=False,
        entry_image_path=str(entry_two),
        occurred_at=BASE + timedelta(seconds=10),
        video_time_s=10.0,
        image_path=str(entry_two),
        confidence=0.7,
    )
    events.close_visit_with_exit(
        visit_id=second_visit,
        face_id=second,
        occurred_at=BASE + timedelta(seconds=12),
        video_time_s=12.0,
        source_id="clip.mp4",
        image_path=str(exit_two),
        confidence=0.6,
    )

    events.create_visit_with_entry(
        face_id=None,
        track_id=3,
        source_id="clip.mp4",
        entry_at=BASE + timedelta(seconds=20),
        is_anonymous=True,
        entry_image_path=str(tmp_path / "missing.jpg"),
        occurred_at=BASE + timedelta(seconds=20),
        video_time_s=20.0,
        image_path=str(tmp_path / "missing.jpg"),
        confidence=0.5,
    )
    return db


def test_builds_summary_and_recent_rows(seeded, tmp_path):
    data = DashboardBuilder(
        seeded,
        tmp_path / "dashboard",
        max_events=10,
        max_visits=10,
    ).build()

    assert data.summary.unique_faces == 2
    assert data.summary.visits == 3
    assert data.summary.open_visits == 1
    assert data.summary.anonymous_visits == 1
    assert data.summary.entries == 3
    assert data.summary.exits == 2
    assert len(data.events) == 5
    assert len(data.visits) == 3

    page = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "Unique faces" in page
    assert "Face 1" in page
    assert "Anonymous" in page
    assert "no image" in page
    assert "clip&lt;script&gt;.mp4" in page
    assert "<script>" not in page


def test_copies_existing_images(seeded, tmp_path):
    data = DashboardBuilder(
        seeded,
        tmp_path / "dashboard",
    ).build()

    copied = sorted((tmp_path / "dashboard" / "images").glob("*.jpg"))
    assert len(copied) == 4
    assert data.events[0].thumbnail is None
    assert all(event.thumbnail for event in data.events[1:])
    page = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "images/event_" in page


def test_limits_recent_rows(seeded, tmp_path):
    data = DashboardBuilder(
        seeded,
        tmp_path / "dashboard",
        max_events=2,
        max_visits=1,
    ).build()

    assert len(data.events) == 2
    assert len(data.visits) == 1


def test_rejects_invalid_limits(seeded, tmp_path):
    with pytest.raises(ValueError, match="max_events"):
        DashboardBuilder(seeded, tmp_path / "dashboard", max_events=0)
    with pytest.raises(ValueError, match="max_visits"):
        DashboardBuilder(seeded, tmp_path / "dashboard", max_visits=0)


def test_empty_database_renders_empty_states(db, tmp_path):
    data = DashboardBuilder(db, tmp_path / "dashboard").build()

    page = (tmp_path / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert data.summary.unique_faces == 0
    assert data.events == ()
    assert data.visits == ()
    assert "No events recorded yet." in page
    assert "No visits recorded yet." in page


def test_preserves_unrelated_files_in_images(seeded, tmp_path):
    images = tmp_path / "dashboard" / "images"
    images.mkdir(parents=True)
    keep = images / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    stale = images / "event_old.jpg"
    stale.write_bytes(b"old")

    DashboardBuilder(seeded, tmp_path / "dashboard").build()

    assert keep.is_file()
    assert not stale.exists()
