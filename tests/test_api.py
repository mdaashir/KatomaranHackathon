"""Contract tests for the read-only FastAPI layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api import create_app
from src.storage import Database, EventRepository, FaceRepository

UTC = timezone.utc
BASE = datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)


def _image(path: Path) -> str:
    path.write_bytes(b"\xff\xd8\xff\xd9")
    return str(path)


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "faces.db"
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    database = Database(db_path)
    database.initialize()

    faces = FaceRepository(database)
    events = EventRepository(database)
    face_id = faces.create(
        vector=np.zeros(512, dtype=np.float32).tobytes(),
        dim=512,
        quality=0.9,
        model_name="buffalo_l",
        now=BASE,
    )
    entry = _image(log_dir / "entry.jpg")
    exit_image = _image(log_dir / "exit.jpg")
    visit_id, _ = events.create_visit_with_entry(
        face_id=face_id,
        track_id=7,
        source_id="clip.mp4",
        entry_at=BASE,
        is_anonymous=False,
        entry_image_path=entry,
        occurred_at=BASE,
        video_time_s=0.0,
        image_path=entry,
        confidence=0.9,
    )
    events.close_visit_with_exit(
        visit_id=visit_id,
        face_id=face_id,
        occurred_at=BASE + timedelta(seconds=3),
        video_time_s=3.0,
        source_id="clip.mp4",
        image_path=exit_image,
        confidence=0.8,
    )
    database.close()

    (log_dir / "events.log").write_text(
        "2025-06-20T18:39:03.000Z | FACE_DETECTED | track=7 frame=0\n"
        "2025-06-20T18:39:03.000Z | ENTRY | visit=1\n"
        "2025-06-20T18:39:04.000Z | EXIT | visit=1\n",
        encoding="utf-8",
    )

    return create_app(db_path=db_path, log_dir=log_dir)


def test_health_and_stats(app):
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["integrity"] == "ok"

    stats = client.get("/stats").json()
    assert stats["unique_faces"] == 1
    assert stats["entries"] == 1
    assert stats["exits"] == 1
    assert stats["open_visits"] == 0
    assert stats["currently_in_frame"] == 0


def test_events_and_filters(app):
    client = TestClient(app)

    events = client.get("/events").json()
    assert events["total"] == 2
    assert len(events["items"]) == 2
    assert events["items"][0]["event_type"] == "exit"
    assert events["items"][0]["image_url"].startswith("/images/")

    entries = client.get("/events", params={"event_type": "entry"}).json()
    assert entries["total"] == 1
    assert entries["items"][0]["event_type"] == "entry"

    by_date = client.get("/events", params={"date": "2025-06-20"}).json()
    assert by_date["total"] == 2

    by_face = client.get("/events", params={"face_id": 1}).json()
    assert by_face["total"] == 2


def test_event_image_is_served(app):
    client = TestClient(app)
    image_url = client.get("/events").json()["items"][0]["image_url"]

    response = client.get(image_url)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/jpeg")


def test_image_traversal_is_rejected(app):
    client = TestClient(app)

    assert client.get("/images/../faces.db").status_code == 404
    assert client.get("/images/%2e%2e/faces.db").status_code == 404


def test_log_tail(app):
    client = TestClient(app)

    data = client.get("/logs/tail", params={"lines": 2}).json()
    assert len(data["lines"]) == 2
    assert "EXIT" in data["lines"][-1]

    filtered = client.get("/logs/tail", params={"text": "ENTRY"}).json()
    assert filtered["lines"] == ["2025-06-20T18:39:03.000Z | ENTRY | visit=1"]


def test_missing_database_is_empty(tmp_path):
    app = create_app(
        db_path=tmp_path / "missing.db",
        log_dir=tmp_path / "logs",
    )
    client = TestClient(app)

    stats = client.get("/stats").json()
    assert stats["unique_faces"] == 0
    assert stats["entries"] == 0
    assert client.get("/health").json()["status"] == "degraded"
    assert client.get("/events").json()["items"] == []
