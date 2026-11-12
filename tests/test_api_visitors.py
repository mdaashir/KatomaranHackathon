"""Visitor gallery and system snapshot contract tests."""

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
def client(tmp_path):
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
    return TestClient(
        create_app(
            db_path=db_path,
            log_dir=log_dir,
            input_root=tmp_path,
        )
    )


def test_visitors_list_and_detail(client):
    listing = client.get("/visitors").json()
    assert listing["total"] == 1
    visitor = listing["items"][0]
    assert visitor["face_id"] == 1
    assert visitor["visit_count"] == 1
    assert visitor["thumbnail_url"].startswith("/images/")

    detail = client.get("/visitors/1").json()
    assert detail["face_id"] == 1
    assert detail["visits"][0]["status"] == "closed"
    assert detail["visits"][0]["entry_image_url"].startswith("/images/")
    assert detail["visits"][0]["exit_image_url"].startswith("/images/")

    assert client.get("/visitors/999").status_code == 404


def test_system_snapshot(client):
    data = client.get("/system").json()
    assert data["cpu"]["percent"] >= 0
    assert data["memory"]["total_bytes"] > 0
    assert data["database"]["size_bytes"] > 0
    assert data["models"]["recognition"] == "buffalo_l"
    assert "providers" in data["gpu"]
    assert data["uptime_seconds"] >= 0
