"""Tests for the event logger: images, DB transactions and events.log."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.models import StorageConfig
from src.events import EventLogError, EventLogger, ImageStore
from src.logging_setup import (
    configure_logging,
    shutdown_logging,
)
from src.storage import (
    Database,
    EventRepository,
    FaceRepository,
    VisitRepository,
)
from src.tracking.types import VisitEvent

UTC = timezone.utc
BASE = datetime(2025, 6, 20, 18, 39, 4, tzinfo=UTC)


def _storage(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        db_path=str(tmp_path / "faces.db"),
        log_dir=str(tmp_path / "images"),
        image_size=64,
        jpeg_quality=85,
    )


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "faces.db")
    database.initialize()
    faces = FaceRepository(database)
    for _ in range(2):
        faces.create(
            vector=np.zeros(512, dtype=np.float32).tobytes(),
            dim=512,
            quality=0.8,
            model_name="buffalo_l",
            now=BASE,
        )
    yield database
    database.close()


@pytest.fixture
def logging_setup(tmp_path):
    setup = configure_logging(tmp_path / "logfiles", level="INFO")
    yield setup
    shutdown_logging()


@pytest.fixture
def logger(db, tmp_path, logging_setup):
    storage = _storage(tmp_path)
    return EventLogger(
        db,
        ImageStore(storage),
        events_logger=logging_setup.events_logger,
    )


def _event(
    event_type: str = "entry",
    *,
    visit_id: int = 1,
    face_id: int | None = 1,
    is_anonymous: bool = False,
    seconds: float = 0.0,
    track_id: int = 7,
) -> VisitEvent:
    return VisitEvent(
        event_type=event_type,
        visit_id=visit_id,
        track_id=track_id,
        face_id=face_id,
        frame_index=int(seconds * 25),
        occurred_at=BASE + timedelta(seconds=seconds),
        video_time_s=seconds,
        source_id="clip.mp4",
        window_id="clip.mp4",
        bbox=(10.0, 20.0, 80.0, 100.0),
        confidence=0.9,
        is_anonymous=is_anonymous,
    )


def _crop() -> np.ndarray:
    return np.full((60, 50, 3), 120, dtype=np.uint8)


def _flush_log(logging_setup) -> str:
    for handler in logging_setup.events_logger.handlers:
        handler.flush()
    return logging_setup.events_log_path.read_text(encoding="utf-8")


def test_entry_persists_image_visit_event_and_log(db, logger, logging_setup):
    event_id = logger.handle(_event(), _crop())

    visit = VisitRepository(db).get(1)
    events = EventRepository(db).list_for_visit(1)
    assert event_id == 1
    assert visit.status == "open"
    assert visit.face_id == 1
    assert visit.entry_image_path is not None
    assert Path(visit.entry_image_path).is_file()
    assert [item.event_type for item in events] == ["entry"]
    assert events[0].image_path == visit.entry_image_path
    assert logger.visit_mapping == {1: 1}

    text = _flush_log(logging_setup)
    assert "| ENTRY |" in text
    assert "visit=1" in text
    assert "face=1" in text
    assert "track=7" in text


def test_exit_closes_visit_and_writes_exit_image(db, logger):
    logger.handle(_event(), _crop())
    event_id = logger.handle(_event("exit", seconds=3.0), _crop())

    visit = VisitRepository(db).get(1)
    events = EventRepository(db).list_for_visit(1)
    assert event_id == 2
    assert visit.status == "closed"
    assert visit.exit_at == "2025-06-20T18:39:07.000Z"
    assert visit.exit_image_path is not None
    assert Path(visit.exit_image_path).is_file()
    assert [item.event_type for item in events] == ["entry", "exit"]
    assert EventRepository(db).count_by_type("entry") == 1
    assert EventRepository(db).count_by_type("exit") == 1


def test_anonymous_entry_can_be_linked_to_a_face(db, logger):
    logger.handle(_event(face_id=None, is_anonymous=True), _crop())

    assert logger.link_face(1, 2) is True

    visit = VisitRepository(db).get(1)
    entry = EventRepository(db).list_for_visit(1)[0]
    assert visit.face_id == 2
    assert visit.is_anonymous is False
    assert entry.face_id == 2
    assert entry.is_anonymous is False


def test_exit_automatically_links_a_recognized_anonymous_visit(db, logger):
    logger.handle(_event(face_id=None, is_anonymous=True), _crop())
    logger.handle(
        _event("exit", face_id=2, is_anonymous=False, seconds=2.0),
        _crop(),
    )

    entry = EventRepository(db).list_for_visit(1)[0]
    visit = VisitRepository(db).get(1)
    assert entry.face_id == 2
    assert entry.is_anonymous is False
    assert visit.face_id == 2
    assert visit.status == "closed"


def test_exit_does_not_relink_an_already_linked_face(db, logger, logging_setup):
    logger.handle(_event(face_id=1), _crop())
    logger.handle(_event("exit", face_id=1, seconds=1.0), _crop())

    text = _flush_log(logging_setup)
    assert "| EXIT |" in text
    assert "VISIT_LINKED" not in text


def test_exit_with_conflicting_face_is_rejected(db, logger):
    logger.handle(_event(face_id=1), _crop())

    with pytest.raises(EventLogError, match="linked to Face ID"):
        logger.handle(_event("exit", face_id=2, seconds=1.0), _crop())


def test_missing_crop_writes_a_placeholder(db, logger):
    logger.handle(_event(), None)

    visit = VisitRepository(db).get(1)
    image = cv2.imread(visit.entry_image_path)
    assert image.shape == (64, 64, 3)


def test_exit_without_entry_is_rejected(logger):
    with pytest.raises(EventLogError, match="no entry"):
        logger.handle(_event("exit"), _crop())


def test_duplicate_entry_is_rejected(logger):
    logger.handle(_event(), _crop())

    with pytest.raises(EventLogError, match="already has an entry"):
        logger.handle(_event(), _crop())


def test_db_failure_removes_the_written_image(db, tmp_path, logging_setup):
    class _FailingRepository:
        def create_visit_with_entry(self, **kwargs):
            raise RuntimeError("db down")

    logger = EventLogger(
        db,
        ImageStore(_storage(tmp_path)),
        event_repository=_FailingRepository(),
        events_logger=logging_setup.events_logger,
    )

    with pytest.raises(RuntimeError, match="db down"):
        logger.handle(_event(), _crop())

    assert list((tmp_path / "images").rglob("*.jpg")) == []


def test_exit_db_failure_removes_the_exit_image(db, tmp_path, logging_setup):
    image_store = ImageStore(_storage(tmp_path))
    real = EventRepository(db)

    class _ExitFailRepository:
        def create_visit_with_entry(self, **kwargs):
            return real.create_visit_with_entry(**kwargs)

        def visit_face(self, visit_id):
            return real.visit_face(visit_id)

        def link_face(self, visit_id, face_id):
            return real.link_face(visit_id, face_id)

        def close_visit_with_exit(self, **kwargs):
            raise RuntimeError("exit db down")

    logger = EventLogger(
        db,
        image_store,
        event_repository=_ExitFailRepository(),
        events_logger=logging_setup.events_logger,
    )
    logger.handle(_event(), _crop())

    with pytest.raises(RuntimeError, match="exit db down"):
        logger.handle(_event("exit", seconds=1.0), _crop())

    assert list((tmp_path / "images" / "exits").rglob("*.jpg")) == []


def test_visit_mapping_is_isolated_per_logger(db, tmp_path, logging_setup):
    first = EventLogger(
        db,
        ImageStore(_storage(tmp_path)),
        events_logger=logging_setup.events_logger,
    )
    first.handle(_event(), _crop())

    second = EventLogger(
        db,
        ImageStore(_storage(tmp_path)),
        events_logger=logging_setup.events_logger,
    )
    assert second.visit_mapping == {}
