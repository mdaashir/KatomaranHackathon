"""Tests for the SQLite storage layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
import struct

import pytest

from src.storage import (
    Database,
    DuplicateEventError,
    EventRepository,
    FaceRepository,
    StorageError,
    VisitRepository,
    utc_iso,
)

NOW = datetime(2025, 6, 20, 18, 39, 3, tzinfo=timezone.utc)


def _vector(values: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


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


def _create_face(faces: FaceRepository, quality: float = 0.8) -> int:
    return faces.create(
        vector=_vector(),
        dim=4,
        quality=quality,
        model_name="buffalo_l",
        now=NOW,
    )


def _create_visit(visits: VisitRepository, face_id: int | None) -> int:
    return visits.create(
        face_id=face_id,
        track_id=7,
        source_id="clip.mp4",
        entry_at=NOW,
        is_anonymous=face_id is None,
    )


def test_schema_initializes_and_uses_wal(db):
    connection = db.connect()
    tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }

    assert {"faces", "embeddings", "visits", "events"} <= tables
    assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert db.integrity_check() == "ok"


def test_create_face_and_count(faces):
    face_id = _create_face(faces)

    assert face_id == 1
    face = faces.get(face_id)
    assert face is not None
    assert face.is_active is True
    assert face.model_name == "buffalo_l"
    assert face.best_quality == pytest.approx(0.8)
    assert faces.count_active() == 1

    embeddings = faces.list_embeddings()
    assert len(embeddings) == 1
    assert embeddings[0][0] == face_id
    assert embeddings[0][2] == 4
    assert len(embeddings[0][1]) == 16


def test_create_face_rejects_wrong_vector_size(faces):
    with pytest.raises(StorageError, match="expected 16 bytes"):
        faces.create(
            vector=b"\x00" * 12,
            dim=4,
            quality=0.8,
            model_name="buffalo_l",
            now=NOW,
        )

    assert faces.count_active() == 0


@pytest.mark.parametrize("quality", [-0.1, 1.1])
def test_face_quality_must_be_in_range(faces, quality):
    with pytest.raises(StorageError, match="quality"):
        faces.create(
            vector=_vector(),
            dim=4,
            quality=quality,
            model_name="buffalo_l",
            now=NOW,
        )

    assert faces.count_active() == 0


@pytest.mark.parametrize("quality", [-0.1, 1.1])
def test_embedding_quality_must_be_in_range(faces, quality):
    face_id = _create_face(faces)

    with pytest.raises(StorageError, match="quality"):
        faces.add_embedding(
            face_id,
            vector=_vector(),
            dim=4,
            quality=quality,
            now=NOW,
        )

    assert len(faces.list_embeddings()) == 1


def test_add_embedding_updates_best_quality(faces):
    face_id = _create_face(faces, quality=0.5)

    embedding_id = faces.add_embedding(
        face_id,
        vector=_vector((0.4, 0.5, 0.6, 0.7)),
        dim=4,
        quality=0.9,
        now=NOW,
    )

    assert embedding_id == 2
    assert faces.get(face_id).best_quality == pytest.approx(0.9)
    assert len(faces.list_embeddings()) == 2


def test_count_embeddings(faces):
    face_id = _create_face(faces)

    assert faces.count_embeddings(face_id) == 1
    faces.add_embedding(
        face_id,
        vector=_vector(),
        dim=4,
        quality=0.9,
        now=NOW,
    )
    assert faces.count_embeddings(face_id) == 2
    assert faces.count_embeddings(999) == 0


def test_add_embedding_unknown_face(faces):
    with pytest.raises(StorageError, match="does not exist"):
        faces.add_embedding(
            999,
            vector=_vector(),
            dim=4,
            quality=0.9,
            now=NOW,
        )


def test_deactivate(faces):
    face_id = _create_face(faces)

    assert faces.deactivate(face_id) is True
    assert faces.count_active() == 0
    assert faces.get(face_id).is_active is False
    assert faces.deactivate(face_id) is False


def test_visit_lifecycle(visits, faces):
    face_id = _create_face(faces)
    visit_id = _create_visit(visits, face_id)

    visit = visits.get(visit_id)
    assert visit is not None
    assert visit.status == "open"
    assert visit.exit_at is None
    assert visit.face_id == face_id
    assert visit.is_anonymous is False
    assert visits.count_open() == 1

    open_visit = visits.find_open_by_face(face_id)
    assert open_visit is not None
    assert open_visit.visit_id == visit_id

    visits.close(
        visit_id,
        exit_at=NOW + timedelta(seconds=3),
        exit_image_path="logs/exits/x.jpg",
    )

    closed = visits.get(visit_id)
    assert closed.status == "closed"
    assert closed.exit_at == "2025-06-20T18:39:06.000Z"
    assert closed.exit_image_path == "logs/exits/x.jpg"
    assert visits.count_open() == 0
    assert visits.count_all() == 1


def test_close_visit_twice_raises(visits, faces):
    visit_id = _create_visit(visits, _create_face(faces))
    visits.close(visit_id, exit_at=NOW + timedelta(seconds=1))

    with pytest.raises(StorageError, match="already closed"):
        visits.close(visit_id, exit_at=NOW + timedelta(seconds=2))


def test_close_before_entry_raises(visits, faces):
    face_id = _create_face(faces)
    visit_id = _create_visit(visits, face_id)

    with pytest.raises(StorageError, match="precedes"):
        visits.close(visit_id, exit_at=NOW - timedelta(seconds=1))

    assert visits.get(visit_id).status == "open"


def test_anonymous_visit(visits):
    visit_id = visits.create(
        face_id=None,
        track_id=3,
        source_id="clip.mp4",
        entry_at=NOW,
        is_anonymous=True,
    )

    visit = visits.get(visit_id)
    assert visit.face_id is None
    assert visit.is_anonymous is True


def test_anonymous_flag_required_for_unlinked_visit(visits):
    with pytest.raises(StorageError, match="is_anonymous"):
        visits.create(
            face_id=None,
            track_id=3,
            source_id="clip.mp4",
            entry_at=NOW,
            is_anonymous=False,
        )


def test_visit_rejects_negative_track_id(visits):
    with pytest.raises(StorageError, match="track_id"):
        visits.create(
            face_id=None,
            track_id=-1,
            source_id="clip.mp4",
            entry_at=NOW,
            is_anonymous=True,
        )


def test_linked_visit_cannot_be_anonymous(visits, faces):
    face_id = _create_face(faces)
    with pytest.raises(StorageError, match="is_anonymous"):
        visits.create(
            face_id=face_id,
            track_id=3,
            source_id="clip.mp4",
            entry_at=NOW,
            is_anonymous=True,
        )


def test_duplicate_event_rejected(events, visits, faces):
    face_id = _create_face(faces)
    visit_id = _create_visit(visits, face_id)

    entry_id = events.insert(
        visit_id=visit_id,
        face_id=face_id,
        event_type="entry",
        occurred_at=NOW,
        video_time_s=1.0,
        source_id="clip.mp4",
        image_path="logs/entries/a.jpg",
    )
    assert entry_id == 1

    with pytest.raises(DuplicateEventError, match="already has a entry event"):
        events.insert(
            visit_id=visit_id,
            face_id=face_id,
            event_type="entry",
            occurred_at=NOW,
            video_time_s=1.0,
            source_id="clip.mp4",
            image_path="logs/entries/b.jpg",
        )

    events.insert(
        visit_id=visit_id,
        face_id=face_id,
        event_type="exit",
        occurred_at=NOW + timedelta(seconds=2),
        video_time_s=3.0,
        source_id="clip.mp4",
        image_path="logs/exits/a.jpg",
    )

    assert events.count_by_type("entry") == 1
    assert events.count_by_type("exit") == 1
    assert [event.event_type for event in events.list_for_visit(visit_id)] == [
        "entry",
        "exit",
    ]


def test_create_visit_with_entry_is_atomic(events, visits, faces):
    face_id = _create_face(faces)

    visit_id, event_id = events.create_visit_with_entry(
        face_id=face_id,
        track_id=7,
        source_id="clip.mp4",
        entry_at=NOW,
        is_anonymous=False,
        entry_image_path="entries/e.jpg",
        occurred_at=NOW,
        video_time_s=1.0,
        image_path="entries/e.jpg",
        bbox_json="[1, 2, 3, 4]",
        confidence=0.9,
        similarity=0.7,
    )

    visit = visits.get(visit_id)
    event = events.get(event_id)
    assert visit.status == "open"
    assert visit.entry_image_path == "entries/e.jpg"
    assert event.event_type == "entry"
    assert event.face_id == face_id
    assert event.similarity == pytest.approx(0.7)


def test_create_visit_with_entry_rolls_back_on_bad_event(
    events, visits
):
    with pytest.raises(StorageError, match="is_anonymous"):
        events.create_visit_with_entry(
            face_id=None,
            track_id=1,
            source_id="clip.mp4",
            entry_at=NOW,
            is_anonymous=False,
            entry_image_path="entries/e.jpg",
            occurred_at=NOW,
            video_time_s=0.0,
            image_path="entries/e.jpg",
        )

    assert visits.count_all() == 0


def test_close_visit_with_exit_is_atomic(events, visits):
    visit_id, _ = events.create_visit_with_entry(
        face_id=None,
        track_id=1,
        source_id="clip.mp4",
        entry_at=NOW,
        is_anonymous=True,
        entry_image_path="entries/e.jpg",
        occurred_at=NOW,
        video_time_s=0.0,
        image_path="entries/e.jpg",
    )

    event_id = events.close_visit_with_exit(
        visit_id=visit_id,
        face_id=None,
        occurred_at=NOW + timedelta(seconds=3),
        video_time_s=3.0,
        source_id="clip.mp4",
        image_path="exits/e.jpg",
        is_anonymous=True,
    )

    visit = visits.get(visit_id)
    event = events.get(event_id)
    assert visit.status == "closed"
    assert visit.exit_image_path == "exits/e.jpg"
    assert event.event_type == "exit"

    with pytest.raises(StorageError, match="already closed"):
        events.close_visit_with_exit(
            visit_id=visit_id,
            face_id=None,
            occurred_at=NOW + timedelta(seconds=4),
            video_time_s=4.0,
            source_id="clip.mp4",
            image_path="exits/e2.jpg",
            is_anonymous=True,
        )


def test_link_face_updates_visit_and_entry(events, visits, faces):
    face_id = _create_face(faces)
    visit_id, event_id = events.create_visit_with_entry(
        face_id=None,
        track_id=1,
        source_id="clip.mp4",
        entry_at=NOW,
        is_anonymous=True,
        entry_image_path="entries/e.jpg",
        occurred_at=NOW,
        video_time_s=0.0,
        image_path="entries/e.jpg",
    )

    assert events.link_face(visit_id, face_id) is True
    assert visits.get(visit_id).face_id == face_id
    assert events.get(event_id).face_id == face_id
    assert events.link_face(visit_id, face_id) is True
    assert events.link_face(visit_id, face_id + 1) is False


def test_anonymous_count_and_event_time_range(events, visits):
    events.create_visit_with_entry(
        face_id=None,
        track_id=1,
        source_id="clip.mp4",
        entry_at=NOW,
        is_anonymous=True,
        entry_image_path="entries/a.jpg",
        occurred_at=NOW,
        video_time_s=0.0,
        image_path="entries/a.jpg",
    )
    events.create_visit_with_entry(
        face_id=None,
        track_id=2,
        source_id="clip.mp4",
        entry_at=NOW + timedelta(seconds=5),
        is_anonymous=True,
        entry_image_path="entries/b.jpg",
        occurred_at=NOW + timedelta(seconds=5),
        video_time_s=5.0,
        image_path="entries/b.jpg",
    )

    assert visits.count_anonymous() == 2
    first, last = events.time_range()
    assert first == "2025-06-20T18:39:03.000Z"
    assert last == "2025-06-20T18:39:08.000Z"


def test_event_foreign_key_enforced(events):
    with pytest.raises(sqlite3.IntegrityError):
        events.insert(
            visit_id=999,
            face_id=None,
            event_type="entry",
            occurred_at=NOW,
            video_time_s=1.0,
            source_id="clip.mp4",
            image_path="logs/entries/a.jpg",
            is_anonymous=True,
        )


def test_event_anonymous_consistency(events, visits, faces):
    face_id = _create_face(faces)
    visit_id = _create_visit(visits, face_id)

    with pytest.raises(StorageError, match="is_anonymous"):
        events.insert(
            visit_id=visit_id,
            face_id=None,
            event_type="entry",
            occurred_at=NOW,
            video_time_s=1.0,
            source_id="clip.mp4",
            image_path="logs/entries/a.jpg",
            is_anonymous=False,
        )

    with pytest.raises(StorageError, match="is_anonymous"):
        events.insert(
            visit_id=visit_id,
            face_id=face_id,
            event_type="entry",
            occurred_at=NOW,
            video_time_s=1.0,
            source_id="clip.mp4",
            image_path="logs/entries/a.jpg",
            is_anonymous=True,
        )

    assert events.count_by_type("entry") == 0


def test_event_validation(events):
    with pytest.raises(StorageError, match="event_type"):
        events.insert(
            visit_id=1,
            face_id=None,
            event_type="arrival",
            occurred_at=NOW,
            video_time_s=1.0,
            source_id="clip.mp4",
            image_path="logs/entries/a.jpg",
        )

    with pytest.raises(StorageError, match="image_path"):
        events.insert(
            visit_id=1,
            face_id=None,
            event_type="entry",
            occurred_at=NOW,
            video_time_s=1.0,
            source_id="clip.mp4",
            image_path="",
        )

    with pytest.raises(StorageError, match="video_time_s"):
        events.insert(
            visit_id=1,
            face_id=None,
            event_type="entry",
            occurred_at=NOW,
            video_time_s=-1.0,
            source_id="clip.mp4",
            image_path="logs/entries/a.jpg",
        )


def test_transaction_rolls_back(db, faces):
    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO faces
                    (created_at, first_seen_at, model_name, best_quality, is_active)
                VALUES ('x', 'x', 'm', 0.5, 1)
                """
            )
            raise RuntimeError("boom")

    assert faces.count_active() == 0


def test_get_missing_returns_none(faces, visits, events):
    assert faces.get(123) is None
    assert visits.get(123) is None
    assert events.get(123) is None


def test_utc_iso_normalizes_naive_datetime():
    assert utc_iso(datetime(2025, 6, 20, 18, 39, 3)) == (
        "2025-06-20T18:39:03.000Z"
    )
