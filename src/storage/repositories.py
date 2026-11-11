"""Repository layer over the SQLite schema.

Vectors are stored as raw float32 little-endian bytes. The repository layer
does not import NumPy; callers serialise and deserialise embeddings.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.storage.db import Database

_EVENT_TYPES = {"entry", "exit"}


class StorageError(RuntimeError):
    """Raised when a storage operation violates a documented invariant."""


class DuplicateEventError(StorageError):
    """Raised when a visit already has an event of the requested type."""


def utc_iso(moment: datetime) -> str:
    """Format a datetime as UTC ISO-8601 with milliseconds and a ``Z``.

    Naive datetimes are treated as UTC; aware datetimes are converted.
    """

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (
        moment.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _validate_vector(vector: bytes, dim: int) -> None:
    if not isinstance(vector, (bytes, bytearray)):
        raise StorageError(
            f"vector: expected bytes, got {type(vector).__name__}"
        )
    if dim < 1:
        raise StorageError("dim: must be at least 1")
    if len(vector) != dim * 4:
        raise StorageError(
            f"vector: expected {dim * 4} bytes for dim={dim}, got {len(vector)}"
        )


def _validate_quality(quality: float) -> None:
    if not 0.0 <= quality <= 1.0:
        raise StorageError(
            f"quality: expected a value between 0 and 1, got {quality}"
        )


@dataclass(frozen=True)
class FaceRecord:
    """A registered identity."""

    face_id: int
    created_at: str
    first_seen_at: str
    model_name: str
    best_quality: float
    is_active: bool


@dataclass(frozen=True)
class VisitRecord:
    """One observed presence of a person."""

    visit_id: int
    face_id: int | None
    track_id: int
    source_id: str
    entry_at: str
    exit_at: str | None
    status: str
    is_anonymous: bool
    entry_image_path: str | None
    exit_image_path: str | None


@dataclass(frozen=True)
class EventRecord:
    """One entry or exit audit row."""

    event_id: int
    visit_id: int
    face_id: int | None
    event_type: str
    occurred_at: str
    video_time_s: float
    source_id: str
    image_path: str
    bbox_json: str | None
    confidence: float | None
    similarity: float | None
    is_anonymous: bool


def _row_to_face(row: sqlite3.Row) -> FaceRecord:
    return FaceRecord(
        face_id=int(row["face_id"]),
        created_at=str(row["created_at"]),
        first_seen_at=str(row["first_seen_at"]),
        model_name=str(row["model_name"]),
        best_quality=float(row["best_quality"]),
        is_active=bool(row["is_active"]),
    )


def _row_to_visit(row: sqlite3.Row) -> VisitRecord:
    return VisitRecord(
        visit_id=int(row["visit_id"]),
        face_id=int(row["face_id"]) if row["face_id"] is not None else None,
        track_id=int(row["track_id"]),
        source_id=str(row["source_id"]),
        entry_at=str(row["entry_at"]),
        exit_at=str(row["exit_at"]) if row["exit_at"] is not None else None,
        status=str(row["status"]),
        is_anonymous=bool(row["is_anonymous"]),
        entry_image_path=(
            str(row["entry_image_path"])
            if row["entry_image_path"] is not None
            else None
        ),
        exit_image_path=(
            str(row["exit_image_path"])
            if row["exit_image_path"] is not None
            else None
        ),
    )


def _row_to_event(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        event_id=int(row["event_id"]),
        visit_id=int(row["visit_id"]),
        face_id=int(row["face_id"]) if row["face_id"] is not None else None,
        event_type=str(row["event_type"]),
        occurred_at=str(row["occurred_at"]),
        video_time_s=float(row["video_time_s"]),
        source_id=str(row["source_id"]),
        image_path=str(row["image_path"]),
        bbox_json=str(row["bbox_json"]) if row["bbox_json"] is not None else None,
        confidence=(
            float(row["confidence"]) if row["confidence"] is not None else None
        ),
        similarity=(
            float(row["similarity"]) if row["similarity"] is not None else None
        ),
        is_anonymous=bool(row["is_anonymous"]),
    )


class FaceRepository:
    """Create, query and deactivate registered identities."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def create(
        self,
        *,
        vector: bytes,
        dim: int,
        quality: float,
        model_name: str,
        now: datetime,
    ) -> int:
        """Insert a face and its first embedding atomically.

        Returns:
            The new ``face_id``.
        """

        _validate_vector(vector, dim)
        _validate_quality(quality)
        if not model_name:
            raise StorageError("model_name: must not be empty")

        timestamp = utc_iso(now)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO faces
                    (created_at, first_seen_at, model_name, best_quality, is_active)
                VALUES (?, ?, ?, ?, 1)
                """,
                (timestamp, timestamp, model_name, float(quality)),
            )
            face_id = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO embeddings
                    (face_id, vector, dim, quality, created_at, source_event_id)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (face_id, bytes(vector), dim, float(quality), timestamp),
            )
        return face_id

    def get(self, face_id: int) -> FaceRecord | None:
        """Return a face by ID, or ``None``."""

        row = self._db.connect().execute(
            "SELECT * FROM faces WHERE face_id = ?", (face_id,)
        ).fetchone()
        return _row_to_face(row) if row is not None else None

    def count_active(self) -> int:
        """Return the unique-visitor count derived from the database."""

        row = self._db.connect().execute(
            "SELECT COUNT(*) FROM faces WHERE is_active = 1"
        ).fetchone()
        return int(row[0])

    def list_embeddings(self) -> list[tuple[int, bytes, int, float]]:
        """Return ``(face_id, vector, dim, quality)`` for every embedding."""

        rows = self._db.connect().execute(
            """
            SELECT face_id, vector, dim, quality
            FROM embeddings
            ORDER BY face_id, embedding_id
            """
        ).fetchall()
        return [
            (
                int(row["face_id"]),
                bytes(row["vector"]),
                int(row["dim"]),
                float(row["quality"]),
            )
            for row in rows
        ]

    def count_embeddings(self, face_id: int) -> int:
        """Return the number of stored embeddings for a face."""

        row = self._db.connect().execute(
            "SELECT COUNT(*) FROM embeddings WHERE face_id = ?",
            (face_id,),
        ).fetchone()
        return int(row[0])

    def add_embedding(
        self,
        face_id: int,
        *,
        vector: bytes,
        dim: int,
        quality: float,
        now: datetime,
    ) -> int:
        """Attach an additional pose variant to an active face.

        Returns:
            The new ``embedding_id``.
        """

        _validate_vector(vector, dim)
        _validate_quality(quality)
        timestamp = utc_iso(now)
        with self._db.transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM faces WHERE face_id = ? AND is_active = 1",
                (face_id,),
            ).fetchone()
            if exists is None:
                raise StorageError(f"face {face_id} does not exist or is inactive")

            cursor = connection.execute(
                """
                INSERT INTO embeddings
                    (face_id, vector, dim, quality, created_at, source_event_id)
                VALUES (?, ?, ?, ?, ?, NULL)
                """,
                (face_id, bytes(vector), dim, float(quality), timestamp),
            )
            connection.execute(
                """
                UPDATE faces
                SET best_quality = MAX(best_quality, ?)
                WHERE face_id = ?
                """,
                (float(quality), face_id),
            )
            embedding_id = int(cursor.lastrowid)
        return embedding_id

    def deactivate(self, face_id: int) -> bool:
        """Mark a face inactive. Returns ``True`` when a row changed."""

        with self._db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE faces SET is_active = 0 WHERE face_id = ? AND is_active = 1",
                (face_id,),
            )
            return cursor.rowcount == 1


class VisitRepository:
    """Create, close and query visits."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def create(
        self,
        *,
        face_id: int | None,
        track_id: int,
        source_id: str,
        entry_at: datetime,
        is_anonymous: bool,
        entry_image_path: str | None = None,
    ) -> int:
        """Insert an open visit and return its ``visit_id``."""

        if not source_id:
            raise StorageError("source_id: must not be empty")
        if track_id < 0:
            raise StorageError("track_id: must not be negative")
        if face_id is None and not is_anonymous:
            raise StorageError(
                "is_anonymous: must be true when face_id is not set"
            )
        if face_id is not None and is_anonymous:
            raise StorageError(
                "is_anonymous: must be false when face_id is set"
            )

        timestamp = utc_iso(entry_at)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO visits
                    (face_id, track_id, source_id, entry_at, exit_at, status,
                     is_anonymous, entry_image_path, exit_image_path)
                VALUES (?, ?, ?, ?, NULL, 'open', ?, ?, NULL)
                """,
                (
                    face_id,
                    track_id,
                    source_id,
                    timestamp,
                    int(is_anonymous),
                    entry_image_path,
                ),
            )
            visit_id = int(cursor.lastrowid)
        return visit_id

    def get(self, visit_id: int) -> VisitRecord | None:
        """Return a visit by ID, or ``None``."""

        row = self._db.connect().execute(
            "SELECT * FROM visits WHERE visit_id = ?", (visit_id,)
        ).fetchone()
        return _row_to_visit(row) if row is not None else None

    def find_open_by_face(self, face_id: int) -> VisitRecord | None:
        """Return the newest open visit for a Face ID, or ``None``."""

        row = self._db.connect().execute(
            """
            SELECT * FROM visits
            WHERE face_id = ? AND status = 'open'
            ORDER BY visit_id DESC
            LIMIT 1
            """,
            (face_id,),
        ).fetchone()
        return _row_to_visit(row) if row is not None else None

    def close(
        self,
        visit_id: int,
        *,
        exit_at: datetime,
        exit_image_path: str | None = None,
    ) -> None:
        """Close an open visit exactly once."""

        timestamp = utc_iso(exit_at)
        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT entry_at, status FROM visits WHERE visit_id = ?",
                (visit_id,),
            ).fetchone()
            if row is None:
                raise StorageError(f"visit {visit_id} does not exist")
            if row["status"] != "open":
                raise StorageError(f"visit {visit_id} is already closed")
            if timestamp < str(row["entry_at"]):
                raise StorageError(
                    f"visit {visit_id}: exit_at precedes entry_at"
                )

            connection.execute(
                """
                UPDATE visits
                SET exit_at = ?, status = 'closed', exit_image_path = ?
                WHERE visit_id = ?
                """,
                (timestamp, exit_image_path, visit_id),
            )

    def count_open(self) -> int:
        """Return the number of visits still open."""

        row = self._db.connect().execute(
            "SELECT COUNT(*) FROM visits WHERE status = 'open'"
        ).fetchone()
        return int(row[0])

    def count_all(self) -> int:
        """Return the total number of visits."""

        row = self._db.connect().execute(
            "SELECT COUNT(*) FROM visits"
        ).fetchone()
        return int(row[0])

    def count_anonymous(self) -> int:
        """Return the number of anonymous visits."""

        row = self._db.connect().execute(
            "SELECT COUNT(*) FROM visits WHERE is_anonymous = 1"
        ).fetchone()
        return int(row[0])


class EventRepository:
    """Insert and query entry/exit events."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def insert(
        self,
        *,
        visit_id: int,
        face_id: int | None,
        event_type: str,
        occurred_at: datetime,
        video_time_s: float,
        source_id: str,
        image_path: str,
        bbox_json: str | None = None,
        confidence: float | None = None,
        similarity: float | None = None,
        is_anonymous: bool = False,
    ) -> int:
        """Insert one event.

        Raises:
            DuplicateEventError: if the visit already has this event type.
            StorageError: if a required string is empty.
        """

        try:
            with self._db.transaction() as connection:
                event_id = self._insert_event(
                    connection,
                    visit_id=visit_id,
                    face_id=face_id,
                    event_type=event_type,
                    occurred_at=occurred_at,
                    video_time_s=video_time_s,
                    source_id=source_id,
                    image_path=image_path,
                    bbox_json=bbox_json,
                    confidence=confidence,
                    similarity=similarity,
                    is_anonymous=is_anonymous,
                )
        except sqlite3.IntegrityError as exc:
            message = str(exc)
            if (
                "UNIQUE constraint failed: events.visit_id, events.event_type"
                in message
            ):
                raise DuplicateEventError(
                    f"visit {visit_id} already has a {event_type} event"
                ) from exc
            raise

        return event_id

    def create_visit_with_entry(
        self,
        *,
        face_id: int | None,
        track_id: int,
        source_id: str,
        entry_at: datetime,
        is_anonymous: bool,
        entry_image_path: str,
        occurred_at: datetime,
        video_time_s: float,
        image_path: str,
        bbox_json: str | None = None,
        confidence: float | None = None,
        similarity: float | None = None,
    ) -> tuple[int, int]:
        """Create a visit and its entry event in one transaction.

        Returns:
            ``(visit_id, event_id)``.
        """

        if track_id < 0:
            raise StorageError("track_id: must not be negative")
        entry_timestamp = utc_iso(entry_at)
        with self._db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO visits
                    (face_id, track_id, source_id, entry_at, exit_at, status,
                     is_anonymous, entry_image_path, exit_image_path)
                VALUES (?, ?, ?, ?, NULL, 'open', ?, ?, NULL)
                """,
                (
                    face_id,
                    track_id,
                    source_id,
                    entry_timestamp,
                    int(is_anonymous),
                    entry_image_path,
                ),
            )
            visit_id = int(cursor.lastrowid)
            event_id = self._insert_event(
                connection,
                visit_id=visit_id,
                face_id=face_id,
                event_type="entry",
                occurred_at=occurred_at,
                video_time_s=video_time_s,
                source_id=source_id,
                image_path=image_path,
                bbox_json=bbox_json,
                confidence=confidence,
                similarity=similarity,
                is_anonymous=is_anonymous,
            )
        return visit_id, event_id

    def close_visit_with_exit(
        self,
        *,
        visit_id: int,
        face_id: int | None,
        occurred_at: datetime,
        video_time_s: float,
        source_id: str,
        image_path: str,
        bbox_json: str | None = None,
        confidence: float | None = None,
        similarity: float | None = None,
        is_anonymous: bool = False,
    ) -> int:
        """Insert the exit event and close the visit in one transaction."""

        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT status FROM visits WHERE visit_id = ?",
                (visit_id,),
            ).fetchone()
            if row is None:
                raise StorageError(f"visit {visit_id} does not exist")
            if row["status"] != "open":
                raise StorageError(f"visit {visit_id} is already closed")

            event_id = self._insert_event(
                connection,
                visit_id=visit_id,
                face_id=face_id,
                event_type="exit",
                occurred_at=occurred_at,
                video_time_s=video_time_s,
                source_id=source_id,
                image_path=image_path,
                bbox_json=bbox_json,
                confidence=confidence,
                similarity=similarity,
                is_anonymous=is_anonymous,
            )
            cursor = connection.execute(
                """
                UPDATE visits
                SET exit_at = ?, status = 'closed', exit_image_path = ?
                WHERE visit_id = ? AND status = 'open'
                """,
                (utc_iso(occurred_at), image_path, visit_id),
            )
            if cursor.rowcount != 1:
                raise StorageError(f"visit {visit_id} could not be closed")
        return event_id

    def link_face(self, visit_id: int, face_id: int) -> bool:
        """Link an anonymous visit and its entry event to a Face ID."""

        with self._db.transaction() as connection:
            row = connection.execute(
                "SELECT face_id FROM visits WHERE visit_id = ?",
                (visit_id,),
            ).fetchone()
            if row is None:
                raise StorageError(f"visit {visit_id} does not exist")
            if row["face_id"] is not None:
                return int(row["face_id"]) == face_id

            connection.execute(
                "UPDATE visits SET face_id = ?, is_anonymous = 0 "
                "WHERE visit_id = ?",
                (face_id, visit_id),
            )
            connection.execute(
                "UPDATE events SET face_id = ?, is_anonymous = 0 "
                "WHERE visit_id = ? AND event_type = 'entry'",
                (face_id, visit_id),
            )
        return True

    def visit_face(self, visit_id: int) -> int | None:
        """Return the Face ID linked to a visit, or ``None``."""

        row = self._db.connect().execute(
            "SELECT face_id FROM visits WHERE visit_id = ?",
            (visit_id,),
        ).fetchone()
        if row is None:
            raise StorageError(f"visit {visit_id} does not exist")
        return int(row["face_id"]) if row["face_id"] is not None else None

    def _insert_event(
        self,
        connection: sqlite3.Connection,
        *,
        visit_id: int,
        face_id: int | None,
        event_type: str,
        occurred_at: datetime,
        video_time_s: float,
        source_id: str,
        image_path: str,
        bbox_json: str | None,
        confidence: float | None,
        similarity: float | None,
        is_anonymous: bool,
    ) -> int:
        if event_type not in _EVENT_TYPES:
            raise StorageError(
                f"event_type: expected one of {sorted(_EVENT_TYPES)}, "
                f"got '{event_type}'"
            )
        if not source_id:
            raise StorageError("source_id: must not be empty")
        if not image_path:
            raise StorageError("image_path: must not be empty")
        if video_time_s < 0:
            raise StorageError("video_time_s: must not be negative")
        if face_id is None and not is_anonymous:
            raise StorageError(
                "is_anonymous: must be true when face_id is not set"
            )
        if face_id is not None and is_anonymous:
            raise StorageError(
                "is_anonymous: must be false when face_id is set"
            )

        cursor = connection.execute(
            """
            INSERT INTO events
                (visit_id, face_id, event_type, occurred_at,
                 video_time_s, source_id, image_path, bbox_json,
                 confidence, similarity, is_anonymous)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                visit_id,
                face_id,
                event_type,
                utc_iso(occurred_at),
                float(video_time_s),
                source_id,
                image_path,
                bbox_json,
                confidence,
                similarity,
                int(is_anonymous),
            ),
        )
        return int(cursor.lastrowid)

    def get(self, event_id: int) -> EventRecord | None:
        """Return an event by ID, or ``None``."""

        row = self._db.connect().execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return _row_to_event(row) if row is not None else None

    def list_for_visit(self, visit_id: int) -> list[EventRecord]:
        """Return all events for a visit in insertion order."""

        rows = self._db.connect().execute(
            "SELECT * FROM events WHERE visit_id = ? ORDER BY event_id",
            (visit_id,),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def count_by_type(self, event_type: str) -> int:
        """Return the number of events of one type."""

        if event_type not in _EVENT_TYPES:
            raise StorageError(f"event_type: unknown '{event_type}'")
        row = self._db.connect().execute(
            "SELECT COUNT(*) FROM events WHERE event_type = ?",
            (event_type,),
        ).fetchone()
        return int(row[0])

    def time_range(self) -> tuple[str | None, str | None]:
        """Return the first and last event timestamps, or ``(None, None)``."""

        row = self._db.connect().execute(
            "SELECT MIN(occurred_at), MAX(occurred_at) FROM events"
        ).fetchone()
        first = str(row[0]) if row[0] is not None else None
        last = str(row[1]) if row[1] is not None else None
        return first, last
