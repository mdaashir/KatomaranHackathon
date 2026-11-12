"""Persist lifecycle events to images, SQLite and ``events.log``."""

from __future__ import annotations

import json
import logging

import numpy as np

from src.events.image_store import ImageStore
from src.logging_setup import get_logger, log_event
from src.storage.db import Database
from src.storage.repositories import EventRepository
from src.tracking.types import VisitEvent


class EventLogError(RuntimeError):
    """Raised when an event cannot be persisted consistently."""


class EventLogger:
    """Synchronous event writer.

    Order per event: atomic image write, one DB transaction, then the audit
    line. If the DB transaction fails, the image is removed.
    """

    def __init__(
        self,
        database: Database,
        image_store: ImageStore,
        *,
        event_repository: EventRepository | None = None,
        events_logger: logging.Logger | None = None,
    ) -> None:
        self._images = image_store
        self._events = event_repository or EventRepository(database)
        self._log = events_logger or get_logger("events")
        self._visit_map: dict[int, int] = {}
        self._event_ids: dict[int, int] = {}

    @property
    def visit_mapping(self) -> dict[int, int]:
        """Map from lifecycle visit ID to database visit ID."""

        return dict(self._visit_map)

    def event_id_for(self, lifecycle_visit_id: int) -> int | None:
        """Return the entry/exit event ID last written for a lifecycle visit."""

        return self._event_ids.get(lifecycle_visit_id)

    def handle(
        self,
        event: VisitEvent,
        crop: np.ndarray | None = None,
    ) -> int:
        """Persist one entry or exit and return the database event ID."""

        if event.event_type == "entry":
            return self._handle_entry(event, crop)
        if event.event_type == "exit":
            return self._handle_exit(event, crop)
        raise EventLogError(f"unknown event type '{event.event_type}'")

    def link_face(self, lifecycle_visit_id: int, face_id: int) -> bool:
        """Link an anonymous visit to a Face ID discovered after entry."""

        database_visit_id = self._visit_map.get(lifecycle_visit_id)
        if database_visit_id is None:
            return False

        existing = self._events.visit_face(database_visit_id)
        if existing is not None:
            return existing == face_id

        linked = self._events.link_face(database_visit_id, face_id)
        if linked:
            log_event(
                self._log,
                "VISIT_LINKED",
                visit=database_visit_id,
                lifecycle_visit=lifecycle_visit_id,
                face=face_id,
            )
        return linked

    def _handle_entry(
        self,
        event: VisitEvent,
        crop: np.ndarray | None,
    ) -> int:
        if event.visit_id in self._visit_map:
            raise EventLogError(
                f"lifecycle visit {event.visit_id} already has an entry"
            )

        path = self._images.save(
            crop,
            event_type="entry",
            occurred_at=event.occurred_at,
            face_id=event.face_id,
            visit_id=event.visit_id,
            track_id=event.track_id,
        )
        try:
            visit_id, event_id = self._events.create_visit_with_entry(
                face_id=event.face_id,
                track_id=event.track_id,
                source_id=event.source_id,
                entry_at=event.occurred_at,
                is_anonymous=event.is_anonymous,
                entry_image_path=str(path.as_posix()),
                occurred_at=event.occurred_at,
                video_time_s=event.video_time_s,
                image_path=str(path.as_posix()),
                bbox_json=json.dumps(list(event.bbox)),
                confidence=event.confidence,
                similarity=event.similarity,
            )
        except Exception:
            self._images.remove(path)
            raise

        self._visit_map[event.visit_id] = visit_id
        self._event_ids[event.visit_id] = event_id
        log_event(
            self._log,
            "ENTRY",
            visit=visit_id,
            lifecycle_visit=event.visit_id,
            face=event.face_id,
            track=event.track_id,
            anon=event.is_anonymous,
            frame=event.frame_index,
            video_t=round(event.video_time_s, 3),
            conf=round(event.confidence, 3),
            img=path.as_posix(),
        )
        return event_id

    def _handle_exit(
        self,
        event: VisitEvent,
        crop: np.ndarray | None,
    ) -> int:
        database_visit_id = self._visit_map.get(event.visit_id)
        if database_visit_id is None:
            raise EventLogError(
                f"exit for lifecycle visit {event.visit_id} has no entry"
            )

        if event.face_id is not None:
            existing_face = self._events.visit_face(database_visit_id)
            if existing_face is not None and existing_face != event.face_id:
                raise EventLogError(
                    f"visit {database_visit_id} is linked to Face ID "
                    f"{existing_face}, not {event.face_id}"
                )
            if existing_face is None:
                self.link_face(event.visit_id, event.face_id)

        path = self._images.save(
            crop,
            event_type="exit",
            occurred_at=event.occurred_at,
            face_id=event.face_id,
            visit_id=event.visit_id,
            track_id=event.track_id,
        )
        try:
            event_id = self._events.close_visit_with_exit(
                visit_id=database_visit_id,
                face_id=event.face_id,
                occurred_at=event.occurred_at,
                video_time_s=event.video_time_s,
                source_id=event.source_id,
                image_path=str(path.as_posix()),
                bbox_json=json.dumps(list(event.bbox)),
                confidence=event.confidence,
                similarity=event.similarity,
                is_anonymous=event.is_anonymous,
            )
        except Exception:
            self._images.remove(path)
            raise

        self._event_ids[event.visit_id] = event_id
        log_event(
            self._log,
            "EXIT",
            visit=database_visit_id,
            lifecycle_visit=event.visit_id,
            face=event.face_id,
            track=event.track_id,
            anon=event.is_anonymous,
            frame=event.frame_index,
            video_t=round(event.video_time_s, 3),
            conf=round(event.confidence, 3),
            img=path.as_posix(),
        )
        return event_id
