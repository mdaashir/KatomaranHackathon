"""DB-derived unique visitor count and run summary."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.storage.db import Database
from src.storage.repositories import (
    EventRepository,
    FaceRepository,
    VisitRepository,
)


@dataclass(frozen=True)
class CounterSummary:
    """Summary derived entirely from SQLite."""

    unique_faces: int
    visits: int
    open_visits: int
    anonymous_visits: int
    entries: int
    exits: int
    first_event_at: str | None
    last_event_at: str | None

    def to_dict(self) -> dict[str, object]:
        """Return the summary as a plain dictionary."""

        return asdict(self)


class UniqueCounter:
    """Count registered faces and summarize visits/events from the DB."""

    def __init__(
        self,
        database: Database,
        *,
        face_repository: FaceRepository | None = None,
        visit_repository: VisitRepository | None = None,
        event_repository: EventRepository | None = None,
    ) -> None:
        self._faces = face_repository or FaceRepository(database)
        self._visits = visit_repository or VisitRepository(database)
        self._events = event_repository or EventRepository(database)

    def get_unique_count(self) -> int:
        """Return ``COUNT(*)`` of active registered faces."""

        return self._faces.count_active()

    def get_summary(self) -> CounterSummary:
        """Return a DB-derived summary of the current run."""

        first_event_at, last_event_at = self._events.time_range()
        return CounterSummary(
            unique_faces=self._faces.count_active(),
            visits=self._visits.count_all(),
            open_visits=self._visits.count_open(),
            anonymous_visits=self._visits.count_anonymous(),
            entries=self._events.count_by_type("entry"),
            exits=self._events.count_by_type("exit"),
            first_event_at=first_event_at,
            last_event_at=last_event_at,
        )
