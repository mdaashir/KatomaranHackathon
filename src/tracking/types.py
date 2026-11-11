"""Tracking and visit lifecycle data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal


class TrackState(str, Enum):
    """Lifecycle state of one ByteTrack track."""

    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    EXITED = "exited"
    DROPPED = "dropped"


@dataclass(frozen=True)
class TrackedBox:
    """A tracker output box in native coordinates."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class Track:
    """Mutable lifecycle state for one tracker ID."""

    track_id: int
    state: TrackState
    first_seen_frame: int
    last_seen_frame: int
    first_seen_time: datetime
    last_seen_time: datetime
    hits: int
    missed_frames: int
    bbox: tuple[float, float, float, float]
    confidence: float
    face_id: int | None = None
    visit_id: int | None = None


@dataclass
class Visit:
    """One continuous presence, possibly represented by several tracks."""

    visit_id: int
    track_ids: set[int]
    face_id: int | None
    entry_frame: int
    entry_time: datetime
    last_seen_frame: int
    last_seen_time: datetime
    last_bbox: tuple[float, float, float, float]
    last_confidence: float
    last_track_id: int
    status: Literal["open", "closed"]
    source_id: str
    window_id: str
    is_anonymous: bool


@dataclass(frozen=True)
class VisitEvent:
    """One entry or exit audit event emitted by the lifecycle."""

    event_type: Literal["entry", "exit"]
    visit_id: int
    track_id: int
    face_id: int | None
    frame_index: int
    occurred_at: datetime
    source_id: str
    window_id: str
    bbox: tuple[float, float, float, float]
    confidence: float
    is_anonymous: bool


@dataclass
class LifecycleSummary:
    """Counters useful for logs and tests."""

    open_visits: int = 0
    closed_visits: int = 0
    active_tracks: int = 0
    lost_tracks: int = 0
    exited_tracks: int = 0
    dropped_tracks: int = 0
    entry_events: int = 0
    exit_events: int = 0
