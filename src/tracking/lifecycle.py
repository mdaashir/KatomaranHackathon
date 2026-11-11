"""Per-track state machine and visit lifecycle.

The lifecycle is deliberately storage-agnostic: it returns ``VisitEvent``
objects and optionally calls an injected sink. Phase 6 maps those events to
the database, images and ``events.log``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Mapping, Sequence

from src.config.models import TrackingConfig
from src.tracking.types import (
    LifecycleSummary,
    Track,
    TrackState,
    TrackedBox,
    Visit,
    VisitEvent,
)


class TrackLifecycle:
    """Turn tracker boxes into exactly one entry and one exit per visit.

    Rules:
        * A track is confirmed after ``confirm_hits`` observations.
        * A confirmed track gets one entry, timestamped at first sight.
        * A missing track becomes LOST; it exits after
          ``exit_timeout_frames`` source frames.
        * A reacquired LOST track continues the same visit.
        * A closed visit is never reopened; a later appearance creates a new
          visit, reusing the Face ID when recognition supplies one.
        * A window change or ``flush()`` closes every open visit.
        * When a new track carries the same Face ID as an open visit, it
          attaches to that visit instead of creating a second entry.
    """

    def __init__(
        self,
        config: TrackingConfig,
        *,
        event_sink: Callable[[VisitEvent], None] | None = None,
    ) -> None:
        self._config = config
        self._event_sink = event_sink
        self._tracks: dict[int, Track] = {}
        self._visits: dict[int, Visit] = {}
        self._next_visit_id = 1
        self._window_id: str | None = None
        self._entry_events = 0
        self._exit_events = 0

    @property
    def window_id(self) -> str | None:
        """Current source window, or ``None`` before the first frame."""

        return self._window_id

    @property
    def tracks(self) -> dict[int, Track]:
        """Snapshot of current track states."""

        return dict(self._tracks)

    @property
    def visits(self) -> dict[int, Visit]:
        """Snapshot of every visit created so far."""

        return dict(self._visits)

    @property
    def open_visits(self) -> list[Visit]:
        """Visits still waiting for an exit."""

        return [
            visit for visit in self._visits.values() if visit.status == "open"
        ]

    @property
    def summary(self) -> LifecycleSummary:
        """Current lifecycle counters."""

        tracks = list(self._tracks.values())
        return LifecycleSummary(
            open_visits=len(self.open_visits),
            closed_visits=sum(
                1 for visit in self._visits.values() if visit.status == "closed"
            ),
            active_tracks=sum(
                1 for track in tracks if track.state == TrackState.ACTIVE
            ),
            lost_tracks=sum(
                1 for track in tracks if track.state == TrackState.LOST
            ),
            exited_tracks=sum(
                1 for track in tracks if track.state == TrackState.EXITED
            ),
            dropped_tracks=sum(
                1 for track in tracks if track.state == TrackState.DROPPED
            ),
            entry_events=self._entry_events,
            exit_events=self._exit_events,
        )

    def observe(
        self,
        tracked_boxes: Sequence[TrackedBox],
        *,
        frame_index: int,
        timestamp: datetime,
        video_time_s: float,
        source_id: str,
        window_id: str,
        face_ids: Mapping[int, int | None] | None = None,
    ) -> list[VisitEvent]:
        """Update lifecycle state and return any entry/exit events.

        Args:
            tracked_boxes: ByteTrack output for this detection cycle.
            frame_index: Global source-frame index, used for timeouts.
            timestamp: Absolute UTC time of this frame.
            video_time_s: Seconds from the start of the current source window.
            source_id: Clip or camera identifier for events.
            window_id: Logical window; a change flushes the previous window.
            face_ids: Optional recognized Face ID per tracker ID. Recognition
                must be resolved before confirmation to avoid duplicate entries.

        Returns:
            Entry and exit events produced by this cycle.
        """

        if not source_id:
            raise ValueError("source_id must not be empty")
        if not window_id:
            raise ValueError("window_id must not be empty")

        events: list[VisitEvent] = []
        if self._window_id is not None and window_id != self._window_id:
            events.extend(self.flush())
            self._tracks.clear()
        self._window_id = window_id

        present = {box.track_id: box for box in tracked_boxes}
        for track in list(self._tracks.values()):
            if (
                track.track_id not in present
                and track.state not in {TrackState.EXITED, TrackState.DROPPED}
            ):
                events.extend(self._handle_missing(track, frame_index))

        recognized = face_ids or {}
        for box in tracked_boxes:
            events.extend(
                self._handle_present(
                    box,
                    frame_index,
                    timestamp,
                    video_time_s,
                    source_id,
                    window_id,
                    recognized,
                )
            )
        return events

    def assign_face(self, track_id: int, face_id: int) -> bool:
        """Attach a recognized Face ID to a track, before or after entry.

        Assigning before confirmation lets the visit be created with the right
        identity. Assigning after entry only fills an anonymous open visit; the
        already-emitted entry event keeps its original ``is_anonymous`` value
        until the event logger updates it.
        """

        track = self._tracks.get(track_id)
        if track is None:
            return False
        if track.face_id is not None and track.face_id != face_id:
            return False

        track.face_id = face_id
        if track.visit_id is None:
            return True

        visit = self._visits.get(track.visit_id)
        if visit is None or visit.status != "open":
            return False
        if visit.face_id is None:
            visit.face_id = face_id
            visit.is_anonymous = False
            return True
        return visit.face_id == face_id

    def open_visit_for_face(self, face_id: int) -> Visit | None:
        """Return the open visit for a Face ID, if one exists."""

        for visit in self._visits.values():
            if visit.status == "open" and visit.face_id == face_id:
                return visit
        return None

    def flush(self) -> list[VisitEvent]:
        """Close every open visit, for example at end-of-stream."""

        events: list[VisitEvent] = []
        for track in list(self._tracks.values()):
            if track.state == TrackState.NEW:
                track.state = TrackState.DROPPED
                continue
            if track.state in {TrackState.ACTIVE, TrackState.LOST}:
                events.extend(self._exit_track(track))

        for visit in list(self._visits.values()):
            if visit.status == "open" and not visit.track_ids:
                events.append(self._close_visit(visit, visit.last_track_id))
        return events

    def _handle_missing(
        self,
        track: Track,
        frame_index: int,
    ) -> list[VisitEvent]:
        track.missed_frames = max(0, frame_index - track.last_seen_frame)
        if track.state == TrackState.NEW:
            if track.missed_frames >= self._config.exit_timeout_frames:
                track.state = TrackState.DROPPED
            return []
        if track.state == TrackState.ACTIVE:
            track.state = TrackState.LOST
        if (
            track.state == TrackState.LOST
            and track.missed_frames >= self._config.exit_timeout_frames
        ):
            return self._exit_track(track)
        return []

    def _handle_present(
        self,
        box: TrackedBox,
        frame_index: int,
        timestamp: datetime,
        video_time_s: float,
        source_id: str,
        window_id: str,
        face_ids: Mapping[int, int | None],
    ) -> list[VisitEvent]:
        track = self._tracks.get(box.track_id)
        if track is None or track.state in {
            TrackState.EXITED,
            TrackState.DROPPED,
        }:
            track = Track(
                track_id=box.track_id,
                state=TrackState.NEW,
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
                first_seen_time=timestamp,
                last_seen_time=timestamp,
                first_seen_video_s=video_time_s,
                last_seen_video_s=video_time_s,
                hits=1,
                missed_frames=0,
                bbox=box.bbox,
                confidence=box.confidence,
            )
            self._tracks[box.track_id] = track
        else:
            track.bbox = box.bbox
            track.confidence = box.confidence
            track.last_seen_frame = frame_index
            track.last_seen_time = timestamp
            track.last_seen_video_s = video_time_s
            track.hits += 1
            track.missed_frames = 0
            if track.state == TrackState.LOST:
                track.state = TrackState.ACTIVE

        if box.track_id in face_ids:
            recognized = face_ids[box.track_id]
            if track.face_id is None:
                track.face_id = recognized

        events: list[VisitEvent] = []
        if (
            track.state == TrackState.NEW
            and track.hits >= self._config.confirm_hits
        ):
            events.extend(
                self._confirm(track, source_id, window_id)
            )
        elif track.state == TrackState.ACTIVE and track.visit_id is not None:
            visit = self._visits.get(track.visit_id)
            if visit is not None and visit.status == "open":
                self._update_visit(visit, track)
        return events

    def _confirm(
        self,
        track: Track,
        source_id: str,
        window_id: str,
    ) -> list[VisitEvent]:
        face_id = track.face_id
        if face_id is not None:
            existing = self.open_visit_for_face(face_id)
            if existing is not None:
                existing.track_ids.add(track.track_id)
                existing.last_track_id = track.track_id
                track.visit_id = existing.visit_id
                track.state = TrackState.ACTIVE
                self._update_visit(existing, track)
                return []

        visit = Visit(
            visit_id=self._next_visit_id,
            track_ids={track.track_id},
            face_id=face_id,
            entry_frame=track.first_seen_frame,
            entry_time=track.first_seen_time,
            entry_video_s=track.first_seen_video_s,
            last_seen_frame=track.last_seen_frame,
            last_seen_time=track.last_seen_time,
            last_seen_video_s=track.last_seen_video_s,
            last_bbox=track.bbox,
            last_confidence=track.confidence,
            last_track_id=track.track_id,
            status="open",
            source_id=source_id,
            window_id=window_id,
            is_anonymous=face_id is None,
        )
        self._visits[visit.visit_id] = visit
        self._next_visit_id += 1
        track.visit_id = visit.visit_id
        track.state = TrackState.ACTIVE

        event = VisitEvent(
            event_type="entry",
            visit_id=visit.visit_id,
            track_id=track.track_id,
            face_id=face_id,
            frame_index=track.first_seen_frame,
            occurred_at=track.first_seen_time,
            video_time_s=track.first_seen_video_s,
            source_id=source_id,
            window_id=window_id,
            bbox=track.bbox,
            confidence=track.confidence,
            is_anonymous=visit.is_anonymous,
        )
        return [self._emit(event)]

    def _exit_track(self, track: Track) -> list[VisitEvent]:
        track.state = TrackState.EXITED
        if track.visit_id is None:
            return []

        visit = self._visits.get(track.visit_id)
        if visit is None or visit.status != "open":
            return []

        visit.track_ids.discard(track.track_id)
        if visit.track_ids:
            return []
        return [self._close_visit(visit, track.track_id)]

    def _close_visit(self, visit: Visit, track_id: int) -> VisitEvent:
        visit.status = "closed"
        event = VisitEvent(
            event_type="exit",
            visit_id=visit.visit_id,
            track_id=track_id,
            face_id=visit.face_id,
            frame_index=visit.last_seen_frame,
            occurred_at=visit.last_seen_time,
            video_time_s=visit.last_seen_video_s,
            source_id=visit.source_id,
            window_id=visit.window_id,
            bbox=visit.last_bbox,
            confidence=visit.last_confidence,
            is_anonymous=visit.face_id is None,
        )
        return self._emit(event)

    def _update_visit(self, visit: Visit, track: Track) -> None:
        if track.last_seen_frame >= visit.last_seen_frame:
            visit.last_seen_frame = track.last_seen_frame
            visit.last_seen_time = track.last_seen_time
            visit.last_seen_video_s = track.last_seen_video_s
            visit.last_bbox = track.bbox
            visit.last_confidence = track.confidence
            visit.last_track_id = track.track_id

    def _emit(self, event: VisitEvent) -> VisitEvent:
        if event.event_type == "entry":
            self._entry_events += 1
        else:
            self._exit_events += 1
        if self._event_sink is not None:
            self._event_sink(event)
        return event
