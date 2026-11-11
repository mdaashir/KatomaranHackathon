"""Frame loop and orchestration.

This module is deliberately free of construction details: ``main.py`` builds
the source, models, registry and event sink and injects them here.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol, Sequence

import numpy as np

from src.config.models import AppConfig
from src.counting import UniqueCounter
from src.detection import FaceDetection
from src.events import EventLogger
from src.logging_setup import get_logger, log_event
from src.pipeline.crops import CropCache
from src.pipeline.types import PipelineResult
from src.recognition import ArcFaceEmbedder, FaceRegistry, QualityGate, crop_face
from src.tracking import (
    ByteTrackAdapter,
    TrackLifecycle,
    TrackState,
    TrackedBox,
)
from src.video import Frame, VideoSource


class DetectorLike(Protocol):
    def detect(self, image: np.ndarray) -> list[FaceDetection]: ...


class TrackerLike(Protocol):
    def update(self, detections: Sequence[FaceDetection]) -> list[TrackedBox]: ...

    def reset(self) -> None: ...


class Pipeline:
    """Consume a video source and persist visits end to end."""

    def __init__(
        self,
        config: AppConfig,
        source: VideoSource,
        detector: DetectorLike,
        lifecycle: TrackLifecycle,
        quality_gate: QualityGate,
        embedder: ArcFaceEmbedder,
        registry: FaceRegistry,
        event_logger: EventLogger,
        counter: UniqueCounter,
        *,
        tracker_factory: Callable[[float, int], TrackerLike] | None = None,
        crop_cache: CropCache | None = None,
        events_logger: logging.Logger | None = None,
        app_logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._detector = detector
        self._lifecycle = lifecycle
        self._quality_gate = quality_gate
        self._embedder = embedder
        self._registry = registry
        self._event_logger = event_logger
        self._counter = counter
        self._tracker_factory = tracker_factory or (
            lambda fps, skip: ByteTrackAdapter(
                config.tracking,
                fps=fps,
                frame_skip=skip,
            )
        )
        self._crops = crop_cache or CropCache()
        self._events_logger = events_logger or get_logger("events")
        self._app_logger = app_logger or get_logger(__name__)
        self._tracker: TrackerLike | None = None
        self._last_attempt: dict[int, float] = {}
        self._window_id: str | None = None

    def run(
        self,
        *,
        stop_event: threading.Event | None = None,
        max_frames: int | None = None,
    ) -> PipelineResult:
        """Process the source and return a DB-derived run summary."""

        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must not be negative")

        stop = stop_event or threading.Event()
        frames = 0
        cycles = 0
        log_event(
            self._events_logger,
            "PIPELINE_STARTED",
            source=self._source.source_id,
            frame_skip=self._config.frame_skip,
        )

        try:
            for frame in self._source.frames():
                if stop.is_set():
                    break
                if max_frames is not None and frames >= max_frames:
                    break

                if (
                    self._window_id is not None
                    and frame.window_id != self._window_id
                ):
                    self._flush_window()
                self._window_id = frame.window_id

                if frame.global_index % self._config.frame_skip == 0:
                    self._process_detection_frame(frame)
                    cycles += 1

                frames += 1

                if frame.is_window_end:
                    self._flush_window()
                    self._window_id = None
        finally:
            try:
                self._flush_window()
            finally:
                self._source.close()

        if stop.is_set():
            log_event(
                self._events_logger,
                "SHUTDOWN",
                reason="signal",
                frames=frames,
            )

        summary = self._counter.get_summary()
        log_event(
            self._events_logger,
            "SUMMARY",
            unique=summary.unique_faces,
            visits=summary.visits,
            entries=summary.entries,
            exits=summary.exits,
            anonymous=summary.anonymous_visits,
            open=summary.open_visits,
        )
        return PipelineResult(
            frames_processed=frames,
            detection_cycles=cycles,
            stopped_by_signal=stop.is_set(),
            summary=summary,
        )

    def _process_detection_frame(self, frame: Frame) -> None:
        if self._tracker is None:
            self._tracker = self._tracker_factory(
                frame.fps,
                self._config.frame_skip,
            )

        detections = self._detector.detect(frame.image)
        boxes = self._tracker.update(detections)
        for box in boxes:
            try:
                crop, _ = crop_face(frame.image, box.bbox)
            except ValueError:
                continue
            self._crops.update(box.track_id, crop)

        previous = {
            track_id: track.state
            for track_id, track in self._lifecycle.tracks.items()
        }
        face_ids = self._recognize(frame, boxes)
        events = self._lifecycle.observe(
            boxes,
            frame_index=frame.global_index,
            timestamp=frame.absolute_time,
            video_time_s=frame.timestamp_s,
            source_id=frame.source_id,
            window_id=frame.window_id,
            face_ids=face_ids,
        )
        self._log_transitions(previous)
        self._persist(events)

    def _recognize(
        self,
        frame: Frame,
        boxes: Sequence[TrackedBox],
    ) -> dict[int, int | None]:
        face_ids: dict[int, int | None] = {}
        now = frame.absolute_time.timestamp()
        retry_interval = self._config.recognition.retry_interval_sec
        tracks = self._lifecycle.tracks

        for box in boxes:
            track = tracks.get(box.track_id)
            if track is not None and track.face_id is not None:
                continue

            last_attempt = self._last_attempt.get(box.track_id)
            if (
                last_attempt is not None
                and now - last_attempt < retry_interval
            ):
                continue
            self._last_attempt[box.track_id] = now

            detection = FaceDetection(
                bbox=box.bbox,
                confidence=box.confidence,
            )
            try:
                quality = self._quality_gate.assess(
                    frame.image,
                    detection,
                )
                if not quality.passed:
                    self._app_logger.debug(
                        "QUALITY_REJECTED track=%s reason=%s",
                        box.track_id,
                        quality.reason,
                    )
                    continue

                result = self._embedder.embed(
                    frame.image,
                    detection,
                    quality.score,
                )
                if result is None:
                    self._app_logger.debug(
                        "POSE_REJECTED track=%s",
                        box.track_id,
                    )
                    continue

                log_event(
                    self._events_logger,
                    "EMBEDDING_GENERATED",
                    track=box.track_id,
                    face="-",
                    dim=int(result.vector.size),
                    quality=round(quality.score, 3),
                )
                match = self._registry.match(result.vector)
                face_id = match.face_id

                if face_id is not None:
                    log_event(
                        self._events_logger,
                        "FACE_RECOGNIZED",
                        track=box.track_id,
                        face=face_id,
                        similarity=round(match.similarity, 3),
                    )
                    if (
                        quality.score
                        >= self._config.recognition.register_quality_threshold
                    ):
                        try:
                            self._registry.add_embedding(
                                face_id,
                                result.vector,
                                quality.score,
                                frame.absolute_time,
                            )
                        except Exception as exc:
                            self._app_logger.warning(
                                "could not add embedding for face %s: %s",
                                face_id,
                                exc,
                            )
                elif (
                    quality.score
                    >= self._config.recognition.register_quality_threshold
                ):
                    face_id = self._registry.register(
                        result.vector,
                        quality.score,
                        frame.absolute_time,
                    )
                    log_event(
                        self._events_logger,
                        "FACE_REGISTERED",
                        track=box.track_id,
                        face=face_id,
                        similarity=round(match.similarity, 3),
                        quality=round(quality.score, 3),
                    )
                else:
                    self._app_logger.debug(
                        "REGISTRATION_REJECTED track=%s quality=%.3f",
                        box.track_id,
                        quality.score,
                    )
                    continue
            except Exception as exc:
                self._app_logger.warning(
                    "recognition failed for track %s: %s",
                    box.track_id,
                    exc,
                )
                continue

            face_ids[box.track_id] = face_id
            if track is not None and track.visit_id is not None:
                if self._lifecycle.assign_face(box.track_id, face_id):
                    self._event_logger.link_face(
                        track.visit_id,
                        face_id,
                    )

        return face_ids

    def _log_transitions(self, previous: dict[int, TrackState]) -> None:
        for track_id, track in self._lifecycle.tracks.items():
            if track_id not in previous:
                bbox = ",".join(f"{value:.1f}" for value in track.bbox)
                log_event(
                    self._events_logger,
                    "FACE_DETECTED",
                    track=track_id,
                    frame=track.first_seen_frame,
                    bbox=bbox,
                    conf=round(track.confidence, 3),
                )
                log_event(
                    self._events_logger,
                    "TRACK_STARTED",
                    track=track_id,
                    frame=track.first_seen_frame,
                )
                continue

            before = previous[track_id]
            if track.state == before:
                continue
            if track.state == TrackState.LOST:
                log_event(
                    self._events_logger,
                    "TRACK_LOST",
                    track=track_id,
                    missed=track.missed_frames,
                )
            elif (
                track.state == TrackState.EXITED
                and before in {TrackState.ACTIVE, TrackState.LOST}
                and track.missed_frames > 0
            ):
                log_event(
                    self._events_logger,
                    "TRACK_LOST",
                    track=track_id,
                    missed=track.missed_frames,
                )
            elif (
                before == TrackState.LOST
                and track.state == TrackState.ACTIVE
            ):
                log_event(
                    self._events_logger,
                    "TRACK_REACQUIRED",
                    track=track_id,
                )

    def _persist(self, events) -> None:
        for event in events:
            crop = self._crops.get(event.track_id)
            try:
                self._event_logger.handle(event, crop)
            except Exception as exc:
                log_event(
                    self._events_logger,
                    "EVENT_ERROR",
                    visit=event.visit_id,
                    type=event.event_type,
                    error=type(exc).__name__,
                )
                self._app_logger.exception(
                    "event persistence failed for visit %s",
                    event.visit_id,
                )
            if event.event_type == "exit":
                self._crops.pop(event.track_id)
                self._last_attempt.pop(event.track_id, None)

    def _flush_window(self) -> None:
        events = self._lifecycle.reset()
        self._persist(events)
        if self._tracker is not None:
            self._tracker.reset()
        self._crops.clear()
        self._last_attempt.clear()
        self._window_id = None
