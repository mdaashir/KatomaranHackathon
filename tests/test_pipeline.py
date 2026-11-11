"""Pipeline orchestration tests with injected fakes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import threading

import numpy as np

from src.config.models import (
    AppConfig,
    DetectorConfig,
    LoggingConfig,
    RecognitionConfig,
    SourceConfig,
    StorageConfig,
    TrackingConfig,
)
from src.counting import CounterSummary
from src.detection import FaceDetection
from src.pipeline import Pipeline
from src.recognition import EmbeddingResult, MatchResult, QualityResult
from src.recognition.runtime import RecognitionError
from src.tracking import TrackedBox, TrackLifecycle
from src.video import Frame

UTC = timezone.utc
BASE = datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)
SUMMARY = CounterSummary(1, 1, 0, 0, 1, 1, None, None)


class FakeSource:
    source_id = "fake"

    def __init__(self, frames):
        self._frames = frames
        self.closed = False

    def frames(self):
        yield from self._frames

    def close(self):
        self.closed = True


class FakeDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [FaceDetection(bbox=(10, 10, 80, 90), confidence=0.9)]


class FakeTracker:
    def __init__(self):
        self.updates = 0
        self.resets = 0
        self.seen_fps = []

    def update(self, detections):
        self.updates += 1
        if not detections:
            return []
        detection = detections[0]
        return [
            TrackedBox(
                track_id=1,
                bbox=detection.bbox,
                confidence=detection.confidence,
            )
        ]

    def reset(self):
        self.resets += 1


class FakeQuality:
    def __init__(self, results=None):
        self.calls = 0
        self.results = list(results or [])

    def assess(self, image, detection):
        self.calls += 1
        if self.results:
            passed = self.results.pop(0)
        else:
            passed = True
        crop = np.zeros((40, 40, 3), dtype=np.uint8)
        return QualityResult(passed, 0.9 if passed else 0.0, None, crop)


class FakeEmbedder:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    def embed(self, image, detection, quality_score):
        self.calls += 1
        if self.error is not None:
            raise self.error
        vector = np.ones(512, dtype=np.float32)
        vector /= np.linalg.norm(vector)
        return EmbeddingResult(
            vector=vector,
            five_points=np.zeros((5, 2), dtype=np.float32),
            aligned=np.zeros((112, 112, 3), dtype=np.uint8),
            quality_score=quality_score,
        )


class FakeRegistry:
    def __init__(self, matched=True):
        self.matched = matched
        self.registered = 0
        self.added = 0

    def match(self, vector):
        if self.matched:
            return MatchResult(1, 0.9)
        return MatchResult(None, 0.1)

    def register(self, vector, quality, now):
        self.registered += 1
        return 2

    def add_embedding(self, face_id, vector, quality, now):
        self.added += 1
        return True


class FakeEventLogger:
    def __init__(self, error=None):
        self.error = error
        self.events = []
        self.crops = []
        self.links = []

    def handle(self, event, crop=None):
        if self.error is not None:
            raise self.error
        self.events.append(event)
        self.crops.append(crop)
        return len(self.events)

    def link_face(self, visit_id, face_id):
        self.links.append((visit_id, face_id))
        return True


class FakeCounter:
    def __init__(self, summary=SUMMARY):
        self._summary = summary

    def get_summary(self):
        return self._summary


def _config() -> AppConfig:
    return AppConfig(
        source=SourceConfig(type="file", path="unused.mp4"),
        frame_skip=2,
        detector=DetectorConfig(),
        recognition=RecognitionConfig(
            model_dir="unused",
            retry_interval_sec=0.0,
        ),
        tracking=TrackingConfig(confirm_hits=1, exit_timeout_frames=2),
        storage=StorageConfig(db_path="unused.db", log_dir="unused-logs"),
        logging=LoggingConfig(),
    )


def _frame(index: int, window: str = "w", is_end: bool = False) -> Frame:
    return Frame(
        index=index,
        global_index=index,
        timestamp_s=index / 25.0,
        absolute_time=BASE + timedelta(seconds=index / 25.0),
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        source_id="fake",
        window_id=window,
        is_window_end=is_end,
        fps=25.0,
    )


def _pipeline(
    frames,
    *,
    embed_error=None,
    event_error=None,
    quality_results=None,
    matched=True,
):
    config = _config()
    source = FakeSource(frames)
    detector = FakeDetector()
    tracker = FakeTracker()
    quality = FakeQuality(quality_results)
    embedder = FakeEmbedder(embed_error)
    registry = FakeRegistry(matched)
    event_logger = FakeEventLogger(event_error)
    counter = FakeCounter()
    lifecycle = TrackLifecycle(config.tracking)
    pipeline = Pipeline(
        config,
        source,
        detector,
        lifecycle,
        quality,
        embedder,
        registry,
        event_logger,
        counter,
        tracker_factory=lambda fps, skip: tracker,
        events_logger=logging.getLogger("test.events"),
        app_logger=logging.getLogger("test.app"),
    )
    return pipeline, {
        "source": source,
        "detector": detector,
        "tracker": tracker,
        "quality": quality,
        "embedder": embedder,
        "registry": registry,
        "event_logger": event_logger,
    }


def test_frame_skip_entry_exit_and_crops():
    pipeline, parts = _pipeline(
        [_frame(index, is_end=(index == 7)) for index in range(8)]
    )

    result = pipeline.run()

    assert parts["detector"].calls == 4
    assert parts["tracker"].updates == 4
    assert [event.event_type for event in parts["event_logger"].events] == [
        "entry",
        "exit",
    ]
    assert all(crop is not None for crop in parts["event_logger"].crops)
    assert parts["embedder"].calls == 1
    assert parts["registry"].added == 1
    assert parts["source"].closed is True
    assert parts["tracker"].resets >= 1
    assert result.frames_processed == 8
    assert result.detection_cycles == 4


def test_window_end_flushes_and_restarts():
    frames = [
        _frame(0, window="a"),
        _frame(1, window="a", is_end=True),
        _frame(2, window="b"),
        _frame(3, window="b", is_end=True),
    ]
    pipeline, parts = _pipeline(frames)

    pipeline.run()

    assert [event.event_type for event in parts["event_logger"].events] == [
        "entry",
        "exit",
        "entry",
        "exit",
    ]
    assert parts["event_logger"].events[0].window_id == "a"
    assert parts["event_logger"].events[2].window_id == "b"
    assert parts["tracker"].resets >= 2


def test_stop_event_prevents_processing():
    pipeline, parts = _pipeline([_frame(index) for index in range(4)])
    stop = threading.Event()
    stop.set()

    result = pipeline.run(stop_event=stop)

    assert result.frames_processed == 0
    assert result.stopped_by_signal is True
    assert parts["detector"].calls == 0
    assert parts["event_logger"].events == []
    assert parts["source"].closed is True


def test_max_frames_limits_work():
    pipeline, parts = _pipeline(
        [_frame(index, is_end=(index == 7)) for index in range(8)]
    )

    result = pipeline.run(max_frames=2)

    assert result.frames_processed == 2
    assert result.detection_cycles == 1
    assert [event.event_type for event in parts["event_logger"].events] == [
        "entry",
        "exit",
    ]


def test_registration_path_when_no_match():
    pipeline, parts = _pipeline(
        [_frame(0, is_end=True)],
        matched=False,
    )

    pipeline.run()

    assert parts["registry"].registered == 1
    entry = parts["event_logger"].events[0]
    assert entry.event_type == "entry"
    assert entry.face_id == 2


def test_recognition_failure_keeps_anonymous_entry():
    pipeline, parts = _pipeline(
        [_frame(index, is_end=(index == 3)) for index in range(4)],
        embed_error=RecognitionError("boom"),
    )

    pipeline.run()

    entry = parts["event_logger"].events[0]
    assert entry.event_type == "entry"
    assert entry.is_anonymous is True
    assert parts["registry"].registered == 0


def test_late_recognition_links_open_visit():
    pipeline, parts = _pipeline(
        [_frame(index, is_end=(index == 3)) for index in range(4)],
        quality_results=[False],
    )

    pipeline.run()

    assert parts["event_logger"].links == [(1, 1)]
    entry = parts["event_logger"].events[0]
    assert entry.event_type == "entry"
    assert entry.is_anonymous is True


def test_event_persistence_failure_does_not_crash():
    pipeline, parts = _pipeline(
        [_frame(index, is_end=(index == 3)) for index in range(4)],
        event_error=RuntimeError("event db down"),
    )

    result = pipeline.run()

    assert result.frames_processed == 4
    assert parts["event_logger"].events == []
