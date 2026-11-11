"""Tests for file, folder-sequence and RTSP video sources."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from src.video import (
    BoundedFrameQueue,
    FileSequenceSource,
    FileSource,
    RTSPSource,
    VideoSourceError,
    next_backoff,
    parse_clip_start,
)

UTC = timezone.utc


def test_parse_clip_start():
    parsed = parse_clip_start("record_20250620_183903.mp4")

    assert parsed == datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)


def test_parse_clip_start_ignores_other_names():
    assert parse_clip_start("video_sample1.mp4") is None
    assert parse_clip_start("record_20250620_1839.mp4") is None


def test_file_source_reads_all_frames(make_video):
    path = make_video("clip.mp4", frames=10, fps=5)
    start = datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC)

    with FileSource(path, start_time=start) as source:
        frames = list(source.frames())

    assert len(frames) == 10
    assert [frame.index for frame in frames] == list(range(10))
    assert [frame.global_index for frame in frames] == list(range(10))
    assert frames[0].source_id == "clip.mp4"
    assert frames[0].image.shape == (120, 160, 3)
    assert frames[0].timestamp_s == pytest.approx(0.0, abs=0.05)
    assert frames[-1].timestamp_s == pytest.approx(1.8, abs=0.1)
    assert frames[0].absolute_time == start
    assert frames[-1].is_window_end is True
    assert all(not frame.is_window_end for frame in frames[:-1])


def test_file_source_is_single_use(make_video):
    path = make_video("clip.mp4", frames=3)
    source = FileSource(path)
    list(source.frames())

    with pytest.raises(VideoSourceError, match="single-use"):
        list(source.frames())


def test_file_source_missing_file(tmp_path):
    with pytest.raises(VideoSourceError, match="file not found"):
        FileSource(tmp_path / "missing.mp4")


def test_file_sequence_marks_each_clip_boundary(make_video, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    make_video("clips/record_20250620_183903.mp4", frames=5, fps=5)
    make_video("clips/record_20250620_184049.mp4", frames=5, fps=5)

    source = FileSequenceSource(clips, glob="*.mp4")
    frames = list(source.frames())

    assert len(frames) == 10
    assert [frame.global_index for frame in frames] == list(range(10))
    assert frames[0].source_id == "record_20250620_183903.mp4"
    assert frames[4].source_id == "record_20250620_183903.mp4"
    assert frames[5].source_id == "record_20250620_184049.mp4"
    assert frames[4].is_window_end is True
    assert frames[5].is_window_end is False
    assert frames[9].is_window_end is True
    assert frames[5].index == 0
    assert frames[4].window_id != frames[5].window_id
    assert frames[5].window_id == "record_20250620_184049.mp4"

    expected_second_start = datetime(2025, 6, 20, 18, 40, 49, tzinfo=UTC)
    delta = abs(
        (frames[5].absolute_time - expected_second_start).total_seconds()
    )
    assert delta < 0.1


def test_file_sequence_skips_corrupt_clip(make_video, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    corrupt = clips / "record_20250620_183903.mp4"
    corrupt.write_text("not a video", encoding="utf-8")
    make_video("clips/record_20250620_184049.mp4", frames=5, fps=5)

    source = FileSequenceSource(clips, glob="*.mp4")
    frames = list(source.frames())

    assert len(frames) == 5
    assert frames[0].source_id == "record_20250620_184049.mp4"


def test_file_sequence_empty_directory(tmp_path):
    with pytest.raises(VideoSourceError, match="no files matched"):
        FileSequenceSource(tmp_path)


def test_file_sequence_close_stops_iteration(make_video, tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    make_video("clips/record_20250620_183903.mp4", frames=5, fps=5)
    make_video("clips/record_20250620_184049.mp4", frames=5, fps=5)

    source = FileSequenceSource(clips, glob="*.mp4")
    iterator = source.frames()
    next(iterator)
    source.close()

    with pytest.raises(StopIteration):
        next(iterator)


def test_bounded_queue_drops_oldest():
    queue = BoundedFrameQueue[int](maxsize=2)

    queue.put_drop_oldest(1)
    queue.put_drop_oldest(2)
    queue.put_drop_oldest(3)

    assert queue.qsize() == 2
    assert [queue.get(), queue.get()] == [2, 3]


def test_bounded_queue_rejects_zero_size():
    with pytest.raises(ValueError, match="at least 1"):
        BoundedFrameQueue[int](maxsize=0)


def test_next_backoff_is_capped():
    assert next_backoff(1, base_sec=1.0, max_sec=30.0) == 1.0
    assert next_backoff(2, base_sec=1.0, max_sec=30.0) == 2.0
    assert next_backoff(3, base_sec=1.0, max_sec=30.0) == 4.0
    assert next_backoff(10, base_sec=1.0, max_sec=30.0) == 30.0


def test_next_backoff_rejects_zero_attempt():
    with pytest.raises(ValueError, match="at least 1"):
        next_backoff(0, base_sec=1.0, max_sec=30.0)


def test_rtsp_retries_then_stops():
    calls: list[str] = []

    def factory(url: str, **kwargs: object) -> None:
        calls.append(url)
        raise ConnectionError("offline")

    source = RTSPSource(
        "rtsp://example/stream",
        source_id="cam-1",
        max_reconnect_attempts=2,
        backoff_base_sec=0.001,
        max_backoff_sec=0.002,
        container_factory=factory,
    )

    assert list(source.frames()) == []
    assert calls == ["rtsp://example/stream", "rtsp://example/stream"]
    source.close()


def test_rtsp_is_single_use():
    def factory(url: str, **kwargs: object) -> None:
        raise ConnectionError("offline")

    source = RTSPSource(
        "rtsp://example/stream",
        max_reconnect_attempts=1,
        backoff_base_sec=0.001,
        max_backoff_sec=0.002,
        container_factory=factory,
    )
    list(source.frames())

    with pytest.raises(VideoSourceError, match="single-use"):
        list(source.frames())
    source.close()


def test_rtsp_rejects_invalid_transport():
    with pytest.raises(ValueError, match="transport"):
        RTSPSource("rtsp://example/stream", transport="http")


def test_rtsp_requires_url():
    with pytest.raises(ValueError, match="url"):
        RTSPSource("")


def test_rtsp_source_id_is_safe_label():
    source = RTSPSource("rtsp://user:secret@example/stream", source_id="cam-1")
    assert source.source_id == "cam-1"
    source.close()


class _FakeRaw:
    def to_ndarray(self, format: str) -> np.ndarray:
        assert format == "bgr24"
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakeStreams:
    def __init__(self) -> None:
        self.video = [object()]


class _FakeContainer:
    def __init__(self) -> None:
        self.streams = _FakeStreams()
        self.closed = False

    def decode(self, stream: object):
        yield _FakeRaw()
        yield _FakeRaw()

    def close(self) -> None:
        self.closed = True


def test_rtsp_yields_frames_from_fake_container():
    containers: list[_FakeContainer] = []

    def factory(url: str, **kwargs: object) -> _FakeContainer:
        container = _FakeContainer()
        containers.append(container)
        return container

    source = RTSPSource(
        "rtsp://example/stream",
        source_id="cam-1",
        max_reconnect_attempts=1,
        backoff_base_sec=0.001,
        max_backoff_sec=0.002,
        container_factory=factory,
    )

    frames = list(source.frames())

    assert len(frames) == 2
    assert frames[0].source_id == "cam-1"
    assert frames[0].image.shape == (2, 2, 3)
    assert frames[0].is_window_end is False
    assert containers[0].closed is True
