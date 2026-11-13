"""Tests for uploads, live frames, and the background job manager."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api import create_app
from src.api.jobs import JobManager, list_videos
from src.api.live import LiveFrameBuffer
from src.api.models import JobRequest
from src.config import load_config
from src.tracking import TrackedBox
from src.video import Frame

UTC = timezone.utc


def _frame() -> Frame:
    return Frame(
        index=0,
        global_index=0,
        timestamp_s=0.0,
        absolute_time=datetime(2025, 6, 20, 18, 39, 3, tzinfo=UTC),
        image=np.zeros((32, 32, 3), dtype=np.uint8),
        source_id="test",
        window_id="window",
    )


def test_live_buffer_only_copies_for_viewers():
    buffer = LiveFrameBuffer()
    boxes = (TrackedBox(track_id=1, bbox=(2, 2, 20, 20), confidence=0.9),)
    buffer.publish(_frame(), boxes, {1: 7})
    assert buffer.wait(0.01) is None

    buffer.acquire()
    try:
        buffer.publish(_frame(), boxes, {1: 7})
        live = buffer.wait(0.1)
        assert live is not None
        assert live.face_ids == {1: 7}
        assert buffer.encode(live).startswith(b"\xff\xd8")
        snapshot = buffer.snapshot()
        assert snapshot["frames_published"] == 1
        assert snapshot["width"] == 32
        assert snapshot["height"] == 32
    finally:
        buffer.release()

    assert buffer.wait(0.01) is None


def _fake_runner(started: threading.Event, release: threading.Event):
    def run(config, *, stop_event, frame_sink):
        started.set()
        release.wait(2)
        return SimpleNamespace(
            stopped_by_signal=stop_event.is_set(),
            to_dict=lambda: {"frames_processed": 0},
        )

    return run


def test_job_manager_runs_and_stops(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-video")
    started = threading.Event()
    release = threading.Event()
    manager = JobManager(
        load_config("config.json"),
        input_root=tmp_path,
        runner=_fake_runner(started, release),
    )

    job = manager.start(JobRequest(source_type="file", path="clip.mp4"))
    assert started.wait(2)
    assert manager.get(job.job_id).status == "running"

    stopped = manager.stop(job.job_id)
    assert stopped is not None
    assert stopped.stop_event.is_set()
    release.set()
    deadline = time.time() + 2
    while manager.get(job.job_id).status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert manager.get(job.job_id).status == "stopped"
    assert manager.list()[0].job_id == job.job_id


def test_job_path_and_rtsp_validation(tmp_path):
    manager = JobManager(
        load_config("config.json"),
        input_root=tmp_path,
        runner=_fake_runner(threading.Event(), threading.Event()),
    )
    with pytest.raises(ValueError, match="inside"):
        manager.start(JobRequest(source_type="file", path="../escape.mp4"))
    with pytest.raises(ValueError, match="inside"):
        manager.start(JobRequest(source_type="file", path="sub/clip.mp4"))
    with pytest.raises(ValueError, match="rtsp"):
        manager.start(JobRequest(source_type="rtsp", url="http://camera"))


def test_webcam_job_uses_device_index(tmp_path):
    manager = JobManager(
        load_config("config.json"),
        input_root=tmp_path,
        runner=_fake_runner(threading.Event(), threading.Event()),
    )
    job = manager.start(JobRequest(source_type="webcam", device_index=2))

    assert job.config.source.type == "webcam"
    assert job.config.source.device_index == 2
    assert job.label == "Webcam 2"


def test_upload_and_job_routes(tmp_path):
    video = tmp_path / "existing.mp4"
    video.write_bytes(b"x")
    started = threading.Event()
    release = threading.Event()
    manager = JobManager(
        load_config("config.json"),
        input_root=tmp_path,
        runner=_fake_runner(started, release),
    )
    app = create_app(
        load_config("config.json"),
        input_root=tmp_path,
        job_manager=manager,
    )
    client = TestClient(app)

    assert client.get("/videos").json()["videos"] == ["existing.mp4"]
    stream = client.get("/stream/status").json()
    assert stream["frames_published"] == 0
    assert stream["has_frame"] is False
    upload = client.post(
        "/upload",
        files={"files": ("new.mp4", b"video-bytes", "video/mp4")},
    )
    assert upload.status_code == 200
    name = upload.json()["uploaded"][0]["name"]
    assert name.endswith(".mp4")
    assert (tmp_path / name).read_bytes() == b"video-bytes"

    created = client.post(
        "/jobs",
        json={"source_type": "file", "path": name},
    )
    assert created.status_code == 201
    job_id = created.json()["job_id"]
    assert started.wait(2)
    assert client.get(f"/jobs/{job_id}").json()["status"] == "running"

    stopped = client.post(f"/jobs/{job_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "running"
    release.set()


def test_upload_rejects_unknown_type(tmp_path):
    manager = JobManager(
        load_config("config.json"),
        input_root=tmp_path,
        runner=_fake_runner(threading.Event(), threading.Event()),
    )
    app = create_app(
        load_config("config.json"),
        input_root=tmp_path,
        job_manager=manager,
    )
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"files": ("notes.txt", b"no", "text/plain")},
    )
    assert response.status_code == 400


def test_list_videos(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.txt").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    assert list_videos(tmp_path) == ["a.mp4"]
    assert list_videos(Path(tmp_path) / "missing") == []
