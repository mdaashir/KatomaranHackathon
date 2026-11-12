"""Config-override and folder-job contract tests."""

from __future__ import annotations

from dataclasses import asdict
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api import create_app
from src.api.jobs import JobManager, list_folders
from src.config import load_config


def _fake_runner(captured: list, started: threading.Event, release: threading.Event):
    def run(config, *, stop_event, frame_sink):
        captured.append(config)
        started.set()
        release.wait(2)
        return SimpleNamespace(
            stopped_by_signal=stop_event.is_set(),
            to_dict=lambda: {"frames_processed": 0},
        )

    return run


def _client(tmp_path, *, captured=None):
    base = load_config("config.json")
    base_path = tmp_path / "config.json"
    base_path.write_text(json.dumps(asdict(base), indent=2), encoding="utf-8")
    overrides_path = tmp_path / "runtime_overrides.json"
    captured = captured if captured is not None else []
    started = threading.Event()
    release = threading.Event()
    manager = JobManager(
        base,
        input_root=tmp_path,
        runner=_fake_runner(captured, started, release),
    )
    app = create_app(
        base,
        config_path=base_path,
        overrides_path=overrides_path,
        input_root=tmp_path,
        job_manager=manager,
    )
    return TestClient(app), base_path, overrides_path, started, release


def test_config_get_apply_and_validation(tmp_path):
    client, _, overrides_path, _, _ = _client(tmp_path)

    snapshot = client.get("/config").json()
    assert snapshot["effective"]["detector"]["conf_threshold"] == 0.5
    assert snapshot["overrides"] == {}

    applied = client.put(
        "/config",
        json={"detector": {"conf_threshold": 0.72}},
    )
    assert applied.status_code == 200
    assert applied.json()["effective"]["detector"]["conf_threshold"] == 0.72
    assert json.loads(overrides_path.read_text())["detector"]["conf_threshold"] == 0.72

    assert client.put("/config", json={"frame_skip": 2}).status_code == 422
    assert (
        client.put(
            "/config",
            json={"detector": {"conf_threshold": 1.5}},
        ).status_code
        == 422
    )
    invalid = client.put(
        "/config",
        json={"recognition": {"min_face_size": 1}},
    )
    assert invalid.status_code == 400
    assert "min_face_size" in invalid.json()["detail"]


def test_override_reaches_next_job(tmp_path):
    captured = []
    client, _, _, started, release = _client(tmp_path, captured=captured)
    (tmp_path / "clip.mp4").write_bytes(b"x")

    client.put("/config", json={"detector": {"conf_threshold": 0.71}})
    created = client.post(
        "/jobs",
        json={"source_type": "file", "path": "clip.mp4"},
    )
    assert created.status_code == 201
    assert started.wait(2)
    assert captured[0].detector.conf_threshold == 0.71
    client.post(f"/jobs/{created.json()['job_id']}/stop")
    release.set()


def test_reset_and_save_default(tmp_path):
    client, base_path, overrides_path, _, _ = _client(tmp_path)
    client.put("/config", json={"detector": {"conf_threshold": 0.74}})

    saved = client.post("/config/save-default")
    assert saved.status_code == 200
    assert saved.json()["overrides"] == {}
    persisted = json.loads(base_path.read_text())
    assert persisted["detector"]["conf_threshold"] == 0.74
    assert not overrides_path.exists()

    client.put("/config", json={"tracking": {"exit_timeout_frames": 90}})
    reset = client.post("/config/reset")
    assert reset.status_code == 200
    assert reset.json()["effective"]["tracking"]["exit_timeout_frames"] == 75


def test_folders_and_folder_job(tmp_path):
    client, _, _, started, release = _client(tmp_path)
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "a.mp4").write_bytes(b"x")
    (batch / "b.txt").write_bytes(b"x")

    assert client.get("/folders").json()["folders"] == ["batch"]
    created = client.post(
        "/jobs",
        json={"source_type": "folder", "path": "batch"},
    )
    assert created.status_code == 201
    assert started.wait(2)
    client.post(f"/jobs/{created.json()['job_id']}/stop")
    release.set()


def test_list_folders_hides_files_and_dotdirs(tmp_path):
    (tmp_path / "batch").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "clip.mp4").write_bytes(b"x")
    assert list_folders(tmp_path) == ["batch"]
    assert list_folders(tmp_path / "missing") == []
