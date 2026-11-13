"""Tests for the server-side camera diagnostic endpoint."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np

from src.api import create_app


class FakeWebcam:
    fps = 25.0

    def __init__(self, index: int) -> None:
        self.index = index
        self.closed = False

    def frames(self):
        yield SimpleNamespace(image=np.zeros((48, 64, 3), dtype=np.uint8))

    def close(self) -> None:
        self.closed = True


def test_webcam_probe_returns_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.app.WebcamSource", FakeWebcam)
    client = TestClient(create_app(input_root=tmp_path))

    response = client.get("/webcam/test", params={"device_index": 1}).json()

    assert response["ok"] is True
    assert response["device_index"] == 1
    assert response["width"] == 64
    assert response["height"] == 48
    assert response["fps"] == 25.0


def test_webcam_probe_reports_failure(tmp_path, monkeypatch):
    def fail(_index: int):
        raise RuntimeError("camera busy")

    monkeypatch.setattr("src.api.app.WebcamSource", fail)
    client = TestClient(create_app(input_root=tmp_path))

    response = client.get("/webcam/test", params={"device_index": 0}).json()

    assert response["ok"] is False
    assert "camera busy" in response["error"]
