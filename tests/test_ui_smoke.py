"""Streamlit boot smoke test with no API running."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_boots_without_api(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("FACETRACKER_REFRESH", "0")
    app_path = Path(__file__).resolve().parents[1] / "ui" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
