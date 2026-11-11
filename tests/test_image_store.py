"""Tests for atomic entry/exit crop storage."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.models import StorageConfig
from src.events import ImageStore, ImageStoreError

UTC = timezone.utc
BASE = datetime(2025, 6, 20, 18, 39, 4, 120000, tzinfo=UTC)


def _config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        db_path=str(tmp_path / "faces.db"),
        log_dir=str(tmp_path / "logs"),
        image_size=128,
        jpeg_quality=85,
    )


def test_save_entry_creates_dated_file(tmp_path):
    store = ImageStore(_config(tmp_path))
    crop = np.full((80, 60, 3), 128, dtype=np.uint8)

    path = store.save(
        crop,
        event_type="entry",
        occurred_at=BASE,
        face_id=17,
        visit_id=42,
        track_id=7,
    )

    assert path.is_file()
    assert path.parent.name == "2025-06-20"
    assert path.parent.parent.name == "entries"
    assert "face17" in path.name
    assert "visit42" in path.name
    assert "track7" in path.name
    assert ":" not in path.name
    image = cv2.imread(str(path))
    assert image is not None
    assert image.shape == (128, 128, 3)


def test_save_exit_uses_exit_directory(tmp_path):
    store = ImageStore(_config(tmp_path))

    path = store.save(
        np.zeros((40, 40, 3), dtype=np.uint8),
        event_type="exit",
        occurred_at=BASE,
        face_id=None,
        visit_id=1,
        track_id=2,
    )

    assert path.parent.parent.name == "exits"
    assert "faceanon" in path.name


def test_missing_crop_writes_placeholder(tmp_path):
    store = ImageStore(_config(tmp_path))

    path = store.save(
        None,
        event_type="entry",
        occurred_at=BASE,
        face_id=None,
        visit_id=1,
        track_id=1,
    )

    image = cv2.imread(str(path))
    assert image.shape == (128, 128, 3)
    assert float(image.mean()) == 0.0


def test_grayscale_crop_is_converted(tmp_path):
    store = ImageStore(_config(tmp_path))

    path = store.save(
        np.full((40, 40), 100, dtype=np.uint8),
        event_type="entry",
        occurred_at=BASE,
        face_id=1,
        visit_id=1,
        track_id=1,
    )

    image = cv2.imread(str(path))
    assert image.shape == (128, 128, 3)


def test_invalid_event_type_rejected(tmp_path):
    store = ImageStore(_config(tmp_path))
    with pytest.raises(ImageStoreError, match="event_type"):
        store.save(
            np.zeros((10, 10, 3), dtype=np.uint8),
            event_type="arrival",
            occurred_at=BASE,
            face_id=1,
            visit_id=1,
            track_id=1,
        )


def test_invalid_crop_shape_rejected(tmp_path):
    store = ImageStore(_config(tmp_path))
    with pytest.raises(ImageStoreError, match="BGR crop"):
        store.save(
            np.zeros((10, 10, 4), dtype=np.uint8),
            event_type="entry",
            occurred_at=BASE,
            face_id=1,
            visit_id=1,
            track_id=1,
        )


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    store = ImageStore(_config(tmp_path))
    path = store.save(
        np.zeros((20, 20, 3), dtype=np.uint8),
        event_type="entry",
        occurred_at=BASE,
        face_id=1,
        visit_id=1,
        track_id=1,
    )

    assert list(path.parent.glob("*.tmp")) == []


def test_remove_missing_file_is_safe(tmp_path):
    store = ImageStore(_config(tmp_path))
    store.remove(tmp_path / "does-not-exist.jpg")
