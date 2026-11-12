"""Atomic JPEG storage for entry/exit face crops."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import cv2
import numpy as np

from src.config.models import StorageConfig
from src.logging_setup import get_logger

logger = get_logger(__name__)


class ImageStoreError(RuntimeError):
    """Raised when a crop cannot be encoded or written."""


class ImageStore:
    """Write face crops to ``logs/entries|exits/YYYY-MM-DD`` atomically."""

    def __init__(self, config: StorageConfig) -> None:
        self._root = Path(config.log_dir)
        self._size = config.image_size
        self._quality = config.jpeg_quality

    def save(
        self,
        crop: np.ndarray | None,
        *,
        event_type: str,
        occurred_at: datetime,
        face_id: int | None,
        visit_id: int,
        track_id: int,
    ) -> Path:
        """Encode and atomically write one crop.

        A black placeholder is written when *crop* is ``None`` so an event is
        never lost; the caller should log the missing crop separately.
        """

        if event_type not in {"entry", "exit"}:
            raise ImageStoreError(
                f"event_type: expected 'entry' or 'exit', got '{event_type}'"
            )

        if crop is None:
            logger.warning(
                "missing crop for visit %s track %s; writing placeholder",
                visit_id,
                track_id,
            )
            image = np.zeros((self._size, self._size, 3), dtype=np.uint8)
        else:
            image = self._prepare(crop)

        success, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._quality)],
        )
        if not success:
            raise ImageStoreError("JPEG encoding failed")

        timestamp = occurred_at.astimezone(timezone.utc)
        date_part = timestamp.strftime("%Y-%m-%d")
        stamp = timestamp.strftime("%Y-%m-%dT%H-%M-%S.%f")[:-3] + "Z"
        face_part = f"face{face_id}" if face_id is not None else "faceanon"
        filename = f"{stamp}_{face_part}_visit{visit_id}_track{track_id}.jpg"
        folder = (
            self._root / ("entries" if event_type == "entry" else "exits") / date_part
        )
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        temporary = path.with_name(path.name + ".tmp")

        try:
            with temporary.open("wb") as handle:
                handle.write(buffer.tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise ImageStoreError(f"cannot write {path}: {exc}") from exc

        return path

    def remove(self, path: Path) -> None:
        """Remove a crop written for a transaction that failed."""

        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove orphan image %s", path)

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ImageStoreError(f"expected a BGR crop, got shape {crop.shape}")

        height, width = crop.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if height >= self._size and width >= self._size
            else cv2.INTER_LINEAR
        )
        return cv2.resize(
            crop,
            (self._size, self._size),
            interpolation=interpolation,
        )
