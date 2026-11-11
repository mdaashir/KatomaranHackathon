"""Best-crop cache used for entry/exit images.

Exit events fire after the person is no longer in the frame, so the pipeline
keeps the largest (closest) crop seen for each tracker ID.
"""

from __future__ import annotations

import numpy as np


class CropCache:
    """Keep the largest crop per track ID."""

    def __init__(self) -> None:
        self._crops: dict[int, np.ndarray] = {}
        self._scores: dict[int, float] = {}

    def update(self, track_id: int, crop: np.ndarray | None) -> None:
        """Store *crop* when it is at least as large as the current best."""

        if crop is None or crop.size == 0:
            return
        score = float(min(crop.shape[0], crop.shape[1]))
        if track_id not in self._scores or score >= self._scores[track_id]:
            self._crops[track_id] = crop
            self._scores[track_id] = score

    def get(self, track_id: int) -> np.ndarray | None:
        """Return the best crop for a track, or ``None``."""

        return self._crops.get(track_id)

    def pop(self, track_id: int) -> np.ndarray | None:
        """Remove and return a track's crop."""

        self._scores.pop(track_id, None)
        return self._crops.pop(track_id, None)

    def clear(self) -> None:
        """Drop every cached crop."""

        self._crops.clear()
        self._scores.clear()

    def __len__(self) -> int:
        return len(self._crops)
