"""Best-crop cache used for entry/exit images.

Exit events fire after the person is no longer in the frame, so the pipeline
keeps the best crop seen for each tracker ID. A quality-passing face crop
always wins over a merely large fallback crop (for example, the back of a head
when someone turns away).
"""

from __future__ import annotations

import numpy as np


class CropCache:
    """Keep the best quality crop, falling back to the largest crop."""

    def __init__(self) -> None:
        self._quality_crops: dict[int, np.ndarray] = {}
        self._quality_scores: dict[int, float] = {}
        self._fallback_crops: dict[int, np.ndarray] = {}
        self._fallback_sizes: dict[int, float] = {}

    def update(
        self,
        track_id: int,
        crop: np.ndarray | None,
        quality_score: float | None = None,
    ) -> None:
        """Store a crop.

        Args:
            track_id: Tracker ID.
            crop: BGR face crop.
            quality_score: Quality-gate score in ``[0, 1]``. When supplied,
                the crop competes only with other quality crops. When omitted,
                it is a fallback and competes only by size.
        """

        if crop is None or crop.size == 0:
            return

        if quality_score is not None:
            previous = self._quality_scores.get(track_id)
            if previous is None or quality_score >= previous:
                self._quality_crops[track_id] = crop
                self._quality_scores[track_id] = float(quality_score)
            return

        size = float(min(crop.shape[0], crop.shape[1]))
        previous_size = self._fallback_sizes.get(track_id)
        if previous_size is None or size >= previous_size:
            self._fallback_crops[track_id] = crop
            self._fallback_sizes[track_id] = size

    def get(self, track_id: int) -> np.ndarray | None:
        """Return the quality crop, else the largest fallback crop."""

        quality = self._quality_crops.get(track_id)
        if quality is not None:
            return quality
        return self._fallback_crops.get(track_id)

    def pop(self, track_id: int) -> np.ndarray | None:
        """Remove and return a track's best crop."""

        quality = self._quality_crops.pop(track_id, None)
        self._quality_scores.pop(track_id, None)
        fallback = self._fallback_crops.pop(track_id, None)
        self._fallback_sizes.pop(track_id, None)
        return quality if quality is not None else fallback

    def clear(self) -> None:
        """Drop every cached crop."""

        self._quality_crops.clear()
        self._quality_scores.clear()
        self._fallback_crops.clear()
        self._fallback_sizes.clear()

    def __len__(self) -> int:
        return len(set(self._quality_crops) | set(self._fallback_crops))
