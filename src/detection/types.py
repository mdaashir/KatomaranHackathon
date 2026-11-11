"""Face detection result types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """One detected face in native image coordinates.

    Attributes:
        bbox: ``(x1, y1, x2, y2)`` in pixels of the source frame.
        confidence: Detector confidence in ``[0, 1]``.
        keypoints: Optional ``5x2`` array of facial landmarks. The bundled
            ``yolov8n-face`` checkpoint is detect-only, so this is normally
            ``None``; Phase 4 derives alignment points with InsightFace.
    """

    bbox: tuple[float, float, float, float]
    confidence: float
    keypoints: np.ndarray | None = None

    @property
    def width(self) -> float:
        """Bounding-box width in pixels."""

        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Bounding-box height in pixels."""

        return self.bbox[3] - self.bbox[1]

    @property
    def size(self) -> float:
        """Smaller bounding-box side, used by the native size gates."""

        return min(self.width, self.height)
