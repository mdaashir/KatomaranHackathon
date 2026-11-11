"""Quality gate applied before embedding and registration."""

from __future__ import annotations

import cv2
import numpy as np

from src.config.models import RecognitionConfig
from src.detection.types import FaceDetection
from src.recognition.types import QualityResult


def crop_face(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    margin: float = 0.2,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a face with a proportional margin and return the crop offset.

    Args:
        image: Source BGR image.
        bbox: ``(x1, y1, x2, y2)`` in native pixels.
        margin: Extra margin as a fraction of the face size.

    Returns:
        ``(crop, (x1, y1, x2, y2))`` with integer, clamped coordinates.
    """

    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    pad_x = width * margin
    pad_y = height * margin

    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(image.shape[1], int(round(x2 + pad_x)))
    bottom = min(image.shape[0], int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid face crop for bbox {bbox}")

    return image[top:bottom, left:right].copy(), (left, top, right, bottom)


class QualityGate:
    """Reject faces that are too small, uncertain or blurry."""

    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config

    def assess(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        *,
        margin: float = 0.2,
    ) -> QualityResult:
        """Evaluate one detection and always return a face crop."""

        crop, _ = crop_face(image, detection.bbox, margin=margin)

        if detection.size < self._config.min_face_size:
            return QualityResult(False, 0.0, "too_small", crop)
        if detection.confidence < self._config.min_confidence:
            return QualityResult(False, 0.0, "low_confidence", crop)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self._config.blur_threshold:
            return QualityResult(False, 0.0, "blurry", crop)

        size_score = min(
            1.0,
            detection.size / (2.0 * self._config.min_face_size),
        )
        confidence_score = min(1.0, max(0.0, detection.confidence))
        sharpness_score = min(
            1.0,
            sharpness / (2.0 * self._config.blur_threshold),
        )
        score = (
            0.4 * size_score
            + 0.3 * confidence_score
            + 0.3 * sharpness_score
        )
        return QualityResult(True, float(score), None, crop)

    def register_quality_ok(self, result: QualityResult) -> bool:
        """Return whether a passed result clears the registration bar."""

        return (
            result.passed
            and result.score >= self._config.register_quality_threshold
        )
