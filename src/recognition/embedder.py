"""ArcFace embedding pipeline: landmarks -> pose gate -> align -> embed."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.config.models import RecognitionConfig
from src.detection.types import FaceDetection
from src.recognition.alignment import (
    align_face,
    five_points_from_106,
    pose_is_frontal,
)
from src.recognition.types import EmbeddingResult


class FaceRuntime(Protocol):
    """Minimal runtime contract used by the embedder and its tests."""

    def landmarks(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray: ...

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray: ...


class ArcFaceEmbedder:
    """Produce a normalised ArcFace vector for one quality-passed detection."""

    def __init__(
        self,
        runtime: FaceRuntime,
        config: RecognitionConfig,
    ) -> None:
        self._runtime = runtime
        self._config = config

    def embed(
        self,
        image: np.ndarray,
        detection: FaceDetection,
        quality_score: float,
    ) -> EmbeddingResult | None:
        """Return an embedding, or ``None`` when the pose is unusable."""

        if detection.keypoints is not None and detection.keypoints.shape == (5, 2):
            five_points = np.asarray(detection.keypoints, dtype=np.float32)
        else:
            landmarks = self._runtime.landmarks(image, detection.bbox)
            five_points = five_points_from_106(landmarks)

        if not pose_is_frontal(five_points, detection.width):
            return None

        aligned = align_face(image, five_points)
        vector = self._runtime.embed(aligned)
        return EmbeddingResult(
            vector=vector,
            five_points=five_points,
            aligned=aligned,
            quality_score=quality_score,
        )
