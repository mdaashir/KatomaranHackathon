"""Tests for the ArcFace embedder with a fake runtime."""

from __future__ import annotations

import numpy as np
import pytest

from src.config.models import RecognitionConfig
from src.detection.types import FaceDetection
from src.recognition.alignment import five_points_from_106
from src.recognition.embedder import ArcFaceEmbedder
from src.recognition.runtime import RecognitionError


def _landmarks(nose_x: float = 120.0) -> np.ndarray:
    points = np.zeros((106, 2), dtype=np.float32)
    points[33:43] = np.asarray(
        [[100 + i, 100 + (i % 2)] for i in range(10)], dtype=np.float32
    )
    points[87:97] = np.asarray(
        [[140 + i, 100 + (i % 2)] for i in range(10)], dtype=np.float32
    )
    points[86] = (nose_x, 115.0)
    points[52] = (105.0, 130.0)
    points[61] = (135.0, 130.0)
    return points


class _FakeRuntime:
    def __init__(self, landmarks: np.ndarray, error: Exception | None = None):
        self._landmarks = landmarks
        self._error = error
        self.landmark_calls: list[tuple[float, float, float, float]] = []
        self.embed_calls: list[np.ndarray] = []

    def landmarks(self, image, bbox):
        self.landmark_calls.append(bbox)
        if self._error is not None:
            raise self._error
        return self._landmarks

    def embed(self, aligned_bgr):
        self.embed_calls.append(aligned_bgr)
        vector = np.ones(512, dtype=np.float32)
        return vector / np.linalg.norm(vector)


def _config() -> RecognitionConfig:
    return RecognitionConfig(model_dir="unused")


def _detection(**kwargs) -> FaceDetection:
    values = {"bbox": (100, 100, 200, 220), "confidence": 0.9}
    values.update(kwargs)
    return FaceDetection(**values)


def test_embed_produces_vector_and_aligned_crop():
    runtime = _FakeRuntime(_landmarks())
    embedder = ArcFaceEmbedder(runtime, _config())

    result = embedder.embed(
        np.zeros((480, 640, 3), dtype=np.uint8),
        _detection(),
        quality_score=0.8,
    )

    assert result is not None
    assert result.vector.shape == (512,)
    assert result.aligned.shape == (112, 112, 3)
    assert result.five_points.shape == (5, 2)
    assert result.quality_score == 0.8
    assert runtime.embed_calls[0].shape == (112, 112, 3)


def test_detection_keypoints_skip_the_landmark_model():
    five = five_points_from_106(_landmarks())
    runtime = _FakeRuntime(_landmarks())
    embedder = ArcFaceEmbedder(runtime, _config())

    result = embedder.embed(
        np.zeros((480, 640, 3), dtype=np.uint8),
        _detection(keypoints=five),
        quality_score=0.8,
    )

    assert result is not None
    assert runtime.landmark_calls == []


def test_profile_pose_is_rejected_without_embedding():
    runtime = _FakeRuntime(_landmarks(nose_x=170.0))
    embedder = ArcFaceEmbedder(runtime, _config())

    result = embedder.embed(
        np.zeros((480, 640, 3), dtype=np.uint8),
        _detection(),
        quality_score=0.8,
    )

    assert result is None
    assert runtime.embed_calls == []


def test_landmark_error_propagates():
    runtime = _FakeRuntime(_landmarks(), error=RecognitionError("boom"))
    embedder = ArcFaceEmbedder(runtime, _config())

    with pytest.raises(RecognitionError, match="boom"):
        embedder.embed(
            np.zeros((480, 640, 3), dtype=np.uint8),
            _detection(),
            quality_score=0.8,
        )
