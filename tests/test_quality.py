"""Tests for the pre-embedding quality gate."""

from __future__ import annotations

import cv2
import numpy as np

from src.config.models import RecognitionConfig
from src.detection.types import FaceDetection
from src.recognition.quality import QualityGate, crop_face


def _config(**overrides) -> RecognitionConfig:
    values = {
        "model_dir": "unused",
        "min_face_size": 60,
        "min_confidence": 0.6,
        "blur_threshold": 50.0,
        "register_quality_threshold": 0.5,
    }
    values.update(overrides)
    return RecognitionConfig(**values)


def _sharp_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)


def _detection(bbox=(100, 100, 200, 220), confidence=0.9) -> FaceDetection:
    return FaceDetection(bbox=bbox, confidence=confidence)


def test_sharp_large_face_passes():
    gate = QualityGate(_config())

    result = gate.assess(_sharp_image(), _detection())

    assert result.passed is True
    assert result.reason is None
    assert 0.0 <= result.score <= 1.0
    assert result.crop.shape[2] == 3
    assert gate.register_quality_ok(result) is True


def test_small_face_is_rejected_but_still_cropped():
    gate = QualityGate(_config())

    result = gate.assess(_sharp_image(), _detection(bbox=(100, 100, 130, 140)))

    assert result.passed is False
    assert result.reason == "too_small"
    assert result.crop.size > 0


def test_low_confidence_is_rejected():
    gate = QualityGate(_config())

    result = gate.assess(_sharp_image(), _detection(confidence=0.2))

    assert result.passed is False
    assert result.reason == "low_confidence"


def test_blurred_face_is_rejected():
    gate = QualityGate(_config())
    blurred = cv2.GaussianBlur(_sharp_image(), (41, 41), 0)

    result = gate.assess(blurred, _detection())

    assert result.passed is False
    assert result.reason == "blurry"


def test_register_quality_threshold_can_reject_a_passed_face():
    gate = QualityGate(_config(register_quality_threshold=0.99))

    result = gate.assess(_sharp_image(), _detection(confidence=0.62))

    assert result.passed is True
    assert gate.register_quality_ok(result) is False


def test_crop_face_uses_margin_and_clamps():
    image = np.zeros((200, 200, 3), dtype=np.uint8)

    crop, (x1, y1, x2, y2) = crop_face(image, (10, 10, 110, 110), margin=0.1)

    assert (x1, y1, x2, y2) == (0, 0, 120, 120)
    assert crop.shape == (120, 120, 3)


def test_crop_face_rejects_invalid_bbox():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    try:
        crop_face(image, (100, 100, 50, 50))
    except ValueError as exc:
        assert "invalid face crop" in str(exc)
    else:
        raise AssertionError("expected ValueError")
