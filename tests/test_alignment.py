"""Tests for 5-point extraction, ArcFace alignment and pose gating."""

from __future__ import annotations

import numpy as np
import pytest

from src.recognition.alignment import (
    ARCFACE_DST,
    align_face,
    five_points_from_106,
    pose_is_frontal,
)


def _landmarks() -> np.ndarray:
    points = np.zeros((106, 2), dtype=np.float32)
    points[33:43] = np.asarray(
        [[100 + i, 100 + (i % 2)] for i in range(10)], dtype=np.float32
    )
    points[87:97] = np.asarray(
        [[140 + i, 100 + (i % 2)] for i in range(10)], dtype=np.float32
    )
    points[86] = (120.0, 115.0)
    points[52] = (105.0, 130.0)
    points[61] = (135.0, 130.0)
    return points


def test_five_points_from_106_uses_eye_centres_and_mouth_corners():
    points = _landmarks()

    five = five_points_from_106(points)

    assert five.shape == (5, 2)
    assert five[0][0] == pytest.approx(104.5)
    assert five[1][0] == pytest.approx(144.5)
    assert five[2].tolist() == [120.0, 115.0]
    assert five[3].tolist() == [105.0, 130.0]
    assert five[4].tolist() == [135.0, 130.0]


def test_five_points_rejects_wrong_shape():
    with pytest.raises(ValueError, match="106x2"):
        five_points_from_106(np.zeros((68, 2), dtype=np.float32))


def test_pose_is_frontal_accepts_symmetric_landmarks():
    five = five_points_from_106(_landmarks())
    assert pose_is_frontal(five, face_width=50.0) is True


def test_pose_rejects_profile_nose_offset():
    five = five_points_from_106(_landmarks())
    five[2] = (160.0, 115.0)
    assert pose_is_frontal(five, face_width=50.0) is False


def test_pose_rejects_tiny_eye_distance():
    five = five_points_from_106(_landmarks())
    five[1] = five[0] + np.asarray([2.0, 0.0], dtype=np.float32)
    assert pose_is_frontal(five, face_width=50.0) is False


def test_pose_rejects_mouth_above_nose():
    five = five_points_from_106(_landmarks())
    five[3][1] = 100.0
    five[4][1] = 100.0
    assert pose_is_frontal(five, face_width=50.0) is False


def test_pose_rejects_zero_face_width():
    five = five_points_from_106(_landmarks())
    assert pose_is_frontal(five, face_width=0.0) is False


def test_align_face_returns_arcface_size():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    five = five_points_from_106(_landmarks())

    aligned = align_face(image, five)

    assert aligned.shape == (112, 112, 3)
    assert aligned.dtype == np.uint8
    assert ARCFACE_DST.shape == (5, 2)


def test_align_face_can_scale_template():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    five = five_points_from_106(_landmarks())

    aligned = align_face(image, five, image_size=224)

    assert aligned.shape == (224, 224, 3)


def test_align_face_rejects_wrong_shape():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="5x2"):
        align_face(image, np.zeros((4, 2), dtype=np.float32))
