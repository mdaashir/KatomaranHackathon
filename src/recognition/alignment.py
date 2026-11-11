"""Five-point alignment to the ArcFace 112x112 template."""

from __future__ import annotations

import cv2
import numpy as np

# Standard ArcFace destination landmarks in a 112x112 crop.
ARCFACE_DST = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# The 106-point InsightFace landmark layout stores each eye as a 10-point
# contour. The mouth corners are 52 and 61; 86 is the nose tip.
_LEFT_EYE_SLICE = slice(33, 43)
_RIGHT_EYE_SLICE = slice(87, 97)
_NOSE_INDEX = 86
_LEFT_MOUTH_INDEX = 52
_RIGHT_MOUTH_INDEX = 61


def five_points_from_106(points: np.ndarray) -> np.ndarray:
    """Reduce 106 InsightFace landmarks to the five ArcFace points.

    Args:
        points: ``(106, 2)`` landmarks in original-image coordinates.

    Raises:
        ValueError: if the shape is not ``(106, 2)``.
    """

    points = np.asarray(points, dtype=np.float32)
    if points.shape != (106, 2):
        raise ValueError(f"expected 106x2 landmarks, got {points.shape}")

    left_eye = points[_LEFT_EYE_SLICE].mean(axis=0)
    right_eye = points[_RIGHT_EYE_SLICE].mean(axis=0)
    nose = points[_NOSE_INDEX]
    left_mouth = points[_LEFT_MOUTH_INDEX]
    right_mouth = points[_RIGHT_MOUTH_INDEX]
    return np.asarray(
        [left_eye, right_eye, nose, left_mouth, right_mouth],
        dtype=np.float32,
    )


def align_face(
    image: np.ndarray,
    five_points: np.ndarray,
    image_size: int = 112,
) -> np.ndarray:
    """Warp a face to the ArcFace template.

    Args:
        image: Source BGR image.
        five_points: ``(5, 2)`` landmarks in source coordinates.
        image_size: Output square size; 112 is the ArcFace standard.

    Raises:
        ValueError: if the landmarks or the affine estimate are invalid.
    """

    points = np.asarray(five_points, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError(f"expected 5x2 landmarks, got {points.shape}")

    destination = ARCFACE_DST * (image_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(
        points,
        destination,
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise ValueError("could not estimate a face alignment transform")

    return cv2.warpAffine(
        image,
        matrix,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def pose_is_frontal(five_points: np.ndarray, face_width: float) -> bool:
    """Return whether five landmarks describe a usable frontal face."""

    if face_width <= 0:
        return False

    points = np.asarray(five_points, dtype=np.float32)
    if points.shape != (5, 2):
        return False

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    if eye_distance < 0.25 * face_width:
        return False

    eye_mid = (left_eye + right_eye) / 2.0
    if abs(float(nose[0] - eye_mid[0])) > 0.35 * eye_distance:
        return False
    if float(nose[1]) <= float(eye_mid[1]):
        return False
    if float(left_mouth[1]) <= float(nose[1]):
        return False
    if float(right_mouth[1]) <= float(nose[1]):
        return False

    mouth_width = float(np.linalg.norm(right_mouth - left_mouth))
    if mouth_width < 0.4 * eye_distance:
        return False
    return True
