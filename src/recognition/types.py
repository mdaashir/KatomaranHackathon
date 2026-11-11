"""Recognition results shared by the quality, embedding and registry layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityResult:
    """Outcome of the pre-embedding quality gate.

    Attributes:
        passed: True when size, confidence and sharpness all pass.
        score: Normalised quality in ``[0, 1]``.
        reason: Failure reason (``too_small``, ``low_confidence``, ``blurry``),
            or ``None`` when passed.
        crop: Face crop with a small margin, always populated for event images.
    """

    passed: bool
    score: float
    reason: str | None
    crop: np.ndarray


@dataclass(frozen=True)
class EmbeddingResult:
    """A successful ArcFace embedding."""

    vector: np.ndarray
    five_points: np.ndarray
    aligned: np.ndarray
    quality_score: float


@dataclass(frozen=True)
class MatchResult:
    """Best registry match.

    ``face_id`` is ``None`` when the best similarity is below the threshold;
    ``similarity`` is still reported so callers can log the near miss.
    """

    face_id: int | None
    similarity: float
