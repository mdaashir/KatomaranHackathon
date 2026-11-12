"""InsightFace recognition and landmark runtime.

Only the recognition and 106-point landmark ONNX models are loaded. YOLO
already supplies the face box, so InsightFace's detector is never run on the
full frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config.models import RecognitionConfig
from src.logging_setup import get_logger
from src.recognition.providers import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    ctx_id_for,
    prepare_cuda_dlls,
    resolve_providers,
)

logger = get_logger(__name__)

RECOGNITION_FILE = "w600k_r50.onnx"
LANDMARK_FILE = "2d106det.onnx"


class RecognitionError(RuntimeError):
    """Raised when the recognition models cannot be loaded or run."""


class InsightFaceRuntime:
    """Lazy loader for the ArcFace recognizer and 106-point landmark model."""

    def __init__(self, config: RecognitionConfig) -> None:
        self._config = config
        self._recognition = None
        self._landmark = None
        self._providers: list[str] | None = None

    @property
    def providers(self) -> list[str]:
        """ONNX Runtime providers actually active for this runtime."""

        self._ensure_loaded()
        assert self._providers is not None
        return list(self._providers)

    @property
    def using_cuda(self) -> bool:
        """True when the CUDA provider is active."""

        return CUDA_PROVIDER in self.providers

    def landmarks(
        self,
        image: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Return 106 landmarks in original-image coordinates."""

        self._ensure_loaded()
        from insightface.app.common import Face

        face = Face(
            bbox=np.asarray(bbox, dtype=np.float32),
            det_score=1.0,
        )
        try:
            points = self._landmark.get(image, face)
        except Exception as exc:
            raise RecognitionError(f"landmark detection failed: {exc}") from exc
        return np.asarray(points, dtype=np.float32)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """Return an L2-normalised 512-d ArcFace embedding."""

        self._ensure_loaded()
        try:
            feature = self._recognition.get_feat(aligned_bgr)
        except Exception as exc:
            raise RecognitionError(f"ArcFace embedding failed: {exc}") from exc

        vector = np.asarray(feature, dtype=np.float32).flatten()
        if vector.size != 512:
            raise RecognitionError(f"expected a 512-d embedding, got {vector.size}")
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise RecognitionError("ArcFace returned a zero embedding")
        return vector / norm

    def _ensure_loaded(self) -> None:
        if self._recognition is not None:
            return

        model_dir = Path(self._config.model_dir)
        recognition_path = model_dir / RECOGNITION_FILE
        landmark_path = model_dir / LANDMARK_FILE
        missing = [
            str(path)
            for path in (recognition_path, landmark_path)
            if not path.is_file()
        ]
        if missing:
            raise RecognitionError(
                "InsightFace models are missing: "
                + ", ".join(missing)
                + ". Run 'python scripts/download_models.py'."
            )

        prepare_cuda_dlls()
        providers = resolve_providers(self._config.device)
        context_id = ctx_id_for(self._config.device, providers)

        try:
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            from insightface.model_zoo.landmark import Landmark
        except ImportError as exc:
            raise RecognitionError(
                "insightface is not installed; run 'uv pip install -r requirements.txt'"
            ) from exc

        try:
            recognition = ArcFaceONNX(str(recognition_path))
            recognition.prepare(ctx_id=context_id, providers=providers)
            landmark = Landmark(str(landmark_path))
            landmark.prepare(ctx_id=context_id, providers=providers)
        except Exception as exc:
            raise RecognitionError(
                f"failed to load InsightFace models from {model_dir}: {exc}"
            ) from exc

        active = [
            provider
            for provider in providers
            if provider in recognition.session.get_providers()
            and provider in landmark.session.get_providers()
        ]
        if CUDA_PROVIDER in providers and CUDA_PROVIDER not in active:
            logger.warning(
                "CUDA requested but active providers are recognition=%s landmark=%s",
                recognition.session.get_providers(),
                landmark.session.get_providers(),
            )

        self._recognition = recognition
        self._landmark = landmark
        self._providers = active or [CPU_PROVIDER]
        logger.info(
            "loaded InsightFace models from %s (providers=%s)",
            model_dir,
            active,
        )
