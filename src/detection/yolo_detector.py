"""YOLO face detector wrapper.

The detector is deliberately the only module that imports Ultralytics. Tests
inject a fake ``model_factory``, so the wrapper contract can be verified
without loading PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config.models import DetectorConfig
from src.detection.types import FaceDetection
from src.logging_setup import get_logger

logger = get_logger(__name__)


class DetectorError(RuntimeError):
    """Raised when the detector cannot load or run."""


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor, list or array to a NumPy array."""

    if value is None:
        return np.empty((0,))
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resolve_device(device: str) -> str | int | None:
    """Map config device names to Ultralytics device values."""

    if device == "auto":
        return None
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        return int(index) if index.isdigit() else 0
    return device


def _load_ultralytics_model(path: str) -> Any:
    """Load a YOLO checkpoint, importing Ultralytics lazily."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError(
            "ultralytics is not installed; run "
            "'uv pip install -r requirements.txt'"
        ) from exc
    return YOLO(path)


class YoloFaceDetector:
    """Detect faces with a YOLOv8 face checkpoint.

    The model is loaded lazily on the first :meth:`detect` call. On CPU, that
    first call also pays PyTorch initialization and Ultralytics warmup
    (roughly 9 seconds in the Phase 3 measurement); later calls are steady
    state.
    """

    def __init__(
        self,
        config: DetectorConfig,
        *,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Args:
            config: Detector settings from ``config.json``.
            model_factory: Optional factory returning a model-like object with
                a ``predict`` method. Tests inject a fake; production uses
                ``ultralytics.YOLO``.
        """

        if config.min_face_size < 1:
            raise ValueError("min_face_size must be at least 1")

        self._config = config
        self._model_path = Path(config.model_path)
        self._model_factory = model_factory or _load_ultralytics_model
        self._device = _resolve_device(config.device)
        self._model: Any | None = None

    @property
    def device(self) -> str | int | None:
        """Resolved Ultralytics device value."""

        return self._device

    @property
    def model_path(self) -> Path:
        """Configured checkpoint path."""

        return self._model_path

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Return faces in native coordinates, filtered by size.

        Args:
            image: BGR image at native resolution.

        Raises:
            DetectorError: if the image is malformed, weights are missing, or
                inference fails.
        """

        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectorError(
                f"expected an HxWx3 BGR image, got shape {image.shape}"
            )

        height, width = image.shape[:2]
        model = self._ensure_model()

        kwargs: dict[str, Any] = {
            "source": image,
            "imgsz": self._config.imgsz,
            "conf": self._config.conf_threshold,
            "verbose": False,
        }
        if self._device is not None:
            kwargs["device"] = self._device

        try:
            results = model.predict(**kwargs)
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = _to_numpy(getattr(boxes, "xyxy", None))
        confidence = _to_numpy(getattr(boxes, "conf", None))
        if xyxy.size == 0 or confidence.size == 0:
            return []

        keypoints_xy: np.ndarray | None = None
        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None:
            keypoints_xy = _to_numpy(getattr(keypoints, "xy", None))
            if keypoints_xy.size == 0:
                keypoints_xy = None

        detections: list[FaceDetection] = []
        for index in range(min(len(xyxy), len(confidence))):
            x1, y1, x2, y2 = (float(value) for value in xyxy[index][:4])
            x1 = min(max(x1, 0.0), float(width - 1))
            y1 = min(max(y1, 0.0), float(height - 1))
            x2 = min(max(x2, 0.0), float(width - 1))
            y2 = min(max(y2, 0.0), float(height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            detection = FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(confidence[index]),
            )
            if detection.size < self._config.min_face_size:
                continue

            keypoint_array = None
            if keypoints_xy is not None and index < len(keypoints_xy):
                keypoint_array = np.asarray(
                    keypoints_xy[index], dtype=np.float32
                )
            detections.append(
                FaceDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    keypoints=keypoint_array,
                )
            )

        return detections

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        if not self._model_path.is_file():
            raise DetectorError(
                f"YOLO face weights not found at '{self._model_path}'. "
                "Download yolov8n-face.pt into models/ (see models/README.md)."
            )

        try:
            self._model = self._model_factory(str(self._model_path))
        except DetectorError:
            raise
        except Exception as exc:
            raise DetectorError(
                f"failed to load YOLO weights '{self._model_path}': {exc}"
            ) from exc

        logger.info(
            "loaded YOLO face model from %s (imgsz=%d, device=%s)",
            self._model_path,
            self._config.imgsz,
            self._device,
        )
        return self._model
