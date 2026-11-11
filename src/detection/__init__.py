"""Face detection layer."""

from src.detection.types import FaceDetection
from src.detection.yolo_detector import DetectorError, YoloFaceDetector

__all__ = ["DetectorError", "FaceDetection", "YoloFaceDetector"]
