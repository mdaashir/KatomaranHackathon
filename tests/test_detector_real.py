"""Real-image detector smoke tests.

These run only when the bundled weights and sample videos are present. They
use the real Ultralytics model, so they also verify the checkpoint loads and
returns the expected detect-only output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ultralytics")

from src.config.models import DetectorConfig
from src.detection import YoloFaceDetector
from src.video import FileSource

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "yolov8n-face.pt"
CROWDED = ROOT / "src" / "video" / "video_sample1.mp4"
FRONT = ROOT / "src" / "video" / "record_20250620_183903.mp4"

pytestmark = pytest.mark.skipif(
    not MODEL.is_file(), reason="yolov8n-face.pt is not available"
)


@pytest.fixture(scope="module")
def detector() -> YoloFaceDetector:
    return YoloFaceDetector(
        DetectorConfig(
            model_path=str(MODEL),
            conf_threshold=0.5,
            imgsz=1280,
            min_face_size=40,
            device="cpu",
        )
    )


def _first_frame(path: Path):
    source = FileSource(path)
    try:
        return next(source.frames()).image
    finally:
        source.close()


def _assert_boxes_in_frame(detections, image) -> None:
    height, width = image.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        assert 0.0 <= x1 < x2 <= width - 1
        assert 0.0 <= y1 < y2 <= height - 1
        assert detection.confidence >= 0.5


def test_real_crowded_frame_detects_faces(detector):
    if not CROWDED.is_file():
        pytest.skip("crowded sample clip is not available")

    image = _first_frame(CROWDED)
    detections = detector.detect(image)

    assert len(detections) >= 5
    _assert_boxes_in_frame(detections, image)


def test_real_front_entry_frame_detects_face(detector):
    if not FRONT.is_file():
        pytest.skip("front-entry clip is not available")

    image = _first_frame(FRONT)
    detections = detector.detect(image)

    assert len(detections) >= 1
    _assert_boxes_in_frame(detections, image)
