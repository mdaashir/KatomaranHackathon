"""Real InsightFace + ArcFace validation on the provided clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("insightface")

from src.config.models import DetectorConfig, RecognitionConfig
from src.detection import YoloFaceDetector
from src.recognition import (
    ArcFaceEmbedder,
    InsightFaceRuntime,
    QualityGate,
)
from src.video import FileSource

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "insightface" / "models" / "buffalo_l"
YOLO_MODEL = ROOT / "models" / "yolov8n-face.pt"
FRONT = ROOT / "src" / "video" / "record_20250620_183903.mp4"
CROWD = ROOT / "src" / "video" / "video_sample1.mp4"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "w600k_r50.onnx").is_file()
    or not (MODEL_DIR / "2d106det.onnx").is_file(),
    reason="InsightFace models are not available",
)


def _config() -> RecognitionConfig:
    return RecognitionConfig(
        model_dir=str(MODEL_DIR),
        device="auto",
        min_face_size=60,
        min_confidence=0.6,
        blur_threshold=60.0,
        similarity_threshold=0.45,
    )


def _detector() -> YoloFaceDetector:
    return YoloFaceDetector(
        DetectorConfig(
            model_path=str(YOLO_MODEL),
            conf_threshold=0.5,
            imgsz=1600,
            min_face_size=40,
            device="auto",
        )
    )


def _frame_at(path: Path, index: int) -> np.ndarray:
    source = FileSource(path)
    try:
        for i, frame in enumerate(source.frames()):
            if i == index:
                return frame.image
    finally:
        source.close()
    raise RuntimeError(f"frame {index} not found in {path}")


@pytest.fixture(scope="module")
def pipeline():
    config = _config()
    runtime = InsightFaceRuntime(config)
    embedder = ArcFaceEmbedder(runtime, config)
    gate = QualityGate(config)
    detector = _detector()
    return detector, gate, embedder, runtime


def _embed_frame(pipeline, path: Path, index: int, max_x: float):
    detector, gate, embedder, _runtime = pipeline
    image = _frame_at(path, index)
    candidates = [
        detection
        for detection in detector.detect(image)
        if (detection.bbox[0] + detection.bbox[2]) / 2 < max_x
    ]
    candidates.sort(key=lambda detection: detection.size, reverse=True)
    if not candidates:
        return None, None

    best_quality = None
    for detection in candidates:
        quality = gate.assess(image, detection)
        if quality.passed:
            return embedder.embed(image, detection, quality.score), quality
        if best_quality is None:
            best_quality = quality
    return None, best_quality


def test_runtime_uses_cuda_when_available(pipeline):
    import onnxruntime as ort
    import torch

    _detector_, _gate, _embedder, runtime = pipeline
    providers = runtime.providers
    cuda_available = (
        "CUDAExecutionProvider" in ort.get_available_providers()
        and torch.cuda.is_available()
    )
    if cuda_available:
        assert "CUDAExecutionProvider" in providers
    else:
        assert "CPUExecutionProvider" in providers


def test_quality_gate_passes_the_near_front_entry_face(pipeline):
    _, quality = _embed_frame(pipeline, FRONT, 0, max_x=2500)

    assert quality is not None
    assert quality.passed is True
    assert quality.score >= 0.5


def test_same_person_embeddings_match(pipeline):
    results = []
    for index in (0, 10, 20):
        result, quality = _embed_frame(pipeline, FRONT, index, max_x=2500)
        assert quality is not None and quality.passed
        assert result is not None
        results.append(result.vector)

    assert len(results) == 3
    assert float(np.dot(results[0], results[1])) > 0.5
    assert float(np.dot(results[0], results[2])) > 0.5


def test_different_person_embeddings_differ(pipeline):
    if not CROWD.is_file():
        pytest.skip("crowded sample clip is not available")

    first, quality = _embed_frame(pipeline, FRONT, 0, max_x=2500)
    assert first is not None and quality is not None and quality.passed

    other, other_quality = _embed_frame(pipeline, CROWD, 0, max_x=10_000)
    if other is None or other_quality is None or not other_quality.passed:
        pytest.skip("no quality-passing different face in the 4K crowd clip")

    assert float(np.dot(first.vector, other.vector)) < 0.35
