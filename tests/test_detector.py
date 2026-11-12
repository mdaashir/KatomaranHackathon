"""Tests for the YOLO detector wrapper using an injected fake model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.models import DetectorConfig
from src.detection import DetectorError, YoloFaceDetector


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)


class _Keypoints:
    def __init__(self, xy):
        self.xy = np.asarray(xy, dtype=np.float32)


class _Result:
    def __init__(self, boxes, keypoints=None):
        self.boxes = boxes
        self.keypoints = keypoints


class _FakeModel:
    def __init__(self, results=None, error: Exception | None = None):
        self.results = results or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    path = tmp_path / "yolov8n-face.pt"
    path.write_bytes(b"fake weights")
    return path


def _config(weights: Path, **overrides) -> DetectorConfig:
    values = {
        "model_path": str(weights),
        "conf_threshold": 0.5,
        "imgsz": 1600,
        "min_face_size": 40,
        "device": "auto",
    }
    values.update(overrides)
    return DetectorConfig(**values)


def _detector(weights: Path, results, **overrides):
    model = _FakeModel(results)
    calls: list[str] = []

    def factory(path: str):
        calls.append(path)
        return model

    detector = YoloFaceDetector(_config(weights, **overrides), model_factory=factory)
    return detector, model, calls


def test_detect_maps_boxes_and_keypoints(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    keypoints = np.array(
        [[[10, 20], [20, 20], [15, 30], [10, 40], [20, 40]]],
        dtype=np.float32,
    )
    results = [
        _Result(
            _Boxes([[10, 20, 110, 140]], [0.9]),
            _Keypoints(keypoints),
        )
    ]
    detector, _, _ = _detector(weights, results)

    detections = detector.detect(image)

    assert len(detections) == 1
    assert detections[0].bbox == (10.0, 20.0, 110.0, 140.0)
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].keypoints is not None
    assert detections[0].keypoints.shape == (5, 2)
    assert detections[0].size == 100.0


def test_filters_faces_below_native_size_gate(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [_Result(_Boxes([[0, 0, 30, 30], [0, 0, 50, 50]], [0.9, 0.8]))]
    detector, _, _ = _detector(weights, results, min_face_size=40)

    detections = detector.detect(image)

    assert len(detections) == 1
    assert detections[0].bbox == (0.0, 0.0, 50.0, 50.0)


def test_clamps_boxes_to_frame_bounds(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = [_Result(_Boxes([[-5, -5, 700, 500]], [0.9]))]
    detector, _, _ = _detector(weights, results)

    detections = detector.detect(image)

    assert detections[0].bbox == (0.0, 0.0, 639.0, 479.0)


def test_predict_receives_config(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector, model, _ = _detector(weights, [_Result(_Boxes([], []))])

    detector.detect(image)

    call = model.calls[0]
    assert call["imgsz"] == 1600
    assert call["conf"] == 0.5
    assert call["verbose"] is False
    assert "device" not in call


def test_cuda_device_maps_to_index(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector, model, _ = _detector(weights, [_Result(_Boxes([], []))], device="cuda:1")

    detector.detect(image)

    assert detector.device == 1
    assert model.calls[0]["device"] == 1


def test_model_is_loaded_once(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector, _, factory_calls = _detector(weights, [_Result(_Boxes([], []))])

    detector.detect(image)
    detector.detect(image)

    assert factory_calls == [str(weights)]


def test_missing_weights_raise_actionable_error(tmp_path):
    config = _config(tmp_path / "missing.pt")
    detector = YoloFaceDetector(config, model_factory=lambda path: None)

    with pytest.raises(DetectorError, match="not found"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_empty_results(weights):
    detector, _, _ = _detector(weights, [])
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_result_without_boxes(weights):
    detector, _, _ = _detector(weights, [_Result(boxes=None)])
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_detection_without_keypoints(weights):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    detector, _, _ = _detector(weights, [_Result(_Boxes([[10, 20, 110, 140]], [0.9]))])

    detections = detector.detect(image)

    assert detections[0].keypoints is None


def test_rejects_non_bgr_image(weights):
    detector, _, _ = _detector(weights, [])
    with pytest.raises(DetectorError, match="HxWx3"):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))


def test_inference_failure_is_wrapped(weights):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    model = _FakeModel(error=RuntimeError("boom"))
    detector = YoloFaceDetector(
        _config(weights),
        model_factory=lambda path: model,
    )

    with pytest.raises(DetectorError, match="inference failed"):
        detector.detect(image)


def test_model_load_failure_is_wrapped(weights):
    def factory(path: str):
        raise RuntimeError("bad checkpoint")

    detector = YoloFaceDetector(
        _config(weights),
        model_factory=factory,
    )

    with pytest.raises(DetectorError, match="failed to load"):
        detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_min_face_size_must_be_positive(weights):
    with pytest.raises(ValueError, match="min_face_size"):
        YoloFaceDetector(_config(weights, min_face_size=0))
