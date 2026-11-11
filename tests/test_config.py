"""Tests for the typed configuration loader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.config import ConfigError, load_config


def test_loads_valid_config(write_config, config_dict):
    config = load_config(write_config(config_dict))

    assert config.frame_skip == 3
    assert config.source.type == "file"
    assert config.detector.imgsz == 1600
    assert config.recognition.similarity_threshold == 0.45
    assert config.tracking.exit_timeout_frames == 75
    assert config.storage.image_size == 256
    assert config.logging.level == "INFO"


def test_missing_top_level_section(write_config, config_dict):
    del config_dict["tracking"]
    with pytest.raises(ConfigError, match="tracking: missing required section"):
        load_config(write_config(config_dict))


def test_missing_frame_skip(write_config, config_dict):
    del config_dict["frame_skip"]
    with pytest.raises(ConfigError, match="frame_skip"):
        load_config(write_config(config_dict))


def test_frame_skip_must_be_positive(write_config, config_dict):
    config_dict["frame_skip"] = 0
    with pytest.raises(ConfigError, match="frame_skip"):
        load_config(write_config(config_dict))


def test_frame_skip_rejects_boolean(write_config, config_dict):
    config_dict["frame_skip"] = True
    with pytest.raises(ConfigError, match="expected an integer"):
        load_config(write_config(config_dict))


def test_rtsp_requires_url(write_config, config_dict):
    config_dict["source"] = {"type": "rtsp"}
    with pytest.raises(ConfigError, match=r"source\.rtsp_url"):
        load_config(write_config(config_dict))


def test_file_requires_path(write_config, config_dict):
    config_dict["source"] = {"type": "file"}
    with pytest.raises(ConfigError, match=r"source\.path"):
        load_config(write_config(config_dict))


def test_folder_requires_glob(write_config, config_dict):
    config_dict["source"] = {"type": "folder", "path": "src/video", "glob": ""}
    with pytest.raises(ConfigError, match=r"source\.glob"):
        load_config(write_config(config_dict))


def test_unknown_top_level_key(write_config, config_dict):
    config_dict["extra"] = 1
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write_config(config_dict))


def test_unknown_section_key(write_config, config_dict):
    config_dict["detector"]["extra"] = 1
    with pytest.raises(ConfigError, match="detector: unknown key"):
        load_config(write_config(config_dict))


def test_similarity_threshold_range(write_config, config_dict):
    config_dict["recognition"]["similarity_threshold"] = 1.5
    with pytest.raises(ConfigError, match="similarity_threshold"):
        load_config(write_config(config_dict))


def test_recognition_gate_must_not_be_smaller(write_config, config_dict):
    config_dict["recognition"]["min_face_size"] = 20
    with pytest.raises(ConfigError, match="recognition.min_face_size"):
        load_config(write_config(config_dict))


def test_tracking_activation_threshold_range(write_config, config_dict):
    config_dict["tracking"]["track_activation_threshold"] = 1.5
    with pytest.raises(ConfigError, match="track_activation_threshold"):
        load_config(write_config(config_dict))


def test_imgsz_must_be_multiple_of_32(write_config, config_dict):
    config_dict["detector"]["imgsz"] = 1000
    with pytest.raises(ConfigError, match="multiple of 32"):
        load_config(write_config(config_dict))


def test_device_accepts_cuda_index(write_config, config_dict):
    config_dict["detector"]["device"] = "cuda:0"
    config = load_config(write_config(config_dict))
    assert config.detector.device == "cuda:0"


def test_device_rejects_unknown(write_config, config_dict):
    config_dict["detector"]["device"] = "tpu"
    with pytest.raises(ConfigError, match="detector.device"):
        load_config(write_config(config_dict))


def test_env_expansion(write_config, config_dict, monkeypatch):
    monkeypatch.setenv("TEST_RTSP_URL", "rtsp://camera/stream1")
    config_dict["source"] = {"type": "rtsp", "rtsp_url": "${TEST_RTSP_URL}"}

    config = load_config(write_config(config_dict))

    assert config.source.rtsp_url == "rtsp://camera/stream1"


def test_missing_env_var(write_config, config_dict):
    config_dict["source"] = {"type": "rtsp", "rtsp_url": "${MISSING_RTSP_URL}"}
    with pytest.raises(ConfigError, match="MISSING_RTSP_URL"):
        load_config(write_config(config_dict))


def test_invalid_json(tmp_path: Path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")


def test_log_level_is_normalized(write_config, config_dict):
    config_dict["logging"]["level"] = "debug"
    config = load_config(write_config(config_dict))
    assert config.logging.level == "DEBUG"


def test_invalid_log_level(write_config, config_dict):
    config_dict["logging"]["level"] = "TRACE"
    with pytest.raises(ConfigError, match="logging.level"):
        load_config(write_config(config_dict))


def test_rejects_non_object_top_level(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ConfigError, match="top-level value must be an object"):
        load_config(path)


def test_does_not_mutate_input_dict(write_config, config_dict):
    original = copy.deepcopy(config_dict)
    load_config(write_config(config_dict))
    assert config_dict == original
