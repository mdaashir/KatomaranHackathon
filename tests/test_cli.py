"""CLI and config-override tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from main import apply_overrides, build_parser, main
from src.config import ConfigError, load_config
from src.storage import Database, FaceRepository


def test_parser_defaults():
    args = build_parser().parse_args([])

    assert args.config == "config.json"
    assert args.source is None
    assert args.path is None
    assert args.max_frames is None
    assert args.report is False
    assert args.start_time is None


def test_start_time_is_parsed():
    args = build_parser().parse_args(
        ["--start-time", "2025-06-20T18:39:03Z"]
    )

    assert args.start_time == datetime(2025, 6, 20, 18, 39, 3, tzinfo=timezone.utc)


def test_invalid_start_time_is_rejected():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--start-time", "not-a-date"])


def test_source_and_log_overrides(write_config, config_dict):
    config = load_config(write_config(config_dict))
    args = build_parser().parse_args(
        [
            "--source",
            "rtsp",
            "--rtsp-url",
            "rtsp://camera/stream",
            "--log-level",
            "debug",
        ]
    )

    overridden = apply_overrides(config, args)

    assert overridden.source.type == "rtsp"
    assert overridden.source.rtsp_url == "rtsp://camera/stream"
    assert overridden.logging.level == "DEBUG"
    assert config.source.type == "file"


def test_missing_path_is_rejected(write_config, config_dict):
    config = load_config(write_config(config_dict))
    args = build_parser().parse_args(["--source", "file", "--path", ""])

    with pytest.raises(ConfigError, match="source.path"):
        apply_overrides(config, args)


def test_report_prints_db_summary(
    write_config, config_dict, capsys
):
    config_path = write_config(config_dict)
    config = load_config(config_path)
    database = Database(config.storage.db_path)
    database.initialize()
    FaceRepository(database).create(
        vector=np.zeros(512, dtype=np.float32).tobytes(),
        dim=512,
        quality=0.8,
        model_name="buffalo_l",
        now=datetime.now(timezone.utc),
    )
    database.close()

    code = main(["--config", str(config_path), "--report"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["unique_faces"] == 1
    assert payload["visits"] == 0
    assert payload["entries"] == 0


def test_missing_config_returns_two(tmp_path, capsys):
    code = main(["--config", str(tmp_path / "missing.json")])

    assert code == 2
    assert "config error" in capsys.readouterr().err


def test_missing_source_returns_one(
    write_config, config_dict, capsys
):
    config_path = write_config(config_dict)

    code = main(
        [
            "--config",
            str(config_path),
            "--source",
            "file",
            "--path",
            "does-not-exist.mp4",
        ]
    )

    assert code == 1
    assert "source error" in capsys.readouterr().err
