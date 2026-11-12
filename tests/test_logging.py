"""Tests for logging configuration and the audit formatter."""

from __future__ import annotations

import re

import pytest

from src.logging_setup import (
    configure_logging,
    get_logger,
    log_event,
    shutdown_logging,
)


@pytest.fixture(autouse=True)
def _cleanup_logging():
    yield
    shutdown_logging()


def test_events_log_format(tmp_path):
    setup = configure_logging(
        tmp_path, level="INFO", max_bytes=1_000_000, backup_count=1
    )

    log_event(
        setup.events_logger,
        "ENTRY",
        visit=1,
        face=None,
        path="logs/a b.jpg",
    )
    shutdown_logging()

    text = (tmp_path / "events.log").read_text(encoding="utf-8").strip()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z \| ENTRY \| "
        r'visit=1 face=- path="logs/a b\.jpg"',
        text,
    )


def test_field_sanitization(tmp_path):
    setup = configure_logging(tmp_path)
    log_event(setup.events_logger, "NOTE", note="line1\nline2|x")
    shutdown_logging()

    text = (tmp_path / "events.log").read_text(encoding="utf-8")
    assert 'note="line1 line2/x"' in text
    assert "\n\n" not in text


def test_configure_twice_does_not_duplicate_handlers(tmp_path):
    first = configure_logging(tmp_path, level="INFO")
    second = configure_logging(tmp_path, level="INFO")

    assert second.app_logger is first.app_logger
    assert second.events_logger is first.events_logger
    assert len(second.app_logger.handlers) == 2
    assert len(second.events_logger.handlers) == 1


def test_debug_file_only_at_debug_level(tmp_path):
    info_setup = configure_logging(tmp_path, level="INFO")
    assert info_setup.debug_log_path is None
    assert not (tmp_path / "debug.log").exists()

    shutdown_logging()
    debug_setup = configure_logging(tmp_path, level="DEBUG")

    assert debug_setup.debug_log_path == tmp_path / "debug.log"
    assert (tmp_path / "debug.log").exists()
    assert len(debug_setup.app_logger.handlers) == 3


def test_invalid_level_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid log level"):
        configure_logging(tmp_path, level="TRACE")


def test_get_logger_uses_ift_namespace():
    assert get_logger("src.config.loader").name == "ift.src.config.loader"
    assert get_logger("ift.events").name == "ift.events"


def test_app_log_receives_regular_records(tmp_path):
    configure_logging(tmp_path)
    get_logger("test").warning("something happened")
    shutdown_logging()

    text = (tmp_path / "app.log").read_text(encoding="utf-8")
    assert "WARNING" in text
    assert "something happened" in text


def test_bool_and_none_fields(tmp_path):
    setup = configure_logging(tmp_path)
    log_event(setup.events_logger, "FLAGS", active=True, missing=None)
    shutdown_logging()

    text = (tmp_path / "events.log").read_text(encoding="utf-8")
    assert "active=true missing=-" in text


def test_quote_in_value_is_escaped(tmp_path):
    setup = configure_logging(tmp_path)
    log_event(setup.events_logger, "NOTE", value='a"b')
    shutdown_logging()

    text = (tmp_path / "events.log").read_text(encoding="utf-8")
    assert 'value="a\\"b"' in text
