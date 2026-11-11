"""Composition root and CLI for the intelligent face tracker."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import signal
import sys
import threading
from typing import Sequence

from src.config import ConfigError, load_config
from src.config.models import AppConfig
from src.counting import UniqueCounter
from src.detection import YoloFaceDetector
from src.events import EventLogger, ImageStore
from src.logging_setup import configure_logging, log_event, shutdown_logging
from src.pipeline import Pipeline, PipelineResult
from src.recognition import (
    ArcFaceEmbedder,
    FaceRegistry,
    InsightFaceRuntime,
    QualityGate,
)
from src.storage import Database, FaceRepository
from src.tracking import TrackLifecycle
from src.video import (
    FileSequenceSource,
    FileSource,
    RTSPSource,
    VideoSource,
    VideoSourceError,
)


def build_source(
    config: AppConfig,
    *,
    start_time: datetime | None = None,
) -> VideoSource:
    """Create the configured video source."""

    if config.source.type == "file":
        return FileSource(config.source.path, start_time=start_time)
    if config.source.type == "folder":
        return FileSequenceSource(
            config.source.path,
            glob=config.source.glob,
        )
    if config.source.type == "rtsp":
        return RTSPSource(
            config.source.rtsp_url,
            transport=config.source.rtsp_transport,
            max_backoff_sec=config.source.reconnect_max_backoff_sec,
            source_id="rtsp",
        )
    raise ConfigError(f"unsupported source type '{config.source.type}'")


def apply_overrides(
    config: AppConfig,
    args: argparse.Namespace,
) -> AppConfig:
    """Apply CLI source/log overrides without mutating the loaded config."""

    source = config.source
    if args.source is not None:
        source = replace(source, type=args.source)
    if args.path is not None:
        source = replace(source, path=args.path)
    if args.rtsp_url is not None:
        source = replace(source, rtsp_url=args.rtsp_url)

    if source.type in {"file", "folder"} and not source.path:
        raise ConfigError("source.path is required for file/folder sources")
    if source.type == "rtsp" and not source.rtsp_url:
        raise ConfigError("source.rtsp_url is required for rtsp sources")

    logging_config = config.logging
    if args.log_level is not None:
        logging_config = replace(logging_config, level=args.log_level)

    return replace(config, source=source, logging=logging_config)


def run_pipeline(
    config: AppConfig,
    *,
    source: VideoSource | None = None,
    stop_event: threading.Event | None = None,
    max_frames: int | None = None,
    verify_db: bool = False,
    start_time: datetime | None = None,
) -> PipelineResult:
    """Wire every dependency and run one pipeline."""

    setup = configure_logging(
        config.storage.log_dir,
        level=config.logging.level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
    )
    database = Database(config.storage.db_path)
    try:
        database.initialize()
        if verify_db:
            log_event(
                setup.events_logger,
                "DB_INTEGRITY",
                result=database.integrity_check(),
            )

        faces = FaceRepository(database)
        runtime = InsightFaceRuntime(config.recognition)
        video_source = source or build_source(
            config,
            start_time=start_time,
        )
        try:
            pipeline = Pipeline(
                config,
                video_source,
                YoloFaceDetector(config.detector),
                TrackLifecycle(config.tracking),
                QualityGate(config.recognition),
                ArcFaceEmbedder(runtime, config.recognition),
                FaceRegistry(faces, config.recognition),
                EventLogger(
                    database,
                    ImageStore(config.storage),
                    events_logger=setup.events_logger,
                ),
                UniqueCounter(database, face_repository=faces),
                events_logger=setup.events_logger,
                app_logger=setup.app_logger,
            )
            return pipeline.run(
                stop_event=stop_event,
                max_frames=max_frames,
            )
        finally:
            video_source.close()
    except Exception:
        setup.app_logger.exception("pipeline run failed")
        raise
    finally:
        database.close()
        shutdown_logging()


def report(config: AppConfig) -> dict[str, object]:
    """Read the database and return the current summary."""

    database = Database(config.storage.db_path)
    try:
        database.initialize()
        return UniqueCounter(database).get_summary().to_dict()
    finally:
        database.close()


def _parse_start_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected an ISO-8601 timestamp, e.g. 2025-06-20T18:39:03Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Intelligent face tracker: detect, recognize, track and count",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="path to config.json (default: config.json)",
    )
    parser.add_argument(
        "--source",
        choices=["file", "folder", "rtsp"],
        default=None,
        help="override source.type",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="override source.path",
    )
    parser.add_argument(
        "--rtsp-url",
        default=None,
        help="override source.rtsp_url",
    )
    parser.add_argument(
        "--start-time",
        type=_parse_start_time,
        default=None,
        help="ISO-8601 start time for a single file source",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many decoded frames",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="override logging.level",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the DB summary and exit",
    )
    parser.add_argument(
        "--verify-db",
        action="store_true",
        help="run SQLite PRAGMA integrity_check at startup",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    try:
        config = apply_overrides(load_config(args.config), args)
    except ConfigError as exc:
        sys.stderr.write(f"config error: {exc}\n")
        return 2

    if args.report:
        try:
            sys.stdout.write(
                json.dumps(report(config), indent=2) + "\n"
            )
        except Exception as exc:
            sys.stderr.write(f"report error: {exc}\n")
            return 1
        return 0

    try:
        source = build_source(config, start_time=args.start_time)
    except VideoSourceError as exc:
        sys.stderr.write(f"source error: {exc}\n")
        return 1

    stop_event = threading.Event()

    def _stop(_signum, _frame) -> None:
        stop_event.set()
        source.close()

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    try:
        result = run_pipeline(
            config,
            source=source,
            stop_event=stop_event,
            max_frames=args.max_frames,
            verify_db=args.verify_db,
            start_time=args.start_time,
        )
    except Exception as exc:
        sys.stderr.write(f"run error: {exc}\n")
        return 1

    sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
