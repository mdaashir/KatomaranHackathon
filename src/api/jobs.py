"""Background pipeline job manager and input helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import logging
from pathlib import Path
import queue
import re
import threading
from typing import Callable
from urllib.parse import urlparse
import uuid

from src.api.live import LiveFrameBuffer
from src.api.models import JobRequest
from src.config.models import AppConfig
from src.pipeline import PipelineResult

LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
Runner = Callable[..., PipelineResult]


def input_root_for(config: AppConfig) -> Path:
    """Return the folder uploads and job paths are confined to."""

    path = Path(config.source.path)
    return path if config.source.type == "folder" else path.parent


def list_videos(root: Path) -> list[str]:
    """List uploaded/available video files in deterministic order."""

    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def list_folders(root: Path) -> list[str]:
    """List immediate, non-hidden subfolders of the video root."""

    if not root.is_dir():
        return []
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def _now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


@dataclass
class Job:
    job_id: str
    source_type: str
    label: str
    status: str
    created_at: str
    config: AppConfig = field(repr=False)
    stop_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "source_type": self.source_type,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    """Run at most one pipeline at a time on a daemon worker thread."""

    def __init__(
        self,
        config: AppConfig,
        *,
        input_root: Path | None = None,
        frame_buffer: LiveFrameBuffer | None = None,
        runner: Runner | None = None,
    ) -> None:
        self._config = config
        self._input_root = (input_root or input_root_for(config)).resolve()
        self._frame_buffer = frame_buffer or LiveFrameBuffer()
        self._runner = runner or _default_runner
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(
            target=self._work,
            name="pipeline-job",
            daemon=True,
        )
        self._worker.start()

    @property
    def input_root(self) -> Path:
        return self._input_root

    @property
    def frame_buffer(self) -> LiveFrameBuffer:
        return self._frame_buffer

    def update_config(self, config: AppConfig) -> None:
        """Use *config* for jobs started after this call."""

        with self._lock:
            self._config = config

    def start(self, request: JobRequest) -> Job:
        """Validate a request, enqueue it, and return the job record."""

        source = self._source_for(request)
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            source_type=request.source_type,
            label=self._label(request, source.path, source.rtsp_url),
            status="queued",
            created_at=_now(),
            config=replace(self._config, source=source),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        self._queue.put(job)
        return job

    def stop(self, job_id: str) -> Job | None:
        """Request a running job stop; cancel it if still queued."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"completed", "failed", "stopped"}:
                return job
            job.stop_event.set()
            if job.status == "queued":
                job.status = "stopped"
                job.finished_at = _now()
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(
                self._jobs.values(),
                key=lambda job: job.created_at,
                reverse=True,
            )

    def _source_for(self, request: JobRequest):
        source = self._config.source
        if request.source_type in {"file", "folder"}:
            raw = request.path or ""
            name = Path(raw).name
            if not name or name != raw:
                raise ValueError("path must be a name inside the video folder")

            def matches_type(path: Path) -> bool:
                if request.source_type == "file":
                    return path.is_file()
                return path.is_dir()

            try:
                matches = [
                    path
                    for path in self._input_root.iterdir()
                    if path.name == name and matches_type(path)
                ]
            except OSError as exc:
                raise ValueError("video folder is unavailable") from exc
            if not matches:
                raise ValueError("video file not found")
            candidate = matches[0].resolve()
            if not candidate.is_relative_to(self._input_root):
                raise ValueError("path must be inside the video folder")
            return replace(
                source,
                type=request.source_type,
                path=str(candidate),
                glob=("**/*.mp4" if request.source_type == "folder" else source.glob),
            )

        if request.source_type == "webcam":
            index = (
                request.device_index
                if request.device_index is not None
                else source.device_index
            )
            if index < 0:
                raise ValueError("device_index must not be negative")
            return replace(source, type="webcam", device_index=index)

        url = (request.url or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.netloc:
            raise ValueError("a valid rtsp:// or rtsps:// URL is required")
        return replace(source, type="rtsp", rtsp_url=url)

    def _label(self, request: JobRequest, path: str, rtsp_url: str) -> str:
        if request.source_type == "webcam":
            index = request.device_index if request.device_index is not None else 0
            return f"Webcam {index}"
        if request.source_type == "rtsp":
            return re.sub(r"(rtsps?://)[^/@]+@", r"\1***@", rtsp_url)
        return Path(path).name or request.source_type

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                if job.status == "stopped":
                    continue
                job.status = "running"
                job.started_at = _now()
            try:
                result = self._runner(
                    job.config,
                    stop_event=job.stop_event,
                    frame_sink=self._frame_buffer.publish,
                )
            except Exception as exc:
                LOGGER.exception("pipeline job %s failed", job.job_id)
                with self._lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.finished_at = _now()
            else:
                with self._lock:
                    job.status = "stopped" if result.stopped_by_signal else "completed"
                    job.result = result.to_dict()
                    job.finished_at = _now()


def _default_runner(config: AppConfig, **kwargs: object) -> PipelineResult:
    # Imported lazily so the API reader/tests do not construct models.
    from main import run_pipeline

    return run_pipeline(config, **kwargs)
