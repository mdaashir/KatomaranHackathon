"""RTSP source with a bounded frame queue and automatic reconnect."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable, Iterator

import av

from src.logging_setup import get_logger, log_event
from src.video.bounded_queue import BoundedFrameQueue
from src.video.helpers import frame_to_bgr, stream_fps, video_stream
from src.video.types import Frame, VideoSourceError

logger = get_logger(__name__)
_STOP = object()


def next_backoff(attempt: int, base_sec: float, max_sec: float) -> float:
    """Return the capped exponential backoff for a 1-based *attempt*."""

    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    return min(base_sec * (2 ** (attempt - 1)), max_sec)


class RTSPSource:
    """Decode a live RTSP stream without ever blocking the consumer.

    A background reader reconnects with capped exponential backoff. The
    consumer queue is bounded, so a slow pipeline drops stale frames instead
    of accumulating latency.
    """

    def __init__(
        self,
        url: str,
        *,
        source_id: str = "rtsp",
        transport: str = "tcp",
        max_backoff_sec: float = 30.0,
        backoff_base_sec: float = 1.0,
        queue_size: int = 3,
        open_timeout_sec: float = 10.0,
        max_reconnect_attempts: int | None = None,
        container_factory: Callable[..., Any] | None = None,
    ) -> None:
        """
        Args:
            url: RTSP URL. Never logged verbatim; use *source_id* in logs.
            source_id: Safe label for logs and the database.
            transport: ``tcp`` (recommended) or ``udp``.
            max_backoff_sec: Maximum delay between reconnect attempts.
            backoff_base_sec: First reconnect delay; doubles each attempt.
            queue_size: Maximum buffered frames (drop-oldest when full).
            open_timeout_sec: Socket/open timeout passed to PyAV.
            max_reconnect_attempts: Stop after this many consecutive failures;
                ``None`` retries forever.
            container_factory: Injectable factory for tests.
        """

        if not url:
            raise ValueError("url must not be empty")
        if transport not in {"tcp", "udp"}:
            raise ValueError("transport must be 'tcp' or 'udp'")
        if max_backoff_sec <= 0:
            raise ValueError("max_backoff_sec must be positive")
        if backoff_base_sec <= 0:
            raise ValueError("backoff_base_sec must be positive")
        if open_timeout_sec <= 0:
            raise ValueError("open_timeout_sec must be positive")
        if max_reconnect_attempts is not None and max_reconnect_attempts < 1:
            raise ValueError("max_reconnect_attempts must be at least 1")

        self._url = url
        self._source_id = source_id
        self._transport = transport
        self._max_backoff_sec = max_backoff_sec
        self._backoff_base_sec = backoff_base_sec
        self._open_timeout_sec = open_timeout_sec
        self._max_reconnect_attempts = max_reconnect_attempts
        self._container_factory = container_factory or av.open

        self._queue: BoundedFrameQueue[object] = BoundedFrameQueue(queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._index = 0
        self._first_monotonic: float | None = None
        self._fps: float = 25.0

    @property
    def source_id(self) -> str:
        """Safe label for logs and the database."""

        return self._source_id

    def frames(self) -> Iterator[Frame]:
        """Start the reader thread and yield frames until closed or stopped."""

        if self._started:
            raise VideoSourceError(f"{self._source_id}: source is single-use")
        if self._closed:
            raise VideoSourceError(f"{self._source_id}: source is closed")
        self._started = True

        self._thread = threading.Thread(
            target=self._reader,
            name=f"rtsp-{self._source_id}",
            daemon=True,
        )
        self._thread.start()

        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    return
                yield item  # type: ignore[misc]
        finally:
            self.close()

    def _reader(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            container: Any | None = None
            try:
                container = self._open_container()
                log_event(
                    logger,
                    "SOURCE_RECONNECTED",
                    source=self._source_id,
                    attempt=attempt,
                )
                attempt = 0

                stream = video_stream(container, self._source_id)
                self._fps = stream_fps(stream)
                for raw in container.decode(stream):
                    if self._stop.is_set():
                        break
                    self._queue.put_drop_oldest(self._make_frame(raw))

                if not self._stop.is_set():
                    raise ConnectionError("stream ended")
            except Exception as exc:
                if self._stop.is_set():
                    break
                attempt += 1
                log_event(
                    logger,
                    "SOURCE_LOST",
                    source=self._source_id,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if (
                    self._max_reconnect_attempts is not None
                    and attempt >= self._max_reconnect_attempts
                ):
                    break
                delay = next_backoff(
                    attempt,
                    self._backoff_base_sec,
                    self._max_backoff_sec,
                )
                self._stop.wait(delay)
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        logger.warning("%s: error closing container", self._source_id)

        self._queue.put_drop_oldest(_STOP)

    def _open_container(self) -> Any:
        options = {
            "rtsp_transport": self._transport,
            "stimeout": str(int(self._open_timeout_sec * 1_000_000)),
            "fflags": "nobuffer",
            "flags": "low_delay",
        }
        try:
            return self._container_factory(
                self._url,
                options=options,
                timeout=self._open_timeout_sec,
            )
        except TypeError:
            return self._container_factory(self._url, options=options)

    def _make_frame(self, raw: Any) -> Frame:
        now = datetime.now(timezone.utc)
        monotonic = time.monotonic()
        if self._first_monotonic is None:
            self._first_monotonic = monotonic
        timestamp_s = monotonic - self._first_monotonic

        index = self._index
        self._index += 1
        return Frame(
            index=index,
            global_index=index,
            timestamp_s=timestamp_s,
            absolute_time=now,
            image=frame_to_bgr(raw),
            source_id=self._source_id,
            window_id=self._source_id,
            is_window_end=False,
            fps=self._fps,
        )

    def close(self) -> None:
        """Stop the reader, release the container, and wake the consumer."""

        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._queue.put_drop_oldest(_STOP)

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "%s: reader did not stop within 2 seconds",
                    self._source_id,
                )

    def __enter__(self) -> RTSPSource:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
