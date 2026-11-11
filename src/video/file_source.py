"""Decode a single local video file with PyAV."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import av

from src.logging_setup import get_logger
from src.video.helpers import frame_to_bgr, mtime_utc, stream_fps, video_stream
from src.video.types import Frame, VideoSourceError

logger = get_logger(__name__)


class FileSource:
    """Read one video file, lazily and exactly once."""

    def __init__(
        self,
        path: str | Path,
        *,
        start_time: datetime | None = None,
        source_id: str | None = None,
        window_id: str | None = None,
    ) -> None:
        """
        Args:
            path: Video file to decode.
            start_time: UTC time of frame zero. Defaults to the file mtime.
            source_id: Identifier for logs and the database. Defaults to the
                file name.
            window_id: Logical track window. Defaults to ``source_id``.
        """

        self._path = Path(path)
        if not self._path.is_file():
            raise VideoSourceError(f"{self._path}: file not found")

        self._start_time = start_time or mtime_utc(self._path)
        self._source_id = source_id or self._path.name
        self._window_id = window_id or self._source_id
        self._closed = False
        self._started = False

    @property
    def source_id(self) -> str:
        """Stable identifier for this clip."""

        return self._source_id

    @property
    def path(self) -> Path:
        """The file being decoded."""

        return self._path

    def frames(self) -> Iterator[Frame]:
        """Yield frames until the file ends.

        Raises:
            VideoSourceError: if the file cannot be opened or decoded, or if
                the source is iterated more than once.
        """

        if self._started:
            raise VideoSourceError(f"{self._source_id}: source is single-use")
        if self._closed:
            raise VideoSourceError(f"{self._source_id}: source is closed")
        self._started = True

        try:
            container = av.open(str(self._path))
        except Exception as exc:
            raise VideoSourceError(
                f"{self._source_id}: cannot open video: {exc}"
            ) from exc

        try:
            stream = video_stream(container, self._source_id)
            fps = stream_fps(stream)
            iterator = container.decode(stream)
            pending: Any | None = None
            index = 0

            while True:
                if self._closed:
                    return
                try:
                    raw = next(iterator)
                except StopIteration:
                    if pending is not None:
                        yield self._make_frame(pending, index, fps, True)
                    break
                except Exception as exc:
                    raise VideoSourceError(
                        f"{self._source_id}: decode failed at frame {index}: {exc}"
                    ) from exc

                if pending is not None:
                    yield self._make_frame(pending, index, fps, False)
                    index += 1
                pending = raw
        finally:
            container.close()

    def _make_frame(
        self, raw: Any, index: int, fps: float, is_window_end: bool
    ) -> Frame:
        timestamp_s = float(raw.time) if raw.time is not None else index / fps
        try:
            image = frame_to_bgr(raw)
        except Exception as exc:
            raise VideoSourceError(
                f"{self._source_id}: cannot convert frame {index}: {exc}"
            ) from exc

        return Frame(
            index=index,
            global_index=index,
            timestamp_s=timestamp_s,
            absolute_time=self._start_time + timedelta(seconds=timestamp_s),
            image=image,
            source_id=self._source_id,
            window_id=self._window_id,
            is_window_end=is_window_end,
        )

    def close(self) -> None:
        """Request the generator to stop and release the decoder.

        The decoder is owned by :meth:`frames` and is closed when the
        generator unwinds. Closing the PyAV container from outside a suspended
        generator can crash the native decoder, so this method only sets a
        stop flag. Safe to call more than once.
        """

        self._closed = True

    def __enter__(self) -> FileSource:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
