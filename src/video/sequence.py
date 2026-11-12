"""Read an ordered folder of clips as independent observation windows."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterator, Mapping

from src.logging_setup import get_logger
from src.video.file_source import FileSource
from src.video.helpers import mtime_utc
from src.video.types import Frame, VideoSourceError

logger = get_logger(__name__)

_CLIP_PATTERN = re.compile(r"^record_(\d{8})_(\d{6})\.mp4$", re.IGNORECASE)


def parse_clip_start(path: str | Path) -> datetime | None:
    """Parse ``record_YYYYMMDD_HHMMSS.mp4`` into a UTC datetime.

    Returns ``None`` when the name does not match the expected pattern.
    The filename timestamp is treated as UTC because the dataset does not
    carry a timezone; relative ordering and durations remain correct.
    """

    match = _CLIP_PATTERN.match(Path(path).name)
    if match is None:
        return None
    date_part, time_part = match.groups()
    return datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


class FileSequenceSource:
    """Iterate clips in filename order and mark each clip boundary.

    Each clip is decoded by its own :class:`FileSource`. A corrupt clip is
    logged and skipped; the next clip still runs. Tracks must be flushed when
    ``Frame.is_window_end`` is true, and also whenever ``window_id`` changes:
    a clip that fails mid-stream cannot mark its final frame as a window end.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        glob: str = "record_*.mp4",
        source_id: str | None = None,
        start_times: Mapping[str, datetime] | None = None,
    ) -> None:
        """
        Args:
            directory: Folder containing the clips.
            glob: Filename pattern, relative to *directory*.
            source_id: Identifier for the folder. Frames use the file name.
            start_times: Optional per-file overrides for frame-zero time.
        """

        self._directory = Path(directory)
        if not self._directory.is_dir():
            raise VideoSourceError(f"{self._directory}: directory not found")

        self._paths = sorted(
            path for path in self._directory.glob(glob) if path.is_file()
        )
        if not self._paths:
            raise VideoSourceError(f"{self._directory}: no files matched '{glob}'")

        self._source_id = source_id or self._directory.name
        self._start_times = dict(start_times or {})
        self._current: FileSource | None = None
        self._closed = False
        self._started = False

    @property
    def source_id(self) -> str:
        """Identifier for the folder sequence."""

        return self._source_id

    @property
    def paths(self) -> list[Path]:
        """The ordered clip paths."""

        return list(self._paths)

    def frames(self) -> Iterator[Frame]:
        """Yield frames from every clip, in filename order."""

        if self._started:
            raise VideoSourceError(f"{self._source_id}: source is single-use")
        if self._closed:
            raise VideoSourceError(f"{self._source_id}: source is closed")
        self._started = True

        global_index = 0
        for path in self._paths:
            if self._closed:
                return

            start_time = self._start_times.get(path.name)
            if start_time is None:
                start_time = parse_clip_start(path)
            if start_time is None:
                start_time = mtime_utc(path)
                logger.warning(
                    "%s: no record_YYYYMMDD_HHMMSS name; using mtime %s",
                    path.name,
                    start_time.isoformat(),
                )

            child = FileSource(
                path,
                start_time=start_time,
                source_id=path.name,
                window_id=path.name,
            )
            self._current = child
            try:
                for frame in child.frames():
                    if self._closed:
                        return
                    yield replace(frame, global_index=global_index)
                    global_index += 1
            except VideoSourceError as exc:
                if self._closed:
                    return
                logger.error("skipping %s: %s", path.name, exc)
            finally:
                child.close()
                self._current = None

    def close(self) -> None:
        """Close the active clip and prevent further iteration."""

        self._closed = True
        if self._current is not None:
            self._current.close()
            self._current = None

    def __enter__(self) -> FileSequenceSource:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
