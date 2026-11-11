"""Bounded queue that drops the oldest item instead of blocking a live feed."""

from __future__ import annotations

import queue
import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedFrameQueue(Generic[T]):
    """Thread-safe queue with a hard size limit and drop-oldest behaviour."""

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()

    def put_drop_oldest(self, item: T) -> None:
        """Insert *item*, evicting the oldest entry when the queue is full."""

        with self._lock:
            while True:
                try:
                    self._queue.put_nowait(item)
                    return
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        continue

    def get(self) -> T:
        """Block until an item is available."""

        return self._queue.get()

    def qsize(self) -> int:
        """Return the current number of queued items."""

        return self._queue.qsize()
