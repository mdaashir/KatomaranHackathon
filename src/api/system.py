"""Host, GPU-provider and model snapshot for the System page."""

from __future__ import annotations

import platform
from pathlib import Path
import time

from src.config import AppConfig


class SystemModel:
    def __init__(
        self,
        config: AppConfig,
        *,
        db_path: str | Path,
        log_dir: str | Path,
        started_at: float,
    ) -> None:
        self._config = config
        self._db_path = Path(db_path)
        self._log_dir = Path(log_dir)
        self._started_at = started_at

    def snapshot(self) -> dict[str, object]:
        import onnxruntime
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(self._log_dir))
        process = psutil.Process()
        providers = onnxruntime.get_available_providers()
        return {
            "uptime_seconds": max(0.0, time.monotonic() - self._started_at),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=None),
                "logical_count": psutil.cpu_count(),
            },
            "memory": {
                "used_bytes": memory.used,
                "available_bytes": memory.available,
                "total_bytes": memory.total,
                "percent": memory.percent,
                "process_rss_bytes": process.memory_info().rss,
            },
            "disk": {
                "used_bytes": disk.used,
                "total_bytes": disk.total,
                "percent": disk.percent,
            },
            "gpu": {
                "available": "CUDAExecutionProvider" in providers,
                "providers": providers,
            },
            "models": {
                "detector": self._config.detector.model_path,
                "detector_device": self._config.detector.device,
                "recognition": self._config.recognition.model_name,
                "recognition_dir": self._config.recognition.model_dir,
                "recognition_device": self._config.recognition.device,
            },
            "database": {
                "path": str(self._db_path),
                "size_bytes": (
                    self._db_path.stat().st_size if self._db_path.is_file() else 0
                ),
            },
            "log_dir": str(self._log_dir),
        }
