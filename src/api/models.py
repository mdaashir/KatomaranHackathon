"""Request/response models for the control API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class JobRequest(BaseModel):
    """Start one pipeline job from a file, folder, or RTSP stream."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["file", "folder", "rtsp"]
    path: str | None = None
    url: str | None = None
