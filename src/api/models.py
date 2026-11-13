"""Request/response models for the control API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class JobRequest(BaseModel):
    """Start one pipeline job from a file, folder, RTSP stream or webcam."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["file", "folder", "rtsp", "webcam"]
    path: str | None = None
    url: str | None = None
    device_index: int | None = Field(None, ge=0, le=64)
