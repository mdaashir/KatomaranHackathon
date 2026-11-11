"""Tracking layer: ByteTrack wrapper and visit lifecycle."""

from src.tracking.lifecycle import TrackLifecycle
from src.tracking.tracker import ByteTrackAdapter
from src.tracking.types import (
    LifecycleSummary,
    Track,
    TrackedBox,
    TrackState,
    Visit,
    VisitEvent,
)

__all__ = [
    "ByteTrackAdapter",
    "LifecycleSummary",
    "Track",
    "TrackLifecycle",
    "TrackState",
    "TrackedBox",
    "Visit",
    "VisitEvent",
]
