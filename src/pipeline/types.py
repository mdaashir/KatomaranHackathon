"""Pipeline result types."""

from __future__ import annotations

from dataclasses import dataclass

from src.counting.counter import CounterSummary


@dataclass(frozen=True)
class PipelineResult:
    """Summary of one pipeline run."""

    frames_processed: int
    detection_cycles: int
    stopped_by_signal: bool
    summary: CounterSummary

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly run summary."""

        return {
            "frames_processed": self.frames_processed,
            "detection_cycles": self.detection_cycles,
            "stopped_by_signal": self.stopped_by_signal,
            **self.summary.to_dict(),
        }
