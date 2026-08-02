"""Replaceable frame-bank interfaces."""

from typing import Protocol

from triage_eg.common.schemas import FrameRecord, ShotRecord, VideoRecord


class ShotDetector(Protocol):
    """Contract for shot boundary detectors such as a future TransNetV2 adapter."""

    def detect(self, video: VideoRecord) -> list[ShotRecord]:
        """Detect shots for one video."""
        ...


class FrameSelector(Protocol):
    """Contract for selecting retrieval frames from detected shots."""

    def select(self, video: VideoRecord, shots: list[ShotRecord]) -> list[FrameRecord]:
        """Select retrieval frames and preserve actual source coordinates."""
        ...
