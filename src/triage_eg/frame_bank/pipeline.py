"""Composable shot-detection and retrieval-frame selection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from triage_eg.common.schemas import FrameRecord, ShotRecord, VideoRecord
from triage_eg.frame_bank.interfaces import FrameSelector, ShotDetector


@dataclass(frozen=True)
class FrameBankReport:
    """Aggregate frame-bank coverage statistics."""

    total_shots: int
    total_selected_frames: int
    frames_per_video: dict[str, int]
    average_frames_per_video: float


@dataclass(frozen=True)
class FrameBankResult:
    """Frame-bank records and their summary report."""

    shots: list[ShotRecord]
    frames: list[FrameRecord]
    report: FrameBankReport


class FrameBankPipeline:
    """Connect a replaceable shot detector and frame selector."""

    def __init__(self, detector: ShotDetector, selector: FrameSelector) -> None:
        self._detector = detector
        self._selector = selector

    def run(self, videos: list[VideoRecord]) -> FrameBankResult:
        """Build frame-bank metadata for all videos."""

        shots: list[ShotRecord] = []
        frames: list[FrameRecord] = []
        counts: dict[str, int] = {}
        for video in videos:
            video_shots = self._detector.detect(video)
            video_frames = self._selector.select(video, video_shots)
            shots.extend(video_shots)
            frames.extend(video_frames)
            counts[video.video_id] = len(video_frames)
        total_frames = len(frames)
        average = total_frames / len(videos) if videos else 0.0
        report = FrameBankReport(len(shots), total_frames, counts, average)
        return FrameBankResult(shots, frames, report)
