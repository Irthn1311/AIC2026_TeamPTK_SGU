"""Metadata-only shot detector used by the v0.1 executable template."""

from dataclasses import dataclass

from triage_eg.common.schemas import ShotRecord, VideoRecord
from triage_eg.data.frame_mapping import frame_to_timestamp_ms


@dataclass(frozen=True)
class DummyShotDetector:
    """Represent every non-empty video as exactly one shot."""

    detector_name: str = "dummy"
    detector_version: str = "0.1"

    def detect(self, video: VideoRecord) -> list[ShotRecord]:
        """Create one inclusive full-video shot without decoding media."""

        if video.total_frames == 0:
            return []
        end_frame = video.total_frames - 1
        return [
            ShotRecord(
                shot_id=f"{video.video_id}:shot:000000",
                video_id=video.video_id,
                start_frame=0,
                end_frame=end_frame,
                start_time_ms=0,
                end_time_ms=frame_to_timestamp_ms(end_frame, video.fps),
                detector_name=self.detector_name,
                detector_version=self.detector_version,
                confidence=None,
            )
        ]
