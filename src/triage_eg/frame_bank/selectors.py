"""Frame-selection policies for the retrieval frame bank."""

from dataclasses import dataclass
from hashlib import sha256

from triage_eg.common.schemas import FrameRecord, ShotRecord, SourcePolicy, VideoRecord
from triage_eg.data.frame_mapping import frame_to_timestamp_ms, validate_frame_in_video


@dataclass(frozen=True)
class CenterFrameSelector:
    """Select the integer center of every shot."""

    extraction_version: str = "0.1"

    def select(self, video: VideoRecord, shots: list[ShotRecord]) -> list[FrameRecord]:
        """Create stable metadata records; no image is decoded in v0.1."""

        frames: list[FrameRecord] = []
        for shot in shots:
            if shot.video_id != video.video_id:
                raise ValueError(f"Shot {shot.shot_id} does not belong to video {video.video_id}")
            center_frame = (shot.start_frame + shot.end_frame) // 2
            validate_frame_in_video(center_frame, video)
            identity = (
                f"{video.dataset_version}|{video.video_id}|{center_frame}|"
                f"{SourcePolicy.SHOT_CENTER}|{self.extraction_version}"
            )
            digest = sha256(identity.encode()).hexdigest()[:20]
            frames.append(
                FrameRecord(
                    frame_uid=f"frm_{digest}",
                    video_id=video.video_id,
                    actual_frame_id=center_frame,
                    timestamp_ms=frame_to_timestamp_ms(center_frame, video.fps),
                    image_path=f"frames/{video.video_id}/{center_frame:09d}.jpg",
                    source_policy=SourcePolicy.SHOT_CENTER,
                    shot_id=shot.shot_id,
                    dataset_version=video.dataset_version,
                    extraction_version=self.extraction_version,
                )
            )
        return frames
