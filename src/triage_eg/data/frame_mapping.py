"""Conversions between actual source frame identifiers and timestamps."""

from triage_eg.common.schemas import VideoRecord


def frame_to_timestamp_ms(frame_id: int, fps: float) -> int:
    """Map a zero-based actual frame ID to its nearest timestamp in milliseconds."""

    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return round(frame_id * 1000 / fps)


def timestamp_ms_to_frame(timestamp_ms: int, fps: float) -> int:
    """Map a timestamp to its nearest zero-based actual source frame ID."""

    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    return round(timestamp_ms * fps / 1000)


def validate_frame_in_video(frame_id: int, video_record: VideoRecord) -> None:
    """Raise when an actual frame ID lies outside a video."""

    if not 0 <= frame_id < video_record.total_frames:
        raise ValueError(
            f"Frame {frame_id} is outside video {video_record.video_id} "
            f"with {video_record.total_frames} frames"
        )

