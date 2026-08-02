import pytest

from triage_eg.common.schemas import VideoRecord
from triage_eg.data.frame_mapping import (
    frame_to_timestamp_ms,
    timestamp_ms_to_frame,
    validate_frame_in_video,
)


def test_frame_timestamp_round_trip():
    assert frame_to_timestamp_ms(50, 25.0) == 2000
    assert timestamp_ms_to_frame(2000, 25.0) == 50


def test_validate_actual_frame_id():
    video = VideoRecord("v", "v.mp4", "b", 25, 10, 400, None, None, None, "v1")
    validate_frame_in_video(9, video)
    with pytest.raises(ValueError):
        validate_frame_in_video(10, video)
