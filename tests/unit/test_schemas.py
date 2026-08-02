from dataclasses import dataclass

import pytest

from triage_eg.common.schemas import FrameRecord, VideoRecord, dataclass_to_dict


def test_video_schema_validation():
    with pytest.raises(ValueError, match="fps"):
        VideoRecord("v", "v.mp4", "b", 0, 1, 1, None, None, None, "v1")


def test_frame_source_policy_validation():
    with pytest.raises(ValueError, match="source_policy"):
        FrameRecord("f", "v", 0, 0, "x.jpg", "UNKNOWN", None, "v1", "e1")


def test_dataclass_to_dict():
    @dataclass(frozen=True)
    class Example:
        value: int

    assert dataclass_to_dict(Example(3)) == {"value": 3}
