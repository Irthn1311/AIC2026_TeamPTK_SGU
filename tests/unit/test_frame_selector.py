from triage_eg.common.schemas import ShotRecord, VideoRecord
from triage_eg.frame_bank.selectors import CenterFrameSelector


def test_center_frame_selector_uses_actual_coordinates():
    video = VideoRecord("v", "v.mp4", "b", 25, 100, 4000, None, None, None, "v1")
    shot = ShotRecord("s", "v", 10, 20, 400, 800, "dummy", "0.1", None)
    first = CenterFrameSelector().select(video, [shot])[0]
    second = CenterFrameSelector().select(video, [shot])[0]
    assert first.actual_frame_id == 15
    assert first.timestamp_ms == 600
    assert first.frame_uid == second.frame_uid
    assert first.source_policy == "SHOT_CENTER"
