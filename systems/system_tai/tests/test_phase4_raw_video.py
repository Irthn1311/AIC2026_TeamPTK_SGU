from __future__ import annotations

from pathlib import Path

import pytest

from system_tai.refinement.video import (
    OpenCVVideoDecoder,
    RawVideoError,
    RawVideoRegistry,
    VideoProbe,
)


def test_raw_video_exact_stem_resolution_missing_and_extensions(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    (root / "L21_V001.mp4").touch()
    (root / "L21_V001-extra.mp4").touch()
    (root / "L21_V002.WEBM").touch()
    registry = RawVideoRegistry.from_bounded_roots(("L21_V001", "L21_V002", "L21_V003"), (root,))
    assert registry.get("L21_V001").raw_video_path.name == "L21_V001.mp4"
    assert registry.get("L21_V002").raw_video_path.name == "L21_V002.WEBM"
    assert registry.get("L21_V003").raw_video_path is None
    assert registry.get("L21_V003").warnings


def test_raw_video_ambiguous_paths_fail(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "L21_V001.mp4").touch()
    (second / "L21_V001.mkv").touch()
    with pytest.raises(RawVideoError, match="ambiguous"):
        RawVideoRegistry.from_bounded_roots(("L21_V001",), (first, second))


@pytest.mark.parametrize(
    ("fps", "count"),
    [(0.0, 10), (float("nan"), 10), (30.0, 0)],
)
def test_invalid_probe_metadata_rejected(tmp_path: Path, fps: float, count: int) -> None:
    with pytest.raises(RawVideoError):
        VideoProbe("L21_V001", tmp_path / "v.mp4", "fake", fps, count, 10, 10, 1)


class FakeCapture:
    def __init__(self, values, *, fail_at=None):
        self.values = values
        self.position = 0
        self.fail_at = fail_at
        self.released = False

    def isOpened(self):
        return True

    def get(self, key):
        if key == 5:
            return self.position
        return self.values[key]

    def set(self, _key, value):
        self.position = int(value)
        return True

    def read(self):
        if self.position == self.fail_at:
            return False, None
        value = self.position
        self.position += 1
        return True, value

    def release(self):
        self.released = True


class FakeCV2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_COUNT = 2
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_POS_FRAMES = 5

    def __init__(self):
        self.captures = []

    def VideoCapture(self, _path):
        capture = FakeCapture({1: 10, 2: 100, 3: 16, 4: 9})
        self.captures.append(capture)
        return capture


def test_opencv_probe_decode_tracks_absolute_ids_and_releases(tmp_path: Path) -> None:
    from system_tai.refinement.video import DecodeRequest, RawVideoRecord

    path = tmp_path / "L21_V001.mp4"
    path.touch()
    cv2 = FakeCV2()
    decoder = OpenCVVideoDecoder(cv2_module=cv2)
    probe = decoder.probe(RawVideoRecord("L21_V001", path))
    result = decoder.decode(DecodeRequest(probe, (5, 8), 10))
    assert [frame.absolute_frame_id for frame in result.frames] == [5, 8]
    assert [frame.image for frame in result.frames] == [5, 8]
    assert result.decoded_frame_count == 4
    assert all(capture.released for capture in cv2.captures)
